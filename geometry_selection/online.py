"""Reusable controller for gradient-free selection during denoising.

The controller owns stochastic branching and the conservative selection policy.
It does not know about a particular video generator or geometry backbone.  A
pipeline supplies an ``x0`` predictor, while a frozen geometry adapter supplies
``GeometryScoreReport`` objects for the decoded provisional videos.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol, Sequence

import torch

from .scorer import GeometryScoreReport
from .selection import CandidateScore, SelectionConfig, SelectionResult, select_candidate


def flow_match_predicted_x0(
    scheduler: Any,
    model_output: torch.Tensor,
    sample: torch.Tensor,
    *,
    step_index: int,
) -> torch.Tensor:
    """Predict ``x0`` without advancing a Diffusers flow-matching scheduler.

    Diffusers 0.38's :class:`FlowMatchEulerDiscreteScheduler` has no
    ``convert_model_output`` method.  Its own ``step`` implements the clean
    prediction as ``x_t - sigma_t * model_output`` before advancing the
    mutable scheduler index.  Online candidate evaluation must use this pure
    version so it cannot change the one scheduler transition of the main
    denoising trajectory.
    """

    if not isinstance(step_index, int) or isinstance(step_index, bool) or step_index < 0:
        raise ValueError("step_index must be a non-negative integer")
    sigmas = getattr(scheduler, "sigmas", None)
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1:
        raise TypeError("scheduler must expose one-dimensional tensor sigmas")
    if step_index >= sigmas.numel():
        raise IndexError("step_index exceeds scheduler sigmas")
    if sample.shape != model_output.shape:
        raise ValueError("sample and model_output must have the same shape")
    sigma = sigmas[step_index].to(device=sample.device, dtype=torch.float32)
    return sample.float() - sigma * model_output.float()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Content hash including tensor shape and dtype for decision provenance."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor_sha256 expects a torch.Tensor")
    value = tensor.detach().contiguous().cpu()
    header = f"{value.dtype}|{tuple(value.shape)}|".encode("utf-8")
    digest = hashlib.sha256(header)
    # NumPy does not support bfloat16.  A byte view preserves the exact tensor
    # representation for every PyTorch dtype while retaining dtype in header.
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class OnlineBranchConfig:
    """Frozen parameters for a branching event.

    ``candidate_count`` includes the unmodified incumbent.  Perturbations are
    RMS-normalised inside the mutable latent support, so the scale has a stable
    interpretation across resolution and sequence length.
    """

    selection_steps: tuple[int, ...]
    candidate_count: int = 3
    perturbation_scale: float = 0.01
    random_seed: int = 0

    def validate(self) -> None:
        if not self.selection_steps:
            raise ValueError("selection_steps must not be empty")
        if any(
            not isinstance(step, int) or isinstance(step, bool) or step < 0
            for step in self.selection_steps
        ):
            raise ValueError("selection_steps must contain non-negative integers")
        if tuple(sorted(set(self.selection_steps))) != self.selection_steps:
            raise ValueError("selection_steps must be sorted and unique")
        if not isinstance(self.candidate_count, int) or isinstance(self.candidate_count, bool):
            raise TypeError("candidate_count must be an integer")
        if self.candidate_count < 2:
            raise ValueError("candidate_count must include incumbent and one challenger")
        if not math.isfinite(self.perturbation_scale) or self.perturbation_scale <= 0.0:
            raise ValueError("perturbation_scale must be finite and positive")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise TypeError("random_seed must be an integer")


@dataclass(frozen=True)
class OnlineSelectionContext:
    """Generator-owned state exposed at one pre-scheduler selection point."""

    step_index: int
    timestep: Any
    incumbent_latents: torch.Tensor
    incumbent_x0: torch.Tensor
    mutable_mask: torch.Tensor | None
    predict_x0: Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class OnlineCandidate:
    candidate_id: str
    latents: torch.Tensor
    x0: torch.Tensor
    is_incumbent: bool
    perturbation_rms: float
    branch_seed: int | None
    noise_sha256: str | None
    latent_sha256: str
    x0_sha256: str


class GeometryReportFn(Protocol):
    """Score decoded provisional videos while preserving candidate order."""

    def __call__(
        self,
        candidates: Sequence[OnlineCandidate],
        context: OnlineSelectionContext,
    ) -> Sequence[GeometryScoreReport]: ...


@dataclass(frozen=True)
class OnlineSchedulerTransition:
    """Recomputed model output bound to one selected latent state.

    It prevents the easy-to-miss error of applying a scheduler update with a
    latent selected by the online controller but a model output predicted from
    the earlier incumbent state.
    """

    selected_latents: torch.Tensor
    model_output: torch.Tensor
    step_index: int


def prepare_online_scheduler_transition(
    *,
    selected_latents: torch.Tensor,
    predict_model_output: Callable[[torch.Tensor], torch.Tensor],
    step_index: int,
) -> OnlineSchedulerTransition:
    """Recompute the denoiser output for the exact latent selected online."""

    if not isinstance(selected_latents, torch.Tensor):
        raise TypeError("selected_latents must be a torch.Tensor")
    if not callable(predict_model_output):
        raise TypeError("predict_model_output must be callable")
    if not isinstance(step_index, int) or isinstance(step_index, bool) or step_index < 0:
        raise ValueError("step_index must be a non-negative integer")
    model_output = predict_model_output(selected_latents)
    if not isinstance(model_output, torch.Tensor):
        raise TypeError("selected-branch model output must be a torch.Tensor")
    if model_output.shape != selected_latents.shape or model_output.device != selected_latents.device:
        raise ValueError("selected-branch model output must match selected latents")
    if not bool(torch.isfinite(model_output).all()):
        raise ValueError("selected-branch model output must be finite")
    return OnlineSchedulerTransition(
        selected_latents=selected_latents.detach(),
        model_output=model_output.detach(),
        step_index=step_index,
    )


def finish_online_scheduler_transition(
    *,
    scheduler: Any,
    timestep: Any,
    transition: OnlineSchedulerTransition,
    selector: Any,
) -> torch.Tensor:
    """Advance exactly once and record the state produced by a selection."""

    if not isinstance(transition, OnlineSchedulerTransition):
        raise TypeError("transition must be an OnlineSchedulerTransition")
    if not callable(getattr(selector, "record_scheduler_output", None)):
        raise TypeError("online selector must record the scheduler output")
    result = scheduler.step(
        transition.model_output,
        timestep,
        transition.selected_latents,
        return_dict=False,
    )
    if not isinstance(result, tuple) or len(result) < 1 or not isinstance(result[0], torch.Tensor):
        raise TypeError("scheduler.step must return a tensor as its first tuple element")
    next_latents = result[0]
    if (
        next_latents.shape != transition.selected_latents.shape
        or next_latents.device != transition.selected_latents.device
        or not bool(torch.isfinite(next_latents).all())
    ):
        raise ValueError("scheduler returned invalid selected-branch latents")
    selector.record_scheduler_output(transition.step_index, next_latents)
    return next_latents


@dataclass(frozen=True)
class OnlineSelectionOutcome:
    selected_latents: torch.Tensor
    selection: SelectionResult
    step_index: int
    timestep: int | float | str
    parent_latent_sha256: str
    selected_latent_sha256: str
    candidate_perturbation_rms: dict[str, float]
    candidates: tuple[OnlineCandidate, ...]
    geometry_artifacts: tuple[dict[str, Any], ...]

    def metadata(self) -> dict[str, Any]:
        # Keep the complete geometry reports, not only their scalar scores.
        # The temporary decoded frames are normally deleted, therefore the
        # serialized reports and graph diagnostics are the decision evidence
        # required to reproduce or audit a selection event afterwards.
        return {
            "step_index": self.step_index,
            "timestep": self.timestep,
            "parent_latent_sha256": self.parent_latent_sha256,
            "selected_latent_sha256": self.selected_latent_sha256,
            "selected_candidate_id": self.selection.selected_candidate_id,
            "incumbent_candidate_id": self.selection.incumbent_candidate_id,
            "decision": self.selection.decision,
            "score_improvement": self.selection.score_improvement,
            "motion_ratio": self.selection.motion_ratio,
            "common_local_edges": self.selection.common_local_edges,
            "common_long_range_edges": self.selection.common_long_range_edges,
            "comparable_scores": self.selection.comparable_scores,
            "candidate_perturbation_rms": self.candidate_perturbation_rms,
            "geometry_artifacts": list(self.geometry_artifacts),
            "candidates": {
                candidate.candidate_id: {
                    "branch_seed": candidate.branch_seed,
                    "noise_sha256": candidate.noise_sha256,
                    "latent_sha256": candidate.latent_sha256,
                    "x0_sha256": candidate.x0_sha256,
                }
                for candidate in self.candidates
            },
            "selection": {
                "selected_candidate_id": self.selection.selected_candidate_id,
                "incumbent_candidate_id": self.selection.incumbent_candidate_id,
                "decision": self.selection.decision,
                "score_improvement": self.selection.score_improvement,
                "motion_ratio": self.selection.motion_ratio,
                "common_local_edges": self.selection.common_local_edges,
                "common_long_range_edges": self.selection.common_long_range_edges,
                "comparable_scores": self.selection.comparable_scores,
                "candidate_ids": [
                    candidate.candidate_id for candidate in self.selection.candidates
                ],
            },
            "candidate_reports": {
                candidate.candidate_id: candidate.report.to_dict()
                for candidate in self.selection.candidates
            },
        }


def _candidate_seed(random_seed: int, step_index: int, candidate_index: int) -> int:
    """Stable seed that does not depend on Python's salted ``hash``."""

    payload = f"online-geometry-selection:{random_seed}:{step_index}:{candidate_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _masked_rms(tensor: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    values = tensor.float()
    if mask is None:
        return torch.sqrt(values.square().mean()).clamp_min(1e-8)
    support = mask.to(device=values.device, dtype=values.dtype)
    if support.shape != values.shape:
        support = torch.broadcast_to(support, values.shape)
    count = support.sum()
    if float(count.detach().cpu()) <= 0.0:
        raise ValueError("mutable_mask has empty support")
    return torch.sqrt((values.square() * support).sum() / count).clamp_min(1e-8)


class OnlineGeometrySelectionController:
    """Branch latents, score their predicted clean videos, and select safely."""

    def __init__(
        self,
        branch_config: OnlineBranchConfig,
        selection_config: SelectionConfig,
        report_fn: GeometryReportFn,
    ) -> None:
        branch_config.validate()
        selection_config.validate()
        self.branch_config = branch_config
        self.selection_config = selection_config
        self.report_fn = report_fn
        self.events: list[dict[str, Any]] = []

    def is_active(self, step_index: int) -> bool:
        return step_index in self.branch_config.selection_steps

    def _branch(self, context: OnlineSelectionContext) -> tuple[OnlineCandidate, ...]:
        incumbent = context.incumbent_latents.detach()
        candidates: list[OnlineCandidate] = [
            OnlineCandidate(
                candidate_id="incumbent",
                latents=incumbent,
                x0=context.incumbent_x0.detach(),
                is_incumbent=True,
                perturbation_rms=0.0,
                branch_seed=None,
                noise_sha256=None,
                latent_sha256=tensor_sha256(incumbent),
                x0_sha256=tensor_sha256(context.incumbent_x0),
            )
        ]
        latent_rms = _masked_rms(incumbent, context.mutable_mask)
        for index in range(1, self.branch_config.candidate_count):
            generator = torch.Generator(device=incumbent.device)
            branch_seed = _candidate_seed(
                self.branch_config.random_seed, context.step_index, index
            )
            generator.manual_seed(branch_seed)
            noise = torch.randn(
                incumbent.shape,
                dtype=incumbent.dtype,
                device=incumbent.device,
                generator=generator,
            )
            if context.mutable_mask is not None:
                mask = context.mutable_mask.to(device=noise.device, dtype=noise.dtype)
                noise = noise * torch.broadcast_to(mask, noise.shape)
            noise_rms = _masked_rms(noise, context.mutable_mask)
            delta = noise * (
                self.branch_config.perturbation_scale * latent_rms / noise_rms
            )
            candidate_latents = (incumbent + delta).detach()
            x0 = context.predict_x0(candidate_latents).detach()
            candidates.append(
                OnlineCandidate(
                    candidate_id=f"branch_{index:02d}",
                    latents=candidate_latents,
                    x0=x0,
                    is_incumbent=False,
                    perturbation_rms=float(_masked_rms(delta, context.mutable_mask).cpu()),
                    branch_seed=branch_seed,
                    noise_sha256=tensor_sha256(noise),
                    latent_sha256=tensor_sha256(candidate_latents),
                    x0_sha256=tensor_sha256(x0),
                )
            )
        return tuple(candidates)

    def __call__(self, context: OnlineSelectionContext) -> OnlineSelectionOutcome:
        if not self.is_active(context.step_index):
            raise ValueError(f"controller is inactive at denoising step {context.step_index}")
        candidates = self._branch(context)
        reports = tuple(self.report_fn(candidates, context))
        if len(reports) != len(candidates):
            raise ValueError("geometry report function must return one report per candidate")
        artifacts = getattr(self.report_fn, "last_artifacts", ())
        if not isinstance(artifacts, (list, tuple)) or not all(
            isinstance(value, dict) for value in artifacts
        ):
            raise TypeError("geometry report function has invalid last_artifacts")
        scored = [
            CandidateScore(
                candidate_id=candidate.candidate_id,
                seed=index,
                # Online candidates have no exported video or reusable cache
                # key.  Their actual x0 and latent content hashes are logged
                # in OnlineSelectionOutcome.metadata instead.
                video_sha256="",
                geometry_cache_key="",
                is_incumbent=candidate.is_incumbent,
                report=report,
            )
            for index, (candidate, report) in enumerate(zip(candidates, reports, strict=True))
        ]
        selection = select_candidate(scored, self.selection_config)
        selected = next(
            candidate
            for candidate in candidates
            if candidate.candidate_id == selection.selected_candidate_id
        )
        timestep = context.timestep
        if hasattr(timestep, "item"):
            timestep = timestep.item()
        outcome = OnlineSelectionOutcome(
            selected_latents=selected.latents.detach(),
            selection=selection,
            step_index=context.step_index,
            timestep=timestep,
            parent_latent_sha256=tensor_sha256(context.incumbent_latents),
            selected_latent_sha256=tensor_sha256(selected.latents),
            candidate_perturbation_rms={
                candidate.candidate_id: candidate.perturbation_rms
                for candidate in candidates
            },
            candidates=candidates,
            geometry_artifacts=tuple(artifacts),
        )
        event = outcome.metadata()
        # The full frozen thresholds are needed to interpret an abstention or
        # a motion-guard rejection independently of the original YAML file.
        event["selection_config"] = asdict(self.selection_config)
        self.events.append(event)
        return outcome

    def record_scheduler_output(self, step_index: int, next_latents: torch.Tensor) -> None:
        """Bind a logged selection to the next denoising state it actually produced."""

        if not self.events or self.events[-1].get("step_index") != step_index:
            raise ValueError("scheduler output does not match the latest selection event")
        self.events[-1]["next_step_input_latent_sha256"] = tensor_sha256(next_latents)
