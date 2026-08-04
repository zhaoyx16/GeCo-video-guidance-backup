from __future__ import annotations

import json

import pytest
import torch

from geometry_selection.online import (
    OnlineBranchConfig,
    OnlineGeometrySelectionController,
    OnlineSelectionContext,
    flow_match_predicted_x0,
)
from geometry_selection.scorer import GeometryScoreReport, PairScore, ScorerConfig
from geometry_selection.selection import SelectionConfig


def _report(score: float, motion: float) -> GeometryScoreReport:
    pairs = (
        PairScore(0, 1, 1, 0.8, 0.8, 100, score, 0.0, score, "ok"),
        PairScore(1, 0, 1, 0.8, 0.8, 100, score, 0.0, score, "ok"),
        PairScore(1, 2, 1, 0.8, 0.8, 100, score, 0.0, score, "ok"),
        PairScore(2, 1, 1, 0.8, 0.8, 100, score, 0.0, score, "ok"),
        PairScore(0, 3, 3, 0.5, 0.5, 100, score, 0.0, score, "ok"),
        PairScore(3, 0, 3, 0.5, 0.5, 100, score, 0.0, score, "ok"),
    )
    return GeometryScoreReport(
        total_score=score,
        local_score=score,
        long_range_score=score,
        camera_path_length=motion,
        normalized_translation_motion=motion,
        camera_angular_path_deg=10.0,
        normalized_camera_motion=motion,
        normalized_net_translation_motion=motion,
        local_edge_fraction=1.0,
        long_range_edge_fraction=1.0,
        valid_local_edges=2,
        valid_long_range_edges=1,
        status="ok",
        pairs=pairs,
        config=ScorerConfig(require_long_range=False),
        keyframe_indices=(0, 1, 2, 3),
    )


def _context(*, mask: torch.Tensor | None = None) -> tuple[OnlineSelectionContext, list[torch.Tensor]]:
    seen: list[torch.Tensor] = []
    incumbent = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)

    def predict_x0(latents: torch.Tensor) -> torch.Tensor:
        seen.append(latents.detach().clone())
        return latents * 0.5

    return (
        OnlineSelectionContext(
            step_index=7,
            timestep=321,
            incumbent_latents=incumbent,
            incumbent_x0=incumbent * 0.5,
            mutable_mask=mask,
            predict_x0=predict_x0,
        ),
        seen,
    )


def test_branch_config_rejects_ambiguous_or_invalid_schedule() -> None:
    with pytest.raises(ValueError, match="sorted and unique"):
        OnlineBranchConfig((8, 7)).validate()
    with pytest.raises(ValueError, match="include incumbent"):
        OnlineBranchConfig((7,), candidate_count=1).validate()
    with pytest.raises(ValueError, match="positive"):
        OnlineBranchConfig((7,), perturbation_scale=0.0).validate()


def test_flow_match_x0_is_pure_and_matches_the_scheduler_formula() -> None:
    class Scheduler:
        sigmas = torch.tensor([1.0, 0.4, 0.0])
        step_index = None

    sample = torch.tensor([2.0, -1.0], dtype=torch.bfloat16)
    velocity = torch.tensor([0.5, -0.25], dtype=torch.bfloat16)
    predicted = flow_match_predicted_x0(Scheduler(), velocity, sample, step_index=1)

    assert torch.allclose(predicted, torch.tensor([1.8, -0.9]), atol=1e-3)
    assert Scheduler.step_index is None
    with pytest.raises(IndexError, match="exceeds"):
        flow_match_predicted_x0(Scheduler(), velocity, sample, step_index=3)


def test_controller_branches_deterministically_and_selects_geometry_winner() -> None:
    context, calls = _context()

    def reports(candidates, _context):
        assert [candidate.candidate_id for candidate in candidates] == [
            "incumbent",
            "branch_01",
            "branch_02",
        ]
        return [_report(0.30, 1.0), _report(0.10, 1.0), _report(0.20, 1.0)]

    controller = OnlineGeometrySelectionController(
        OnlineBranchConfig((7,), candidate_count=3, perturbation_scale=0.02, random_seed=13),
        SelectionConfig(min_relative_improvement=0.01),
        reports,
    )
    outcome = controller(context)

    assert len(calls) == 2  # incumbent reuses the already computed x0 prediction.
    assert outcome.selection.selected_candidate_id == "branch_01"
    assert outcome.selection.decision == "select_geometry_best"
    assert not torch.equal(outcome.selected_latents, context.incumbent_latents)
    assert torch.equal(context.incumbent_latents, torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2))
    assert controller.events == [outcome.metadata()]
    assert outcome.parent_latent_sha256 == outcome.candidates[0].latent_sha256
    assert outcome.selected_latent_sha256 == outcome.candidates[1].latent_sha256
    assert outcome.candidates[1].branch_seed is not None
    assert outcome.candidates[1].noise_sha256 is not None
    controller.record_scheduler_output(7, outcome.selected_latents + 1.0)
    assert "next_step_input_latent_sha256" in controller.events[0]
    json.dumps(outcome.metadata(), allow_nan=False)


def test_branch_respects_immutable_conditioning_support_and_abstains() -> None:
    mask = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
    mask[:, :, 1] = 1.0
    context, calls = _context(mask=mask)

    controller = OnlineGeometrySelectionController(
        OnlineBranchConfig((7,), candidate_count=3, perturbation_scale=0.05, random_seed=7),
        SelectionConfig(min_relative_improvement=0.01),
        lambda candidates, _: [_report(0.30, 1.0), _report(0.01, 0.10), _report(0.02, 0.20)],
    )
    outcome = controller(context)

    assert outcome.selection.decision == "abstain_margin_or_motion_guard"
    assert outcome.selection.selected_candidate_id == "incumbent"
    assert torch.equal(outcome.selected_latents, context.incumbent_latents)
    for candidate in calls:
        assert torch.equal(candidate[:, :, 0], context.incumbent_latents[:, :, 0])


def test_controller_rejects_calls_outside_its_frozen_schedule() -> None:
    context, _ = _context()
    inactive = OnlineSelectionContext(
        **{**context.__dict__, "step_index": 6}
    )
    controller = OnlineGeometrySelectionController(
        OnlineBranchConfig((7,)),
        SelectionConfig(),
        lambda candidates, _: [_report(0.2, 1.0) for _ in candidates],
    )
    assert controller.is_active(7)
    assert not controller.is_active(6)
    with pytest.raises(ValueError, match="inactive"):
        controller(inactive)
