"""Convert independent-window pose-graph consistency into a ranking report."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Sequence

import numpy as np

from .cache import canonical_hash
from .graph import (
    PoseGraphConfig,
    RelativePoseMeasurement,
    invert_se3,
    optimize_pose_graph,
    pose_edge_residual,
)
from .scorer import GeometryScoreReport, PairScore, ScorerConfig, score_geometry
from .schema import GeometryPrediction
from .window_graph import (
    IndependentWindow,
    InsufficientGraphEvidenceError,
    WindowGraphConfig,
    build_window_graph_measurements,
)


@dataclass(frozen=True)
class GraphScoreConfig:
    window: WindowGraphConfig = field(default_factory=WindowGraphConfig)
    optimizer: PoseGraphConfig = field(default_factory=PoseGraphConfig)
    switch_penalty_weight: float = 0.25
    # Candidate comparison already requires identical accepted evidence.
    # Non-zero missing-edge offsets can only distort relative improvements.
    missing_loop_penalty_weight: float = 0.0
    missing_local_penalty_weight: float = 0.0
    missing_scale_penalty_weight: float = 0.0
    scale_residual_weight: float = 0.10
    # Held-out mode fits one deterministic observation per adjacent frame pair,
    # then scores the remaining overlapping-window observations and every loop
    # edge without allowing those observations to influence the fitted poses.
    # This prevents a graph from receiving a low score merely by fitting the
    # same constraints that are later used for evaluation.
    use_heldout_residuals: bool = False
    heldout_local_weight: float = 1.0
    heldout_loop_weight: float = 0.5
    require_convergence: bool = True

    def validate(self) -> None:
        self.window.validate()
        self.optimizer.validate()
        for name in (
            "switch_penalty_weight",
            "missing_loop_penalty_weight",
            "missing_local_penalty_weight",
            "missing_scale_penalty_weight",
            "scale_residual_weight",
            "heldout_local_weight",
            "heldout_loop_weight",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(self.require_convergence, bool):
            raise TypeError("require_convergence must be boolean")
        if not isinstance(self.use_heldout_residuals, bool):
            raise TypeError("use_heldout_residuals must be boolean")


def _scaled_residual(
    states: np.ndarray,
    edge: RelativePoseMeasurement,
    config: PoseGraphConfig,
) -> np.ndarray:
    residual = pose_edge_residual(
        states[edge.source],
        states[edge.target],
        edge.target_from_source,
    )
    residual = residual.copy()
    residual[:3] /= config.translation_scale
    residual[3:] /= config.rotation_scale
    return residual


def _pair_report(
    edge: RelativePoseMeasurement,
    states: np.ndarray,
    node_frames: tuple[int, ...],
    config: PoseGraphConfig,
) -> PairScore:
    residual = _scaled_residual(states, edge, config)
    translation = float(np.linalg.norm(residual[:3]))
    rotation = float(np.linalg.norm(residual[3:]))
    return PairScore(
        source=edge.source,
        target=edge.target,
        temporal_gap=abs(node_frames[edge.target] - node_frames[edge.source]),
        overlap=float(np.clip(edge.confidence, 0.0, 1.0)),
        comparable_fraction=float(np.clip(edge.confidence, 0.0, 1.0)),
        valid_pixels=1,
        depth_error=translation,
        cycle_error=rotation,
        score=float(np.linalg.norm(residual)),
        status="ok",
        mean_confidence_evidence=edge.confidence,
    )


def _initial_states_from_local_edges(
    node_count: int,
    edges: Sequence[RelativePoseMeasurement],
) -> np.ndarray:
    """Build a deterministic local-chain initialization without held-out edges."""

    states: list[np.ndarray | None] = [None] * node_count
    states[0] = np.eye(4, dtype=np.float64)
    adjacency: list[list[tuple[str, int, np.ndarray]]] = [
        [] for _ in range(node_count)
    ]
    for edge in edges:
        adjacency[edge.source].append(
            (edge.provenance, edge.target, invert_se3(edge.target_from_source))
        )
        adjacency[edge.target].append(
            (edge.provenance, edge.source, edge.target_from_source)
        )
    queue = [0]
    while queue:
        source = queue.pop(0)
        assert states[source] is not None
        for _, target, source_world_to_target_world in sorted(adjacency[source]):
            if states[target] is not None:
                continue
            states[target] = states[source] @ source_world_to_target_world
            queue.append(target)
    if any(state is None for state in states):
        raise InsufficientGraphEvidenceError(
            "held-out fit edges do not connect all graph nodes"
        )
    return np.stack(states)


def _partition_local_edges(
    edges: Sequence[RelativePoseMeasurement],
) -> tuple[tuple[RelativePoseMeasurement, ...], tuple[tuple[RelativePoseMeasurement, RelativePoseMeasurement], ...]]:
    """Keep one edge per frame pair for fitting and hold duplicate estimates out."""

    groups: dict[tuple[int, int], list[RelativePoseMeasurement]] = {}
    for edge in edges:
        groups.setdefault((edge.source, edge.target), []).append(edge)
    fit: list[RelativePoseMeasurement] = []
    heldout: list[tuple[RelativePoseMeasurement, RelativePoseMeasurement]] = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda edge: edge.provenance)
        reference = ordered[0]
        fit.append(reference)
        heldout.extend((reference, alternate) for alternate in ordered[1:])
    return tuple(fit), tuple(heldout)


def _heldout_local_pair_report(
    reference: RelativePoseMeasurement,
    alternate: RelativePoseMeasurement,
    node_frames: tuple[int, ...],
    config: PoseGraphConfig,
) -> PairScore:
    states = np.repeat(np.eye(4, dtype=np.float64)[None], len(node_frames), axis=0)
    states[reference.target] = invert_se3(reference.target_from_source)
    report = _pair_report(alternate, states, node_frames, config)
    return replace(
        report,
        overlap=min(reference.confidence, alternate.confidence),
        comparable_fraction=min(reference.confidence, alternate.confidence),
        mean_confidence_evidence=min(reference.confidence, alternate.confidence),
    )


def _residual_rms(pairs: Sequence[PairScore]) -> float | None:
    values = [pair.score for pair in pairs if pair.status == "ok" and np.isfinite(pair.score)]
    if not values:
        return None
    return float(np.sqrt(np.mean(np.square(values))))


def score_window_pose_graph(
    global_prediction: GeometryPrediction,
    windows: Sequence[IndependentWindow],
    *,
    direct_config: ScorerConfig | None = None,
    graph_config: GraphScoreConfig | None = None,
) -> GeometryScoreReport:
    """Score one candidate using separately estimated local/loop windows.

    ``global_prediction`` is used only for camera-motion diagnostics and the
    direct-score ablation fields.  Every pose-graph measurement is constructed
    from ``windows`` and never from that global prediction.
    """

    resolved = graph_config or GraphScoreConfig()
    resolved.validate()
    direct = direct_config or ScorerConfig()
    base = score_geometry(global_prediction, direct)
    try:
        measurements = build_window_graph_measurements(windows, resolved.window)
    except InsufficientGraphEvidenceError as error:
        return replace(
            base,
            total_score=float("inf"),
            local_score=None,
            long_range_score=None,
            status="invalid_insufficient_graph_evidence",
            pairs=(),
            score_kind="pose_graph",
            graph_diagnostics={"error": str(error)},
        )
    heldout_local_pairs: tuple[PairScore, ...] = ()
    if resolved.use_heldout_residuals:
        fit_local, heldout_local = _partition_local_edges(
            measurements.local_measurements
        )
        fit_initial = _initial_states_from_local_edges(
            len(measurements.node_frame_indices), fit_local
        )
        # Loop measurements are validation evidence in held-out mode.  They
        # never participate in pose fitting or switch-variable optimization.
        optimized = optimize_pose_graph(
            fit_initial,
            fit_local,
            (),
            resolved.optimizer,
        )
        heldout_local_pairs = tuple(
            _heldout_local_pair_report(
                reference,
                alternate,
                measurements.node_frame_indices,
                resolved.optimizer,
            )
            for reference, alternate in heldout_local
        )
    else:
        fit_local = measurements.local_measurements
        optimized = optimize_pose_graph(
            measurements.initial_world_from_camera,
            measurements.local_measurements,
            measurements.loop_measurements,
            resolved.optimizer,
        )
    missing_loops = max(
        measurements.potential_loop_edges - len(measurements.loop_measurements),
        0,
    )
    missing_fraction = missing_loops / max(measurements.potential_loop_edges, 1)
    missing_local_fraction = (
        measurements.potential_local_edges - len(measurements.local_measurements)
    ) / max(measurements.potential_local_edges, 1)
    missing_scale_fraction = (
        measurements.potential_scale_constraints - len(measurements.scale_constraints)
    ) / max(measurements.potential_scale_constraints, 1)
    switch_penalty = (
        float(np.mean([(1.0 - value) ** 2 for value in optimized.loop_switches]))
        if optimized.loop_switches
        else 0.0
    )
    loop_pairs = tuple(
        _pair_report(
            edge,
            optimized.optimized_world_from_camera,
            measurements.node_frame_indices,
            resolved.optimizer,
        )
        for edge in measurements.loop_measurements
    )
    heldout_local_rms = _residual_rms(heldout_local_pairs)
    heldout_loop_rms = _residual_rms(loop_pairs)
    if resolved.use_heldout_residuals:
        components = []
        if heldout_local_rms is not None and resolved.heldout_local_weight > 0:
            components.append((resolved.heldout_local_weight, heldout_local_rms))
        if heldout_loop_rms is not None and resolved.heldout_loop_weight > 0:
            components.append((resolved.heldout_loop_weight, heldout_loop_rms))
        if components:
            total_score = sum(weight * value for weight, value in components) / sum(
                weight for weight, _ in components
            )
        else:
            total_score = float("inf")
        total_score += (
            resolved.missing_loop_penalty_weight * missing_fraction
            + resolved.missing_local_penalty_weight * missing_local_fraction
            + resolved.missing_scale_penalty_weight * missing_scale_fraction
            + resolved.scale_residual_weight * measurements.scale_residual_rms
        )
    else:
        total_score = (
            optimized.normalized_cost
            + resolved.switch_penalty_weight * switch_penalty
            + resolved.missing_loop_penalty_weight * missing_fraction
            + resolved.missing_local_penalty_weight * missing_local_fraction
            + resolved.missing_scale_penalty_weight * missing_scale_fraction
            + resolved.scale_residual_weight * measurements.scale_residual_rms
        )
    if optimized.degenerate:
        status = "invalid_graph_degenerate"
        total_score = float("inf")
    elif resolved.window.require_loop_edges and not measurements.loop_measurements:
        status = "invalid_no_supported_loop_edges"
        total_score = float("inf")
    elif resolved.use_heldout_residuals and not np.isfinite(total_score):
        status = "invalid_no_heldout_graph_evidence"
    elif resolved.require_convergence and not optimized.converged:
        status = "invalid_graph_not_converged"
        total_score = float("inf")
    else:
        status = (
            "ok_pose_graph_heldout"
            if resolved.use_heldout_residuals
            else "ok_pose_graph"
        )

    local_pairs = (
        heldout_local_pairs
        if resolved.use_heldout_residuals
        else tuple(
            _pair_report(
                edge,
                optimized.optimized_world_from_camera,
                measurements.node_frame_indices,
                resolved.optimizer,
            )
            for edge in measurements.local_measurements
        )
    )
    diagnostics = {
        "graph_score_config_sha256": canonical_hash(asdict(resolved)),
        "optimizer": {
            key: value
            for key, value in asdict(optimized).items()
            if key != "optimized_world_from_camera"
        },
        "window_scale_ids": list(measurements.window_scale_ids),
        "window_scales": list(measurements.window_scales),
        "candidate_depth_normalizer": measurements.candidate_depth_normalizer,
        "scale_residual_rms": measurements.scale_residual_rms,
        "scale_rank": measurements.scale_rank,
        "potential_loop_edges": measurements.potential_loop_edges,
        "potential_loop_edge_ids": list(measurements.potential_loop_edge_ids),
        "accepted_loop_edges": len(measurements.loop_measurements),
        "accepted_loop_edge_ids": list(measurements.accepted_loop_edge_ids),
        "missing_loop_fraction": missing_fraction,
        "potential_local_edges": measurements.potential_local_edges,
        "potential_local_edge_ids": list(measurements.potential_local_edge_ids),
        "accepted_local_edges": len(measurements.local_measurements),
        "accepted_local_edge_ids": list(measurements.accepted_local_edge_ids),
        "missing_local_fraction": missing_local_fraction,
        "potential_scale_constraints": measurements.potential_scale_constraints,
        "potential_scale_constraint_ids": list(
            measurements.potential_scale_constraint_ids
        ),
        "accepted_scale_constraints": len(measurements.scale_constraints),
        "accepted_scale_constraint_ids": list(
            measurements.accepted_scale_constraint_ids
        ),
        "missing_scale_fraction": missing_scale_fraction,
        "switch_penalty": switch_penalty,
        "score_evidence_mode": (
            "heldout" if resolved.use_heldout_residuals else "in_sample"
        ),
        "fit_local_edge_provenance": [edge.provenance for edge in fit_local],
        "heldout_local_edge_count": len(heldout_local_pairs),
        "heldout_local_residual_rms": heldout_local_rms,
        "heldout_loop_edge_count": (
            len(loop_pairs) if resolved.use_heldout_residuals else 0
        ),
        "heldout_loop_residual_rms": (
            heldout_loop_rms if resolved.use_heldout_residuals else None
        ),
        "rejected_loop_edges": [asdict(item) for item in measurements.rejected_loop_edges],
    }
    return replace(
        base,
        total_score=float(total_score),
        local_score=(
            heldout_local_rms
            if resolved.use_heldout_residuals
            else optimized.local_residual_rms
        ),
        long_range_score=(
            heldout_loop_rms
            if resolved.use_heldout_residuals
            else optimized.loop_residual_rms
        ),
        local_edge_fraction=(
            len(measurements.local_measurements)
            / max(measurements.potential_local_edges, 1)
        ),
        long_range_edge_fraction=(
            len(measurements.loop_measurements)
            / max(measurements.potential_loop_edges, 1)
        ),
        valid_local_edges=(
            len(heldout_local_pairs)
            if resolved.use_heldout_residuals
            else len(measurements.local_measurements)
        ),
        valid_long_range_edges=len(measurements.loop_measurements),
        status=status,
        pairs=local_pairs + loop_pairs,
        keyframe_indices=measurements.node_frame_indices,
        score_kind="pose_graph",
        potential_local_edges=measurements.potential_local_edges,
        potential_long_range_edges=measurements.potential_loop_edges,
        graph_diagnostics=diagnostics,
    )
