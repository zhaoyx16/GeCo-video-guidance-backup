"""Convert independent-window pose-graph consistency into a ranking report."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Sequence

import numpy as np

from .graph import (
    PoseGraphConfig,
    RelativePoseMeasurement,
    optimize_pose_graph,
    pose_edge_residual,
)
from .scorer import GeometryScoreReport, PairScore, ScorerConfig, score_geometry
from .schema import GeometryPrediction
from .window_graph import (
    IndependentWindow,
    WindowGraphConfig,
    build_window_graph_measurements,
)


@dataclass(frozen=True)
class GraphScoreConfig:
    window: WindowGraphConfig = field(default_factory=WindowGraphConfig)
    optimizer: PoseGraphConfig = field(default_factory=PoseGraphConfig)
    switch_penalty_weight: float = 0.25
    missing_loop_penalty_weight: float = 0.25
    scale_residual_weight: float = 0.10
    require_convergence: bool = True

    def validate(self) -> None:
        self.window.validate()
        self.optimizer.validate()
        for name in (
            "switch_penalty_weight",
            "missing_loop_penalty_weight",
            "scale_residual_weight",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(self.require_convergence, bool):
            raise TypeError("require_convergence must be boolean")


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
    measurements = build_window_graph_measurements(windows, resolved.window)
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
    switch_penalty = (
        float(np.mean([(1.0 - value) ** 2 for value in optimized.loop_switches]))
        if optimized.loop_switches
        else 0.0
    )
    total_score = (
        optimized.normalized_cost
        + resolved.switch_penalty_weight * switch_penalty
        + resolved.missing_loop_penalty_weight * missing_fraction
        + resolved.scale_residual_weight * measurements.scale_residual_rms
    )
    if optimized.degenerate:
        status = "invalid_graph_degenerate"
        total_score = float("inf")
    elif resolved.window.require_loop_edges and not measurements.loop_measurements:
        status = "invalid_no_supported_loop_edges"
        total_score = float("inf")
    elif resolved.require_convergence and not optimized.converged:
        status = "invalid_graph_not_converged"
        total_score = float("inf")
    else:
        status = "ok_pose_graph"

    local_pairs = tuple(
        _pair_report(
            edge,
            optimized.optimized_world_from_camera,
            measurements.node_frame_indices,
            resolved.optimizer,
        )
        for edge in measurements.local_measurements
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
    diagnostics = {
        "optimizer": {
            key: value
            for key, value in asdict(optimized).items()
            if key != "optimized_world_from_camera"
        },
        "window_scales": list(measurements.window_scales),
        "scale_residual_rms": measurements.scale_residual_rms,
        "scale_rank": measurements.scale_rank,
        "potential_loop_edges": measurements.potential_loop_edges,
        "accepted_loop_edges": len(measurements.loop_measurements),
        "missing_loop_fraction": missing_fraction,
        "switch_penalty": switch_penalty,
        "rejected_loop_edges": [asdict(item) for item in measurements.rejected_loop_edges],
    }
    return replace(
        base,
        total_score=float(total_score),
        local_score=optimized.local_residual_rms,
        long_range_score=optimized.loop_residual_rms,
        local_edge_fraction=(
            len(measurements.local_measurements)
            / max(len(measurements.local_measurements), 1)
        ),
        long_range_edge_fraction=(
            len(measurements.loop_measurements)
            / max(measurements.potential_loop_edges, 1)
        ),
        valid_local_edges=len(measurements.local_measurements),
        valid_long_range_edges=len(measurements.loop_measurements),
        status=status,
        pairs=local_pairs + loop_pairs,
        keyframe_indices=measurements.node_frame_indices,
        score_kind="pose_graph",
        potential_local_edges=len(measurements.local_measurements),
        potential_long_range_edges=measurements.potential_loop_edges,
        graph_diagnostics=diagnostics,
    )
