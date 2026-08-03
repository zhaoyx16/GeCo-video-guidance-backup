"""Build an auditable pose graph from independent VGGT-Omega windows.

Every window is a separate model forward pass.  Relative camera rotations are
invariant to a window's world-frame gauge, but monocular translations are only
defined up to one scale per forward pass.  Before constructing SE(3) edges, we
therefore solve a window-scale graph from frames shared by overlapping windows.
The scale ratio between two windows is measured from their shared-frame depth
maps and, when observable, shared camera baselines.

This module deliberately rejects edges derived from one global camera output:
such edges close algebraically and cannot measure loop-closure inconsistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .graph import RelativePoseMeasurement, invert_se3
from .projection import camera_centers
from .schema import GeometryPrediction
from .scorer import ScorerConfig, score_pair


@dataclass(frozen=True)
class IndependentWindow:
    window_id: str
    kind: Literal["local", "loop"]
    prediction: GeometryPrediction
    independent_run_id: str

    def validate(self) -> None:
        if not self.window_id.strip() or not self.independent_run_id.strip():
            raise ValueError("window_id and independent_run_id must be non-empty")
        if self.kind not in {"local", "loop"}:
            raise ValueError("window kind must be 'local' or 'loop'")
        self.prediction.validate()


@dataclass(frozen=True)
class WindowGraphConfig:
    min_shared_frames_for_scale: int = 2
    confidence_quantile: float = 0.20
    confidence_evidence_floor: float = 1e-3
    depth_sample_stride: int = 8
    min_depth_scale_pixels: int = 32
    min_camera_baseline: float = 1e-4
    scale_huber_delta: float = 0.10
    scale_max_iterations: int = 20
    scale_tolerance: float = 1e-9
    min_loop_node_gap: int = 3
    require_reobservation_support: bool = True
    min_loop_overlap: float = 0.15
    min_loop_comparable_fraction: float = 0.05
    min_loop_valid_pixels: int = 64
    require_loop_edges: bool = True

    def validate(self) -> None:
        if self.min_shared_frames_for_scale < 1:
            raise ValueError("min_shared_frames_for_scale must be positive")
        if not 0.0 <= self.confidence_quantile <= 1.0:
            raise ValueError("confidence_quantile must be in [0,1]")
        if self.confidence_evidence_floor < 0:
            raise ValueError("confidence_evidence_floor must be non-negative")
        if self.depth_sample_stride < 1 or self.min_depth_scale_pixels < 1:
            raise ValueError("depth sampling values must be positive")
        if self.min_camera_baseline <= 0 or self.scale_huber_delta <= 0:
            raise ValueError("scale thresholds must be positive")
        if self.scale_max_iterations < 1 or self.scale_tolerance <= 0:
            raise ValueError("scale solver settings must be positive")
        if self.min_loop_node_gap < 2:
            raise ValueError("min_loop_node_gap must be at least 2")
        for name in ("min_loop_overlap", "min_loop_comparable_fraction"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.min_loop_valid_pixels < 1:
            raise ValueError("min_loop_valid_pixels must be positive")
        if not isinstance(self.require_loop_edges, bool) or not isinstance(
            self.require_reobservation_support, bool
        ):
            raise TypeError("loop requirement flags must be boolean")


@dataclass(frozen=True)
class WindowScaleConstraint:
    first_window: int
    second_window: int
    log_second_scale_from_first: float
    confidence: float
    shared_frames: tuple[int, ...]
    depth_samples: int
    baseline_samples: int
    dispersion: float


@dataclass(frozen=True)
class WindowGraphMeasurements:
    node_frame_indices: tuple[int, ...]
    initial_world_from_camera: np.ndarray
    local_measurements: tuple[RelativePoseMeasurement, ...]
    loop_measurements: tuple[RelativePoseMeasurement, ...]
    window_scales: tuple[float, ...]
    scale_constraints: tuple[WindowScaleConstraint, ...]
    scale_residual_rms: float
    scale_rank: int
    potential_loop_edges: int
    rejected_loop_edges: tuple["RejectedLoopMeasurement", ...]


@dataclass(frozen=True)
class RejectedLoopMeasurement:
    window_id: str
    source_frame: int
    target_frame: int
    reason: str
    overlap: float
    comparable_fraction: float
    valid_pixels: int


def _confidence_evidence(confidence: np.ndarray) -> np.ndarray:
    values = np.asarray(confidence, dtype=np.float64) - 1.0
    return np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)


def _confidence_threshold(evidence: np.ndarray, config: WindowGraphConfig) -> float:
    positive = evidence[evidence > 0]
    if positive.size == 0:
        return float("inf")
    return max(
        float(np.quantile(positive, config.confidence_quantile)),
        config.confidence_evidence_floor,
    )


def _frame_quality(prediction: GeometryPrediction, local_index: int) -> float:
    evidence = _confidence_evidence(prediction.confidence[local_index])
    positive = evidence[evidence > 0]
    if positive.size == 0:
        return 0.0
    median = float(np.median(positive))
    return median / (1.0 + median)


def _frame_positions(window: IndependentWindow) -> dict[int, int]:
    return {
        int(frame): position
        for position, frame in enumerate(window.prediction.keyframe_indices)
    }


def _shared_scale_samples(
    first: IndependentWindow,
    second: IndependentWindow,
    config: WindowGraphConfig,
) -> tuple[list[float], int, int, tuple[int, ...]]:
    first_positions = _frame_positions(first)
    second_positions = _frame_positions(second)
    shared = tuple(sorted(set(first_positions) & set(second_positions)))
    if len(shared) < config.min_shared_frames_for_scale:
        return [], 0, 0, shared

    samples: list[float] = []
    depth_samples = 0
    stride = config.depth_sample_stride
    for frame in shared:
        first_index = first_positions[frame]
        second_index = second_positions[frame]
        first_depth = first.prediction.depth[first_index].astype(np.float64, copy=False)[::stride, ::stride]
        second_depth = second.prediction.depth[second_index].astype(np.float64, copy=False)[::stride, ::stride]
        if first_depth.shape != second_depth.shape:
            raise ValueError(
                f"shared frame {frame} has different depth shapes across windows"
            )
        first_evidence = _confidence_evidence(first.prediction.confidence[first_index])[::stride, ::stride]
        second_evidence = _confidence_evidence(second.prediction.confidence[second_index])[::stride, ::stride]
        valid = (
            np.isfinite(first_depth)
            & np.isfinite(second_depth)
            & (first_depth > 0)
            & (second_depth > 0)
            & (first_evidence >= _confidence_threshold(first_evidence, config))
            & (second_evidence >= _confidence_threshold(second_evidence, config))
        )
        if int(valid.sum()) >= config.min_depth_scale_pixels:
            log_ratios = np.log(first_depth[valid]) - np.log(second_depth[valid])
            samples.append(float(np.median(log_ratios)))
            depth_samples += int(valid.sum())

    first_centers = camera_centers(first.prediction.world_to_camera)
    second_centers = camera_centers(second.prediction.world_to_camera)
    baseline_samples = 0
    for left_position, left_frame in enumerate(shared):
        for right_frame in shared[left_position + 1 :]:
            first_distance = float(
                np.linalg.norm(
                    first_centers[first_positions[left_frame]]
                    - first_centers[first_positions[right_frame]]
                )
            )
            second_distance = float(
                np.linalg.norm(
                    second_centers[second_positions[left_frame]]
                    - second_centers[second_positions[right_frame]]
                )
            )
            if (
                first_distance >= config.min_camera_baseline
                and second_distance >= config.min_camera_baseline
            ):
                samples.append(float(np.log(first_distance) - np.log(second_distance)))
                baseline_samples += 1
    return samples, depth_samples, baseline_samples, shared


def build_scale_constraints(
    windows: Sequence[IndependentWindow],
    config: WindowGraphConfig,
) -> tuple[WindowScaleConstraint, ...]:
    constraints: list[WindowScaleConstraint] = []
    for first_index, first in enumerate(windows):
        for second_index in range(first_index + 1, len(windows)):
            second = windows[second_index]
            samples, depth_count, baseline_count, shared = _shared_scale_samples(
                first, second, config
            )
            if not samples:
                continue
            values = np.asarray(samples, dtype=np.float64)
            estimate = float(np.median(values))
            dispersion = float(1.4826 * np.median(np.abs(values - estimate)))
            shared_quality = []
            first_positions = _frame_positions(first)
            second_positions = _frame_positions(second)
            for frame in shared:
                shared_quality.append(
                    np.sqrt(
                        _frame_quality(first.prediction, first_positions[frame])
                        * _frame_quality(second.prediction, second_positions[frame])
                    )
                )
            confidence = float(np.mean(shared_quality)) / max(
                1.0 + dispersion / config.scale_huber_delta,
                1.0,
            )
            if confidence <= 0 or not np.isfinite(confidence):
                continue
            constraints.append(
                WindowScaleConstraint(
                    first_window=first_index,
                    second_window=second_index,
                    log_second_scale_from_first=estimate,
                    confidence=confidence,
                    shared_frames=shared,
                    depth_samples=depth_count,
                    baseline_samples=baseline_count,
                    dispersion=dispersion,
                )
            )
    return tuple(constraints)


def solve_window_scales(
    num_windows: int,
    constraints: Sequence[WindowScaleConstraint],
    config: WindowGraphConfig,
) -> tuple[np.ndarray, float, int]:
    if num_windows < 1:
        raise ValueError("num_windows must be positive")
    if num_windows == 1:
        return np.ones(1, dtype=np.float64), 0.0, 0
    if not constraints:
        raise ValueError("overlapping windows provide no scale constraints")

    matrix = np.zeros((len(constraints), num_windows - 1), dtype=np.float64)
    target = np.empty(len(constraints), dtype=np.float64)
    base_weights = np.empty(len(constraints), dtype=np.float64)
    for row, constraint in enumerate(constraints):
        if constraint.first_window != 0:
            matrix[row, constraint.first_window - 1] = -1.0
        if constraint.second_window != 0:
            matrix[row, constraint.second_window - 1] = 1.0
        target[row] = constraint.log_second_scale_from_first
        base_weights[row] = constraint.confidence

    rank = int(np.linalg.matrix_rank(matrix))
    if rank < num_windows - 1:
        raise ValueError(
            "window overlap graph is disconnected; every window scale must be linked to the anchor"
        )
    solution = np.zeros(num_windows - 1, dtype=np.float64)
    for _ in range(config.scale_max_iterations):
        residual = matrix @ solution - target
        absolute = np.abs(residual)
        robust = np.ones_like(absolute)
        outside = absolute > config.scale_huber_delta
        robust[outside] = config.scale_huber_delta / absolute[outside]
        weights = base_weights * robust
        weighted_matrix = matrix * np.sqrt(weights)[:, None]
        weighted_target = target * np.sqrt(weights)
        updated = np.linalg.lstsq(weighted_matrix, weighted_target, rcond=None)[0]
        if np.linalg.norm(updated - solution) < config.scale_tolerance:
            solution = updated
            break
        solution = updated
    final_residual = matrix @ solution - target
    rms = float(np.sqrt(np.average(final_residual**2, weights=base_weights)))
    log_scales = np.concatenate([[0.0], solution])
    if not np.isfinite(log_scales).all():
        raise ValueError("window scale solver produced non-finite values")
    return np.exp(log_scales), rms, rank


def _scaled_relative_transform(
    prediction: GeometryPrediction,
    source_position: int,
    target_position: int,
    scale: float,
) -> np.ndarray:
    source_world_from_camera = invert_se3(prediction.world_to_camera[source_position])
    target_world_from_camera = invert_se3(prediction.world_to_camera[target_position])
    target_from_source = prediction.world_to_camera[target_position] @ source_world_from_camera
    target_from_source = target_from_source.copy()
    target_from_source[:3, 3] *= scale
    return target_from_source


def _initial_states(
    node_frames: tuple[int, ...],
    local_edges: Sequence[RelativePoseMeasurement],
) -> np.ndarray:
    node_count = len(node_frames)
    states: list[np.ndarray | None] = [None] * node_count
    states[0] = np.eye(4, dtype=np.float64)
    adjacency: list[list[tuple[float, int, np.ndarray]]] = [[] for _ in range(node_count)]
    for edge in local_edges:
        adjacency[edge.source].append(
            (edge.confidence, edge.target, invert_se3(edge.target_from_source))
        )
        adjacency[edge.target].append(
            (edge.confidence, edge.source, edge.target_from_source)
        )
    queue = [0]
    while queue:
        source = queue.pop(0)
        for _, target, source_world_to_target_world in sorted(
            adjacency[source], key=lambda item: item[0], reverse=True
        ):
            if states[target] is not None:
                continue
            states[target] = states[source] @ source_world_to_target_world
            queue.append(target)
    missing = [node_frames[index] for index, state in enumerate(states) if state is None]
    if missing:
        raise ValueError(f"local measurements do not connect graph nodes: {missing}")
    return np.stack(states)


def build_window_graph_measurements(
    windows: Sequence[IndependentWindow],
    config: WindowGraphConfig | None = None,
) -> WindowGraphMeasurements:
    resolved = config or WindowGraphConfig()
    resolved.validate()
    if len(windows) < 2:
        raise ValueError("at least two independent windows are required")
    for window in windows:
        window.validate()
    if len({window.window_id for window in windows}) != len(windows):
        raise ValueError("window IDs must be unique")
    if len({window.independent_run_id for window in windows}) != len(windows):
        raise ValueError("each window must come from a distinct model forward pass")

    local_windows = [window for window in windows if window.kind == "local"]
    loop_windows = [window for window in windows if window.kind == "loop"]
    if not local_windows:
        raise ValueError("at least one local window is required")
    node_frames = tuple(
        sorted(
            {
                int(frame)
                for window in local_windows
                for frame in window.prediction.keyframe_indices
            }
        )
    )
    if len(node_frames) < 2:
        raise ValueError("local windows must cover at least two graph nodes")
    node_lookup = {frame: index for index, frame in enumerate(node_frames)}
    if any(
        int(frame) not in node_lookup
        for window in loop_windows
        for frame in window.prediction.keyframe_indices
    ):
        raise ValueError("loop windows may only reference nodes covered by local windows")

    constraints = build_scale_constraints(windows, resolved)
    scales, scale_rms, scale_rank = solve_window_scales(
        len(windows), constraints, resolved
    )
    local_edges: list[RelativePoseMeasurement] = []
    loop_edges: list[RelativePoseMeasurement] = []
    rejected_loops: list[RejectedLoopMeasurement] = []
    potential_loop_edges = 0
    for window_index, window in enumerate(windows):
        frames = tuple(int(frame) for frame in window.prediction.keyframe_indices)
        if window.kind == "local":
            pairs = [(position, position + 1) for position in range(len(frames) - 1)]
        else:
            eligible = [
                (source, target)
                for source in range(len(frames))
                for target in range(source + 1, len(frames))
                if abs(node_lookup[frames[target]] - node_lookup[frames[source]])
                >= resolved.min_loop_node_gap
            ]
            pairs = (
                [max(eligible, key=lambda pair: abs(frames[pair[1]] - frames[pair[0]]))]
                if eligible
                else []
            )
            potential_loop_edges += int(bool(pairs))
        if not pairs:
            continue
        per_edge_normalizer = float(len(pairs))
        for source_position, target_position in pairs:
            source_frame = frames[source_position]
            target_frame = frames[target_position]
            confidence = np.sqrt(
                _frame_quality(window.prediction, source_position)
                * _frame_quality(window.prediction, target_position)
            ) / per_edge_normalizer
            if window.kind == "loop" and resolved.require_reobservation_support:
                support = score_pair(
                    window.prediction,
                    source_position,
                    target_position,
                    ScorerConfig(
                        local_offsets=(1,),
                        min_long_range_gap=2,
                        min_pair_overlap=resolved.min_loop_overlap,
                        min_long_range_overlap=resolved.min_loop_overlap,
                        min_valid_pixels=resolved.min_loop_valid_pixels,
                        min_comparable_fraction=resolved.min_loop_comparable_fraction,
                        min_valid_local_fraction=0.0,
                        min_valid_long_range_fraction=0.0,
                        require_long_range=False,
                    ),
                )
                if support.status != "ok":
                    rejected_loops.append(
                        RejectedLoopMeasurement(
                            window_id=window.window_id,
                            source_frame=source_frame,
                            target_frame=target_frame,
                            reason=support.status,
                            overlap=support.overlap,
                            comparable_fraction=support.comparable_fraction,
                            valid_pixels=support.valid_pixels,
                        )
                    )
                    continue
                confidence *= np.sqrt(
                    max(support.overlap, 0.0)
                    * max(support.comparable_fraction, 0.0)
                )
            if confidence <= 0:
                continue
            edge = RelativePoseMeasurement(
                source=node_lookup[source_frame],
                target=node_lookup[target_frame],
                target_from_source=_scaled_relative_transform(
                    window.prediction,
                    source_position,
                    target_position,
                    float(scales[window_index]),
                ),
                confidence=float(confidence),
                independently_estimated=True,
                provenance=(
                    f"{window.kind}:{window.window_id}:{window.independent_run_id}:"
                    f"{source_frame}->{target_frame}"
                ),
            )
            (local_edges if window.kind == "local" else loop_edges).append(edge)
    if not local_edges:
        raise ValueError("independent local windows produced no valid pose measurements")
    if resolved.require_loop_edges and not loop_edges:
        raise ValueError("independent loop windows produced no valid loop measurements")
    initial = _initial_states(node_frames, local_edges)
    return WindowGraphMeasurements(
        node_frame_indices=node_frames,
        initial_world_from_camera=initial,
        local_measurements=tuple(local_edges),
        loop_measurements=tuple(loop_edges),
        window_scales=tuple(float(value) for value in scales),
        scale_constraints=constraints,
        scale_residual_rms=scale_rms,
        scale_rank=scale_rank,
        potential_loop_edges=potential_loop_edges,
        rejected_loop_edges=tuple(rejected_loops),
    )


def make_window_schedule(
    keyframe_indices: Sequence[int],
    *,
    local_window_size: int = 4,
    local_stride: int = 2,
    loop_context: int = 2,
    min_loop_node_gap: int = 3,
    max_loop_windows: int = 2,
) -> tuple[tuple[str, Literal["local", "loop"], tuple[int, ...]], ...]:
    frames = tuple(int(frame) for frame in keyframe_indices)
    if len(frames) < local_window_size or len(set(frames)) != len(frames):
        raise ValueError("keyframes must be unique and cover at least one local window")
    if frames != tuple(sorted(frames)):
        raise ValueError("keyframes must be sorted")
    if not 2 <= local_window_size <= len(frames):
        raise ValueError("local_window_size is invalid")
    if not 1 <= local_stride < local_window_size:
        raise ValueError("local_stride must overlap consecutive local windows")
    if loop_context < 2 or 2 * loop_context > len(frames):
        raise ValueError("loop_context must provide at least two frames per endpoint")
    if max_loop_windows < 0:
        raise ValueError("max_loop_windows must be non-negative")

    starts = list(range(0, len(frames) - local_window_size + 1, local_stride))
    final_start = len(frames) - local_window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    schedule: list[tuple[str, Literal["local", "loop"], tuple[int, ...]]] = []
    for index, start in enumerate(starts):
        schedule.append((f"local-{index:02d}", "local", frames[start : start + local_window_size]))

    blocks = [frames[start : start + loop_context] for start in range(0, len(frames) - loop_context + 1, local_stride)]
    loop_candidates = []
    for first_index, first in enumerate(blocks):
        for second in blocks[first_index + 1 :]:
            first_node = frames.index(first[-1])
            second_node = frames.index(second[0])
            if second_node - first_node < min_loop_node_gap:
                continue
            combined = tuple(dict.fromkeys(first + second))
            loop_candidates.append((second_node - first_node, combined))
    loop_candidates.sort(key=lambda item: (-item[0], item[1]))
    for index, (_, combined) in enumerate(loop_candidates[:max_loop_windows]):
        schedule.append((f"loop-{index:02d}", "loop", combined))
    return tuple(schedule)
