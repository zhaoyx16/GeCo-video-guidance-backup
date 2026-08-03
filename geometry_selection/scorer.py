"""Confidence-aware static-scene geometry scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from .projection import (
    bilinear_sample,
    camera_centers,
    camera_to_world,
    project_camera,
    relative_depth_error,
    unproject_z_depth,
    world_to_camera,
)
from .schema import GeometryPrediction


@dataclass(frozen=True)
class ScorerConfig:
    local_offsets: tuple[int, ...] = (1, 2)
    min_long_range_gap: int = 3
    confidence_quantile: float = 0.20
    depth_edge_relative_threshold: float = 0.10
    occlusion_relative_tolerance: float = 0.05
    min_pair_overlap: float = 0.05
    min_long_range_overlap: float = 0.15
    min_valid_pixels: int = 64
    pixel_stride: int = 4
    huber_delta: float = 0.05
    cycle_weight: float = 0.25
    local_weight: float = 1.0
    long_range_weight: float = 0.5
    require_long_range: bool = False

    def validate(self) -> None:
        if not self.local_offsets or any(offset < 1 for offset in self.local_offsets):
            raise ValueError("local_offsets must contain positive integers")
        if self.min_long_range_gap < 2:
            raise ValueError("min_long_range_gap must be at least 2")
        for name, value in (
            ("confidence_quantile", self.confidence_quantile),
            ("min_pair_overlap", self.min_pair_overlap),
            ("min_long_range_overlap", self.min_long_range_overlap),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {value}")
        if self.depth_edge_relative_threshold <= 0:
            raise ValueError("depth_edge_relative_threshold must be positive")
        if self.occlusion_relative_tolerance < 0:
            raise ValueError("occlusion_relative_tolerance must be non-negative")
        if self.min_valid_pixels < 1 or self.pixel_stride < 1:
            raise ValueError("min_valid_pixels and pixel_stride must be positive")
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if min(self.cycle_weight, self.local_weight, self.long_range_weight) < 0:
            raise ValueError("score weights must be non-negative")
        if self.local_weight + self.long_range_weight <= 0:
            raise ValueError("at least one score weight must be positive")


@dataclass(frozen=True)
class PairScore:
    source: int
    target: int
    temporal_gap: int
    overlap: float
    comparable_fraction: float
    valid_pixels: int
    depth_error: float
    cycle_error: float
    score: float
    status: str


@dataclass(frozen=True)
class GeometryScoreReport:
    total_score: float
    local_score: float | None
    long_range_score: float | None
    camera_path_length: float
    normalized_camera_motion: float
    valid_local_edges: int
    valid_long_range_edges: int
    status: str
    pairs: tuple[PairScore, ...]
    config: ScorerConfig

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pairs"] = [asdict(pair) for pair in self.pairs]
        payload["config"] = asdict(self.config)
        return _json_safe(payload)


def _json_safe(value: Any) -> Any:
    """Map non-finite diagnostics to null so results remain strict JSON."""

    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _confidence_mask(confidence: np.ndarray, quantile: float) -> np.ndarray:
    finite_positive = np.isfinite(confidence) & (confidence > 0)
    if not finite_positive.any():
        return np.zeros_like(confidence, dtype=bool)
    threshold = np.quantile(confidence[finite_positive], quantile)
    return finite_positive & (confidence >= threshold)


def _normalized_confidence(confidence: np.ndarray) -> np.ndarray:
    """Convert raw positive Omega confidence to a bounded relative weight."""

    finite_positive = np.isfinite(confidence) & (confidence > 0)
    result = np.zeros_like(confidence, dtype=np.float64)
    if not finite_positive.any():
        return result
    cap = float(np.quantile(confidence[finite_positive], 0.95))
    if cap <= 0:
        return result
    result[finite_positive] = np.clip(confidence[finite_positive] / cap, 0.0, 1.0)
    return result


def _depth_edge_mask(depth: np.ndarray, relative_threshold: float) -> np.ndarray:
    horizontal = np.zeros_like(depth, dtype=np.float64)
    vertical = np.zeros_like(depth, dtype=np.float64)
    horizontal[:, 1:] = np.abs(np.diff(depth, axis=1)) / np.maximum(
        np.minimum(depth[:, 1:], depth[:, :-1]), 1e-8
    )
    vertical[1:, :] = np.abs(np.diff(depth, axis=0)) / np.maximum(
        np.minimum(depth[1:, :], depth[:-1, :]), 1e-8
    )
    return (horizontal > relative_threshold) | (vertical > relative_threshold)


def _huber_mean(values: np.ndarray, delta: float, weights: np.ndarray | None = None) -> float:
    if values.size == 0:
        return float("nan")
    absolute = np.abs(values.astype(np.float64, copy=False))
    loss = np.where(absolute <= delta, 0.5 * absolute**2 / delta, absolute - 0.5 * delta)
    if weights is None:
        return float(np.mean(loss))
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != loss.shape:
        raise ValueError(f"weight shape {weights.shape} does not match residual shape {loss.shape}")
    finite_positive = np.isfinite(weights) & (weights > 0)
    if not finite_positive.any():
        return float("nan")
    return float(np.average(loss[finite_positive], weights=weights[finite_positive]))


def _source_zbuffer_mask(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    valid: np.ndarray,
    height: int,
    width: int,
    tolerance: float,
) -> np.ndarray:
    """Keep only the nearest projected source surface per target pixel."""

    result = np.zeros_like(valid, dtype=bool)
    if not valid.any():
        return result
    xi = np.clip(np.rint(x[valid]).astype(np.int64), 0, width - 1)
    yi = np.clip(np.rint(y[valid]).astype(np.int64), 0, height - 1)
    zi = z[valid]
    flat = yi * width + xi
    zbuffer = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(zbuffer, flat, zi)
    visible = zi <= zbuffer[flat] * (1.0 + tolerance)
    result_indices = np.flatnonzero(valid)
    result.flat[result_indices[visible]] = True
    return result


def score_pair(
    prediction: GeometryPrediction,
    source: int,
    target: int,
    config: ScorerConfig,
) -> PairScore:
    if source == target:
        raise ValueError("source and target frames must differ")
    frames = prediction.num_frames
    if not 0 <= source < frames or not 0 <= target < frames:
        raise IndexError(f"pair ({source},{target}) outside {frames} frames")

    source_depth = prediction.depth[source].astype(np.float64, copy=False)
    target_depth = prediction.depth[target].astype(np.float64, copy=False)
    source_conf = prediction.confidence[source].astype(np.float64, copy=False)
    target_conf = prediction.confidence[target].astype(np.float64, copy=False)
    source_conf_weight = _normalized_confidence(source_conf)[
        :: config.pixel_stride, :: config.pixel_stride
    ]
    target_conf_weight_image = _normalized_confidence(target_conf)

    source_points_camera, source_x, source_y = unproject_z_depth(
        source_depth,
        prediction.intrinsics[source],
        stride=config.pixel_stride,
    )
    source_reliable = _confidence_mask(source_conf, config.confidence_quantile)[
        :: config.pixel_stride, :: config.pixel_stride
    ]
    source_edges = _depth_edge_mask(
        source_depth, config.depth_edge_relative_threshold
    )[:: config.pixel_stride, :: config.pixel_stride]
    source_reliable &= ~source_edges
    source_reliable &= np.isfinite(source_points_camera).all(axis=-1)
    source_reliable &= source_points_camera[..., 2] > 0

    reliable_count = int(source_reliable.sum())
    if reliable_count == 0:
        return PairScore(
            source, target, abs(target - source), 0.0, 0.0, 0,
            float("nan"), float("nan"), float("inf"), "no_reliable_source_pixels",
        )

    points_world = camera_to_world(source_points_camera, prediction.world_to_camera[source])
    points_target = world_to_camera(points_world, prediction.world_to_camera[target])
    target_x, target_y, projected_z = project_camera(points_target, prediction.intrinsics[target])
    sampled_depth = bilinear_sample(target_depth, target_x, target_y)
    sampled_conf = bilinear_sample(target_conf, target_x, target_y)
    sampled_target_weight = bilinear_sample(target_conf_weight_image, target_x, target_y)
    target_edges = _depth_edge_mask(target_depth, config.depth_edge_relative_threshold)
    sampled_target_edges = bilinear_sample(target_edges.astype(np.float64), target_x, target_y)
    target_conf_threshold = np.quantile(
        target_conf[np.isfinite(target_conf) & (target_conf > 0)],
        config.confidence_quantile,
    )

    projected_valid = (
        source_reliable
        & sampled_depth.in_bounds
        & np.isfinite(projected_z)
        & (projected_z > 0)
    )
    projected_valid &= _source_zbuffer_mask(
        target_x,
        target_y,
        projected_z,
        projected_valid,
        target_depth.shape[0],
        target_depth.shape[1],
        config.occlusion_relative_tolerance,
    )
    overlap = float(projected_valid.sum() / max(reliable_count, 1))
    target_reliable = (
        np.isfinite(sampled_conf.values)
        & (sampled_conf.values >= target_conf_threshold)
        & np.isfinite(sampled_target_edges.values)
        & (sampled_target_edges.values < 0.01)
    )

    # If the source surface projects behind the target surface, the source is
    # likely occluded and should not be penalised. A source point in front of
    # the target surface remains comparable and contributes an inconsistency.
    behind_target_surface = projected_z > sampled_depth.values * (
        1.0 + config.occlusion_relative_tolerance
    )
    comparable = projected_valid & target_reliable & ~behind_target_surface
    valid_pixels = int(comparable.sum())
    comparable_fraction = float(valid_pixels / max(reliable_count, 1))

    minimum_overlap = (
        config.min_long_range_overlap
        if abs(target - source) >= config.min_long_range_gap
        else config.min_pair_overlap
    )
    if overlap < minimum_overlap:
        status = "insufficient_overlap"
    elif valid_pixels < config.min_valid_pixels:
        status = "insufficient_comparable_pixels"
    else:
        status = "ok"

    if status != "ok":
        return PairScore(
            source, target, abs(target - source), overlap, comparable_fraction,
            valid_pixels, float("nan"), float("nan"), float("inf"), status,
        )

    depth_residual = relative_depth_error(projected_z[comparable], sampled_depth.values[comparable])
    pair_weights = np.sqrt(
        np.clip(source_conf_weight[comparable], 0.0, 1.0)
        * np.clip(sampled_target_weight.values[comparable], 0.0, 1.0)
    )

    # Unproject the target surface sampled at the projected source pixels,
    # return it to the source camera, and measure the normalized pixel cycle.
    target_z = sampled_depth.values
    target_fx = prediction.intrinsics[target, 0, 0]
    target_fy = prediction.intrinsics[target, 1, 1]
    target_cx = prediction.intrinsics[target, 0, 2]
    target_cy = prediction.intrinsics[target, 1, 2]
    target_points = np.stack(
        [
            (target_x - target_cx) / target_fx * target_z,
            (target_y - target_cy) / target_fy * target_z,
            target_z,
        ],
        axis=-1,
    )
    target_world = camera_to_world(target_points, prediction.world_to_camera[target])
    source_cycle_camera = world_to_camera(target_world, prediction.world_to_camera[source])
    cycle_x, cycle_y, cycle_z = project_camera(source_cycle_camera, prediction.intrinsics[source])
    diagonal = float(np.hypot(*prediction.image_size_hw))
    cycle_residual = np.sqrt((cycle_x - source_x) ** 2 + (cycle_y - source_y) ** 2) / max(diagonal, 1.0)
    cycle_valid = comparable & np.isfinite(cycle_residual) & (cycle_z > 0)
    if int(cycle_valid.sum()) < config.min_valid_pixels:
        return PairScore(
            source, target, abs(target - source), overlap, comparable_fraction,
            int(cycle_valid.sum()), float("nan"), float("nan"), float("inf"),
            "insufficient_cycle_pixels",
        )

    depth_error = _huber_mean(depth_residual, config.huber_delta, pair_weights)
    cycle_weights = np.sqrt(
        np.clip(source_conf_weight[cycle_valid], 0.0, 1.0)
        * np.clip(sampled_target_weight.values[cycle_valid], 0.0, 1.0)
    )
    cycle_error = _huber_mean(cycle_residual[cycle_valid], config.huber_delta, cycle_weights)
    score = depth_error + config.cycle_weight * cycle_error
    return PairScore(
        source=source,
        target=target,
        temporal_gap=abs(target - source),
        overlap=overlap,
        comparable_fraction=comparable_fraction,
        valid_pixels=int(cycle_valid.sum()),
        depth_error=depth_error,
        cycle_error=cycle_error,
        score=score,
        status="ok",
    )


def _candidate_pairs(num_frames: int, config: ScorerConfig) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    local = sorted(
        {
            (source, source + offset)
            for offset in config.local_offsets
            for source in range(num_frames - offset)
        }
    )
    local_set = set(local)
    long_range = [
        (source, target)
        for source in range(num_frames)
        for target in range(source + config.min_long_range_gap, num_frames)
        if (source, target) not in local_set
    ]
    return local, long_range


def _mean_valid_score(pairs: list[PairScore]) -> float | None:
    scores = [pair.score for pair in pairs if pair.status == "ok" and np.isfinite(pair.score)]
    return float(np.mean(scores)) if scores else None


def score_geometry(
    prediction: GeometryPrediction,
    config: ScorerConfig | None = None,
) -> GeometryScoreReport:
    config = config or ScorerConfig()
    config.validate()
    prediction.validate()
    local_indices, long_indices = _candidate_pairs(prediction.num_frames, config)
    local_pairs = [
        score_pair(prediction, source, target, config)
        for first, second in local_indices
        for source, target in ((first, second), (second, first))
    ]
    long_pairs = [
        score_pair(prediction, source, target, config)
        for first, second in long_indices
        for source, target in ((first, second), (second, first))
    ]
    local_score = _mean_valid_score(local_pairs)
    long_score = _mean_valid_score(long_pairs)

    if local_score is None:
        total = float("inf")
        status = "no_valid_local_edges"
    elif config.require_long_range and long_score is None:
        total = float("inf")
        status = "no_valid_long_range_edges"
    else:
        components = [(config.local_weight, local_score)]
        if long_score is not None and config.long_range_weight > 0:
            components.append((config.long_range_weight, long_score))
        weight_sum = sum(weight for weight, _ in components)
        total = sum(weight * score for weight, score in components) / weight_sum
        status = "ok" if long_score is not None else "ok_local_only"

    centers = camera_centers(prediction.world_to_camera)
    camera_path_length = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
    median_depth = float(np.median(prediction.depth[np.isfinite(prediction.depth)]))
    normalized_camera_motion = camera_path_length / max(median_depth, 1e-8)
    return GeometryScoreReport(
        total_score=float(total),
        local_score=local_score,
        long_range_score=long_score,
        camera_path_length=camera_path_length,
        normalized_camera_motion=normalized_camera_motion,
        valid_local_edges=sum(pair.status == "ok" for pair in local_pairs),
        valid_long_range_edges=sum(pair.status == "ok" for pair in long_pairs),
        status=status,
        pairs=tuple(local_pairs + long_pairs),
        config=config,
    )
