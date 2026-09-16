"""Geometry-aware candidate scoring and source-aware reranking for C2F."""

from __future__ import annotations

from typing import Any

import torch


def _require_candidate_shapes(
    source_index: torch.Tensor,
    source_time: torch.Tensor,
    candidate_confidence: torch.Tensor,
    candidate_valid: torch.Tensor,
    token_grid: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    if source_index.ndim != 4:
        raise ValueError("candidate tensors must have shape (B,T-1,S,K)")
    if not (
        source_time.shape == source_index.shape
        and candidate_confidence.shape == source_index.shape
        and candidate_valid.shape == source_index.shape
    ):
        raise ValueError("all candidate tensors must have the same shape")
    num_frames, token_h, token_w = token_grid
    batch, target_frames, spatial_tokens, candidates = source_index.shape
    expected = (num_frames - 1, token_h * token_w)
    if (target_frames, spatial_tokens) != expected:
        raise ValueError(
            f"candidate shape must contain target/spatial dimensions {expected}, "
            f"got {(target_frames, spatial_tokens)}"
        )
    return batch, target_frames, spatial_tokens, candidates


def _bilinear_sample(
    frames: torch.Tensor,
    frame_index: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a `(T,H,W)` tensor at a per-item frame and image coordinate."""
    _, height, width = frames.shape
    finite = torch.isfinite(u) & torch.isfinite(v)
    in_bounds = finite & (u >= 0.0) & (u <= width - 1) & (v >= 0.0) & (v <= height - 1)
    safe_u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, width - 1)
    safe_v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, height - 1)
    x0 = torch.floor(safe_u).long()
    y0 = torch.floor(safe_v).long()
    x1 = (x0 + 1).clamp_max(width - 1)
    y1 = (y0 + 1).clamp_max(height - 1)
    wx = safe_u - x0.float()
    wy = safe_v - y0.float()
    flat = frames.reshape(frames.shape[0], height * width)

    def gather(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        linear = y * width + x
        return flat[frame_index.reshape(-1), linear.reshape(-1)].reshape_as(safe_u)

    value = (
        gather(x0, y0) * (1.0 - wx) * (1.0 - wy)
        + gather(x1, y0) * wx * (1.0 - wy)
        + gather(x0, y1) * (1.0 - wx) * wy
        + gather(x1, y1) * wx * wy
    )
    return value, in_bounds


def _candidate_coordinates(
    source_index: torch.Tensor,
    source_time: torch.Tensor,
    candidate_valid: torch.Tensor,
    token_grid: tuple[int, int, int],
    geometry: dict[str, torch.Tensor | tuple[int, int]],
) -> dict[str, torch.Tensor]:
    num_frames, token_h, token_w = token_grid
    spatial_tokens = token_h * token_w
    batch, target_frames, _, candidates = source_index.shape
    processed_h, processed_w = geometry["image_size_hw"]

    safe_source_time = source_time.clamp(0, num_frames - 1).long()
    safe_source_index = source_index.clamp(0, spatial_tokens - 1).long()
    target_time = torch.arange(1, num_frames, device=source_index.device, dtype=torch.long)
    target_time = target_time.view(1, target_frames, 1, 1).expand(batch, -1, spatial_tokens, candidates)
    target_index = torch.arange(spatial_tokens, device=source_index.device, dtype=torch.long)
    target_index = target_index.view(1, 1, spatial_tokens, 1).expand(batch, target_frames, -1, candidates)

    source_x = safe_source_index.remainder(token_w).float()
    source_y = torch.div(safe_source_index, token_w, rounding_mode="floor").float()
    target_x = target_index.remainder(token_w).float()
    target_y = torch.div(target_index, token_w, rounding_mode="floor").float()
    source_u = (source_x + 0.5) * processed_w / token_w - 0.5
    source_v = (source_y + 0.5) * processed_h / token_h - 0.5
    target_u = (target_x + 0.5) * processed_w / token_w - 0.5
    target_v = (target_y + 0.5) * processed_h / token_h - 0.5
    return {
        "candidate_valid": candidate_valid.bool(),
        "source_time": safe_source_time,
        "source_index": safe_source_index,
        "target_time": target_time,
        "source_u": source_u,
        "source_v": source_v,
        "target_u": target_u,
        "target_v": target_v,
    }


def _unproject_source(
    coordinates: dict[str, torch.Tensor],
    geometry: dict[str, torch.Tensor | tuple[int, int]],
) -> dict[str, torch.Tensor]:
    depth = geometry["depth"]
    confidence = geometry["confidence"]
    intrinsics = geometry["intrinsics"]
    camera_to_world = geometry["camera_to_world"]
    thresholds = geometry["confidence_thresholds"]
    if not all(isinstance(value, torch.Tensor) for value in (depth, confidence, intrinsics, camera_to_world, thresholds)):
        raise TypeError("geometry must be prepared before candidate scoring")

    source_time = coordinates["source_time"]
    source_u = coordinates["source_u"]
    source_v = coordinates["source_v"]
    source_depth, source_depth_in_bounds = _bilinear_sample(depth, source_time, source_u, source_v)
    source_confidence, source_conf_in_bounds = _bilinear_sample(
        confidence, source_time, source_u, source_v
    )
    source_k = intrinsics[source_time]
    source_camera = torch.stack(
        (
            (source_u - source_k[..., 0, 2]) / source_k[..., 0, 0] * source_depth,
            (source_v - source_k[..., 1, 2]) / source_k[..., 1, 1] * source_depth,
            source_depth,
            torch.ones_like(source_depth),
        ),
        dim=-1,
    )
    world_point = torch.matmul(camera_to_world[source_time], source_camera.unsqueeze(-1)).squeeze(-1)
    source_observable = (
        coordinates["candidate_valid"]
        & source_depth_in_bounds
        & source_conf_in_bounds
        & torch.isfinite(source_depth)
        & torch.isfinite(source_confidence)
        & (source_depth > 0.0)
        & (source_confidence >= thresholds[source_time])
    )
    return {
        **coordinates,
        "source_depth": source_depth,
        "source_observable": source_observable,
        "world_point": world_point,
    }


def _project_world_point(
    world_point: torch.Tensor,
    view_time: torch.Tensor,
    geometry: dict[str, torch.Tensor | tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    world_to_camera = geometry["world_to_camera"]
    intrinsics = geometry["intrinsics"]
    if not isinstance(world_to_camera, torch.Tensor) or not isinstance(intrinsics, torch.Tensor):
        raise TypeError("geometry must be prepared before candidate scoring")
    view_point = torch.matmul(world_to_camera[view_time], world_point.unsqueeze(-1)).squeeze(-1)
    view_depth = view_point[..., 2]
    safe_depth = torch.where(view_depth.abs() > 1e-8, view_depth, torch.ones_like(view_depth))
    view_k = intrinsics[view_time]
    u = view_k[..., 0, 0] * view_point[..., 0] / safe_depth + view_k[..., 0, 2]
    v = view_k[..., 1, 1] * view_point[..., 1] / safe_depth + view_k[..., 1, 2]
    return u, v, view_depth


def candidate_pairwise_geometry(
    source_index: torch.Tensor,
    source_time: torch.Tensor,
    candidate_confidence: torch.Tensor,
    candidate_valid: torch.Tensor,
    token_grid: tuple[int, int, int],
    geometry: dict[str, torch.Tensor | tuple[int, int]],
    *,
    weighting: str,
    depth_tolerance: float = 0.15,
    max_error_tokens: float = 1.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Score every feature candidate against source-target geometry."""
    _require_candidate_shapes(
        source_index, source_time, candidate_confidence, candidate_valid, token_grid
    )
    if weighting not in {"hard", "soft"}:
        raise ValueError("weighting must be 'hard' or 'soft'")
    if depth_tolerance <= 0.0 or max_error_tokens <= 0.0:
        raise ValueError("geometry tolerances must be positive")

    coordinates = _candidate_coordinates(
        source_index, source_time, candidate_valid, token_grid, geometry
    )
    projected = _unproject_source(coordinates, geometry)
    depth = geometry["depth"]
    confidence = geometry["confidence"]
    thresholds = geometry["confidence_thresholds"]
    processed_h, processed_w = geometry["image_size_hw"]
    if not all(isinstance(value, torch.Tensor) for value in (depth, confidence, thresholds)):
        raise TypeError("geometry must be prepared before candidate scoring")

    geometry_u, geometry_v, projected_depth = _project_world_point(
        projected["world_point"], projected["target_time"], geometry
    )
    target_depth, target_depth_in_bounds = _bilinear_sample(
        depth, projected["target_time"], geometry_u, geometry_v
    )
    target_confidence, target_conf_in_bounds = _bilinear_sample(
        confidence, projected["target_time"], geometry_u, geometry_v
    )
    finite = (
        torch.isfinite(geometry_u)
        & torch.isfinite(geometry_v)
        & torch.isfinite(projected_depth)
        & torch.isfinite(target_depth)
        & torch.isfinite(target_confidence)
    )
    observable = (
        projected["source_observable"]
        & target_depth_in_bounds
        & target_conf_in_bounds
        & finite
        & (projected_depth > 0.0)
        & (target_depth > 0.0)
        & (target_confidence >= thresholds[projected["target_time"]])
    )
    depth_ratio = (projected_depth - target_depth) / target_depth.clamp_min(1e-8)
    token_h, token_w = token_grid[1:]
    dx_tokens = (geometry_u - projected["target_u"]) / (processed_w / token_w)
    dy_tokens = (geometry_v - projected["target_v"]) / (processed_h / token_h)
    reprojection_error = torch.sqrt(dx_tokens.square() + dy_tokens.square())
    occluded = observable & (depth_ratio > depth_tolerance)
    front_conflict = observable & (depth_ratio < -depth_tolerance)
    comparable = observable & ~occluded & ~front_conflict
    accepted = comparable & (reprojection_error <= max_error_tokens)

    if weighting == "hard":
        weight = accepted.to(torch.float32)
    else:
        depth_weight = (1.0 - depth_ratio.abs() / depth_tolerance).clamp(0.0, 1.0)
        reprojection_weight = (1.0 - reprojection_error / max_error_tokens).clamp(0.0, 1.0)
        weight = (depth_weight * reprojection_weight) * comparable.to(torch.float32)
    weight = weight * candidate_confidence.gt(0).to(weight.dtype)
    return weight, {
        "observable": observable,
        "accepted": accepted,
        "occluded": occluded,
        "front_conflict": front_conflict,
        "rejected_reprojection": comparable & ~accepted,
        "depth_relative_delta": depth_ratio,
        "reprojection_error_tokens": reprojection_error,
    }


def candidate_source_support(
    source_index: torch.Tensor,
    source_time: torch.Tensor,
    candidate_confidence: torch.Tensor,
    candidate_valid: torch.Tensor,
    token_grid: tuple[int, int, int],
    geometry: dict[str, torch.Tensor | tuple[int, int]],
    *,
    history: int = 3,
    depth_tolerance: float = 0.15,
    max_error_tokens: float = 1.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Estimate each source candidate's support from earlier geometry views."""
    _require_candidate_shapes(
        source_index, source_time, candidate_confidence, candidate_valid, token_grid
    )
    if history < 1:
        raise ValueError("source support history must be positive")
    if depth_tolerance <= 0.0 or max_error_tokens <= 0.0:
        raise ValueError("geometry tolerances must be positive")

    coordinates = _candidate_coordinates(
        source_index, source_time, candidate_valid, token_grid, geometry
    )
    projected = _unproject_source(coordinates, geometry)
    depth = geometry["depth"]
    confidence = geometry["confidence"]
    intrinsics = geometry["intrinsics"]
    camera_to_world = geometry["camera_to_world"]
    world_to_camera = geometry["world_to_camera"]
    thresholds = geometry["confidence_thresholds"]
    processed_h, processed_w = geometry["image_size_hw"]
    if not all(
        isinstance(value, torch.Tensor)
        for value in (depth, confidence, intrinsics, camera_to_world, world_to_camera, thresholds)
    ):
        raise TypeError("geometry must be prepared before candidate scoring")

    support_masks = []
    conflict_masks = []
    unknown_masks = []
    eligible_masks = []
    roundtrip_errors = []
    support_depth_deltas = []
    for history_lag in range(1, history + 1):
        support_time_raw = projected["source_time"] - history_lag
        eligible = projected["candidate_valid"] & (support_time_raw >= 0)
        support_time = support_time_raw.clamp_min(0)
        support_u, support_v, projected_support_depth = _project_world_point(
            projected["world_point"], support_time, geometry
        )
        observed_support_depth, support_depth_in_bounds = _bilinear_sample(
            depth, support_time, support_u, support_v
        )
        observed_support_confidence, support_conf_in_bounds = _bilinear_sample(
            confidence, support_time, support_u, support_v
        )
        finite = (
            torch.isfinite(support_u)
            & torch.isfinite(support_v)
            & torch.isfinite(projected_support_depth)
            & torch.isfinite(observed_support_depth)
            & torch.isfinite(observed_support_confidence)
        )
        observable = (
            eligible
            & projected["source_observable"]
            & support_depth_in_bounds
            & support_conf_in_bounds
            & finite
            & (projected_support_depth > 0.0)
            & (observed_support_depth > 0.0)
            & (observed_support_confidence >= thresholds[support_time])
        )
        support_depth_delta = (
            projected_support_depth - observed_support_depth
        ) / observed_support_depth.clamp_min(1e-8)
        occluded = observable & (support_depth_delta > depth_tolerance)
        front_conflict = observable & (support_depth_delta < -depth_tolerance)
        depth_consistent = observable & ~occluded & ~front_conflict

        support_k = intrinsics[support_time]
        support_camera = torch.stack(
            (
                (support_u - support_k[..., 0, 2])
                / support_k[..., 0, 0]
                * observed_support_depth,
                (support_v - support_k[..., 1, 2])
                / support_k[..., 1, 1]
                * observed_support_depth,
                observed_support_depth,
                torch.ones_like(observed_support_depth),
            ),
            dim=-1,
        )
        support_world = torch.matmul(
            camera_to_world[support_time], support_camera.unsqueeze(-1)
        ).squeeze(-1)
        returned_source = torch.matmul(
            world_to_camera[projected["source_time"]], support_world.unsqueeze(-1)
        ).squeeze(-1)
        returned_depth = returned_source[..., 2]
        safe_returned_depth = torch.where(
            returned_depth.abs() > 1e-8, returned_depth, torch.ones_like(returned_depth)
        )
        source_k = intrinsics[projected["source_time"]]
        returned_u = (
            source_k[..., 0, 0] * returned_source[..., 0] / safe_returned_depth
            + source_k[..., 0, 2]
        )
        returned_v = (
            source_k[..., 1, 1] * returned_source[..., 1] / safe_returned_depth
            + source_k[..., 1, 2]
        )
        token_h, token_w = token_grid[1:]
        dx_tokens = (returned_u - projected["source_u"]) / (processed_w / token_w)
        dy_tokens = (returned_v - projected["source_v"]) / (processed_h / token_h)
        roundtrip_error = torch.sqrt(dx_tokens.square() + dy_tokens.square())
        returned_depth_delta = (
            returned_depth - projected["source_depth"]
        ) / projected["source_depth"].clamp_min(1e-8)
        roundtrip_finite = (
            torch.isfinite(returned_u)
            & torch.isfinite(returned_v)
            & torch.isfinite(returned_depth)
            & torch.isfinite(roundtrip_error)
            & torch.isfinite(returned_depth_delta)
            & (returned_depth > 0.0)
        )
        supported = (
            depth_consistent
            & roundtrip_finite
            & (returned_depth_delta.abs() <= depth_tolerance)
            & (roundtrip_error <= max_error_tokens)
        )
        conflict = observable & ~occluded & ~supported
        unknown = eligible & ~supported & ~conflict
        support_masks.append(supported)
        conflict_masks.append(conflict)
        unknown_masks.append(unknown)
        eligible_masks.append(eligible)
        roundtrip_errors.append(roundtrip_error)
        support_depth_deltas.append(support_depth_delta)

    support = torch.stack(support_masks, dim=-1)
    conflict = torch.stack(conflict_masks, dim=-1)
    unknown = torch.stack(unknown_masks, dim=-1)
    eligible = torch.stack(eligible_masks, dim=-1)
    support_count = support.sum(dim=-1)
    conflict_count = conflict.sum(dim=-1)
    score = (1.0 + support_count.float()) / (
        1.0 + support_count.float() + conflict_count.float()
    )
    score = torch.where(candidate_valid, score, torch.ones_like(score))
    return score, {
        "support": support,
        "conflict": conflict,
        "unknown": unknown,
        "eligible": eligible,
        "support_count": support_count,
        "conflict_count": conflict_count,
        "unknown_count": unknown.sum(dim=-1),
        "roundtrip_error_tokens": torch.stack(roundtrip_errors, dim=-1),
        "support_depth_relative_delta": torch.stack(support_depth_deltas, dim=-1),
    }


def _gather_candidate(values: torch.Tensor, choice: torch.Tensor) -> torch.Tensor:
    return values.gather(-1, choice.unsqueeze(-1)).squeeze(-1)


def select_candidate_source(
    source_index: torch.Tensor,
    source_time: torch.Tensor,
    candidate_confidence: torch.Tensor,
    candidate_valid: torch.Tensor,
    pair_weight: torch.Tensor,
    source_score: torch.Tensor,
    *,
    policy: str,
) -> dict[str, torch.Tensor]:
    """Select one candidate with V, P, or S while preserving recent-lag tie breaks."""
    if policy not in {"V", "P", "S"}:
        raise ValueError("policy must be one of V, P, or S")
    tensors = (source_time, candidate_confidence, candidate_valid, pair_weight, source_score)
    if any(tensor.shape != source_index.shape for tensor in tensors):
        raise ValueError("all candidate tensors must share one shape")

    feature_score = candidate_confidence.float() * candidate_valid.to(torch.float32)
    if policy == "V":
        selection_score = feature_score
    elif policy == "P":
        selection_score = feature_score * pair_weight.float()
    else:
        selection_score = feature_score * pair_weight.float() * source_score.float()
    choice = selection_score.argmax(dim=-1)
    selected_confidence = _gather_candidate(candidate_confidence, choice)
    selected_pair_weight = _gather_candidate(pair_weight, choice)
    selected_source_score = _gather_candidate(source_score, choice)
    selected_score = _gather_candidate(selection_score, choice)
    selected_valid = selected_score > 0.0
    update_weight = selected_confidence.float() * selected_pair_weight.float()
    update_weight = update_weight * selected_valid.to(update_weight.dtype)
    return {
        "choice": choice,
        "source_index": _gather_candidate(source_index, choice),
        "source_time": _gather_candidate(source_time, choice),
        "confidence": selected_confidence,
        "pair_weight": selected_pair_weight,
        "source_score": selected_source_score,
        "selection_score": selected_score,
        "valid": selected_valid,
        "update_weight": update_weight,
    }


def candidate_policy_selections(
    source_index: torch.Tensor,
    source_time: torch.Tensor,
    candidate_confidence: torch.Tensor,
    candidate_valid: torch.Tensor,
    pair_weight: torch.Tensor,
    source_score: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        policy: select_candidate_source(
            source_index,
            source_time,
            candidate_confidence,
            candidate_valid,
            pair_weight,
            source_score,
            policy=policy,
        )
        for policy in ("V", "P", "S")
    }


def uniform_scale_to_target_update(
    base_residual: torch.Tensor,
    base_weight: torch.Tensor,
    target_residual: torch.Tensor,
    target_weight: torch.Tensor,
    *,
    clamp_max: float = 1.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match a geometry-free base update to a geometry-selected target norm."""
    if base_residual.shape != target_residual.shape:
        raise ValueError("base and target residual tensors must have the same shape")
    if base_weight.shape != target_weight.shape:
        raise ValueError("base and target weights must have the same shape")
    if tuple(base_weight.shape) != tuple(base_residual.shape[: base_weight.ndim]):
        raise ValueError("weight dimensions must match the leading residual dimensions")
    if clamp_max <= 0.0:
        raise ValueError("clamp_max must be positive")
    while base_weight.ndim < base_residual.ndim:
        base_weight = base_weight.unsqueeze(-1)
        target_weight = target_weight.unsqueeze(-1)
    base_update = base_residual.float() * base_weight.float()
    target_update = target_residual.float() * target_weight.float()
    base_norm = torch.linalg.vector_norm(base_update)
    target_norm = torch.linalg.vector_norm(target_update)
    unconstrained_scale = torch.where(
        base_norm > eps,
        target_norm / base_norm,
        torch.zeros_like(base_norm),
    )
    scale = unconstrained_scale.clamp(0.0, clamp_max)
    actual_norm = base_norm * scale
    relative_error = (actual_norm - target_norm).abs() / target_norm.clamp_min(eps)
    return scale, {
        "base_update_norm": base_norm,
        "target_update_norm": target_norm,
        "unconstrained_scale": unconstrained_scale,
        "actual_update_norm": actual_norm,
        "norm_match_relative_error": relative_error,
    }
