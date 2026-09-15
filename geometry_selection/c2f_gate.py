"""Vectorized external-geometry validation for C2F token correspondences."""

from __future__ import annotations

from typing import Any

import torch


REQUIRED_GEOMETRY_KEYS = ("world_to_camera", "intrinsics", "depth", "confidence")


def prepare_geometry_bundle(
    bundle: dict[str, Any],
    *,
    device: torch.device,
    expected_frames: int,
    confidence_percentile: float = 20.0,
) -> dict[str, torch.Tensor | tuple[int, int]]:
    """Validate a saved geometry bundle and move its small tensors to one device."""
    missing = [key for key in REQUIRED_GEOMETRY_KEYS if key not in bundle]
    if missing:
        raise ValueError(f"geometry bundle is missing keys: {missing}")

    world_to_camera = torch.as_tensor(bundle["world_to_camera"], dtype=torch.float32, device=device)
    intrinsics = torch.as_tensor(bundle["intrinsics"], dtype=torch.float32, device=device)
    depth = torch.as_tensor(bundle["depth"], dtype=torch.float32, device=device)
    confidence = torch.as_tensor(bundle["confidence"], dtype=torch.float32, device=device)
    if world_to_camera.shape != (expected_frames, 4, 4):
        raise ValueError(
            f"world_to_camera must have shape {(expected_frames, 4, 4)}, got {tuple(world_to_camera.shape)}"
        )
    if intrinsics.shape != (expected_frames, 3, 3):
        raise ValueError(f"intrinsics must have shape {(expected_frames, 3, 3)}, got {tuple(intrinsics.shape)}")
    if depth.ndim != 3 or depth.shape[0] != expected_frames:
        raise ValueError(f"depth must have shape (T,H,W) with T={expected_frames}, got {tuple(depth.shape)}")
    if confidence.shape != depth.shape:
        raise ValueError(f"confidence shape {tuple(confidence.shape)} differs from depth {tuple(depth.shape)}")

    thresholds = bundle.get("confidence_thresholds")
    if thresholds is None:
        thresholds = torch.quantile(
            confidence.reshape(expected_frames, -1),
            float(confidence_percentile) / 100.0,
            dim=1,
        )
    else:
        thresholds = torch.as_tensor(thresholds, dtype=torch.float32, device=device)
    if thresholds.shape != (expected_frames,):
        raise ValueError(f"confidence_thresholds must have shape {(expected_frames,)}, got {tuple(thresholds.shape)}")

    return {
        "world_to_camera": world_to_camera,
        "camera_to_world": torch.linalg.inv(world_to_camera),
        "intrinsics": intrinsics,
        "depth": depth,
        "confidence": confidence,
        "confidence_thresholds": thresholds,
        "image_size_hw": (int(depth.shape[1]), int(depth.shape[2])),
    }


def _bilinear_sample(
    frames: torch.Tensor,
    frame_index: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one scalar image per item from a `(T,H,W)` tensor."""
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


def geometry_validation_gate(
    source_index: torch.Tensor,
    source_time: torch.Tensor,
    token_grid: tuple[int, int, int],
    geometry: dict[str, torch.Tensor | tuple[int, int]],
    *,
    depth_tolerance: float = 0.15,
    max_error_tokens: float = 1.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return a hard positive gate for the per-token sources selected by C2F."""
    num_frames, token_h, token_w = token_grid
    spatial_tokens = token_h * token_w
    expected = (source_index.shape[0], num_frames - 1, spatial_tokens)
    if tuple(source_index.shape) != expected or tuple(source_time.shape) != expected:
        raise ValueError(
            f"source tensors must both have shape {expected}; got {tuple(source_index.shape)} and "
            f"{tuple(source_time.shape)}"
        )
    if not 0.0 <= depth_tolerance:
        raise ValueError("depth_tolerance must be non-negative")
    if max_error_tokens < 0.0:
        raise ValueError("max_error_tokens must be non-negative")

    depth = geometry["depth"]
    confidence = geometry["confidence"]
    intrinsics = geometry["intrinsics"]
    world_to_camera = geometry["world_to_camera"]
    camera_to_world = geometry["camera_to_world"]
    thresholds = geometry["confidence_thresholds"]
    processed_h, processed_w = geometry["image_size_hw"]
    if not isinstance(depth, torch.Tensor) or not isinstance(confidence, torch.Tensor):
        raise TypeError("geometry bundle must be prepared with prepare_geometry_bundle")

    source_time = source_time.long()
    source_index = source_index.long()
    if bool((source_time < 0).any()) or bool((source_time >= num_frames).any()):
        raise ValueError("source_time is outside the geometry frame range")
    if bool((source_index < 0).any()) or bool((source_index >= spatial_tokens).any()):
        raise ValueError("source_index is outside the spatial token range")

    batch = source_index.shape[0]
    target_time = torch.arange(1, num_frames, device=source_index.device, dtype=torch.long)
    target_time = target_time.view(1, num_frames - 1, 1).expand(batch, -1, spatial_tokens)
    target_index = torch.arange(spatial_tokens, device=source_index.device, dtype=torch.long)
    target_index = target_index.view(1, 1, spatial_tokens).expand(batch, num_frames - 1, -1)

    source_x = source_index.remainder(token_w).float()
    source_y = torch.div(source_index, token_w, rounding_mode="floor").float()
    target_x = target_index.remainder(token_w).float()
    target_y = torch.div(target_index, token_w, rounding_mode="floor").float()
    source_u = (source_x + 0.5) * processed_w / token_w - 0.5
    source_v = (source_y + 0.5) * processed_h / token_h - 0.5
    c2f_target_u = (target_x + 0.5) * processed_w / token_w - 0.5
    c2f_target_v = (target_y + 0.5) * processed_h / token_h - 0.5

    source_depth, source_in_bounds = _bilinear_sample(depth, source_time, source_u, source_v)
    source_confidence, source_conf_in_bounds = _bilinear_sample(confidence, source_time, source_u, source_v)
    source_k = intrinsics[source_time]
    source_points = torch.stack(
        (
            (source_u - source_k[..., 0, 2]) / source_k[..., 0, 0] * source_depth,
            (source_v - source_k[..., 1, 2]) / source_k[..., 1, 1] * source_depth,
            source_depth,
            torch.ones_like(source_depth),
        ),
        dim=-1,
    )
    world_points = torch.matmul(camera_to_world[source_time], source_points.unsqueeze(-1)).squeeze(-1)
    target_points = torch.matmul(world_to_camera[target_time], world_points.unsqueeze(-1)).squeeze(-1)
    projected_depth = target_points[..., 2]
    target_k = intrinsics[target_time]
    safe_projected_depth = torch.where(
        projected_depth.abs() > 1e-8,
        projected_depth,
        torch.ones_like(projected_depth),
    )
    geometry_u = target_k[..., 0, 0] * target_points[..., 0] / safe_projected_depth + target_k[..., 0, 2]
    geometry_v = target_k[..., 1, 1] * target_points[..., 1] / safe_projected_depth + target_k[..., 1, 2]
    target_depth, target_in_bounds = _bilinear_sample(depth, target_time, geometry_u, geometry_v)
    target_confidence, target_conf_in_bounds = _bilinear_sample(
        confidence, target_time, geometry_u, geometry_v
    )

    finite = (
        torch.isfinite(source_depth)
        & torch.isfinite(source_confidence)
        & torch.isfinite(geometry_u)
        & torch.isfinite(geometry_v)
        & torch.isfinite(projected_depth)
        & torch.isfinite(target_depth)
        & torch.isfinite(target_confidence)
    )
    positive = (source_depth > 0.0) & (projected_depth > 0.0) & (target_depth > 0.0)
    in_bounds = source_in_bounds & source_conf_in_bounds & target_in_bounds & target_conf_in_bounds
    confident = (
        (source_confidence >= thresholds[source_time])
        & (target_confidence >= thresholds[target_time])
    )
    valid = finite & positive & in_bounds & confident
    depth_ratio = (projected_depth - target_depth) / target_depth.clamp_min(1e-8)
    dx_tokens = (geometry_u - c2f_target_u) / (processed_w / token_w)
    dy_tokens = (geometry_v - c2f_target_v) / (processed_h / token_h)
    reprojection_error = torch.sqrt(dx_tokens.square() + dy_tokens.square())

    occluded = valid & (depth_ratio > depth_tolerance)
    front_conflict = valid & (depth_ratio < -depth_tolerance)
    depth_consistent = valid & ~occluded & ~front_conflict
    accepted = depth_consistent & (reprojection_error <= max_error_tokens)
    rejected_reprojection = depth_consistent & ~accepted
    abstained = ~accepted & ~front_conflict & ~rejected_reprojection
    gate = accepted.to(torch.float32).unsqueeze(-1)
    return gate, {
        "accepted": accepted,
        "rejected_reprojection": rejected_reprojection,
        "front_conflict": front_conflict,
        "occluded": occluded,
        "abstained": abstained,
        "valid_geometry": valid,
        "reprojection_error_tokens": reprojection_error,
        "target_depth_relative_delta": depth_ratio,
    }


def uniform_update_scale(
    residual: torch.Tensor,
    base_weight: torch.Tensor,
    geometry_gate: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match a uniform residual update to the L2 norm of a spatially gated one."""
    if residual.ndim < 4:
        raise ValueError("residual must have shape (B,T,S,...)")
    if tuple(base_weight.shape[:3]) != tuple(residual.shape[:3]):
        raise ValueError("base_weight and residual leading dimensions differ")
    if tuple(geometry_gate.shape[:3]) != tuple(residual.shape[:3]):
        raise ValueError("geometry_gate and residual leading dimensions differ")
    while base_weight.ndim < residual.ndim:
        base_weight = base_weight.unsqueeze(-1)
    while geometry_gate.ndim < residual.ndim:
        geometry_gate = geometry_gate.unsqueeze(-1)

    reduce_dims = tuple(range(3, residual.ndim))
    residual_sq = residual.float().square().sum(dim=reduce_dims)
    base_scalar = base_weight.float().reshape(*residual.shape[:3], -1)[..., 0]
    gate_scalar = geometry_gate.float().reshape(*residual.shape[:3], -1)[..., 0]
    base_norm_sq = (residual_sq * base_scalar.square()).sum()
    gated_norm_sq = (residual_sq * base_scalar.square() * gate_scalar.square()).sum()
    base_norm = torch.sqrt(base_norm_sq.clamp_min(0.0))
    gated_norm = torch.sqrt(gated_norm_sq.clamp_min(0.0))
    scale = torch.where(base_norm > eps, gated_norm / base_norm, torch.zeros_like(base_norm)).clamp(0.0, 1.0)
    return scale, {
        "base_update_norm": base_norm,
        "gated_update_norm": gated_norm,
        "uniform_update_norm": base_norm * scale,
    }
