"""Offline 3D correspondence maps for frozen Wan attention guidance.

The first pass generates a normal RGB draft. A separate geometry process
estimates cameras and depth from that draft, then serializes a sparse map from
target Wan tokens to visible source-anchor tokens. The second diffusion pass
only loads this map; it does not decode an x0 estimate or run a geometry model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class DraftGeometryMap:
    """Up to K geometry candidates for every target token at t=1..T-1."""

    source_time: torch.Tensor  # [T-1, P, K], long
    source_index: torch.Tensor  # [T-1, P, K], long
    confidence: torch.Tensor  # [T-1, Ht, Wt, K], transport-only confidence
    conflict_confidence: torch.Tensor  # [T-1, Ht, Wt], suppression confidence
    observed_background_time: torch.Tensor  # [T-1, P, K], long
    observed_background_index: torch.Tensor  # [T-1, P, K], long
    observed_background_confidence: torch.Tensor  # [T-1, Ht, Wt, K]
    temporal_support_count: torch.Tensor  # [T-1, Ht, Wt], integer frame support
    temporal_observation_count: torch.Tensor  # [T-1], sampled RGB frames per token
    pair_stats: list[dict[str, float | int | str]]
    metadata: dict[str, Any]


def frame_to_latent_index(frame_index: int, temporal_scale: int, latent_frames: int) -> int:
    """Map a decoded video frame to Wan's causal temporal latent index."""
    if frame_index == 0:
        return 0
    return min(latent_frames - 1, (frame_index - 1) // temporal_scale + 1)


def _confidence_mask(confidence: torch.Tensor, percentile: float, floor: float) -> torch.Tensor:
    finite = torch.isfinite(confidence)
    if not finite.any():
        return torch.zeros_like(confidence, dtype=torch.bool)
    threshold = torch.maximum(
        torch.quantile(confidence[finite], percentile / 100.0),
        torch.tensor(floor, device=confidence.device, dtype=confidence.dtype),
    )
    return finite & (confidence > 0) & (confidence >= threshold)


def _sample_scalar(map_hw: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    height, width = map_hw.shape
    grid = torch.stack(
        [2.0 * u / max(width - 1, 1) - 1.0, 2.0 * v / max(height - 1, 1) - 1.0],
        dim=-1,
    ).reshape(1, -1, 1, 2)
    sampled = F.grid_sample(
        map_hw.float().reshape(1, 1, height, width),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.reshape(-1)


def _pixel_centres_to_token_indices(
    coordinate: torch.Tensor,
    image_extent: int,
    token_extent: int,
) -> torch.Tensor:
    """Map pixel-centre coordinates to containing token indices."""
    return torch.floor(
        (coordinate + 0.5) * token_extent / image_extent
    ).long()


def _sample_conservative_source_depth(
    depth_hw: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    radius: int,
    max_relative_spread: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use a local minimum and abstain near discontinuous depth boundaries."""
    if radius < 0:
        raise ValueError("source_depth_edge_radius must be non-negative")
    if max_relative_spread < 0.0:
        raise ValueError("max_source_depth_spread must be non-negative")

    height, width = depth_hw.shape
    finite_positive = torch.isfinite(depth_hw) & (depth_hw > 0)
    depth_for_min = torch.where(
        finite_positive,
        depth_hw.float(),
        torch.full_like(depth_hw.float(), torch.inf),
    )
    depth_for_max = torch.where(
        finite_positive,
        depth_hw.float(),
        torch.full_like(depth_hw.float(), -torch.inf),
    )
    kernel = 2 * radius + 1
    local_min = -F.max_pool2d(
        -depth_for_min.reshape(1, 1, height, width),
        kernel_size=kernel,
        stride=1,
        padding=radius,
    ).reshape(height, width)
    local_max = F.max_pool2d(
        depth_for_max.reshape(1, 1, height, width),
        kernel_size=kernel,
        stride=1,
        padding=radius,
    ).reshape(height, width)

    pixel_x = torch.floor(u + 0.5).long().clamp(0, width - 1)
    pixel_y = torch.floor(v + 0.5).long().clamp(0, height - 1)
    sampled_min = local_min[pixel_y, pixel_x]
    sampled_max = local_max[pixel_y, pixel_x]
    relative_spread = (sampled_max - sampled_min) / (
        sampled_max.abs() + sampled_min.abs() + 1e-6
    )
    valid = (
        torch.isfinite(sampled_min)
        & torch.isfinite(sampled_max)
        & (sampled_min > 0)
        & (relative_spread <= max_relative_spread)
    )
    return sampled_min, valid, relative_spread


def _project_token_centres(
    source_depth: torch.Tensor,
    source_intrinsic: torch.Tensor,
    source_extrinsic: torch.Tensor,
    target_intrinsic: torch.Tensor,
    target_extrinsic: torch.Tensor,
    token_height: int,
    token_width: int,
) -> tuple[torch.Tensor, ...]:
    image_height, image_width = source_depth.shape
    device = source_depth.device
    token_y, token_x = torch.meshgrid(
        (torch.arange(token_height, device=device, dtype=torch.float32) + 0.5)
        * image_height
        / token_height
        - 0.5,
        (torch.arange(token_width, device=device, dtype=torch.float32) + 0.5)
        * image_width
        / token_width
        - 0.5,
        indexing="ij",
    )
    pixel_y = torch.floor(token_y + 0.5).long().clamp(0, image_height - 1).reshape(-1)
    pixel_x = torch.floor(token_x + 0.5).long().clamp(0, image_width - 1).reshape(-1)
    source_index = torch.arange(token_height * token_width, device=device)
    pixels = torch.stack(
        [
            token_x.reshape(-1),
            token_y.reshape(-1),
            torch.ones_like(token_x).reshape(-1),
        ],
        dim=-1,
    ).T
    source_z = source_depth[pixel_y, pixel_x].float()
    source_xyz = (torch.linalg.inv(source_intrinsic.float()) @ pixels) * source_z.unsqueeze(0)

    r_source, t_source = source_extrinsic[:, :3].float(), source_extrinsic[:, 3].float()
    r_target, t_target = target_extrinsic[:, :3].float(), target_extrinsic[:, 3].float()
    r_relative = r_target @ r_source.T
    t_relative = t_target - r_relative @ t_source
    target_xyz = r_relative @ source_xyz + t_relative.unsqueeze(1)
    target_z = target_xyz[2]
    projected = target_intrinsic.float() @ target_xyz
    u = projected[0] / projected[2].clamp_min(1e-6)
    v = projected[1] / projected[2].clamp_min(1e-6)
    inside = (
        (source_z > 0)
        & (target_z > 0)
        & (u >= 0)
        & (u <= image_width - 1)
        & (v >= 0)
        & (v <= image_height - 1)
    )
    return source_index, pixel_x, pixel_y, u, v, target_z, inside


def _project_token_samples(
    source_depth: torch.Tensor,
    source_intrinsic: torch.Tensor,
    source_extrinsic: torch.Tensor,
    target_intrinsic: torch.Tensor,
    target_extrinsic: torch.Tensor,
    token_height: int,
    token_width: int,
    samples_per_axis: int,
) -> tuple[torch.Tensor, ...]:
    """Project a regular sub-token sample grid instead of only token centres."""
    if samples_per_axis < 1:
        raise ValueError("samples_per_axis must be positive")
    image_height, image_width = source_depth.shape
    device = source_depth.device
    offsets = (
        torch.arange(samples_per_axis, device=device, dtype=torch.float32) + 0.5
    ) / samples_per_axis
    token_y, token_x, offset_y, offset_x = torch.meshgrid(
        torch.arange(token_height, device=device, dtype=torch.float32),
        torch.arange(token_width, device=device, dtype=torch.float32),
        offsets,
        offsets,
        indexing="ij",
    )
    pixel_y_float = (
        (token_y + offset_y) * image_height / token_height - 0.5
    )
    pixel_x_float = (
        (token_x + offset_x) * image_width / token_width - 0.5
    )
    pixel_y = (
        torch.floor(pixel_y_float + 0.5)
        .long()
        .clamp(0, image_height - 1)
        .reshape(-1)
    )
    pixel_x = (
        torch.floor(pixel_x_float + 0.5)
        .long()
        .clamp(0, image_width - 1)
        .reshape(-1)
    )
    token_id = (
        token_y.long() * token_width + token_x.long()
    ).reshape(-1)
    pixels = torch.stack(
        [
            pixel_x_float.reshape(-1),
            pixel_y_float.reshape(-1),
            torch.ones_like(pixel_x_float).reshape(-1),
        ],
        dim=-1,
    ).T
    source_z = source_depth[pixel_y, pixel_x].float()
    source_xyz = (
        torch.linalg.inv(source_intrinsic.float()) @ pixels
    ) * source_z.unsqueeze(0)

    r_source, t_source = source_extrinsic[:, :3].float(), source_extrinsic[:, 3].float()
    r_target, t_target = target_extrinsic[:, :3].float(), target_extrinsic[:, 3].float()
    r_relative = r_target @ r_source.T
    t_relative = t_target - r_relative @ t_source
    target_xyz = r_relative @ source_xyz + t_relative.unsqueeze(1)
    target_z = target_xyz[2]
    projected = target_intrinsic.float() @ target_xyz
    u = projected[0] / projected[2].clamp_min(1e-6)
    v = projected[1] / projected[2].clamp_min(1e-6)
    inside = (
        (source_z > 0)
        & (target_z > 0)
        & (u >= 0)
        & (u <= image_width - 1)
        & (v >= 0)
        & (v <= image_height - 1)
    )
    return token_id, pixel_x, pixel_y, u, v, target_z, inside


def _source_free_space_violation_map(
    *,
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    source_confidence: torch.Tensor,
    target_confidence: torch.Tensor,
    source_intrinsic: torch.Tensor,
    target_intrinsic: torch.Tensor,
    source_extrinsic: torch.Tensor,
    target_extrinsic: torch.Tensor,
    token_height: int,
    token_width: int,
    confidence_percentile: float,
    confidence_floor: float,
    source_depth_percentile: float,
    behind_threshold: float,
    front_threshold: float,
    spatial_samples_per_axis: int,
    min_spatial_support: int,
    source_depth_edge_radius: int,
    max_source_depth_spread: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """Find target draft points that occupy source-observed free space."""
    if front_threshold <= behind_threshold:
        raise ValueError(
            "source_free_space_violation requires front_threshold > behind_threshold"
        )

    spatial_tokens = token_height * token_width
    (
        target_token_id,
        target_pixel_x,
        target_pixel_y,
        source_u,
        source_v,
        target_depth_in_source,
        inside_source,
    ) = _project_token_samples(
        target_depth,
        target_intrinsic,
        target_extrinsic,
        source_intrinsic,
        source_extrinsic,
        token_height,
        token_width,
        spatial_samples_per_axis,
    )

    source_confident = _confidence_mask(
        source_confidence,
        confidence_percentile,
        confidence_floor,
    )
    target_confident = _confidence_mask(
        target_confidence,
        confidence_percentile,
        confidence_floor,
    )
    (
        sampled_source_depth,
        source_depth_edge_valid,
        source_depth_relative_spread,
    ) = _sample_conservative_source_depth(
        source_depth,
        source_u,
        source_v,
        radius=source_depth_edge_radius,
        max_relative_spread=max_source_depth_spread,
    )
    sampled_source_confident = (
        _sample_scalar(source_confident.float(), source_u, source_v) > 0.5
    )
    target_token_confident = target_confident[target_pixel_y, target_pixel_x]

    if not 0.0 < source_depth_percentile <= 100.0:
        raise ValueError("source_depth_percentile must be in (0, 100]")
    if source_depth_percentile < 100.0:
        reliable_depth = (
            source_confident
            & torch.isfinite(source_depth)
            & (source_depth > 0)
        )
        if reliable_depth.any():
            source_depth_cutoff = torch.quantile(
                source_depth[reliable_depth].float(),
                source_depth_percentile / 100.0,
            )
            source_depth_valid = sampled_source_depth <= source_depth_cutoff
        else:
            source_depth_cutoff = torch.tensor(
                float("nan"),
                device=source_depth.device,
            )
            source_depth_valid = torch.zeros_like(sampled_source_confident)
    else:
        source_depth_cutoff = torch.tensor(
            float("inf"),
            device=source_depth.device,
        )
        source_depth_valid = torch.ones_like(sampled_source_confident)

    # Positive error means the target draft point lies before the clean
    # source-view surface on the same ray. That segment was observed as empty
    # in the source frame, so the target point violates static free space.
    free_space_error = (sampled_source_depth - target_depth_in_source) / (
        sampled_source_depth.abs() + target_depth_in_source.abs() + 1e-6
    )
    geometry_valid = free_space_error >= behind_threshold
    confidence_score = (
        (free_space_error - behind_threshold)
        / (front_threshold - behind_threshold)
    ).clamp(0.0, 1.0)
    source_x = _pixel_centres_to_token_indices(
        source_u,
        source_depth.shape[1],
        token_width,
    )
    source_y = _pixel_centres_to_token_indices(
        source_v,
        source_depth.shape[0],
        token_height,
    )
    source_in_token_grid = (
        (source_x >= 0)
        & (source_x < token_width)
        & (source_y >= 0)
        & (source_y < token_height)
    )
    valid = (
        inside_source
        & source_in_token_grid
        & sampled_source_confident
        & target_token_confident
        & source_depth_valid
        & source_depth_edge_valid
        & geometry_valid
    )
    source_token_id = (
        source_y.clamp(0, token_height - 1) * token_width
        + source_x.clamp(0, token_width - 1)
    )

    source_for_target = torch.zeros(
        spatial_tokens,
        dtype=torch.long,
        device=source_depth.device,
    )
    confidence_for_target = torch.zeros(
        spatial_tokens,
        dtype=torch.float32,
        device=source_depth.device,
    )
    sample_count = spatial_samples_per_axis**2
    if min_spatial_support < 1 or min_spatial_support > sample_count:
        raise ValueError(
            "min_spatial_support must be in "
            f"[1, {sample_count}] for spatial_samples_per_axis="
            f"{spatial_samples_per_axis}"
        )
    spatial_support_count = torch.zeros(
        spatial_tokens,
        dtype=torch.long,
        device=source_depth.device,
    )
    if valid.any():
        spatial_support_count.scatter_add_(
            0,
            target_token_id[valid],
            torch.ones_like(target_token_id[valid]),
        )
        confidence_for_target.scatter_add_(
            0,
            target_token_id[valid],
            confidence_score[valid],
        )
        confidence_for_target = confidence_for_target / sample_count
        best_score = torch.full(
            (spatial_tokens,),
            -torch.inf,
            dtype=torch.float32,
            device=source_depth.device,
        )
        best_score.scatter_reduce_(
            0,
            target_token_id[valid],
            confidence_score[valid],
            reduce="amax",
            include_self=True,
        )
        best_sample = valid & (
            confidence_score
            >= best_score[target_token_id].clamp_min(0.0) - 1e-8
        )
        source_for_target[target_token_id[best_sample]] = source_token_id[best_sample]
    spatial_support_valid = spatial_support_count >= min_spatial_support
    confidence_for_target = confidence_for_target * spatial_support_valid
    source_for_target = torch.where(
        confidence_for_target > 0,
        source_for_target,
        torch.zeros_like(source_for_target),
    )

    stats = {
        "in_bounds_fraction": float(inside_source.float().mean().item()),
        "geometry_valid_fraction": float(valid.float().mean().item()),
        "source_depth_valid_fraction": float(source_depth_valid.float().mean().item()),
        "source_depth_edge_valid_fraction": float(
            source_depth_edge_valid.float().mean().item()
        ),
        "median_source_depth_relative_spread": (
            float(
                source_depth_relative_spread[
                    torch.isfinite(source_depth_relative_spread)
                ].median().item()
            )
            if torch.isfinite(source_depth_relative_spread).any()
            else float("nan")
        ),
        "source_depth_edge_radius": source_depth_edge_radius,
        "max_source_depth_spread": max_source_depth_spread,
        "source_depth_cutoff": float(source_depth_cutoff.item()),
        "visible_target_tokens": int((confidence_for_target > 0).sum().item()),
        "visible_target_fraction": float(
            (confidence_for_target > 0).float().mean().item()
        ),
        "spatial_samples_per_axis": spatial_samples_per_axis,
        "spatial_sample_count": sample_count,
        "min_spatial_support": min_spatial_support,
        "mean_spatial_support_fraction": float(
            (spatial_support_count.float() / sample_count).mean().item()
        ),
        "median_spatial_support_fraction_visible": (
            float(
                (
                    spatial_support_count[confidence_for_target > 0].float()
                    / sample_count
                ).median().item()
            )
            if (confidence_for_target > 0).any()
            else float("nan")
        ),
        "median_signed_depth_error_visible": (
            float(free_space_error[valid].median().item())
            if valid.any()
            else float("nan")
        ),
    }
    return source_for_target, confidence_for_target, stats


def _pair_map(
    *,
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    source_confidence: torch.Tensor,
    target_confidence: torch.Tensor,
    source_intrinsic: torch.Tensor,
    target_intrinsic: torch.Tensor,
    source_extrinsic: torch.Tensor,
    target_extrinsic: torch.Tensor,
    token_height: int,
    token_width: int,
    confidence_percentile: float,
    confidence_floor: float,
    source_depth_percentile: float,
    visibility_mode: str,
    behind_threshold: float,
    front_threshold: float,
    spatial_samples_per_axis: int,
    min_spatial_support: int,
    source_depth_edge_radius: int,
    max_source_depth_spread: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    if visibility_mode == "source_free_space_violation":
        return _source_free_space_violation_map(
            source_depth=source_depth,
            target_depth=target_depth,
            source_confidence=source_confidence,
            target_confidence=target_confidence,
            source_intrinsic=source_intrinsic,
            target_intrinsic=target_intrinsic,
            source_extrinsic=source_extrinsic,
            target_extrinsic=target_extrinsic,
            token_height=token_height,
            token_width=token_width,
            confidence_percentile=confidence_percentile,
            confidence_floor=confidence_floor,
            source_depth_percentile=source_depth_percentile,
            behind_threshold=behind_threshold,
            front_threshold=front_threshold,
            spatial_samples_per_axis=spatial_samples_per_axis,
            min_spatial_support=min_spatial_support,
            source_depth_edge_radius=source_depth_edge_radius,
            max_source_depth_spread=max_source_depth_spread,
        )

    spatial_tokens = token_height * token_width
    (
        token_id,
        pixel_x,
        pixel_y,
        u,
        v,
        projected_depth,
        inside,
    ) = _project_token_centres(
        source_depth,
        source_intrinsic,
        source_extrinsic,
        target_intrinsic,
        target_extrinsic,
        token_height,
        token_width,
    )
    source_confident = _confidence_mask(source_confidence, confidence_percentile, confidence_floor)
    target_confident = _confidence_mask(target_confidence, confidence_percentile, confidence_floor)
    sampled_target_depth = _sample_scalar(target_depth, u, v)
    sampled_target_confident = _sample_scalar(target_confident.float(), u, v) > 0.5
    source_token_confident = source_confident[pixel_y, pixel_x]
    source_depth_at_token = source_depth[pixel_y, pixel_x]
    if not 0.0 < source_depth_percentile <= 100.0:
        raise ValueError("source_depth_percentile must be in (0, 100]")
    if source_depth_percentile < 100.0:
        reliable_depth = (
            source_confident
            & torch.isfinite(source_depth)
            & (source_depth > 0)
        )
        if reliable_depth.any():
            source_depth_cutoff = torch.quantile(
                source_depth[reliable_depth].float(),
                source_depth_percentile / 100.0,
            )
            source_depth_valid = source_depth_at_token <= source_depth_cutoff
        else:
            source_depth_cutoff = torch.tensor(
                float("nan"),
                device=source_depth.device,
            )
            source_depth_valid = torch.zeros_like(source_token_confident)
    else:
        source_depth_cutoff = torch.tensor(
            float("inf"),
            device=source_depth.device,
        )
        source_depth_valid = torch.ones_like(source_token_confident)

    signed_relative_error = (projected_depth - sampled_target_depth) / (
        sampled_target_depth.abs() + projected_depth.abs() + 1e-6
    )
    if visibility_mode == "symmetric":
        geometry_valid = signed_relative_error.abs() <= behind_threshold
        depth_score = (1.0 - signed_relative_error.abs() / behind_threshold).clamp(0.0, 1.0)
    elif visibility_mode == "front_tolerant":
        # A source anchor slightly behind the target depth is occluded. A source
        # in front can indicate a surface that disappeared in the draft, so it
        # receives a wider tolerance instead of being rejected symmetrically.
        geometry_valid = (
            (signed_relative_error <= behind_threshold)
            & (signed_relative_error >= -front_threshold)
        )
        normalizer = torch.where(
            signed_relative_error >= 0,
            torch.full_like(signed_relative_error, behind_threshold),
            torch.full_like(signed_relative_error, front_threshold),
        )
        depth_score = (1.0 - signed_relative_error.abs() / normalizer).clamp(0.0, 1.0)
    elif visibility_mode == "source_zbuffer":
        geometry_valid = torch.ones_like(inside)
        depth_score = torch.ones_like(projected_depth)
    elif visibility_mode == "unsupported_foreground":
        # Positive signed error means the anchor surface reprojects behind the
        # target draft depth. When every clean anchor predicts a farther static
        # surface, the closer target surface is unsupported foreground and is a
        # useful candidate for hallucinated geometry. The lower threshold
        # rejects normal depth noise; the upper threshold saturates confidence.
        if front_threshold <= behind_threshold:
            raise ValueError(
                "unsupported_foreground requires front_threshold > behind_threshold"
            )
        geometry_valid = signed_relative_error >= behind_threshold
        depth_score = (
            (signed_relative_error - behind_threshold)
            / (front_threshold - behind_threshold)
        ).clamp(0.0, 1.0)
    elif visibility_mode == "missing_source_surface":
        # Negative signed error means the clean-anchor surface reprojects in
        # front of the surface visible in the target draft. In a static scene,
        # that nearer anchor surface should still occlude the farther target
        # surface, so its absence is a candidate deletion/deformation.
        if front_threshold <= behind_threshold:
            raise ValueError(
                "missing_source_surface requires "
                "front_threshold > behind_threshold"
            )
        missing_surface_error = -signed_relative_error
        geometry_valid = missing_surface_error >= behind_threshold
        depth_score = (
            (missing_surface_error - behind_threshold)
            / (front_threshold - behind_threshold)
        ).clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unknown visibility_mode: {visibility_mode}")

    valid = (
        inside
        & source_token_confident
        & sampled_target_confident
        & source_depth_valid
        & geometry_valid
    )
    target_x = _pixel_centres_to_token_indices(
        u,
        source_depth.shape[1],
        token_width,
    )
    target_y = _pixel_centres_to_token_indices(
        v,
        source_depth.shape[0],
        token_height,
    )
    valid = valid & (target_x >= 0) & (target_x < token_width) & (target_y >= 0) & (target_y < token_height)
    target_id = target_y.clamp(0, token_height - 1) * token_width + target_x.clamp(0, token_width - 1)

    z_buffer = torch.full(
        (spatial_tokens,),
        float("inf"),
        device=source_depth.device,
        dtype=torch.float32,
    )
    if valid.any():
        z_buffer.scatter_reduce_(
            0,
            target_id[valid],
            projected_depth[valid],
            reduce="amin",
            include_self=True,
        )
    visible = valid & (projected_depth <= z_buffer[target_id] + 1e-5)

    source_for_target = torch.zeros(spatial_tokens, dtype=torch.long, device=source_depth.device)
    confidence_for_target = torch.zeros(spatial_tokens, dtype=torch.float32, device=source_depth.device)
    if visible.any():
        source_for_target[target_id[visible]] = token_id[visible]
        confidence_for_target[target_id[visible]] = depth_score[visible]

    stats = {
        "in_bounds_fraction": float(inside.float().mean().item()),
        "geometry_valid_fraction": float(valid.float().mean().item()),
        "source_depth_valid_fraction": float(source_depth_valid.float().mean().item()),
        "source_depth_cutoff": float(source_depth_cutoff.item()),
        "visible_target_tokens": int((confidence_for_target > 0).sum().item()),
        "visible_target_fraction": float((confidence_for_target > 0).float().mean().item()),
        "median_signed_depth_error_visible": (
            float(signed_relative_error[visible].median().item()) if visible.any() else float("nan")
        ),
    }
    return source_for_target, confidence_for_target, stats


def build_anchor_transport_map(
    *,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    depth_map: torch.Tensor,
    confidence_map: torch.Tensor,
    selected_video_frames: list[int],
    anchor_video_frames: list[int],
    target_video_frames: list[int],
    token_grid: tuple[int, int, int],
    temporal_scale: int,
    memory_slots: int = 3,
    min_anchor_support: int = 1,
    min_temporal_support: int = 1,
    min_temporal_observations: int = 1,
    confidence_percentile: float = 20.0,
    confidence_floor: float = 0.2,
    source_depth_percentile: float = 100.0,
    visibility_mode: str = "front_tolerant",
    behind_threshold: float = 0.08,
    front_threshold: float = 0.35,
    rgb_map: torch.Tensor | None = None,
    appearance_l1_threshold: float = 0.0,
    spatial_samples_per_axis: int = 1,
    min_spatial_support: int = 1,
    source_depth_edge_radius: int = 1,
    max_source_depth_spread: float = 0.10,
    metadata: dict[str, Any] | None = None,
) -> DraftGeometryMap:
    """Build direct clean-anchor-to-target maps without temporal chaining."""
    if len(selected_video_frames) != intrinsic.shape[0]:
        raise ValueError("selected_video_frames and geometry sequence length differ")
    if depth_map.ndim != 4 or confidence_map.ndim != 4:
        raise ValueError("Expected depth/confidence [F,H,W,1]")
    if rgb_map is not None and (
        rgb_map.ndim != 4
        or rgb_map.shape[0] != len(selected_video_frames)
        or rgb_map.shape[-1] != 3
    ):
        raise ValueError("Expected rgb_map [F,H,W,3] aligned with selected_video_frames")
    appearance_modes = {
        "source_free_space_violation",
        "missing_source_surface",
    }
    if (
        appearance_l1_threshold > 0.0
        and visibility_mode not in appearance_modes
    ):
        raise ValueError(
            "appearance_l1_threshold is currently defined only for "
            f"{sorted(appearance_modes)}"
        )
    if memory_slots < 1:
        raise ValueError("memory_slots must be positive")
    if min_anchor_support < 1:
        raise ValueError("min_anchor_support must be positive")
    if min_temporal_support < 1:
        raise ValueError("min_temporal_support must be positive")
    if min_temporal_observations < 1:
        raise ValueError("min_temporal_observations must be positive")
    if spatial_samples_per_axis < 1:
        raise ValueError("spatial_samples_per_axis must be positive")
    if (
        min_spatial_support < 1
        or min_spatial_support > spatial_samples_per_axis**2
    ):
        raise ValueError(
            "min_spatial_support must be in "
            f"[1, {spatial_samples_per_axis**2}]"
        )
    if source_depth_edge_radius < 0:
        raise ValueError("source_depth_edge_radius must be non-negative")
    if max_source_depth_spread < 0.0:
        raise ValueError("max_source_depth_spread must be non-negative")
    if not set(anchor_video_frames).issubset(selected_video_frames):
        raise ValueError("Every anchor frame must be present in selected_video_frames")
    if not set(target_video_frames).issubset(selected_video_frames):
        raise ValueError("Every target frame must be present in selected_video_frames")

    latent_frames, token_height, token_width = token_grid
    spatial_tokens = token_height * token_width
    device = depth_map.device
    if appearance_l1_threshold < 0.0:
        raise ValueError("appearance_l1_threshold must be non-negative")
    if appearance_l1_threshold > 0.0 and rgb_map is None:
        raise ValueError(
            "rgb_map is required when appearance_l1_threshold is enabled"
        )
    rgb_tokens = None
    if rgb_map is not None:
        rgb_tokens = F.interpolate(
            rgb_map.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2),
            size=(token_height, token_width),
            mode="area",
        ).permute(0, 2, 3, 1).reshape(
            len(selected_video_frames),
            spatial_tokens,
            3,
        )
    source_time = torch.zeros(
        (latent_frames - 1, spatial_tokens, memory_slots),
        dtype=torch.long,
        device=device,
    )
    source_index = torch.zeros_like(source_time)
    transport_confidence = torch.zeros(
        (latent_frames - 1, token_height, token_width, memory_slots),
        dtype=torch.float32,
        device=device,
    )
    conflict_confidence = torch.zeros(
        (latent_frames - 1, token_height, token_width),
        dtype=torch.float32,
        device=device,
    )
    observed_background_time = torch.zeros_like(source_time)
    observed_background_index = torch.zeros_like(source_index)
    observed_background_confidence = torch.zeros_like(transport_confidence)
    temporal_support_count = torch.zeros(
        (latent_frames - 1, token_height, token_width),
        dtype=torch.long,
        device=device,
    )
    temporal_observation_count = torch.zeros(
        (latent_frames - 1,),
        dtype=torch.long,
        device=device,
    )
    sequence_lookup = {frame: index for index, frame in enumerate(selected_video_frames)}
    pair_stats: list[dict[str, float | int | str]] = []
    frame_candidates_by_time: dict[int, list[tuple[int, list[tuple]]]] = {}

    for target_frame in target_video_frames:
        target_sequence_index = sequence_lookup[target_frame]
        target_t = frame_to_latent_index(target_frame, temporal_scale, latent_frames)
        if target_t <= 0:
            continue
        candidates = []
        for source_frame in anchor_video_frames:
            source_t = frame_to_latent_index(source_frame, temporal_scale, latent_frames)
            if source_t >= target_t:
                continue
            source_sequence_index = sequence_lookup[source_frame]
            source_for_target, confidence_for_target, stats = _pair_map(
                source_depth=depth_map[source_sequence_index, ..., 0],
                target_depth=depth_map[target_sequence_index, ..., 0],
                source_confidence=confidence_map[source_sequence_index, ..., 0],
                target_confidence=confidence_map[target_sequence_index, ..., 0],
                source_intrinsic=intrinsic[source_sequence_index],
                target_intrinsic=intrinsic[target_sequence_index],
                source_extrinsic=extrinsic[source_sequence_index],
                target_extrinsic=extrinsic[target_sequence_index],
                token_height=token_height,
                token_width=token_width,
                confidence_percentile=confidence_percentile,
                confidence_floor=confidence_floor,
                source_depth_percentile=source_depth_percentile,
                visibility_mode=visibility_mode,
                behind_threshold=behind_threshold,
                front_threshold=front_threshold,
                spatial_samples_per_axis=spatial_samples_per_axis,
                min_spatial_support=min_spatial_support,
                source_depth_edge_radius=source_depth_edge_radius,
                max_source_depth_spread=max_source_depth_spread,
            )
            if appearance_l1_threshold > 0.0:
                source_rgb_tokens = rgb_tokens[source_sequence_index]
                target_rgb_tokens = rgb_tokens[target_sequence_index]
                appearance_l1 = (
                    target_rgb_tokens
                    - source_rgb_tokens[source_for_target]
                ).abs().mean(dim=-1)
                appearance_valid = appearance_l1 >= appearance_l1_threshold
                stats["visible_target_tokens_before_appearance"] = stats[
                    "visible_target_tokens"
                ]
                stats["visible_target_fraction_before_appearance"] = stats[
                    "visible_target_fraction"
                ]
                confidence_for_target = confidence_for_target * appearance_valid
                source_for_target = torch.where(
                    confidence_for_target > 0,
                    source_for_target,
                    torch.zeros_like(source_for_target),
                )
                visible_count = int((confidence_for_target > 0).sum().item())
                stats["visible_target_tokens"] = visible_count
                stats["visible_target_fraction"] = visible_count / spatial_tokens
                stats["appearance_l1_threshold"] = appearance_l1_threshold
                stats["median_appearance_l1_visible"] = (
                    float(appearance_l1[confidence_for_target > 0].median().item())
                    if visible_count > 0
                    else float("nan")
                )
            stats.update(
                {
                    "source_frame": source_frame,
                    "target_frame": target_frame,
                    "source_latent_index": source_t,
                    "target_latent_index": target_t,
                    "visibility_mode": visibility_mode,
                }
            )
            candidates.append(
                (
                    stats["visible_target_fraction"],
                    source_frame,
                    source_t,
                    source_for_target,
                    confidence_for_target,
                    stats,
                )
            )

        if candidates:
            anchor_support = torch.stack(
                [candidate[4] > 0 for candidate in candidates],
                dim=0,
            ).sum(dim=0)
            support_mask = anchor_support >= min_anchor_support
            consensus_candidates = []
            for (
                _,
                source_frame,
                source_t,
                source_for_target,
                confidence_for_target,
                stats,
            ) in candidates:
                stats = dict(stats)
                stats["visible_target_tokens_before_consensus"] = stats[
                    "visible_target_tokens"
                ]
                stats["visible_target_fraction_before_consensus"] = stats[
                    "visible_target_fraction"
                ]
                stats["min_anchor_support"] = min_anchor_support
                confidence_for_target = confidence_for_target * support_mask
                source_for_target = torch.where(
                    confidence_for_target > 0,
                    source_for_target,
                    torch.zeros_like(source_for_target),
                )
                visible_count = int((confidence_for_target > 0).sum().item())
                visible_fraction = visible_count / spatial_tokens
                stats["visible_target_tokens"] = visible_count
                stats["visible_target_fraction"] = visible_fraction
                consensus_candidates.append(
                    (
                        visible_fraction,
                        source_frame,
                        source_t,
                        source_for_target,
                        confidence_for_target,
                        stats,
                    )
                )
            candidates = consensus_candidates

        # Prefer anchors with broad overlap. Ties prefer the temporally closer
        # clean anchor, which is usually less affected by camera-pose error.
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected_candidates = candidates[:memory_slots]
        for slot, candidate in enumerate(selected_candidates):
            stats = candidate[5]
            stats["memory_slot"] = slot
            pair_stats.append(stats)
        frame_candidates_by_time.setdefault(target_t, []).append(
            (target_frame, selected_candidates)
        )

    conflict_modes = {
        "unsupported_foreground",
        "source_free_space_violation",
        "missing_source_surface",
    }
    for target_t, frame_entries in sorted(frame_candidates_by_time.items()):
        frame_entries = sorted(frame_entries, key=lambda item: item[0])
        observation_count = len(frame_entries)
        temporal_observation_count[target_t - 1] = observation_count
        per_frame_confidence = []
        flattened_candidates = []
        for target_frame, candidates in frame_entries:
            if candidates:
                candidate_confidence = torch.stack(
                    [candidate[4] for candidate in candidates],
                    dim=0,
                )
                per_frame_confidence.append(candidate_confidence.amax(dim=0))
                flattened_candidates.extend(
                    sorted(candidates, key=lambda candidate: candidate[1])
                )
            else:
                per_frame_confidence.append(
                    torch.zeros(
                        spatial_tokens,
                        dtype=torch.float32,
                        device=device,
                    )
                )

        frame_confidence = torch.stack(per_frame_confidence, dim=0)
        support_count = (frame_confidence > 0).sum(dim=0)
        temporal_support_count[target_t - 1] = support_count.reshape(
            token_height,
            token_width,
        )
        enough_observations = observation_count >= min_temporal_observations
        support_mask = (
            support_count >= min_temporal_support
            if enough_observations
            else torch.zeros_like(support_count, dtype=torch.bool)
        )
        support_fraction = support_count.float() / max(observation_count, 1)
        aggregate_confidence = frame_confidence.mean(dim=0) * support_mask

        if visibility_mode in conflict_modes:
            conflict_confidence[target_t - 1] = aggregate_confidence.reshape(
                token_height,
                token_width,
            )

        if flattened_candidates:
            candidate_source_time = torch.tensor(
                [candidate[2] for candidate in flattened_candidates],
                dtype=torch.long,
                device=device,
            )
            candidate_source_index = torch.stack(
                [candidate[3] for candidate in flattened_candidates],
                dim=0,
            )
            candidate_confidence = torch.stack(
                [candidate[4] for candidate in flattened_candidates],
                dim=0,
            )
            selection_confidence = (
                candidate_confidence
                * support_mask.unsqueeze(0)
                * support_fraction.unsqueeze(0)
            )
            selected_slots = min(memory_slots, len(flattened_candidates))
            top_confidence, top_candidate = selection_confidence.T.topk(
                selected_slots,
                dim=-1,
            )
            selected_source_index = candidate_source_index.T.gather(
                1,
                top_candidate,
            )
            selected_source_time = candidate_source_time.reshape(
                1,
                -1,
            ).expand(
                spatial_tokens,
                -1,
            ).gather(
                1,
                top_candidate,
            )
            if visibility_mode not in conflict_modes:
                source_time[target_t - 1, :, :selected_slots] = (
                    selected_source_time
                )
                source_index[target_t - 1, :, :selected_slots] = (
                    selected_source_index
                )
                transport_confidence[target_t - 1, ..., :selected_slots] = (
                    top_confidence.reshape(
                        token_height,
                        token_width,
                        selected_slots,
                    )
                )
            elif visibility_mode == "source_free_space_violation":
                observed_background_time[
                    target_t - 1, :, :selected_slots
                ] = selected_source_time
                observed_background_index[
                    target_t - 1, :, :selected_slots
                ] = selected_source_index
                observed_background_confidence[
                    target_t - 1, ..., :selected_slots
                ] = top_confidence.reshape(
                    token_height,
                    token_width,
                    selected_slots,
                )

        active_temporal = aggregate_confidence > 0
        pair_stats.append(
            {
                "kind": "temporal_aggregate",
                "target_latent_index": target_t,
                "target_frames": ",".join(
                    str(target_frame) for target_frame, _ in frame_entries
                ),
                "temporal_observation_count": observation_count,
                "min_temporal_observations": min_temporal_observations,
                "min_temporal_support": min_temporal_support,
                "active_target_tokens": int(active_temporal.sum().item()),
                "active_target_fraction": float(
                    active_temporal.float().mean().item()
                ),
                "mean_support_fraction_active": (
                    float(support_fraction[active_temporal].mean().item())
                    if active_temporal.any()
                    else 0.0
                ),
            }
        )

    map_metadata = {
        "format_version": 3,
        "selected_video_frames": selected_video_frames,
        "anchor_video_frames": anchor_video_frames,
        "target_video_frames": target_video_frames,
        "token_grid": list(token_grid),
        "temporal_scale": temporal_scale,
        "memory_slots": memory_slots,
        "min_anchor_support": min_anchor_support,
        "min_temporal_support": min_temporal_support,
        "min_temporal_observations": min_temporal_observations,
        "spatial_samples_per_axis": spatial_samples_per_axis,
        "min_spatial_support": min_spatial_support,
        "source_depth_edge_radius": source_depth_edge_radius,
        "max_source_depth_spread": max_source_depth_spread,
        "visibility_mode": visibility_mode,
        "confidence_percentile": confidence_percentile,
        "confidence_floor": confidence_floor,
        "source_depth_percentile": source_depth_percentile,
        "behind_threshold": behind_threshold,
        "front_threshold": front_threshold,
        "appearance_l1_threshold": appearance_l1_threshold,
        "evidence_semantics": {
            "confidence": "same_surface_transport",
            "conflict_confidence": "source_observed_free_space_conflict",
            "observed_background_confidence": (
                "source_ray_background_behind_free_space_conflict"
            ),
        },
    }
    if metadata:
        map_metadata.update(metadata)
    return DraftGeometryMap(
        source_time=source_time,
        source_index=source_index,
        confidence=transport_confidence,
        conflict_confidence=conflict_confidence,
        observed_background_time=observed_background_time,
        observed_background_index=observed_background_index,
        observed_background_confidence=observed_background_confidence,
        temporal_support_count=temporal_support_count,
        temporal_observation_count=temporal_observation_count,
        pair_stats=pair_stats,
        metadata=map_metadata,
    )


def save_draft_geometry_map(transport: DraftGeometryMap, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "source_time": transport.source_time.detach().cpu(),
            "source_index": transport.source_index.detach().cpu(),
            "confidence": transport.confidence.detach().cpu(),
            "conflict_confidence": transport.conflict_confidence.detach().cpu(),
            "observed_background_time": (
                transport.observed_background_time.detach().cpu()
            ),
            "observed_background_index": (
                transport.observed_background_index.detach().cpu()
            ),
            "observed_background_confidence": (
                transport.observed_background_confidence.detach().cpu()
            ),
            "temporal_support_count": transport.temporal_support_count.detach().cpu(),
            "temporal_observation_count": (
                transport.temporal_observation_count.detach().cpu()
            ),
            "pair_stats": transport.pair_stats,
            "metadata": transport.metadata,
        },
        path,
    )


def load_draft_geometry_map(path: str | Path, device: torch.device | str) -> DraftGeometryMap:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"source_time", "source_index", "confidence", "pair_stats", "metadata"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Draft geometry map is missing fields: {sorted(missing)}")
    confidence = payload["confidence"].to(device=device, dtype=torch.float32)
    conflict_confidence = payload.get("conflict_confidence")
    if conflict_confidence is None:
        conflict_confidence = confidence.amax(dim=-1)
    conflict_confidence = conflict_confidence.to(
        device=device,
        dtype=torch.float32,
    )
    observed_background_time = payload.get(
        "observed_background_time",
        torch.zeros_like(payload["source_time"]),
    ).to(device=device, dtype=torch.long)
    observed_background_index = payload.get(
        "observed_background_index",
        torch.zeros_like(payload["source_index"]),
    ).to(device=device, dtype=torch.long)
    observed_background_confidence = payload.get(
        "observed_background_confidence",
        torch.zeros_like(confidence),
    ).to(device=device, dtype=torch.float32)
    temporal_support_count = payload.get("temporal_support_count")
    if temporal_support_count is None:
        temporal_support_count = (conflict_confidence > 0).long()
    temporal_support_count = temporal_support_count.to(
        device=device,
        dtype=torch.long,
    )
    temporal_observation_count = payload.get("temporal_observation_count")
    if temporal_observation_count is None:
        temporal_observation_count = (
            temporal_support_count.flatten(1).amax(dim=1) > 0
        ).long()
    temporal_observation_count = temporal_observation_count.to(
        device=device,
        dtype=torch.long,
    )
    return DraftGeometryMap(
        source_time=payload["source_time"].to(device=device, dtype=torch.long),
        source_index=payload["source_index"].to(device=device, dtype=torch.long),
        confidence=confidence,
        conflict_confidence=conflict_confidence,
        observed_background_time=observed_background_time,
        observed_background_index=observed_background_index,
        observed_background_confidence=observed_background_confidence,
        temporal_support_count=temporal_support_count,
        temporal_observation_count=temporal_observation_count,
        pair_stats=payload["pair_stats"],
        metadata=payload["metadata"],
    )
