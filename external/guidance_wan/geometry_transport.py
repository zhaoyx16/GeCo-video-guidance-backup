"""Geometry-validated source-token maps for frozen Wan attention transport.

The module intentionally contains no model, VAE, or gradient logic.  Given
camera/depth estimates for selected decoded frames, it makes a sparse map from
target Wan spatial tokens to source tokens by 3D reprojection and a depth
visibility test.  A pipeline can use the map to choose *where* a transported
value comes from without changing Q/K attention logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GeometryTransportMap:
    """Maps targets t=1..T-1 to a visible source token, or confidence zero."""

    source_time: torch.Tensor  # [T-1, P, K], long
    source_index: torch.Tensor  # [T-1, P, K], long
    confidence: torch.Tensor  # [T-1, Ht, Wt, K], float
    pair_stats: list[dict[str, float | int]]


def frame_to_latent_index(frame_index: int, temporal_scale: int, latent_frames: int) -> int:
    """Wan's causal VAE mapping, matching the existing debug x0 decode block."""
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
    return finite & (confidence >= threshold)


def _sample_scalar(map_hw: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    height, width = map_hw.shape
    grid = torch.stack(
        [2.0 * u / max(width - 1, 1) - 1.0, 2.0 * v / max(height - 1, 1) - 1.0], dim=-1
    ).reshape(1, -1, 1, 2)
    sampled = F.grid_sample(
        map_hw.float().reshape(1, 1, height, width),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.reshape(-1)


def _project_token_centers(
    source_depth: torch.Tensor,
    source_intrinsic: torch.Tensor,
    source_extrinsic: torch.Tensor,
    target_intrinsic: torch.Tensor,
    target_extrinsic: torch.Tensor,
    token_height: int,
    token_width: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Project source token centres and return source token id, target uv/z, bounds."""
    image_height, image_width = source_depth.shape
    device = source_depth.device
    dtype = torch.float32
    token_y, token_x = torch.meshgrid(
        (torch.arange(token_height, device=device, dtype=dtype) + 0.5) * image_height / token_height,
        (torch.arange(token_width, device=device, dtype=dtype) + 0.5) * image_width / token_width,
        indexing="ij",
    )
    pixel_y = token_y.round().long().clamp(0, image_height - 1).reshape(-1)
    pixel_x = token_x.round().long().clamp(0, image_width - 1).reshape(-1)
    source_index = (torch.arange(token_height, device=device)[:, None] * token_width + torch.arange(token_width, device=device)[None, :]).reshape(-1)
    pixels = torch.stack(
        [pixel_x.float(), pixel_y.float(), torch.ones_like(pixel_x, dtype=dtype)], dim=-1
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
    inside = (source_z > 0) & (target_z > 0) & (u >= 0) & (u <= image_width - 1) & (v >= 0) & (v <= image_height - 1)
    return source_index, pixel_x, pixel_y, u, v, target_z, inside


def build_geometry_transport_map(
    *,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    depth_map: torch.Tensor,
    confidence_map: torch.Tensor,
    selected_video_frames: list[int],
    token_grid: tuple[int, int, int],
    temporal_scale: int,
    confidence_percentile: float = 20.0,
    confidence_floor: float = 0.2,
    depth_relative_threshold: float = 0.15,
    memory_lookback: int = 1,
) -> GeometryTransportMap:
    """Build adjacent selected-frame, z-buffered static-surface transport maps.

    `intrinsic`, `extrinsic`, `depth_map`, and `confidence_map` all index the
    *same* `selected_video_frames` sequence.  Each consecutive pair projects
    source token centres into the target.  A map entry is active only if both
    VGGT confidence maps accept it, it remains inside the target image, and
    the target depth agrees with the projected source depth.  Collisions are
    resolved by retaining the closest projected point (a single-view z-buffer).
    """
    if len(selected_video_frames) != intrinsic.shape[0]:
        raise ValueError("selected_video_frames and geometry sequence length differ")
    if depth_map.ndim != 4 or confidence_map.ndim != 4:
        raise ValueError("Expected depth/confidence [F,H,W,1]")
    latent_frames, token_height, token_width = token_grid
    spatial_tokens = token_height * token_width
    device = depth_map.device
    if memory_lookback < 1:
        raise ValueError("memory_lookback must be at least one")
    source_time = torch.zeros((latent_frames - 1, spatial_tokens, memory_lookback), dtype=torch.long, device=device)
    source_index = torch.zeros_like(source_time)
    transport_confidence = torch.zeros(
        (latent_frames - 1, token_height, token_width, memory_lookback), dtype=torch.float32, device=device
    )
    pair_stats: list[dict[str, float | int]] = []

    for target_sequence_index in range(1, len(selected_video_frames)):
        for lag in range(1, min(memory_lookback, target_sequence_index) + 1):
            source_sequence_index = target_sequence_index - lag
            source_frame = selected_video_frames[source_sequence_index]
            target_frame = selected_video_frames[target_sequence_index]
            source_t = frame_to_latent_index(source_frame, temporal_scale, latent_frames)
            target_t = frame_to_latent_index(target_frame, temporal_scale, latent_frames)
            if target_t <= 0 or target_t == source_t:
                continue

            source_depth = depth_map[source_sequence_index, ..., 0]
            target_depth = depth_map[target_sequence_index, ..., 0]
            source_confidence = confidence_map[source_sequence_index, ..., 0]
            target_confidence = confidence_map[target_sequence_index, ..., 0]
            source_confident = _confidence_mask(source_confidence, confidence_percentile, confidence_floor)
            target_confident = _confidence_mask(target_confidence, confidence_percentile, confidence_floor)
            token_id, pixel_x, pixel_y, u, v, projected_depth, inside = _project_token_centers(
                source_depth,
                intrinsic[source_sequence_index],
                extrinsic[source_sequence_index],
                intrinsic[target_sequence_index],
                extrinsic[target_sequence_index],
                token_height,
                token_width,
            )
            sampled_target_depth = _sample_scalar(target_depth, u, v)
            sampled_target_confidence = _sample_scalar(target_confident.float(), u, v) > 0.5
            source_token_confidence = source_confident[pixel_y, pixel_x]
            depth_error = (sampled_target_depth - projected_depth).abs() / (
                sampled_target_depth.abs() + projected_depth.abs() + 1e-6
            )
            geometry_valid = inside & source_token_confidence & sampled_target_confidence & (depth_error <= depth_relative_threshold)

            target_x = (u * token_width / source_depth.shape[1] - 0.5).round().long()
            target_y = (v * token_height / source_depth.shape[0] - 0.5).round().long()
            target_grid_valid = geometry_valid & (target_x >= 0) & (target_x < token_width) & (target_y >= 0) & (target_y < token_height)
            target_id = target_y.clamp(0, token_height - 1) * token_width + target_x.clamp(0, token_width - 1)

            # Keep the closest source surface for every target token.
            z_buffer = torch.full((spatial_tokens,), float("inf"), device=device, dtype=torch.float32)
            if target_grid_valid.any():
                z_buffer.scatter_reduce_(0, target_id[target_grid_valid], projected_depth[target_grid_valid], reduce="amin", include_self=True)
            visible = target_grid_valid & (projected_depth <= z_buffer[target_id] + 1e-5)
            memory_slot = lag - 1
            if visible.any():
                source_time[target_t - 1, target_id[visible], memory_slot] = source_t
                source_index[target_t - 1, target_id[visible], memory_slot] = token_id[visible]
                depth_score = (1.0 - depth_error[visible] / depth_relative_threshold).clamp(0.0, 1.0)
                transport_confidence[target_t - 1, ..., memory_slot].reshape(-1)[target_id[visible]] = depth_score

            pair_stats.append(
                {
                    "source_frame": source_frame,
                    "target_frame": target_frame,
                    "source_latent_index": source_t,
                    "target_latent_index": target_t,
                    "memory_slot": memory_slot,
                    "source_tokens": spatial_tokens,
                    "in_bounds_fraction": float(inside.float().mean().item()),
                    "geometry_valid_fraction": float(geometry_valid.float().mean().item()),
                    "visible_target_tokens": int((transport_confidence[target_t - 1, ..., memory_slot] > 0).sum().item()),
                    "visible_target_fraction": float((transport_confidence[target_t - 1, ..., memory_slot] > 0).float().mean().item()),
                    "median_depth_error_valid": float(depth_error[geometry_valid].median().item()) if geometry_valid.any() else float("nan"),
                }
            )

    return GeometryTransportMap(source_time, source_index, transport_confidence, pair_stats)
