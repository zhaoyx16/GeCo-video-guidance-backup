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
    confidence: torch.Tensor  # [T-1, Ht, Wt, K], float
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
    return finite & (confidence >= threshold)


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
        / token_height,
        (torch.arange(token_width, device=device, dtype=torch.float32) + 0.5)
        * image_width
        / token_width,
        indexing="ij",
    )
    pixel_y = token_y.round().long().clamp(0, image_height - 1).reshape(-1)
    pixel_x = token_x.round().long().clamp(0, image_width - 1).reshape(-1)
    source_index = torch.arange(token_height * token_width, device=device)
    pixels = torch.stack(
        [pixel_x.float(), pixel_y.float(), torch.ones_like(pixel_x, dtype=torch.float32)],
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
    visibility_mode: str,
    behind_threshold: float,
    front_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
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
    else:
        raise ValueError(f"Unknown visibility_mode: {visibility_mode}")

    valid = inside & source_token_confident & sampled_target_confident & geometry_valid
    target_x = (u * token_width / source_depth.shape[1] - 0.5).round().long()
    target_y = (v * token_height / source_depth.shape[0] - 0.5).round().long()
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
    confidence_percentile: float = 20.0,
    confidence_floor: float = 0.2,
    visibility_mode: str = "front_tolerant",
    behind_threshold: float = 0.08,
    front_threshold: float = 0.35,
    metadata: dict[str, Any] | None = None,
) -> DraftGeometryMap:
    """Build direct clean-anchor-to-target maps without temporal chaining."""
    if len(selected_video_frames) != intrinsic.shape[0]:
        raise ValueError("selected_video_frames and geometry sequence length differ")
    if depth_map.ndim != 4 or confidence_map.ndim != 4:
        raise ValueError("Expected depth/confidence [F,H,W,1]")
    if memory_slots < 1:
        raise ValueError("memory_slots must be positive")
    if not set(anchor_video_frames).issubset(selected_video_frames):
        raise ValueError("Every anchor frame must be present in selected_video_frames")
    if not set(target_video_frames).issubset(selected_video_frames):
        raise ValueError("Every target frame must be present in selected_video_frames")

    latent_frames, token_height, token_width = token_grid
    spatial_tokens = token_height * token_width
    device = depth_map.device
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
    sequence_lookup = {frame: index for index, frame in enumerate(selected_video_frames)}
    pair_stats: list[dict[str, float | int | str]] = []

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
                visibility_mode=visibility_mode,
                behind_threshold=behind_threshold,
                front_threshold=front_threshold,
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

        # Prefer anchors with broad overlap. Ties prefer the temporally closer
        # clean anchor, which is usually less affected by camera-pose error.
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for slot, candidate in enumerate(candidates[:memory_slots]):
            _, _, source_t, source_for_target, confidence_for_target, stats = candidate
            source_time[target_t - 1, :, slot] = source_t
            source_index[target_t - 1, :, slot] = source_for_target
            transport_confidence[target_t - 1, ..., slot] = confidence_for_target.reshape(
                token_height, token_width
            )
            stats["memory_slot"] = slot
            pair_stats.append(stats)

    map_metadata = {
        "format_version": 1,
        "selected_video_frames": selected_video_frames,
        "anchor_video_frames": anchor_video_frames,
        "target_video_frames": target_video_frames,
        "token_grid": list(token_grid),
        "temporal_scale": temporal_scale,
        "memory_slots": memory_slots,
        "visibility_mode": visibility_mode,
        "confidence_percentile": confidence_percentile,
        "confidence_floor": confidence_floor,
        "behind_threshold": behind_threshold,
        "front_threshold": front_threshold,
    }
    if metadata:
        map_metadata.update(metadata)
    return DraftGeometryMap(
        source_time=source_time,
        source_index=source_index,
        confidence=transport_confidence,
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
    return DraftGeometryMap(
        source_time=payload["source_time"].to(device=device, dtype=torch.long),
        source_index=payload["source_index"].to(device=device, dtype=torch.long),
        confidence=payload["confidence"].to(device=device, dtype=torch.float32),
        pair_stats=payload["pair_stats"],
        metadata=payload["metadata"],
    )
