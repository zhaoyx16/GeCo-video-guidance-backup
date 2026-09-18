#!/usr/bin/env python3
"""Detect static-surface identity changes with geometry-aligned DINO features."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from external.guidance_wan.draft_geometry_map import (  # noqa: E402
    DraftGeometryMap,
    frame_to_latent_index,
    load_draft_geometry_map,
    save_draft_geometry_map,
)
from scripts.build_deformation_risk_map import (  # noqa: E402
    local_depth_cv,
    make_contact_sheet,
    overlay_risk,
    pair_projection,
    read_video_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--link_map", required=True)
    parser.add_argument("--geometry", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_frames", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--feature_type",
        choices=("dino", "hog", "edge_birth"),
        default="dino",
    )
    parser.add_argument("--hog_bins", type=int, default=8)
    parser.add_argument("--edge_birth_start", type=float, default=0.20)
    parser.add_argument("--edge_birth_full", type=float, default=0.65)
    parser.add_argument("--edge_presence", type=float, default=0.35)
    parser.add_argument("--edge_source_variation", type=float, default=0.35)
    parser.add_argument("--edge_continuity", type=int, default=7)
    parser.add_argument("--detection_scale", type=int, default=1)
    parser.add_argument("--confidence_percentile", type=float, default=20.0)
    parser.add_argument("--confidence_floor", type=float, default=0.2)
    parser.add_argument("--depth_patch_size", type=int, default=31)
    parser.add_argument("--max_depth_patch_cv", type=float, default=1e6)
    parser.add_argument("--reverse_margin", type=float, default=0.03)
    parser.add_argument("--min_anchor_agreement", type=int, default=2)
    parser.add_argument("--source_consistency", type=float, default=0.70)
    parser.add_argument("--risk_quantile", type=float, default=0.90)
    parser.add_argument("--risk_full_quantile", type=float, default=0.99)
    parser.add_argument("--distance_floor", type=float, default=0.10)
    return parser.parse_args()


def finite_quantile(values: torch.Tensor, quantile: float, fallback: float) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return fallback
    return float(torch.quantile(finite, quantile).item())


@torch.inference_mode()
def extract_dino_features(
    model: torch.nn.Module,
    frames: dict[int, np.ndarray],
    frame_order: list[int],
    *,
    token_height: int,
    token_width: int,
    batch_size: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    image_height = token_height * 14
    image_width = token_width * 14
    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=device, dtype=torch.float32
    ).reshape(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=device, dtype=torch.float32
    ).reshape(1, 3, 1, 1)
    output: dict[int, torch.Tensor] = {}

    for start in range(0, len(frame_order), batch_size):
        batch_frames = frame_order[start : start + batch_size]
        batch = torch.stack(
            [
                torch.from_numpy(frames[frame])
                .permute(2, 0, 1)
                .to(dtype=torch.float32)
                / 255.0
                for frame in batch_frames
            ],
            dim=0,
        ).to(device)
        batch = F.interpolate(
            batch,
            size=(image_height, image_width),
            mode="bilinear",
            align_corners=False,
        )
        batch = (batch - mean) / std
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = model.forward_features(batch)["x_norm_patchtokens"]
        if features.shape[1] != token_height * token_width:
            raise ValueError(
                "DINO token grid mismatch: "
                f"tokens={features.shape[1]} expected={token_height * token_width}"
            )
        features = F.normalize(features.float(), dim=-1).cpu()
        for index, frame in enumerate(batch_frames):
            output[frame] = features[index]
        print(
            f"features {min(start + batch_size, len(frame_order))}/{len(frame_order)}",
            flush=True,
        )
    return output


def source_consistency_score(
    source_features: torch.Tensor,
    source_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean pairwise cosine similarity and valid pair count per token."""
    slots, spatial_tokens, _ = source_features.shape
    similarity_sum = torch.zeros(spatial_tokens, dtype=torch.float32)
    pair_count = torch.zeros(spatial_tokens, dtype=torch.long)
    for left in range(slots):
        for right in range(left + 1, slots):
            valid = source_valid[left] & source_valid[right]
            similarity = (source_features[left] * source_features[right]).sum(dim=-1)
            similarity_sum += similarity * valid.float()
            pair_count += valid.long()
    score = similarity_sum / pair_count.clamp_min(1)
    return score, pair_count


@torch.inference_mode()
def extract_hog_features(
    frames: dict[int, np.ndarray],
    frame_order: list[int],
    *,
    token_height: int,
    token_width: int,
    bins: int,
) -> dict[int, torch.Tensor]:
    if bins < 2:
        raise ValueError("hog_bins must be at least 2")
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).reshape(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    output: dict[int, torch.Tensor] = {}
    for frame_index in frame_order:
        rgb = (
            torch.from_numpy(frames[frame_index])
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        gray = (
            0.2989 * rgb[:, 0:1]
            + 0.5870 * rgb[:, 1:2]
            + 0.1140 * rgb[:, 2:3]
        )
        gradient_x = F.conv2d(gray, sobel_x, padding=1)
        gradient_y = F.conv2d(gray, sobel_y, padding=1)
        magnitude = (gradient_x.square() + gradient_y.square() + 1e-8).sqrt()
        orientation = torch.remainder(
            torch.atan2(gradient_y, gradient_x), torch.pi
        )
        height, width = gray.shape[-2:]
        if height % token_height or width % token_width:
            raise ValueError(
                f"Frame size {(height, width)} is not divisible by "
                f"token grid {(token_height, token_width)}"
            )
        patch_height = height // token_height
        patch_width = width // token_width
        bin_index = torch.floor(orientation * bins / torch.pi).long() % bins
        histograms = []
        for bin_id in range(bins):
            weighted = magnitude * (bin_index == bin_id)
            histograms.append(
                F.avg_pool2d(
                    weighted,
                    kernel_size=(patch_height, patch_width),
                    stride=(patch_height, patch_width),
                )
            )
        histogram = torch.cat(histograms, dim=1)
        histogram = histogram / histogram.sum(dim=1, keepdim=True).clamp_min(1e-8)
        edge_density = F.avg_pool2d(
            magnitude,
            kernel_size=(patch_height, patch_width),
            stride=(patch_height, patch_width),
        )
        density_scale = edge_density.flatten().median().clamp_min(1e-6)
        edge_density = torch.log1p(edge_density / density_scale)
        descriptor = torch.cat([histogram, edge_density], dim=1)
        descriptor = descriptor.permute(0, 2, 3, 1).reshape(
            token_height * token_width, bins + 1
        )
        output[frame_index] = F.normalize(descriptor, dim=-1)
    return output


@torch.inference_mode()
def extract_oriented_edge_features(
    frames: dict[int, np.ndarray],
    frame_order: list[int],
    *,
    token_height: int,
    token_width: int,
    bins: int,
) -> dict[int, torch.Tensor]:
    """Return robustly normalized oriented edge strength per image cell."""
    if bins < 2:
        raise ValueError("hog_bins must be at least 2")
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).reshape(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    output: dict[int, torch.Tensor] = {}
    for frame_index in frame_order:
        rgb = (
            torch.from_numpy(frames[frame_index])
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        gray = (
            0.2989 * rgb[:, 0:1]
            + 0.5870 * rgb[:, 1:2]
            + 0.1140 * rgb[:, 2:3]
        )
        gray = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        gradient_x = F.conv2d(gray, sobel_x, padding=1)
        gradient_y = F.conv2d(gray, sobel_y, padding=1)
        magnitude = (gradient_x.square() + gradient_y.square() + 1e-8).sqrt()
        orientation = torch.remainder(torch.atan2(gradient_y, gradient_x), torch.pi)
        height, width = gray.shape[-2:]
        if height % token_height or width % token_width:
            raise ValueError(
                f"Frame size {(height, width)} is not divisible by "
                f"token grid {(token_height, token_width)}"
            )
        patch_height = height // token_height
        patch_width = width // token_width
        bin_index = torch.floor(orientation * bins / torch.pi).long() % bins
        strengths = []
        for bin_id in range(bins):
            weighted = magnitude * (bin_index == bin_id)
            strengths.append(
                F.avg_pool2d(
                    weighted,
                    kernel_size=(patch_height, patch_width),
                    stride=(patch_height, patch_width),
                )
            )
        strength = torch.cat(strengths, dim=1)
        positive = strength[strength > 0]
        scale = (
            torch.quantile(positive, 0.75)
            if positive.numel()
            else torch.tensor(1.0)
        ).clamp_min(1e-6)
        strength = torch.log1p(strength / scale)
        output[frame_index] = strength.permute(0, 2, 3, 1).reshape(
            token_height * token_width, bins
        )
    return output


def oriented_line_continuity(
    birth_by_orientation: torch.Tensor,
    token_height: int,
    token_width: int,
    length: int,
) -> torch.Tensor:
    """Average each edge-orientation channel along its image-plane tangent."""
    if length < 1 or length % 2 == 0:
        raise ValueError("edge_continuity must be a positive odd integer")
    bins = birth_by_orientation.shape[-1]
    radius = length // 2
    kernel_size = 2 * radius + 1
    kernels = torch.zeros(bins, 1, kernel_size, kernel_size)
    for bin_id in range(bins):
        normal_angle = (bin_id + 0.5) * torch.pi / bins
        tangent_angle = normal_angle + torch.pi / 2
        points: set[tuple[int, int]] = set()
        for offset in range(-radius, radius + 1):
            dx = int(round(offset * math.cos(tangent_angle)))
            dy = int(round(offset * math.sin(tangent_angle)))
            points.add((dy, dx))
        for dy, dx in points:
            kernels[bin_id, 0, radius + dy, radius + dx] = 1.0
        kernels[bin_id] /= kernels[bin_id].sum().clamp_min(1.0)
    grid = birth_by_orientation.reshape(
        token_height, token_width, bins
    ).permute(2, 0, 1).unsqueeze(0)
    continuity = F.conv2d(
        grid,
        kernels,
        padding=radius,
        groups=bins,
    )
    return continuity.squeeze(0).permute(1, 2, 0).reshape(
        token_height * token_width, bins
    )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.source_consistency <= 1.0:
        raise ValueError("source_consistency must lie in [0, 1]")
    if not 0.0 <= args.risk_quantile < args.risk_full_quantile <= 1.0:
        raise ValueError("Require 0 <= risk_quantile < risk_full_quantile <= 1")
    if args.detection_scale < 1:
        raise ValueError("detection_scale must be positive")
    if args.detection_scale > 1 and not args.geometry:
        raise ValueError("geometry is required when detection_scale > 1")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_frames = [
        int(item) for item in args.target_frames.split(",") if item.strip()
    ]

    links = load_draft_geometry_map(args.link_map, "cpu")
    latent_frames, token_height, token_width = tuple(links.metadata["token_grid"])
    detection_height = token_height * args.detection_scale
    detection_width = token_width * args.detection_scale
    temporal_scale = int(links.metadata.get("temporal_scale", 4))
    memory_slots = links.source_time.shape[-1]
    spatial_tokens = token_height * token_width

    required_frames = set(target_frames)
    for target_frame in target_frames:
        target_time = frame_to_latent_index(
            target_frame, temporal_scale, latent_frames
        )
        if target_time <= 0:
            continue
        row = target_time - 1
        for source_time in links.source_time[row].unique().tolist():
            if int(source_time) < target_time:
                required_frames.add(int(source_time) * temporal_scale)
    frame_order = sorted(required_frames)
    video_frames = read_video_frames(Path(args.video), frame_order)
    depth = None
    depth_cv = None
    confidence = None
    intrinsic = None
    extrinsic = None
    frame_lookup = None
    if args.detection_scale > 1:
        geometry = torch.load(
            args.geometry, map_location="cpu", weights_only=False
        )
        frame_indices = [int(frame) for frame in geometry["frame_indices"]]
        frame_lookup = {frame: idx for idx, frame in enumerate(frame_indices)}
        missing_geometry = sorted(set(frame_order).difference(frame_lookup))
        if missing_geometry:
            raise ValueError(
                f"Geometry file is missing required frames: {missing_geometry}"
            )
        depth = geometry["depth"].float()[..., 0]
        depth_cv = local_depth_cv(depth, args.depth_patch_size)
        confidence = geometry["confidence"].float()[..., 0]
        intrinsic = geometry["intrinsic"].float()
        extrinsic = geometry["extrinsic"].float()

    if args.feature_type == "dino":
        print("loading DINOv2...", flush=True)
        dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        dino = dino.to(device).eval()
        dino.requires_grad_(False)
        features = extract_dino_features(
            dino,
            video_frames,
            frame_order,
            token_height=detection_height,
            token_width=detection_width,
            batch_size=args.batch_size,
            device=device,
        )
        del dino
        torch.cuda.empty_cache()
        feature_model = "dinov2_vitl14"
    elif args.feature_type == "hog":
        features = extract_hog_features(
            video_frames,
            frame_order,
            token_height=detection_height,
            token_width=detection_width,
            bins=args.hog_bins,
        )
        feature_model = f"hog_{args.hog_bins}bin_plus_edge_density"
    else:
        if not 0.0 <= args.edge_birth_start < args.edge_birth_full <= 1.0:
            raise ValueError(
                "Require 0 <= edge_birth_start < edge_birth_full <= 1"
            )
        features = extract_oriented_edge_features(
            video_frames,
            frame_order,
            token_height=detection_height,
            token_width=detection_width,
            bins=args.hog_bins,
        )
        feature_model = f"oriented_edge_birth_{args.hog_bins}bin"

    output_confidence = torch.zeros_like(links.confidence)
    pair_stats: list[dict[str, float | int | str]] = []
    raw_report_frames: list[np.ndarray] = []
    overlay_report_frames: list[np.ndarray] = []

    for target_frame in target_frames:
        target_time = frame_to_latent_index(
            target_frame, temporal_scale, latent_frames
        )
        if target_time <= 0:
            continue
        row = target_time - 1
        target_feature = features[target_frame]
        detection_tokens = detection_height * detection_width

        gathered_sources = []
        source_valid = []
        target_distances = []
        source_labels = []
        for slot in range(memory_slots):
            source_times = links.source_time[row, :, slot]
            active_link = links.confidence[row, :, :, slot].reshape(-1) > 0
            active_source_times = source_times[active_link]
            if active_source_times.numel() == 0:
                gathered_sources.append(torch.zeros_like(target_feature))
                source_valid.append(
                    torch.zeros(detection_tokens, dtype=torch.bool)
                )
                target_distances.append(
                    torch.full(
                        (detection_tokens,), torch.nan, dtype=torch.float32
                    )
                )
                source_labels.append(-1)
                continue

            source_time = int(torch.mode(active_source_times).values.item())
            source_frame = source_time * temporal_scale
            source_labels.append(source_frame)
            if args.detection_scale == 1:
                source_indices = links.source_index[row, :, slot].reshape(-1)
                detection_link = active_link
            else:
                assert (
                    frame_lookup is not None
                    and depth is not None
                    and depth_cv is not None
                    and confidence is not None
                    and intrinsic is not None
                    and extrinsic is not None
                )
                source_sequence = frame_lookup[source_frame]
                target_sequence = frame_lookup[target_frame]
                projection = pair_projection(
                    source_depth=depth[source_sequence],
                    target_depth=depth[target_sequence],
                    source_depth_cv=depth_cv[source_sequence],
                    target_depth_cv=depth_cv[target_sequence],
                    source_confidence=confidence[source_sequence],
                    target_confidence=confidence[target_sequence],
                    source_intrinsic=intrinsic[source_sequence],
                    target_intrinsic=intrinsic[target_sequence],
                    source_extrinsic=extrinsic[source_sequence],
                    target_extrinsic=extrinsic[target_sequence],
                    token_height=detection_height,
                    token_width=detection_width,
                    confidence_percentile=args.confidence_percentile,
                    confidence_floor=args.confidence_floor,
                    reverse_margin=args.reverse_margin,
                    max_depth_patch_cv=args.max_depth_patch_cv,
                )
                source_indices = projection["source_index"]
                detection_link = projection["valid"]
            gathered = features[source_frame][source_indices]
            distance = 1.0 - (gathered * target_feature).sum(dim=-1)
            gathered_sources.append(gathered)
            source_valid.append(detection_link)
            target_distances.append(
                torch.where(
                    detection_link,
                    distance,
                    torch.full_like(distance, torch.nan),
                )
            )

        gathered_stack = torch.stack(gathered_sources, dim=0)
        valid_stack = torch.stack(source_valid, dim=0)
        distance_stack = torch.stack(target_distances, dim=0)
        valid_count = valid_stack.sum(dim=0)
        enough = valid_count >= args.min_anchor_agreement
        median_distance = torch.nanmedian(distance_stack, dim=0).values
        if args.feature_type == "edge_birth":
            valid_channels = valid_stack.unsqueeze(-1)
            source_edges = torch.where(
                valid_channels,
                gathered_stack,
                torch.full_like(gathered_stack, torch.nan),
            )
            median_source = torch.nanmedian(source_edges, dim=0).values
            source_deviation = torch.nanmedian(
                (source_edges - median_source.unsqueeze(0)).abs(),
                dim=0,
            ).values / median_source.clamp_min(0.10)
            stable_channels = (
                source_deviation <= args.edge_source_variation
            ) & torch.isfinite(median_source)
            signed_birth = (
                target_feature - median_source
            ) / (target_feature + median_source + 1e-6)
            birth = (
                (signed_birth - args.edge_birth_start)
                / (args.edge_birth_full - args.edge_birth_start)
            ).clamp(0.0, 1.0)
            target_presence = (
                target_feature / max(args.edge_presence, 1e-6)
            ).clamp(0.0, 1.0)
            birth = (
                torch.nan_to_num(birth)
                * target_presence
                * stable_channels.float()
                * enough.unsqueeze(-1).float()
            )
            continuity = oriented_line_continuity(
                birth,
                detection_height,
                detection_width,
                args.edge_continuity,
            )
            raw_gate = (birth * continuity.sqrt()).amax(dim=-1)
            candidate = enough & (raw_gate > 0)
            candidate_values = raw_gate[candidate]
            risk_start = finite_quantile(
                candidate_values, args.risk_quantile, 0.0
            )
            risk_full = max(
                risk_start + 1e-6,
                finite_quantile(
                    candidate_values,
                    args.risk_full_quantile,
                    risk_start + 1e-6,
                ),
            )
            gate = (
                (raw_gate - risk_start) / (risk_full - risk_start)
            ).clamp(0.0, 1.0) * candidate.float()
            source_pair_count = valid_stack.sum(dim=0)
            source_consistency = (
                1.0
                - torch.nan_to_num(
                    source_deviation.mean(dim=-1),
                    nan=1.0,
                    posinf=1.0,
                )
            ).clamp(0.0, 1.0)
            stable_source = stable_channels.any(dim=-1)
            median_distance = 1.0 - signed_birth.amax(dim=-1)
        else:
            source_consistency, source_pair_count = source_consistency_score(
                gathered_stack, valid_stack
            )
            stable_source = (
                source_pair_count > 0
            ) & (source_consistency >= args.source_consistency)
            candidate = enough & stable_source & torch.isfinite(median_distance)
            candidate_values = median_distance[candidate]
            risk_start = max(
                args.distance_floor,
                finite_quantile(
                    candidate_values,
                    args.risk_quantile,
                    args.distance_floor,
                ),
            )
            risk_full = max(
                risk_start + 1e-6,
                finite_quantile(
                    candidate_values,
                    args.risk_full_quantile,
                    risk_start + 1e-6,
                ),
            )
            distance_gate = (
                (median_distance - risk_start) / (risk_full - risk_start)
            ).clamp(0.0, 1.0)
            consistency_gate = (
                (source_consistency - args.source_consistency)
                / max(1.0 - args.source_consistency, 1e-6)
            ).clamp(0.0, 1.0)
            gate = (
                torch.nan_to_num(distance_gate)
                * consistency_gate
                * candidate.float()
            )

        gate_grid = gate.reshape(detection_height, detection_width)
        if args.detection_scale > 1:
            output_gate = F.max_pool2d(
                gate_grid.reshape(1, 1, detection_height, detection_width),
                kernel_size=args.detection_scale,
                stride=args.detection_scale,
            )[0, 0]
        else:
            output_gate = gate_grid
        for slot in range(memory_slots):
            original = links.confidence[row, :, :, slot]
            output_confidence[row, :, :, slot] = (
                original * output_gate
            )

        active = gate > 0
        stat = {
            "target_frame": target_frame,
            "target_latent_index": target_time,
            "source_frames": ",".join(str(frame) for frame in source_labels),
            "risk_coverage": float(active.float().mean().item()),
            "risk_mean_all": float(gate.mean().item()),
            "risk_mean_active": (
                float(gate[active].mean().item()) if active.any() else 0.0
            ),
            "distance_threshold": risk_start,
            "distance_full": risk_full,
            "stable_source_fraction": float(stable_source.float().mean().item()),
            "mean_source_consistency": float(
                source_consistency[source_pair_count > 0].mean().item()
            ),
        }
        pair_stats.append(stat)
        print(json.dumps(stat, sort_keys=True), flush=True)
        frame = video_frames[target_frame]
        raw_report_frames.append(frame)
        overlay_report_frames.append(
            overlay_risk(
                frame,
                gate_grid,
                detection_height,
                detection_width,
                (
                    f"f{target_frame} identity-risk coverage="
                    f"{stat['risk_coverage']:.3f} q={risk_start:.3f}"
                ),
            )
        )

    metadata = dict(links.metadata)
    metadata.update(
        {
            "format_version": 8,
            "map_type": "geometry_aligned_feature_identity_risk",
            "video": str(Path(args.video)),
            "link_map": str(Path(args.link_map)),
            "target_video_frames": target_frames,
            "feature_model": feature_model,
            "feature_image_size": (
                [detection_height * 14, detection_width * 14]
                if args.feature_type == "dino"
                else list(video_frames[frame_order[0]].shape[:2])
            ),
            "detection_token_grid": [detection_height, detection_width],
            "output_token_grid": [token_height, token_width],
            "geometry": str(Path(args.geometry)) if args.geometry else "",
            "min_anchor_agreement": args.min_anchor_agreement,
            "source_consistency": args.source_consistency,
            "risk_quantile": args.risk_quantile,
            "risk_full_quantile": args.risk_full_quantile,
            "distance_floor": args.distance_floor,
            "edge_birth_start": args.edge_birth_start,
            "edge_birth_full": args.edge_birth_full,
            "edge_presence": args.edge_presence,
            "edge_source_variation": args.edge_source_variation,
            "edge_continuity": args.edge_continuity,
        }
    )
    output = DraftGeometryMap(
        source_time=links.source_time.clone(),
        source_index=links.source_index.clone(),
        confidence=output_confidence,
        pair_stats=pair_stats,
        metadata=metadata,
    )
    output_path = output_dir / "feature_identity_risk_map.pt"
    save_draft_geometry_map(output, output_path)
    report_path = output_dir / "feature_identity_risk_report.json"
    report_path.write_text(
        json.dumps({"metadata": metadata, "target_stats": pair_stats}, indent=2)
    )
    make_contact_sheet(
        raw_report_frames,
        overlay_report_frames,
        output_dir / "feature_identity_risk_contact_sheet.png",
    )
    print(f"saved_map={output_path}")
    print(f"saved_report={report_path}")


if __name__ == "__main__":
    main()
