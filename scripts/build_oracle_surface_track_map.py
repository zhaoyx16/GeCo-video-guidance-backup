#!/usr/bin/env python3
"""Track one oracle-selected static surface through fixed-anchor 3D correspondences.

The ROI is used only on a final clean seed frame. It selects persistent source
anchor token IDs. Later targets are gated by those source IDs, so the active
target region can move with camera motion instead of remaining screen-fixed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from external.guidance_wan.draft_geometry_map import (  # noqa: E402
    DraftGeometryMap,
    build_anchor_transport_map,
    frame_to_latent_index,
    save_draft_geometry_map,
)


def parse_ints(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated integer list")
    return values


def parse_roi(text: str) -> tuple[int, int, int, int]:
    values = parse_ints(text)
    if len(values) != 4:
        raise argparse.ArgumentTypeError("--seed_roi must be x0,y0,x1,y1")
    x0, y0, x1, y1 = values
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise argparse.ArgumentTypeError("Invalid ROI bounds")
    return x0, y0, x1, y1


def parse_polygon(text: str) -> list[tuple[float, float]]:
    points = []
    for item in text.split(";"):
        values = [float(value.strip()) for value in item.split(",") if value.strip()]
        if len(values) != 2:
            raise argparse.ArgumentTypeError(
                "--seed_polygon must be x0,y0;x1,y1;... with at least three points"
            )
        points.append((values[0], values[1]))
    if len(points) < 3:
        raise argparse.ArgumentTypeError("--seed_polygon requires at least three points")
    return points


def polygon_mask(
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    points: list[tuple[float, float]],
) -> torch.Tensor:
    """Return a token-center mask using the even-odd polygon rule."""
    inside = torch.zeros_like(center_x, dtype=torch.bool)
    previous_x, previous_y = points[-1]
    for current_x, current_y in points:
        crosses_y = (current_y > center_y) != (previous_y > center_y)
        edge_x = (
            (previous_x - current_x)
            * (center_y - current_y)
            / (previous_y - current_y + 1e-12)
            + current_x
        )
        inside ^= crosses_y & (center_x < edge_x)
        previous_x, previous_y = current_x, current_y
    return inside


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--draft_run_fingerprint", required=True)
    parser.add_argument("--anchor_frames", required=True, type=parse_ints)
    parser.add_argument("--seed_frame", required=True, type=int)
    parser.add_argument("--guide_frames", required=True, type=parse_ints)
    seed_region = parser.add_mutually_exclusive_group(required=True)
    seed_region.add_argument("--seed_roi", type=parse_roi)
    seed_region.add_argument("--seed_polygon", type=parse_polygon)
    parser.add_argument("--image_height", required=True, type=int)
    parser.add_argument("--image_width", required=True, type=int)
    parser.add_argument("--num_frames", required=True, type=int)
    parser.add_argument("--memory_slots", type=int, default=3)
    parser.add_argument("--temporal_scale", type=int, default=4)
    parser.add_argument("--spatial_scale", type=int, default=16)
    parser.add_argument("--patch_height", type=int, default=2)
    parser.add_argument("--patch_width", type=int, default=2)
    parser.add_argument("--confidence_percentile", type=float, default=20.0)
    parser.add_argument("--confidence_floor", type=float, default=0.2)
    parser.add_argument(
        "--visibility_mode",
        choices=("symmetric", "front_tolerant", "source_zbuffer"),
        default="front_tolerant",
    )
    parser.add_argument("--behind_threshold", type=float, default=0.08)
    parser.add_argument("--front_threshold", type=float, default=0.35)
    args = parser.parse_args()

    if args.image_height <= 0 or args.image_width <= 0 or args.num_frames <= 1:
        raise ValueError("Image dimensions and num_frames must be positive")
    if args.seed_polygon is not None:
        if any(
            x < 0 or x > args.image_width or y < 0 or y > args.image_height
            for x, y in args.seed_polygon
        ):
            raise ValueError("Seed polygon lies outside the image")
        polygon_x = [point[0] for point in args.seed_polygon]
        polygon_y = [point[1] for point in args.seed_polygon]
        x0, y0 = min(polygon_x), min(polygon_y)
        x1, y1 = max(polygon_x), max(polygon_y)
    else:
        x0, y0, x1, y1 = args.seed_roi
        if x1 > args.image_width or y1 > args.image_height:
            raise ValueError("ROI lies outside the image")
    if args.seed_frame in args.guide_frames:
        raise ValueError("--seed_frame must not also be present in --guide_frames")

    geometry = torch.load(args.geometry, map_location="cpu", weights_only=False)
    frame_indices = [int(frame) for frame in geometry["frame_indices"]]
    latent_frames = 1 + (args.num_frames - 1) // args.temporal_scale
    token_height = args.image_height // args.spatial_scale // args.patch_height
    token_width = args.image_width // args.spatial_scale // args.patch_width
    token_grid = (latent_frames, token_height, token_width)
    seed_t = frame_to_latent_index(args.seed_frame, args.temporal_scale, latent_frames)
    if seed_t <= 0:
        raise ValueError("--seed_frame must map to a nonzero latent frame")
    guide_frames = sorted(set(args.guide_frames))
    target_frames = sorted({args.seed_frame, *guide_frames})

    base = build_anchor_transport_map(
        intrinsic=geometry["intrinsic"].float(),
        extrinsic=geometry["extrinsic"].float(),
        depth_map=geometry["depth"].float(),
        confidence_map=geometry["confidence"].float(),
        selected_video_frames=frame_indices,
        anchor_video_frames=args.anchor_frames,
        target_video_frames=target_frames,
        token_grid=token_grid,
        temporal_scale=args.temporal_scale,
        memory_slots=args.memory_slots,
        confidence_percentile=args.confidence_percentile,
        confidence_floor=args.confidence_floor,
        visibility_mode=args.visibility_mode,
        behind_threshold=args.behind_threshold,
        front_threshold=args.front_threshold,
        metadata={
            "map_type": "oracle_source_surface_track",
            "geometry": str(args.geometry.resolve()),
            "image_size": [args.image_height, args.image_width],
            "num_frames": args.num_frames,
            "draft_run_fingerprint": args.draft_run_fingerprint,
        },
    )

    token_y, token_x = torch.meshgrid(
        torch.arange(token_height, dtype=torch.float32),
        torch.arange(token_width, dtype=torch.float32),
        indexing="ij",
    )
    center_x = (token_x + 0.5) * args.image_width / token_width
    center_y = (token_y + 0.5) * args.image_height / token_height
    if args.seed_polygon is not None:
        roi_mask = polygon_mask(center_x, center_y, args.seed_polygon)
    else:
        roi_mask = (
            (center_x >= x0)
            & (center_x < x1)
            & (center_y >= y0)
            & (center_y < y1)
        )
    if not roi_mask.any():
        raise RuntimeError("The seed region contains no attention tokens")
    spatial_tokens = token_height * token_width
    seed_row = seed_t - 1
    seed_confidence = base.confidence[seed_row].reshape(spatial_tokens, args.memory_slots)
    seed_valid = (seed_confidence > 0) & roi_mask.reshape(-1, 1)
    seed_source_ids = (
        base.source_time[seed_row] * spatial_tokens + base.source_index[seed_row]
    )[seed_valid].unique()
    if seed_source_ids.numel() == 0:
        raise RuntimeError("The seed ROI contains no valid anchor correspondences")

    gated_confidence = torch.zeros_like(base.confidence)
    guide_latent_to_frame = {}
    for frame in guide_frames:
        target_t = frame_to_latent_index(frame, args.temporal_scale, latent_frames)
        if target_t <= seed_t:
            raise ValueError(
                f"Guide frame {frame} must map after seed frame {args.seed_frame}"
            )
        guide_latent_to_frame[target_t] = frame
        row = target_t - 1
        candidate_source_ids = (
            base.source_time[row] * spatial_tokens + base.source_index[row]
        )
        same_surface = torch.isin(candidate_source_ids, seed_source_ids)
        gated_confidence[row] = base.confidence[row] * same_surface.reshape(
            token_height, token_width, args.memory_slots
        )

    active_by_target = []
    for row in range(gated_confidence.shape[0]):
        active = gated_confidence[row].amax(dim=-1) > 0
        if active.any():
            active_y, active_x = torch.where(active)
            pixel_x0 = int(active_x.min().item() * args.image_width / token_width)
            pixel_y0 = int(active_y.min().item() * args.image_height / token_height)
            pixel_x1 = int((active_x.max().item() + 1) * args.image_width / token_width)
            pixel_y1 = int((active_y.max().item() + 1) * args.image_height / token_height)
            active_by_target.append(
                {
                    "target_latent_index": row + 1,
                    "target_video_frame": guide_latent_to_frame[row + 1],
                    "active_tokens": int(active.sum().item()),
                    "tracked_bbox_token_yx": [
                        [int(active_y.min().item()), int(active_x.min().item())],
                        [int(active_y.max().item()), int(active_x.max().item())],
                    ],
                    "tracked_bbox_pixels_xyxy": [pixel_x0, pixel_y0, pixel_x1, pixel_y1],
                }
            )

    metadata = dict(base.metadata)
    metadata.update(
        {
            "format_version": 9,
            "map_type": "oracle_source_surface_track",
            "seed_video_frame": args.seed_frame,
            "seed_latent_index": seed_t,
            "guide_video_frames": guide_frames,
            "seed_roi_pixels_xyxy": [x0, y0, x1, y1],
            "seed_polygon_pixels_xy": (
                [[x, y] for x, y in args.seed_polygon]
                if args.seed_polygon is not None
                else None
            ),
            "seed_roi_token_indices_yx": [
                [int(token_y[roi_mask].min().item()), int(token_x[roi_mask].min().item())],
                [int(token_y[roi_mask].max().item()), int(token_x[roi_mask].max().item())],
            ],
            "seed_roi_token_count": int(roi_mask.sum().item()),
            "seed_source_id_count": int(seed_source_ids.numel()),
            "seed_source_ids": [int(value) for value in seed_source_ids.tolist()],
            "active_by_target": active_by_target,
        }
    )
    output = DraftGeometryMap(
        source_time=base.source_time,
        source_index=base.source_index,
        confidence=gated_confidence,
        pair_stats=base.pair_stats,
        metadata=metadata,
    )
    save_draft_geometry_map(output, args.output)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"saved_map={args.output}")
    print(f"saved_report={report_path}")
    print(json.dumps({"active_by_target": active_by_target}, indent=2))


if __name__ == "__main__":
    main()
