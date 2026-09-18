#!/usr/bin/env python3
"""Build an oracle ROI gate over fixed-anchor 3D token correspondences."""

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
        raise argparse.ArgumentTypeError("--roi must be x0,y0,x1,y1")
    x0, y0, x1, y1 = values
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise argparse.ArgumentTypeError("Invalid ROI bounds")
    return x0, y0, x1, y1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--draft_run_fingerprint", required=True)
    parser.add_argument("--anchor_frames", required=True, type=parse_ints)
    parser.add_argument("--target_frames", required=True, type=parse_ints)
    parser.add_argument("--roi", required=True, type=parse_roi)
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
    x0, y0, x1, y1 = args.roi
    if x1 > args.image_width or y1 > args.image_height:
        raise ValueError("ROI lies outside the image")

    geometry = torch.load(args.geometry, map_location="cpu", weights_only=False)
    frame_indices = [int(frame) for frame in geometry["frame_indices"]]
    latent_frames = 1 + (args.num_frames - 1) // args.temporal_scale
    token_height = args.image_height // args.spatial_scale // args.patch_height
    token_width = args.image_width // args.spatial_scale // args.patch_width
    token_grid = (latent_frames, token_height, token_width)

    base = build_anchor_transport_map(
        intrinsic=geometry["intrinsic"].float(),
        extrinsic=geometry["extrinsic"].float(),
        depth_map=geometry["depth"].float(),
        confidence_map=geometry["confidence"].float(),
        selected_video_frames=frame_indices,
        anchor_video_frames=args.anchor_frames,
        target_video_frames=args.target_frames,
        token_grid=token_grid,
        temporal_scale=args.temporal_scale,
        memory_slots=args.memory_slots,
        confidence_percentile=args.confidence_percentile,
        confidence_floor=args.confidence_floor,
        visibility_mode=args.visibility_mode,
        behind_threshold=args.behind_threshold,
        front_threshold=args.front_threshold,
        metadata={
            "map_type": "oracle_roi_fixed_anchor_transport",
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
    roi_mask = (
        (center_x >= x0)
        & (center_x < x1)
        & (center_y >= y0)
        & (center_y < y1)
    )
    gated_confidence = base.confidence * roi_mask[None, :, :, None]

    active_by_target = []
    for row in range(gated_confidence.shape[0]):
        active = gated_confidence[row].amax(dim=-1) > 0
        if active.any():
            active_by_target.append(
                {
                    "target_latent_index": row + 1,
                    "active_tokens": int(active.sum().item()),
                    "roi_tokens": int(roi_mask.sum().item()),
                    "roi_valid_fraction": float(
                        active.sum().item() / max(1, roi_mask.sum().item())
                    ),
                }
            )

    metadata = dict(base.metadata)
    metadata.update(
        {
            "format_version": 8,
            "map_type": "oracle_roi_fixed_anchor_transport",
            "oracle_roi_pixels_xyxy": [x0, y0, x1, y1],
            "oracle_roi_token_indices_yx": [
                [int(token_y[roi_mask].min().item()), int(token_x[roi_mask].min().item())],
                [int(token_y[roi_mask].max().item()), int(token_x[roi_mask].max().item())],
            ],
            "oracle_roi_token_count": int(roi_mask.sum().item()),
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
