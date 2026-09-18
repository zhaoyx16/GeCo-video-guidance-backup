#!/usr/bin/env python3
"""Build dense short-baseline geometry links for persistent surface tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from external.guidance_wan.draft_geometry_map import (
    DraftGeometryMap,
    frame_to_latent_index,
    load_draft_geometry_map,
    save_draft_geometry_map,
)
from scripts.build_deformation_risk_map import local_depth_cv, pair_projection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_map", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rolling_offsets", default="4,8,12")
    parser.add_argument("--confidence_percentile", type=float, default=20.0)
    parser.add_argument("--confidence_floor", type=float, default=0.2)
    parser.add_argument("--depth_patch_size", type=int, default=31)
    parser.add_argument("--max_depth_patch_cv", type=float, default=1e6)
    parser.add_argument("--reverse_margin", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rolling_offsets = sorted(
        {int(item) for item in args.rolling_offsets.split(",") if item.strip()}
    )
    if not rolling_offsets or rolling_offsets[0] <= 0:
        raise ValueError("rolling_offsets must contain positive frame offsets")

    template = load_draft_geometry_map(args.template_map, "cpu")
    geometry = torch.load(args.geometry, map_location="cpu", weights_only=False)
    latent_frames, token_height, token_width = tuple(template.metadata["token_grid"])
    temporal_scale = int(template.metadata.get("temporal_scale", 4))
    spatial_tokens = token_height * token_width
    memory_slots = len(rolling_offsets)

    frame_indices = [int(frame) for frame in geometry["frame_indices"]]
    frame_lookup = {frame: idx for idx, frame in enumerate(frame_indices)}
    depth = geometry["depth"].float()[..., 0]
    depth_cv = local_depth_cv(depth, args.depth_patch_size)
    confidence = geometry["confidence"].float()[..., 0]
    intrinsic = geometry["intrinsic"].float()
    extrinsic = geometry["extrinsic"].float()

    source_time = torch.zeros(
        (latent_frames - 1, spatial_tokens, memory_slots),
        dtype=torch.long,
    )
    source_index = torch.zeros_like(source_time)
    link_confidence = torch.zeros(
        (latent_frames - 1, token_height, token_width, memory_slots),
        dtype=torch.float32,
    )
    pair_stats: list[dict[str, float | int]] = []
    used_frames: set[int] = set()

    for target_frame in frame_indices:
        target_time = frame_to_latent_index(
            target_frame,
            temporal_scale,
            latent_frames,
        )
        if target_time <= 0:
            continue
        target_sequence = frame_lookup[target_frame]
        for slot, offset in enumerate(rolling_offsets):
            source_frame = target_frame - offset
            if source_frame not in frame_lookup:
                continue
            source_sequence = frame_lookup[source_frame]
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
                token_height=token_height,
                token_width=token_width,
                confidence_percentile=args.confidence_percentile,
                confidence_floor=args.confidence_floor,
                reverse_margin=args.reverse_margin,
                max_depth_patch_cv=args.max_depth_patch_cv,
            )
            valid = projection["valid"]
            source_time[target_time - 1, :, slot] = frame_to_latent_index(
                source_frame,
                temporal_scale,
                latent_frames,
            )
            source_index[target_time - 1, :, slot] = projection["source_index"]
            link_confidence[target_time - 1, :, :, slot] = valid.reshape(
                token_height,
                token_width,
            ).float()
            if valid.any():
                used_frames.add(source_frame)
            pair_stats.append(
                {
                    "source_frame": source_frame,
                    "target_frame": target_frame,
                    "offset": offset,
                    "valid_coverage": float(valid.float().mean().item()),
                }
            )

    metadata = dict(template.metadata)
    metadata.update(
        {
            "format_version": 6,
            "map_type": "dense_rolling_geometry_links",
            "template_map": str(Path(args.template_map)),
            "geometry": str(Path(args.geometry)),
            "rolling_anchor_offsets": rolling_offsets,
            "anchor_video_frames": sorted(used_frames),
            "memory_slots": memory_slots,
            "confidence_percentile": args.confidence_percentile,
            "confidence_floor": args.confidence_floor,
            "depth_patch_size": args.depth_patch_size,
            "max_depth_patch_cv": args.max_depth_patch_cv,
            "reverse_margin": args.reverse_margin,
        }
    )
    output = DraftGeometryMap(
        source_time=source_time,
        source_index=source_index,
        confidence=link_confidence,
        pair_stats=pair_stats,
        metadata=metadata,
    )
    output_path = Path(args.output)
    save_draft_geometry_map(output, output_path)
    report_path = output_path.with_suffix(".json")
    with report_path.open("w") as handle:
        json.dump(
            {
                "metadata": metadata,
                "mean_valid_coverage": (
                    sum(stat["valid_coverage"] for stat in pair_stats)
                    / max(1, len(pair_stats))
                ),
                "pair_stats": pair_stats,
            },
            handle,
            indent=2,
        )
    print(f"saved_map={output_path}")
    print(f"saved_report={report_path}")
    print(
        "mean_valid_coverage="
        f"{sum(stat['valid_coverage'] for stat in pair_stats) / max(1, len(pair_stats)):.6f}"
    )


if __name__ == "__main__":
    main()
