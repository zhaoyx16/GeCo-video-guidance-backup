#!/usr/bin/env python3
"""Replace a rolling risk map's short-term sources with persistent 3D anchors.

The deformation detector may compare a target against recent frames so it can
find failures anywhere in a long video. Recent frames are poor repair sources,
however, because they may already contain the same accumulating deformation.
For each risky target token, this script finds the earliest cluster of past
source views whose projected depths agree, then stores those source tokens in a
pipeline-compatible transport map.
"""

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


def parse_indices(value: str) -> list[int]:
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            result.append(int(item))
            continue
        pieces = [int(piece) for piece in item.split(":")]
        if len(pieces) == 2:
            start, end = pieces
            stride = 1
        elif len(pieces) == 3:
            start, end, stride = pieces
        else:
            raise ValueError(f"Invalid frame range: {item}")
        if stride <= 0:
            raise ValueError("Frame-range stride must be positive")
        result.extend(range(start, end + 1, stride))
    return sorted(set(result))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Keep a deformation risk map's support, but transport clean latent "
            "patches from the earliest geometrically consistent past views."
        )
    )
    parser.add_argument("--risk_map", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--candidate_frames",
        default="",
        help="Optional past-frame subset/ranges. Empty uses every geometry frame.",
    )
    parser.add_argument("--memory_slots", type=int, default=3)
    parser.add_argument("--min_source_agreement", type=int, default=2)
    parser.add_argument("--source_consensus_threshold", type=float, default=0.05)
    parser.add_argument("--confidence_percentile", type=float, default=None)
    parser.add_argument("--confidence_floor", type=float, default=None)
    parser.add_argument("--depth_patch_size", type=int, default=None)
    parser.add_argument("--max_depth_patch_cv", type=float, default=None)
    args = parser.parse_args()

    if args.memory_slots < 1:
        raise ValueError("memory_slots must be positive")
    if args.min_source_agreement < 1:
        raise ValueError("min_source_agreement must be positive")
    if args.source_consensus_threshold <= 0:
        raise ValueError("source_consensus_threshold must be positive")

    risk = load_draft_geometry_map(args.risk_map, "cpu")
    geometry = torch.load(args.geometry, map_location="cpu", weights_only=False)
    metadata = risk.metadata
    latent_frames, token_height, token_width = tuple(metadata["token_grid"])
    spatial_tokens = token_height * token_width
    temporal_scale = int(metadata.get("temporal_scale", 4))

    frame_indices = [int(frame) for frame in geometry["frame_indices"]]
    frame_lookup = {frame: idx for idx, frame in enumerate(frame_indices)}
    candidate_frames = (
        parse_indices(args.candidate_frames)
        if args.candidate_frames
        else frame_indices
    )
    missing_candidates = set(candidate_frames).difference(frame_lookup)
    if missing_candidates:
        raise ValueError(
            f"Geometry payload is missing candidate frames {sorted(missing_candidates)}"
        )

    target_frames = [
        int(frame)
        for frame in metadata.get("target_video_frames", frame_indices[1:])
    ]
    missing_targets = set(target_frames).difference(frame_lookup)
    if missing_targets:
        raise ValueError(
            f"Geometry payload is missing target frames {sorted(missing_targets)}"
        )

    confidence_percentile = (
        float(args.confidence_percentile)
        if args.confidence_percentile is not None
        else float(metadata.get("confidence_percentile", 20.0))
    )
    confidence_floor = (
        float(args.confidence_floor)
        if args.confidence_floor is not None
        else float(metadata.get("confidence_floor", 0.2))
    )
    depth_patch_size = (
        int(args.depth_patch_size)
        if args.depth_patch_size is not None
        else int(metadata.get("depth_patch_size", 31))
    )
    max_depth_patch_cv = (
        float(args.max_depth_patch_cv)
        if args.max_depth_patch_cv is not None
        else float(metadata.get("max_depth_patch_cv", 0.08))
    )

    depth = geometry["depth"].float()[..., 0]
    depth_cv = local_depth_cv(depth, depth_patch_size)
    confidence = geometry["confidence"].float()[..., 0]
    intrinsic = geometry["intrinsic"].float()
    extrinsic = geometry["extrinsic"].float()

    source_time = torch.zeros(
        (latent_frames - 1, spatial_tokens, args.memory_slots),
        dtype=torch.long,
    )
    source_index = torch.zeros_like(source_time)
    transport_confidence = torch.zeros(
        (
            latent_frames - 1,
            token_height,
            token_width,
            args.memory_slots,
        ),
        dtype=torch.float32,
    )
    pair_stats: list[dict[str, float | int | str]] = []
    used_source_frames: set[int] = set()

    for target_frame in target_frames:
        target_time = frame_to_latent_index(
            target_frame,
            temporal_scale,
            latent_frames,
        )
        if target_time <= 0:
            continue
        risk_gate = risk.confidence[target_time - 1].amax(dim=-1).reshape(-1)
        risk_support = risk_gate > 0
        if not risk_support.any():
            continue

        past_frames = [
            frame for frame in candidate_frames if frame < target_frame
        ]
        if len(past_frames) < args.min_source_agreement:
            continue

        target_sequence = frame_lookup[target_frame]
        projections = []
        for source_frame in past_frames:
            source_sequence = frame_lookup[source_frame]
            projections.append(
                pair_projection(
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
                    confidence_percentile=confidence_percentile,
                    confidence_floor=confidence_floor,
                    reverse_margin=float(metadata.get("reverse_margin", 0.03)),
                    max_depth_patch_cv=max_depth_patch_cv,
                )
            )

        expected = torch.stack(
            [projection["expected_depth"] for projection in projections],
            dim=0,
        )
        valid = torch.stack(
            [projection["valid"] for projection in projections],
            dim=0,
        )
        pair_error = (
            expected[:, None] - expected[None, :]
        ).abs() / (
            expected[:, None].abs()
            + expected[None, :].abs()
            + 1e-6
        )
        agrees = (
            valid[:, None]
            & valid[None, :]
            & (pair_error <= args.source_consensus_threshold)
        )
        agreement_count = agrees.sum(dim=1)
        viable_reference = (
            agreement_count >= args.min_source_agreement
        )
        has_reference = viable_reference.any(dim=0) & risk_support
        earliest_reference = viable_reference.to(torch.int64).argmax(dim=0)

        token_ids = torch.arange(spatial_tokens)
        earliest_cluster = agrees[
            earliest_reference,
            :,
            token_ids,
        ].transpose(0, 1)
        earliest_cluster &= has_reference.unsqueeze(0)

        selected_count = torch.zeros(spatial_tokens, dtype=torch.long)
        remaining = earliest_cluster.clone()
        for slot in range(args.memory_slots):
            has_candidate = remaining.any(dim=0)
            if not has_candidate.any():
                break
            chosen_source = remaining.to(torch.int64).argmax(dim=0)
            chosen_source_index = torch.zeros(
                spatial_tokens,
                dtype=torch.long,
            )
            chosen_source_time = torch.zeros(
                spatial_tokens,
                dtype=torch.long,
            )
            for source_idx, source_frame in enumerate(past_frames):
                chosen = has_candidate & (chosen_source == source_idx)
                if not chosen.any():
                    continue
                chosen_source_index[chosen] = projections[source_idx][
                    "source_index"
                ][chosen]
                chosen_source_time[chosen] = frame_to_latent_index(
                    source_frame,
                    temporal_scale,
                    latent_frames,
                )
                used_source_frames.add(source_frame)

            source_time[target_time - 1, :, slot] = chosen_source_time
            source_index[target_time - 1, :, slot] = chosen_source_index
            transport_confidence[target_time - 1, :, :, slot] = (
                risk_gate * has_candidate.float()
            ).reshape(token_height, token_width)
            remaining[
                chosen_source,
                token_ids,
            ] = False
            selected_count += has_candidate.long()

        retained_support = (
            transport_confidence[target_time - 1].amax(dim=-1) > 0
        )
        stat = {
            "target_frame": target_frame,
            "target_latent_index": target_time,
            "risk_coverage": float(risk_support.float().mean().item()),
            "retained_coverage": float(
                retained_support.float().mean().item()
            ),
            "mean_sources_per_retained_token": float(
                selected_count[has_reference].float().mean().item()
                if has_reference.any()
                else 0.0
            ),
            "earliest_source_frame_used": (
                min(used_source_frames) if used_source_frames else -1
            ),
        }
        pair_stats.append(stat)
        print(json.dumps(stat, sort_keys=True), flush=True)

    output_metadata = dict(metadata)
    output_metadata.update(
        {
            "format_version": 3,
            "map_type": "persistent_geometry_risk_transport",
            "risk_map": str(Path(args.risk_map)),
            "geometry": str(Path(args.geometry)),
            "anchor_video_frames": sorted(used_source_frames),
            "candidate_video_frames": candidate_frames,
            "memory_slots": args.memory_slots,
            "min_source_agreement": args.min_source_agreement,
            "source_consensus_threshold": args.source_consensus_threshold,
            "confidence_percentile": confidence_percentile,
            "confidence_floor": confidence_floor,
            "depth_patch_size": depth_patch_size,
            "max_depth_patch_cv": max_depth_patch_cv,
        }
    )
    transport = DraftGeometryMap(
        source_time=source_time,
        source_index=source_index,
        confidence=transport_confidence,
        pair_stats=pair_stats,
        metadata=output_metadata,
    )
    output_path = Path(args.output)
    save_draft_geometry_map(transport, output_path)
    report_path = output_path.with_suffix(".json")
    with report_path.open("w") as handle:
        json.dump(
            {
                "metadata": output_metadata,
                "target_stats": pair_stats,
            },
            handle,
            indent=2,
        )
    print(f"saved_map={output_path}", flush=True)
    print(f"saved_report={report_path}", flush=True)


if __name__ == "__main__":
    main()
