#!/usr/bin/env python3
"""Trace rolling 3D correspondences backward into a persistent source memory.

Each active deformation-risk correspondence initially points only a few frames
back. The dense source indices stored for those rolling pairs can be composed:
target -> recent source -> older source -> ... . This keeps the detector local
in time while repairing from the earliest surface observation reachable through
overlapping views.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from external.guidance_wan.draft_geometry_map import (
    DraftGeometryMap,
    load_draft_geometry_map,
    save_draft_geometry_map,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compose a rolling risk map's dense 3D links backward, preserving "
            "the original risk support and confidence."
        )
    )
    parser.add_argument("--input_map", required=True)
    parser.add_argument(
        "--link_map",
        default="",
        help=(
            "Optional dense rolling correspondence map used only for track "
            "continuation. The input map still defines risk support."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_hops", type=int, default=30)
    parser.add_argument(
        "--prefer",
        choices=("earliest", "closest"),
        default="earliest",
        help="Choose the oldest or nearest valid source at each chain link.",
    )
    args = parser.parse_args()
    if args.max_hops < 1:
        raise ValueError("max_hops must be positive")

    transport = load_draft_geometry_map(args.input_map, "cpu")
    link_transport = (
        load_draft_geometry_map(args.link_map, "cpu")
        if args.link_map
        else transport
    )
    source_time_dense = transport.source_time
    source_index_dense = transport.source_index
    link_source_time = link_transport.source_time
    link_source_index = link_transport.source_index
    link_confidence = link_transport.confidence.reshape_as(link_source_time)
    if (
        link_source_time.shape[:2] != source_time_dense.shape[:2]
        or link_source_index.shape != link_source_time.shape
    ):
        raise ValueError("Dense link map is incompatible with the input risk map")
    confidence = transport.confidence.clone()
    output_source_time = source_time_dense.clone()
    output_source_index = source_index_dense.clone()

    target_steps, spatial_tokens, memory_slots = source_time_dense.shape
    if confidence.shape[0] != target_steps:
        raise ValueError("Source and confidence time dimensions differ")

    hop_counts: list[int] = []
    endpoint_times: list[int] = []
    stopped_without_link = 0
    active_slots = confidence.reshape(
        target_steps,
        spatial_tokens,
        memory_slots,
    ) > 0

    for target_row in range(target_steps):
        active_positions = active_slots[target_row].nonzero(as_tuple=False)
        for spatial_index, slot in active_positions.tolist():
            current_time = int(
                source_time_dense[target_row, spatial_index, slot].item()
            )
            current_index = int(
                source_index_dense[target_row, spatial_index, slot].item()
            )
            hops = 0

            while current_time > 0 and hops < args.max_hops:
                previous_target_row = current_time - 1
                if previous_target_row >= target_steps:
                    break
                candidate_time = link_source_time[
                    previous_target_row,
                    current_index,
                ]
                candidate_index = link_source_index[
                    previous_target_row,
                    current_index,
                ]
                candidate_confidence = link_confidence[
                    previous_target_row,
                    current_index,
                ]
                valid = (
                    (candidate_confidence > 0)
                    & (candidate_time < current_time)
                    & (candidate_index >= 0)
                )
                if not valid.any():
                    stopped_without_link += 1
                    break

                valid_slots = valid.nonzero(as_tuple=False).flatten()
                valid_times = candidate_time[valid_slots]
                if args.prefer == "earliest":
                    chosen_slot = valid_slots[valid_times.argmin()]
                else:
                    chosen_slot = valid_slots[valid_times.argmax()]
                current_time = int(candidate_time[chosen_slot].item())
                current_index = int(candidate_index[chosen_slot].item())
                hops += 1

            output_source_time[
                target_row,
                spatial_index,
                slot,
            ] = current_time
            output_source_index[
                target_row,
                spatial_index,
                slot,
            ] = current_index
            hop_counts.append(hops)
            endpoint_times.append(current_time)

    metadata = dict(transport.metadata)
    endpoint_token_times = sorted(set(endpoint_times))
    temporal_scale = int(metadata.get("temporal_scale", 4))
    endpoint_video_frames = [
        0 if token_time == 0 else token_time * temporal_scale
        for token_time in endpoint_token_times
    ]
    metadata.update(
        {
            "format_version": 4,
            "map_type": "chained_persistent_geometry_risk_transport",
            "input_map": str(Path(args.input_map)),
            "link_map": (
                str(Path(args.link_map))
                if args.link_map
                else str(Path(args.input_map))
            ),
            "chain_max_hops": args.max_hops,
            "chain_prefer": args.prefer,
            "anchor_video_frames": endpoint_video_frames,
            "persistent_anchor_token_times": endpoint_token_times,
        }
    )
    report = {
        "active_correspondence_slots": len(hop_counts),
        "mean_hops": (
            float(torch.tensor(hop_counts, dtype=torch.float32).mean().item())
            if hop_counts
            else 0.0
        ),
        "max_hops_observed": max(hop_counts, default=0),
        "stopped_without_link": stopped_without_link,
        "endpoint_time_histogram": {
            str(key): value
            for key, value in sorted(Counter(endpoint_times).items())
        },
    }
    output_transport = DraftGeometryMap(
        source_time=output_source_time,
        source_index=output_source_index,
        confidence=confidence,
        pair_stats=transport.pair_stats,
        metadata=metadata,
    )
    output_path = Path(args.output)
    save_draft_geometry_map(output_transport, output_path)
    report_path = output_path.with_suffix(".json")
    with report_path.open("w") as handle:
        json.dump(
            {"metadata": metadata, "chain_report": report},
            handle,
            indent=2,
        )
    print(json.dumps(report, sort_keys=True), flush=True)
    print(f"saved_map={output_path}", flush=True)
    print(f"saved_report={report_path}", flush=True)


if __name__ == "__main__":
    main()
