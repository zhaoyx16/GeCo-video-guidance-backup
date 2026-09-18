#!/usr/bin/env python3
"""Expand top-1 geometry correspondences into causal persistent tracks.

The input map assigns each active target token to one canonical source token.
This script groups repeated observations of that canonical token and gives each
target access to the clean anchor plus a bounded number of earlier observations.
No future target token is ever used as memory.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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
        description="Build a past-only attention-memory map from top-1 tracks."
    )
    parser.add_argument("--input_map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--memory_slots", type=int, default=4)
    parser.add_argument(
        "--history_policy",
        choices=("earliest", "recent"),
        default="earliest",
    )
    parser.add_argument(
        "--min_track_observations",
        type=int,
        default=2,
        help="Suppress tracks with fewer target-frame observations.",
    )
    args = parser.parse_args()

    if args.memory_slots < 1:
        raise ValueError("memory_slots must be positive")
    if args.min_track_observations < 1:
        raise ValueError("min_track_observations must be positive")

    transport = load_draft_geometry_map(args.input_map, "cpu")
    if transport.source_time.shape[-1] != 1:
        raise ValueError("Past-track expansion requires a top-1 input map")

    source_time_top1 = transport.source_time[..., 0]
    source_index_top1 = transport.source_index[..., 0]
    confidence_top1 = transport.confidence.reshape_as(
        transport.source_time
    )[..., 0]
    target_steps, spatial_tokens = source_time_top1.shape
    token_grid = tuple(transport.metadata["token_grid"])
    if token_grid[0] - 1 != target_steps:
        raise ValueError("Map time dimension is incompatible with token_grid")
    token_height, token_width = token_grid[1:]
    if token_height * token_width != spatial_tokens:
        raise ValueError("Map spatial dimension is incompatible with token_grid")

    # One observation per canonical token and target time avoids duplicated
    # same-frame keys when several target patches collapse onto one source.
    observations: dict[
        tuple[int, int],
        dict[int, tuple[int, float]],
    ] = defaultdict(dict)
    query_records = []
    for target_row in range(target_steps):
        target_time = target_row + 1
        active = (confidence_top1[target_row] > 0).nonzero(
            as_tuple=False
        ).flatten()
        for target_spatial in active.tolist():
            canonical = (
                int(source_time_top1[target_row, target_spatial]),
                int(source_index_top1[target_row, target_spatial]),
            )
            confidence = float(
                confidence_top1[target_row, target_spatial]
            )
            query_records.append(
                (
                    canonical,
                    target_time,
                    target_spatial,
                    confidence,
                )
            )
            previous = observations[canonical].get(target_time)
            if previous is None or confidence > previous[1]:
                observations[canonical][target_time] = (
                    target_spatial,
                    confidence,
                )

    source_time = torch.zeros(
        (target_steps, spatial_tokens, args.memory_slots),
        dtype=torch.long,
    )
    source_index = torch.zeros_like(source_time)
    confidence = torch.zeros(
        (
            target_steps,
            token_height,
            token_width,
            args.memory_slots,
        ),
        dtype=torch.float32,
    )
    confidence_flat = confidence.reshape(
        target_steps,
        spatial_tokens,
        args.memory_slots,
    )

    active_queries = 0
    history_slots = 0
    track_lengths = [
        len(time_to_observation)
        for time_to_observation in observations.values()
    ]
    for (
        canonical,
        target_time,
        target_spatial,
        target_confidence,
    ) in query_records:
        ordered_observations = sorted(observations[canonical].items())
        if len(ordered_observations) < args.min_track_observations:
            continue

        anchor_time, anchor_index = canonical
        target_row = target_time - 1
        memories = [
            (anchor_time, anchor_index, target_confidence)
        ]
        past = [
            (
                past_time,
                past_spatial,
                min(target_confidence, past_confidence),
            )
            for past_time, (
                past_spatial,
                past_confidence,
            ) in ordered_observations
            if past_time < target_time
        ]
        if args.history_policy == "recent":
            past = list(reversed(past))
        memories.extend(past[: args.memory_slots - 1])

        seen_flat_tokens = set()
        write_slot = 0
        for memory_time, memory_index, memory_confidence in memories:
            flat_token = memory_time * spatial_tokens + memory_index
            if flat_token in seen_flat_tokens:
                continue
            if memory_time >= target_time:
                raise RuntimeError("Past-track map attempted to use future memory")
            seen_flat_tokens.add(flat_token)
            source_time[target_row, target_spatial, write_slot] = memory_time
            source_index[target_row, target_spatial, write_slot] = memory_index
            confidence_flat[
                target_row,
                target_spatial,
                write_slot,
            ] = memory_confidence
            if write_slot > 0:
                history_slots += 1
            write_slot += 1
            if write_slot == args.memory_slots:
                break
        active_queries += int(write_slot > 0)

    metadata = dict(transport.metadata)
    metadata.update(
        {
            "format_version": 4,
            "map_type": "past_only_geometry_track_attention",
            "input_map": str(Path(args.input_map)),
            "memory_slots": args.memory_slots,
            "history_policy": args.history_policy,
            "min_track_observations": args.min_track_observations,
        }
    )
    output_transport = DraftGeometryMap(
        source_time=source_time,
        source_index=source_index,
        confidence=confidence,
        pair_stats=transport.pair_stats,
        metadata=metadata,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_draft_geometry_map(output_transport, output_path)

    report = {
        "canonical_tracks": len(track_lengths),
        "track_length_histogram": {
            str(length): count
            for length, count in sorted(Counter(track_lengths).items())
        },
        "active_queries": active_queries,
        "history_slots": history_slots,
        "memory_slots": args.memory_slots,
        "history_policy": args.history_policy,
        "min_track_observations": args.min_track_observations,
    }
    report_path = output_path.with_suffix(".json")
    with report_path.open("w") as handle:
        json.dump(
            {"metadata": metadata, "report": report},
            handle,
            indent=2,
        )
    print(json.dumps(report, sort_keys=True), flush=True)
    print(f"saved_map={output_path}", flush=True)
    print(f"saved_report={report_path}", flush=True)


if __name__ == "__main__":
    main()
