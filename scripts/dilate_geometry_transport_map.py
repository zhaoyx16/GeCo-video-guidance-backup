#!/usr/bin/env python3
"""Expand sparse geometry transport support with a local translation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

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
            "Fill unsupported neighbors from the strongest nearby geometry "
            "correspondence while preserving local source-token offsets."
        )
    )
    parser.add_argument("--input_map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spatial_radius", type=int, default=1)
    args = parser.parse_args()
    if args.spatial_radius < 1:
        raise ValueError("spatial_radius must be at least 1")

    transport = load_draft_geometry_map(args.input_map, "cpu")
    _, height_tokens, width_tokens = tuple(transport.metadata["token_grid"])
    target_steps, spatial_tokens, memory_slots = transport.source_time.shape
    if spatial_tokens != height_tokens * width_tokens:
        raise ValueError("Transport spatial shape does not match token_grid")

    source_time_grid = transport.source_time.reshape(
        target_steps,
        height_tokens,
        width_tokens,
        memory_slots,
    )
    source_index_grid = transport.source_index.reshape_as(source_time_grid)
    confidence_grid = transport.confidence.reshape_as(source_time_grid)
    support_score = confidence_grid.amax(dim=-1)
    pooled_score, donor_index = F.max_pool2d(
        support_score.unsqueeze(1),
        kernel_size=2 * args.spatial_radius + 1,
        stride=1,
        padding=args.spatial_radius,
        return_indices=True,
    )
    pooled_score = pooled_score.squeeze(1)
    donor_index = donor_index.squeeze(1)
    fill_support = (support_score <= 0) & (pooled_score > 0)

    donor_flat = donor_index.reshape(
        target_steps,
        spatial_tokens,
        1,
    ).expand(-1, -1, memory_slots)

    def gather_donor(values: torch.Tensor) -> torch.Tensor:
        return values.reshape(
            target_steps,
            spatial_tokens,
            memory_slots,
        ).gather(1, donor_flat).reshape_as(source_time_grid)

    donor_source_time = gather_donor(source_time_grid)
    donor_source_index = gather_donor(source_index_grid)
    donor_confidence = gather_donor(confidence_grid)

    target_y = torch.arange(height_tokens).view(1, height_tokens, 1)
    target_x = torch.arange(width_tokens).view(1, 1, width_tokens)
    donor_y = torch.div(
        donor_index,
        width_tokens,
        rounding_mode="floor",
    )
    donor_x = donor_index.remainder(width_tokens)
    offset_y = (target_y - donor_y).unsqueeze(-1)
    offset_x = (target_x - donor_x).unsqueeze(-1)
    source_y = torch.div(
        donor_source_index,
        width_tokens,
        rounding_mode="floor",
    )
    source_x = donor_source_index.remainder(width_tokens)
    propagated_source_index = (
        (source_y + offset_y).clamp(0, height_tokens - 1) * width_tokens
        + (source_x + offset_x).clamp(0, width_tokens - 1)
    )

    fill_slots = fill_support.unsqueeze(-1)
    source_time = torch.where(
        fill_slots,
        donor_source_time,
        source_time_grid,
    ).reshape_as(transport.source_time)
    source_index = torch.where(
        fill_slots,
        propagated_source_index,
        source_index_grid,
    ).reshape_as(transport.source_index)
    confidence = torch.where(
        fill_slots,
        donor_confidence,
        confidence_grid,
    ).reshape_as(transport.confidence)

    metadata = dict(transport.metadata)
    metadata.update(
        {
            "format_version": max(int(metadata.get("format_version", 1)), 5),
            "input_map": str(Path(args.input_map)),
            "spatial_support_dilation": args.spatial_radius,
        }
    )
    before = float((support_score > 0).float().mean().item())
    after = float((confidence.amax(dim=-1) > 0).float().mean().item())
    report = {
        "spatial_radius": args.spatial_radius,
        "coverage_before": before,
        "coverage_after": after,
        "new_support_fraction": after - before,
    }
    output_path = Path(args.output)
    save_draft_geometry_map(
        DraftGeometryMap(
            source_time=source_time,
            source_index=source_index,
            confidence=confidence,
            pair_stats=transport.pair_stats,
            metadata=metadata,
        ),
        output_path,
    )
    report_path = output_path.with_suffix(".json")
    with report_path.open("w") as handle:
        json.dump(
            {"metadata": metadata, "dilation_report": report},
            handle,
            indent=2,
        )
    print(json.dumps(report, sort_keys=True), flush=True)
    print(f"saved_map={output_path}", flush=True)
    print(f"saved_report={report_path}", flush=True)


if __name__ == "__main__":
    main()
