#!/usr/bin/env python3
"""Keep identity drift that persists on the same canonical 3D surface."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_target_times", type=int, default=3)
    parser.add_argument("--canonical_radius", type=int, default=1)
    args = parser.parse_args()
    if args.min_target_times < 1:
        raise ValueError("min_target_times must be positive")
    if args.canonical_radius < 0:
        raise ValueError("canonical_radius must be non-negative")

    transport = load_draft_geometry_map(args.input_map, "cpu")
    latent_frames, token_height, token_width = tuple(
        transport.metadata["token_grid"]
    )
    spatial_tokens = token_height * token_width
    source_time = transport.source_time
    source_index = transport.source_index
    confidence = transport.confidence.reshape_as(source_time).float()
    target_steps, map_spatial_tokens, memory_slots = source_time.shape
    if map_spatial_tokens != spatial_tokens:
        raise ValueError("Map token dimensions do not match token_grid")

    canonical_counts = torch.zeros(
        latent_frames,
        spatial_tokens,
        dtype=torch.float32,
    )
    for target_row in range(target_steps):
        active = confidence[target_row] > 0
        if not active.any():
            continue
        active_time = source_time[target_row][active]
        active_index = source_index[target_row][active]
        valid = (
            (active_time >= 0)
            & (active_time < latent_frames)
            & (active_index >= 0)
            & (active_index < spatial_tokens)
        )
        canonical_flat = (
            active_time[valid] * spatial_tokens + active_index[valid]
        ).unique()
        canonical_counts.view(-1)[canonical_flat] += 1.0

    if args.canonical_radius > 0:
        canonical_counts_grid = canonical_counts.reshape(
            latent_frames,
            1,
            token_height,
            token_width,
        )
        canonical_counts_lookup = F.max_pool2d(
            canonical_counts_grid,
            kernel_size=2 * args.canonical_radius + 1,
            stride=1,
            padding=args.canonical_radius,
        ).reshape(latent_frames, spatial_tokens)
    else:
        canonical_counts_lookup = canonical_counts

    lookup_flat = canonical_counts_lookup.reshape(-1)
    source_flat = (
        source_time.clamp(0, latent_frames - 1) * spatial_tokens
        + source_index.clamp(0, spatial_tokens - 1)
    )
    persistent_slot = (
        (confidence > 0)
        & (
            lookup_flat[source_flat]
            >= float(args.min_target_times)
        )
    )
    persistent_token = persistent_slot.any(dim=-1, keepdim=True)
    filtered_confidence = confidence * persistent_token.float()

    metadata = dict(transport.metadata)
    metadata.update(
        {
            "format_version": 8,
            "map_type": "persistent_canonical_identity_risk",
            "input_map": str(Path(args.input_map)),
            "identity_min_target_times": args.min_target_times,
            "identity_canonical_radius": args.canonical_radius,
        }
    )
    before = float((confidence.amax(dim=-1) > 0).float().mean().item())
    after = float(
        (filtered_confidence.amax(dim=-1) > 0).float().mean().item()
    )
    canonical_support = canonical_counts > 0
    persistent_canonical = (
        canonical_counts_lookup >= float(args.min_target_times)
    )
    report = {
        "coverage_before": before,
        "coverage_after": after,
        "canonical_tokens_observed": int(canonical_support.sum().item()),
        "persistent_canonical_tokens": int(persistent_canonical.sum().item()),
        "min_target_times": args.min_target_times,
        "canonical_radius": args.canonical_radius,
        "canonical_count_quantiles": {
            str(q): float(
                torch.quantile(
                    canonical_counts[canonical_support],
                    q,
                ).item()
            )
            if canonical_support.any()
            else 0.0
            for q in (0.5, 0.75, 0.9, 0.95)
        },
    }
    output_path = Path(args.output)
    save_draft_geometry_map(
        DraftGeometryMap(
            source_time=source_time.clone(),
            source_index=source_index.clone(),
            confidence=filtered_confidence.reshape(
                target_steps,
                token_height,
                token_width,
                memory_slots,
            ),
            pair_stats=transport.pair_stats,
            metadata=metadata,
        ),
        output_path,
    )
    report_path = output_path.with_suffix(".json")
    with report_path.open("w") as handle:
        json.dump(
            {"metadata": metadata, "track_filter_report": report},
            handle,
            indent=2,
        )
    print(json.dumps(report, sort_keys=True), flush=True)
    print(f"saved_map={output_path}", flush=True)
    print(f"saved_report={report_path}", flush=True)


if __name__ == "__main__":
    main()
