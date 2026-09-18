#!/usr/bin/env python3
"""Keep the earliest stable observations for every geometry-matched target token."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--canonical_window",
        type=int,
        default=2,
        help="Keep source observations no later than earliest_source + this many latent steps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.canonical_window < 0:
        raise ValueError("canonical_window must be non-negative")

    data = torch.load(args.input, map_location="cpu", weights_only=False)
    required = {"source_time", "source_index", "confidence"}
    if not isinstance(data, dict) or not required.issubset(data):
        raise ValueError(f"Expected a geometry transport dictionary with keys {sorted(required)}")

    source_time = data["source_time"]
    confidence = data["confidence"]
    if source_time.ndim != 3 or confidence.ndim != 4:
        raise ValueError("Unexpected source_time or confidence rank")

    num_targets, height, width, slots = confidence.shape
    if source_time.shape != (num_targets, height * width, slots):
        raise ValueError(
            f"source_time shape {tuple(source_time.shape)} is incompatible with "
            f"confidence shape {tuple(confidence.shape)}"
        )

    source_grid = source_time.reshape(num_targets, height, width, slots)
    valid = confidence > 0
    sentinel = torch.iinfo(source_grid.dtype).max
    earliest = torch.where(valid, source_grid, sentinel).amin(dim=-1, keepdim=True)
    keep = valid & (source_grid <= earliest + args.canonical_window)
    canonical_confidence = torch.where(keep, confidence, torch.zeros_like(confidence))

    output = dict(data)
    output["confidence"] = canonical_confidence
    metadata = dict(output.get("metadata", {}))
    metadata.update(
        {
            "canonicalized_from": str(Path(args.input).resolve()),
            "canonicalization": "per_target_token_earliest_window",
            "canonical_window": args.canonical_window,
        }
    )
    output["metadata"] = metadata

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)

    active_before = valid.any(dim=-1)
    active_after = keep.any(dim=-1)
    kept_candidates = source_grid[keep].tolist()
    print("saved:", output_path)
    print(f"target_token_coverage_before={active_before.float().mean().item():.6f}")
    print(f"target_token_coverage_after={active_after.float().mean().item():.6f}")
    print(f"candidate_keep_fraction={keep.sum().item() / max(1, valid.sum().item()):.6f}")
    print("kept_source_times:", Counter(kept_candidates).most_common())


if __name__ == "__main__":
    main()
