#!/usr/bin/env python3
"""Keep only the highest-confidence source anchor per target token."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min_confidence", type=float, default=0.0)
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    confidence = payload["confidence"].float()
    source_time = payload["source_time"].long()
    source_index = payload["source_index"].long()
    if confidence.ndim != 4:
        raise ValueError("Expected confidence [T-1,H,W,K]")
    if source_time.shape != source_index.shape:
        raise ValueError("source_time and source_index shapes differ")
    if source_time.shape[:2] != (
        confidence.shape[0],
        confidence.shape[1] * confidence.shape[2],
    ):
        raise ValueError("Source map shapes do not match confidence")
    if source_time.shape[-1] != confidence.shape[-1]:
        raise ValueError("Source map and confidence slot counts differ")

    top_confidence, top_slot = confidence.max(dim=-1, keepdim=True)
    top_source_time = source_time.reshape(
        confidence.shape[0],
        confidence.shape[1],
        confidence.shape[2],
        confidence.shape[3],
    ).gather(-1, top_slot)
    top_source_index = source_index.reshape_as(confidence).gather(
        -1,
        top_slot,
    )
    active = top_confidence >= args.min_confidence
    active &= top_confidence > 0
    top_confidence = top_confidence * active
    top_source_time = torch.where(
        active,
        top_source_time,
        torch.zeros_like(top_source_time),
    )
    top_source_index = torch.where(
        active,
        top_source_index,
        torch.zeros_like(top_source_index),
    )

    metadata = dict(payload.get("metadata", {}))
    metadata.update(
        {
            "derived_from": str(args.input.resolve()),
            "map_transform": "top1_confidence",
            "memory_slots": 1,
            "top1_min_confidence": args.min_confidence,
        }
    )
    output_payload = {
        "source_time": top_source_time.reshape(
            confidence.shape[0],
            confidence.shape[1] * confidence.shape[2],
            1,
        ),
        "source_index": top_source_index.reshape(
            confidence.shape[0],
            confidence.shape[1] * confidence.shape[2],
            1,
        ),
        "confidence": top_confidence,
        "pair_stats": payload.get("pair_stats", []),
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, args.output)

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "min_confidence": args.min_confidence,
        "active_tokens": int(active.sum().item()),
        "active_fraction": float(active.float().mean().item()),
        "mean_active_confidence": (
            float(top_confidence[active].mean().item())
            if active.any()
            else 0.0
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
