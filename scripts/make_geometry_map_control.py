#!/usr/bin/env python3
"""Create correspondence controls while preserving a geometry map's mask."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_map", required=True)
    parser.add_argument("--output_map", required=True)
    parser.add_argument("--mode", required=True, choices=("identity", "random"))
    parser.add_argument(
        "--evidence",
        choices=("auto", "transport", "observed_background"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = torch.load(args.input_map, map_location="cpu", weights_only=False)
    use_background = (
        args.evidence == "observed_background"
        or (
            args.evidence == "auto"
            and "observed_background_confidence" in payload
            and torch.count_nonzero(
                payload["observed_background_confidence"]
            ).item()
            > 0
        )
    )
    index_key = (
        "observed_background_index" if use_background else "source_index"
    )
    confidence_key = (
        "observed_background_confidence" if use_background else "confidence"
    )
    source_index = payload[index_key].clone()
    confidence = payload[confidence_key]
    if confidence.ndim == 4:
        valid = confidence.reshape(source_index.shape) > 0
    else:
        valid = confidence > 0

    temporal, spatial, slots = source_index.shape
    target_index = torch.arange(spatial, dtype=source_index.dtype)
    replacement = torch.empty_like(source_index)

    if args.mode == "identity":
        replacement[:] = target_index.view(1, spatial, 1)
    else:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        for target_time in range(temporal):
            for slot in range(slots):
                replacement[target_time, :, slot] = torch.randperm(
                    spatial, generator=generator
                )

    payload[index_key] = torch.where(valid, replacement, source_index)
    metadata = dict(payload.get("metadata", {}))
    metadata["correspondence_control"] = args.mode
    metadata["correspondence_control_seed"] = args.seed
    metadata["correspondence_control_evidence"] = (
        "observed_background" if use_background else "transport"
    )
    input_map = Path(args.input_map).resolve()
    metadata["correspondence_control_source"] = str(input_map)
    metadata["correspondence_control_source_sha256"] = sha256_file(input_map)
    payload["metadata"] = metadata

    output = Path(args.output_map)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"saved: {output}")
    print(f"mode: {args.mode}")
    print(f"evidence: {metadata['correspondence_control_evidence']}")
    print(f"valid_fraction: {valid.float().mean().item():.6f}")


if __name__ == "__main__":
    main()
