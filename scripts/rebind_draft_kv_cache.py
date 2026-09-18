#!/usr/bin/env python3
"""Bind an existing draft attention K/V cache to a geometry map.

The cached K/V tensors depend on the draft run and anchor token times, not on
the target geometry map. This utility copies the cache, records the map SHA256,
and verifies that every cached tensor is bit-identical after serialization.
"""

import argparse
import hashlib
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_cache", type=Path, required=True)
    parser.add_argument("--geometry_map", type=Path, required=True)
    parser.add_argument("--output_cache", type=Path, required=True)
    return parser.parse_args()


def assert_same_features(before: dict, after: dict) -> None:
    if before.keys() != after.keys():
        raise RuntimeError("Feature-cache keys changed during rebinding")
    for cache_key in before:
        before_entry = before[cache_key]
        after_entry = after[cache_key]
        if before_entry.keys() != after_entry.keys():
            raise RuntimeError(f"Entry keys changed for {cache_key}")
        for tensor_name in before_entry:
            if not torch.equal(before_entry[tensor_name], after_entry[tensor_name]):
                raise RuntimeError(
                    f"Tensor changed during rebinding: {cache_key}/{tensor_name}"
                )


def main() -> None:
    args = parse_args()
    payload = torch.load(args.input_cache, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("cache_kind") != "attention_kv":
        raise ValueError("Only attention_kv caches can be rebound")
    if metadata.get("format_version") != 2:
        raise ValueError("Only attention_kv format_version=2 is supported")

    map_sha256 = hashlib.sha256(args.geometry_map.read_bytes()).hexdigest()
    metadata["geometry_map_sha256"] = map_sha256
    rebound = {"metadata": metadata, "features": payload["features"]}

    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rebound, args.output_cache)

    verified = torch.load(args.output_cache, map_location="cpu", weights_only=False)
    assert_same_features(payload["features"], verified["features"])
    if verified["metadata"] != metadata:
        raise RuntimeError("Cache metadata changed during serialization")

    print(f"saved: {args.output_cache}")
    print(f"geometry_map_sha256: {map_sha256}")
    print(f"entries_verified: {len(payload['features'])}")


if __name__ == "__main__":
    main()
