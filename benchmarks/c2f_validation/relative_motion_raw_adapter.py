#!/usr/bin/env python3
"""Compute frozen RAFT total motion without a cross-dataset baseline anchor."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path

from metric_adapter_common import (
    load_input_lock,
    load_weight_manifest,
    publish_component,
    require_schedule,
    sha256_file,
    verify_core_and_dependencies,
)


METRIC_ID = "relative_total_motion_raw"
SCHEDULE_ID = "relative_total_motion_percent"
FIXED_ADAPTER = Path(
    "/vol/dissolve/yz10325/evaluator_sources/geometry_selection_evaluator_v13/"
    "protocol/metric_adapters/relative_total_motion_adapter.py"
)
FIXED_ADAPTER_SHA256 = "593632eb5ef7a2a881806657e922482e123598a2b64a3a989cc7f25d0d40fadd"
WEIGHT_ROLES = {"relative_motion_raft_large_checkpoint"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fixed_adapter():
    if file_sha256(FIXED_ADAPTER) != FIXED_ADAPTER_SHA256:
        raise RuntimeError("fixed relative-motion adapter SHA mismatch")
    spec = importlib.util.spec_from_file_location("_fixed_relative_motion_adapter", FIXED_ADAPTER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {FIXED_ADAPTER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-lock", type=Path, required=True)
    parser.add_argument("--core-evaluator", type=Path, required=True)
    parser.add_argument("--weight-manifest", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    core_path, weight_manifest, schedule_path, decoder = verify_core_and_dependencies(
        adapter_path=Path(__file__),
        core_evaluator=args.core_evaluator,
        weight_manifest=args.weight_manifest,
        schedule=args.schedule,
        decoder=args.decoder,
        output=args.output,
    )
    schedule = require_schedule(schedule_path, SCHEDULE_ID)
    if schedule[SCHEDULE_ID].get("adjacent_pairs") != [[index, index + 1] for index in range(120)]:
        raise ValueError("locked motion schedule must contain all 120 adjacent frame pairs")
    entries = load_input_lock(args.input_lock, schedule_path)
    weights = load_weight_manifest(weight_manifest, WEIGHT_ROLES)
    fixed = load_fixed_adapter()

    import torch
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RAFT motion diagnostic must run on a CUDA device")
    core = fixed.load_original_core(core_path)
    expected_opencv = json.loads(os.environ.get("GEOMETRY_EVAL_LOCKED_RUNTIME_IDENTITIES_JSON", "{}"))
    if expected_opencv != {"opencv": fixed.opencv_identity()}:
        raise RuntimeError("OpenCV runtime differs from the evaluator-lock binding")
    checkpoint = torch.load(weights["relative_motion_raft_large_checkpoint"], map_location="cpu")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise RuntimeError("locked RAFT checkpoint does not contain a state dictionary")
    model = raft_large(weights=None, progress=False)
    result = model.load_state_dict(
        {str(key).removeprefix("module."): value for key, value in checkpoint.items()}, strict=True
    )
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("locked RAFT checkpoint does not exactly match RAFT-Large")
    model = model.to(device).eval()
    transform = Raft_Large_Weights.C_T_SKHT_V2.transforms()

    records = []
    for entry in entries:
        with tempfile.TemporaryDirectory(prefix="raft-motion-raw-frames-") as temporary:
            frames = fixed.frames_from_locked_decode(
                core, decoder, Path(entry["metric_video_path"]), Path(temporary)
            )
            stats = core.motion_stats(model, transform, frames, device)
        if stats.get("n_pairs") != 120:
            raise RuntimeError("original RAFT motion core did not consume exactly 120 adjacent pairs")
        raw_total = float(stats["mean_flow_px"]) * 120.0
        records.append(
            {
                "case_id": entry["case_id"],
                "unit_id": "total_motion",
                "value": raw_total,
                "raw_total_motion": raw_total,
                "mean_flow_px": float(stats["mean_flow_px"]),
                "median_flow_px": float(stats["median_flow_px"]),
                "p90_flow_px": float(stats["p90_flow_px"]),
                "n_pairs": 120,
            }
        )

    publish_component(
        args.output,
        METRIC_ID,
        records,
        {
            "core_evaluator_sha256": sha256_file(core_path),
            "fixed_adapter": {"path": str(FIXED_ADAPTER), "sha256": FIXED_ADAPTER_SHA256},
            "weights_enum": "Raft_Large_Weights.C_T_SKHT_V2",
            "long_side": 512,
            "adjacent_pairs": 120,
            "aggregation": "raw sum of mean RAFT flow over all adjacent pairs; paired ratio is computed later",
            "opencv_runtime": expected_opencv["opencv"],
        },
    )


if __name__ == "__main__":
    main()
