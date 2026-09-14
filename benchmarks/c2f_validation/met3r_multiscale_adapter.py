#!/usr/bin/env python3
"""Evaluate all frozen MEt3R temporal scales with the sealed v13 stack."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import tempfile
from pathlib import Path

from metric_adapter_common import (
    decode_all_frames,
    load_input_lock,
    load_weight_manifest,
    publish_component,
    require_schedule,
    sha256_file,
    verify_core_and_dependencies,
)


METRIC_ID = "met3r_multiscale"
SCHEDULE_ID = "met3r"
FIXED_ADAPTER = Path("/vol/dissolve/yz10325/evaluator_sources/met3r_adapter_v13_api_fix_v2/met3r_adapter.py")
FIXED_ADAPTER_SHA256 = "7e816d1bd5542cf3581f9e22b124a8779667ca567ba6d19f22f13f6cdd17612b"
WEIGHT_ROLES = {
    "mast3r_config",
    "mast3r_checkpoint",
    "featup_dino16_jbu_checkpoint",
    "featup_dino_vits16_checkpoint",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fixed_adapter():
    if file_sha256(FIXED_ADAPTER) != FIXED_ADAPTER_SHA256:
        raise RuntimeError("fixed MEt3R adapter SHA mismatch")
    spec = importlib.util.spec_from_file_location("_fixed_met3r_adapter", FIXED_ADAPTER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {FIXED_ADAPTER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_pairs(value: object, count: int, label: str) -> list[list[int]]:
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not isinstance(index, int) or not 0 <= index < 121 for index in pair)
            for pair in value
        )
    ):
        raise ValueError(f"locked MEt3R schedule has invalid {label}")
    return value


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
    schedule = require_schedule(schedule_path, SCHEDULE_ID)[SCHEDULE_ID]
    pair_groups = {
        "half_second": validate_pairs(schedule.get("half_second_pairs"), 10, "half-second pairs"),
        "one_second": validate_pairs(schedule.get("one_second_pairs"), 5, "one-second pairs"),
        "first_last": validate_pairs([schedule.get("first_last_pair")], 1, "first-last pair"),
    }
    entries = load_input_lock(args.input_lock, schedule_path)
    weights = load_weight_manifest(weight_manifest, WEIGHT_ROLES)
    fixed = load_fixed_adapter()
    source_root = fixed.sealed_source_root()
    featup_root = fixed.sealed_source_directory(source_root, "featup", "hubconf.py")
    dino_root = fixed.sealed_source_directory(source_root, "dino", "hubconf.py")
    mast3r_directory = fixed.local_mast3r_directory(weights)

    import torch

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("MEt3R must run on a CUDA device")
    core = fixed.load_original_core(core_path)
    core.backbone_to_weights["mast3r"] = str(mast3r_directory)
    with fixed.locked_local_featup(
        featup_root,
        dino_root,
        weights["featup_dino16_jbu_checkpoint"],
        weights["featup_dino_vits16_checkpoint"],
    ):
        model = core.MEt3R(
            img_size=256,
            use_norm=True,
            backbone="mast3r",
            feature_backbone="dino16",
            feature_backbone_weights=fixed.FEATUP_SENTINEL,
            upsampler="featup",
            distance="cosine",
            freeze=True,
        ).to(device).eval()

    records = []
    with torch.inference_mode():
        for entry in entries:
            with tempfile.TemporaryDirectory(prefix="met3r-multiscale-frames-") as temporary:
                frames = decode_all_frames(decoder, Path(entry["metric_video_path"]), Path(temporary))
                tensors: dict[int, torch.Tensor] = {}
                for pairs in pair_groups.values():
                    for left, right in pairs:
                        for frame_index in (left, right):
                            if frame_index not in tensors:
                                tensors[frame_index] = fixed.frame_tensor(frames[frame_index]).squeeze(0)
                for group, pairs in pair_groups.items():
                    for pair_index, (left, right) in enumerate(pairs):
                        images = torch.stack([tensors[left], tensors[right]], dim=0).unsqueeze(0).to(
                            device, non_blocking=True
                        )
                        score = model(images)[0]
                        records.append(
                            {
                                "case_id": entry["case_id"],
                                "unit_id": f"{group}_{pair_index}",
                                "value": float(score.detach().cpu().reshape(-1)[0].item()),
                            }
                        )

    publish_component(
        args.output,
        METRIC_ID,
        records,
        {
            "core_evaluator_sha256": sha256_file(core_path),
            "fixed_adapter": {"path": str(FIXED_ADAPTER), "sha256": FIXED_ADAPTER_SHA256},
            "pair_groups": pair_groups,
            "preprocessing": "identical to fixed v13 MEt3R adapter: RGB -> bilinear 256x256 -> [-1,1]",
            "local_weight_routing": {
                "mast3r_directory": str(mast3r_directory),
                "mast3r_config": str(weights["mast3r_config"]),
                "mast3r_checkpoint": str(weights["mast3r_checkpoint"]),
                "featup_dino16_jbu_checkpoint": str(weights["featup_dino16_jbu_checkpoint"]),
                "featup_dino_vits16_checkpoint": str(weights["featup_dino_vits16_checkpoint"]),
            },
        },
    )


if __name__ == "__main__":
    main()
