#!/usr/bin/env python3
"""Evaluate Independent LRE with MegaSaM geometry and UFM correspondences only."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from independent_lre_megasam_runner import run_full_megasam_case
from independent_lre_reprojection import load_endpoint_geometry, score_direction
from independent_lre_runtime_trace import collect_loaded_source_files, merge_load_traces, trace_payload
from metric_adapter_common import (
    decode_all_frames,
    load_input_lock,
    load_weight_manifest,
    publish_component,
    read_json,
    require_readonly_regular,
    require_schedule,
    require_sha,
    sha256_file,
    verify_core_and_dependencies,
)
from torch_runtime_guard import current_locked_torch_load_trace, import_locked_torch


METRIC_ID = "long_range_reprojection_error"
WEIGHT_ROLES = {
    "megasam_camera_tracker_checkpoint",
    "depthanything_vitl14_checkpoint",
    "shared_raft_things_checkpoint",
    "unidepth_v2_vitl14_config",
    "unidepth_v2_vitl14_checkpoint",
    "ufm_base_checkpoint",
}
MEGASAM_ROLES = WEIGHT_ROLES - {"ufm_base_checkpoint"}
MIN_VALID_FRACTION = 0.2
COVISIBILITY_THRESHOLD = 0.5
FORBIDDEN_SELECTOR_TOKENS = ("vgg" + "t", "vgg" + "t" + "_" + "omega", "vgg" + "t" + "-" + "omega")
LRE_WEIGHT_SOURCE_REQUIREMENTS = {
    "megasam_camera_tracker_checkpoint": {
        "kind": "url", "origin_class": "versioned_repository", "provider": "github",
        "repository": "mega-sam/mega-sam", "revision": "a27b4e633c5cc0828a62ed943ef9f6505705fd3f",
        "upstream_path": "checkpoints/megasam_final.pth",
    },
    "depthanything_vitl14_checkpoint": {
        "kind": "url", "origin_class": "versioned_repository", "provider": "huggingface",
        "repository": "spaces/LiheYoung/Depth-Anything", "revision": "7f1457e21e74e7aa001c88fc15da5c74598aa3fa",
        "upstream_path": "checkpoints/depth_anything_vitl14.pth",
    },
    "shared_raft_things_checkpoint": {
        "kind": "zip_member_url", "origin_class": "direct_content", "provider": "princeton-vl/RAFT",
        "repository": "RAFT model archive", "source_code_repository": "princeton-vl/RAFT",
        "source_code_revision": "2888e15a51fa41140771d3f498ed8023cff098d1",
        "source_snapshot_labels": ["raft", "megasam", "vbench"],
        "archive_relative_path": "source_archives/raft_models.zip", "archive_member": "models/raft-things.pth",
    },
    "unidepth_v2_vitl14_config": {
        "kind": "url", "origin_class": "versioned_repository", "provider": "huggingface",
        "repository": "lpiccinelli/unidepth-v2-vitl14", "revision": "1d0d3c52f60b5164629d279bb9a7546458e6dcc4",
        "upstream_path": "config.json",
    },
    "unidepth_v2_vitl14_checkpoint": {
        "kind": "url", "origin_class": "versioned_repository", "provider": "huggingface",
        "repository": "lpiccinelli/unidepth-v2-vitl14", "revision": "1d0d3c52f60b5164629d279bb9a7546458e6dcc4",
        "upstream_path": "model.safetensors",
    },
    "ufm_base_checkpoint": {
        "kind": "url", "origin_class": "versioned_repository", "provider": "huggingface",
        "repository": "infinity1096/UFM-Base", "revision": "cb60fe3d33ace8bbe1416a1e93de0807336927c1",
        "upstream_path": "ufm_560_base.pt",
    },
}


def _locked_cases(input_lock: Path, entries: list[dict]) -> set[str] | None:
    """Return the frozen seed-0 case denominator for compared methods."""

    payload = read_json(require_readonly_regular(input_lock, "input lock"), "input lock")
    method_id = payload.get("method_id")
    if method_id in {"wan_unguided_seed0", "wan_lora_dpo_step64_base_traindev"}:
        return None
    binding = payload.get("baseline_metric_eligibility_lock")
    if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
        raise ValueError("compared Independent LRE scoring requires the frozen baseline eligibility lock")
    eligibility_path = require_readonly_regular(Path(binding["path"]), "baseline eligibility lock")
    if sha256_file(eligibility_path) != require_sha(binding.get("sha256"), "baseline eligibility lock SHA"):
        raise ValueError("baseline eligibility lock binding changed")
    eligibility = read_json(eligibility_path, "baseline eligibility lock")
    units = eligibility.get("locked_units")
    expected_schema = (
        "geometry-selection-metric-traindev-eligibility-lock-v1"
        if method_id == "wan_lora_dpo_step64_adapted_traindev"
        else "geometry-selection-metric-eligibility-lock-v1"
    )
    if (
        eligibility.get("schema") != expected_schema
        or not isinstance(units, dict)
        or not isinstance(units.get(METRIC_ID), list)
    ):
        raise ValueError("baseline eligibility lock lacks Independent LRE units")
    allowed_cases = {entry["case_id"] for entry in entries}
    locked: set[str] = set()
    for item in units[METRIC_ID]:
        if (
            not isinstance(item, dict)
            or item.get("unit_id") != "first_last"
            or not isinstance(item.get("case_id"), str)
            or item["case_id"] not in allowed_cases
        ):
            raise ValueError("baseline eligibility LRE units do not match this input lock")
        locked.add(item["case_id"])
    if not locked:
        raise ValueError("baseline eligibility lock contains no Independent LRE cases")
    return locked


def _source_root_from_core(core_path: Path) -> Path:
    expected_runner = Path(__file__).with_name("independent_lre_megasam_runner.py").resolve(strict=True)
    if core_path != expected_runner:
        raise RuntimeError("the locked Independent LRE core must be the co-sealed MegaSaM runner")
    root = expected_runner.parents[2]
    ufm_source = root / "geco" / "external" / "UFM"
    if ufm_source.is_symlink() or not ufm_source.is_dir():
        raise RuntimeError("sealed evaluator source lacks the local UFM runtime")
    return root


def _load_ufm(source_root: Path, checkpoint: Path, device):
    source = str((source_root / "geco" / "external" / "UFM").resolve(strict=True))
    if source not in sys.path:
        sys.path.insert(0, source)
    from uniflowmatch.models.ufm import UniFlowMatchConfidence

    model = UniFlowMatchConfidence.from_pretrained_ckpt(str(checkpoint)).to(device).eval()
    model.requires_grad_(False)
    return model


def _read_rgb(frame: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(frame) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if array.shape != (704, 1280, 3):
        raise RuntimeError(f"locked decoder endpoint raster changed: {frame}")
    return array


def _predict_ufm(model, source_rgb: np.ndarray, target_rgb: np.ndarray, device):
    import torch

    source = torch.from_numpy(np.ascontiguousarray(source_rgb)).to(device=device, dtype=torch.uint8)
    target = torch.from_numpy(np.ascontiguousarray(target_rgb)).to(device=device, dtype=torch.uint8)
    with torch.inference_mode():
        result = model.predict_correspondences_batched(source_image=source, target_image=target)
    if result.flow is None or result.covisibility is None or result.flow.flow_output is None or result.covisibility.mask is None:
        raise RuntimeError("sealed UFM runtime did not return flow and covisibility")
    return result.flow.flow_output, result.covisibility.mask


def _require_lre_weight_sources(weight_manifest: Path) -> dict[str, dict]:
    payload = read_json(require_readonly_regular(weight_manifest, "weight manifest"), "weight manifest")
    sources = payload.get("source_identities")
    if not isinstance(sources, dict) or set(sources) != WEIGHT_ROLES:
        raise RuntimeError("Independent LRE weight sources do not cover the frozen role closure")
    normalized: dict[str, dict] = {}
    for role, expected in LRE_WEIGHT_SOURCE_REQUIREMENTS.items():
        source = sources.get(role)
        if not isinstance(source, dict) or any(source.get(field) != value for field, value in expected.items()):
            raise RuntimeError(f"Independent LRE weight source differs from its frozen MegaSaM/UFM identity: {role}")
        try:
            canonical = json.loads(json.dumps(source, sort_keys=True, separators=(",", ":")))
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"Independent LRE weight source is not canonical JSON: {role}") from error
        if canonical != source:
            raise RuntimeError(f"Independent LRE weight source is noncanonical: {role}")
        normalized[role] = canonical
    return normalized


def _reject_selector_weight_bindings(weights: dict[str, Path], sources: dict[str, dict]) -> None:
    for role, path in weights.items():
        source = sources.get(role)
        if any(token in f"{role}:{path}:{json.dumps(source, sort_keys=True)}".lower() for token in FORBIDDEN_SELECTOR_TOKENS):
            raise RuntimeError("Independent LRE weight binding contains a forbidden selector reference")


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

    torch, _initial_torch_trace = import_locked_torch()

    core_path, weight_manifest, schedule_path, decoder = verify_core_and_dependencies(
        adapter_path=Path(__file__),
        core_evaluator=args.core_evaluator,
        weight_manifest=args.weight_manifest,
        schedule=args.schedule,
        decoder=args.decoder,
        output=args.output,
    )
    schedule = require_schedule(schedule_path, METRIC_ID)
    if (
        schedule[METRIC_ID].get("pairs") != [[0, 120]]
        or schedule[METRIC_ID].get("ufm_covisibility_threshold") != COVISIBILITY_THRESHOLD
        or schedule[METRIC_ID].get("minimum_valid_fraction_per_direction") != MIN_VALID_FRACTION
    ):
        raise ValueError("locked Independent LRE schedule differs from the frozen endpoint contract")
    entries = load_input_lock(args.input_lock, schedule_path)
    locked_cases = _locked_cases(args.input_lock, entries)
    weights = load_weight_manifest(weight_manifest, WEIGHT_ROLES)
    weight_sources = _require_lre_weight_sources(weight_manifest)
    _reject_selector_weight_bindings(weights, weight_sources)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Independent LRE must run on a CUDA device")
    source_root = _source_root_from_core(core_path)
    ufm_model = _load_ufm(source_root, weights["ufm_base_checkpoint"], device)
    runtime_traces: list[list[dict[str, str]]] = [collect_loaded_source_files(source_root)]
    records = []
    for entry in entries:
        if locked_cases is not None and entry["case_id"] not in locked_cases:
            continue
        with tempfile.TemporaryDirectory(prefix="independent-lre-") as temporary:
            workspace = Path(temporary)
            artifacts = run_full_megasam_case(
                decoder=decoder,
                video=Path(entry["metric_video_path"]),
                weights={role: weights[role] for role in MEGASAM_ROLES},
                case_id=entry["case_id"],
                workspace=workspace,
            )
            runtime_traces.append(list(artifacts.runtime_load_trace))
            forward_flow, forward_covisibility = _predict_ufm(
                ufm_model, _read_rgb(artifacts.frames[0]), _read_rgb(artifacts.frames[120]), device
            )
            backward_flow, backward_covisibility = _predict_ufm(
                ufm_model, _read_rgb(artifacts.frames[120]), _read_rgb(artifacts.frames[0]), device
            )
            geometry_0 = load_endpoint_geometry(artifacts.cvd_artifact, 0, device)
            geometry_120 = load_endpoint_geometry(artifacts.cvd_artifact, 120, device)
            forward = score_direction(
                geometry_0,
                geometry_120,
                forward_flow,
                forward_covisibility,
                covisibility_threshold=COVISIBILITY_THRESHOLD,
            )
            backward = score_direction(
                geometry_120,
                geometry_0,
                backward_flow,
                backward_covisibility,
                covisibility_threshold=COVISIBILITY_THRESHOLD,
            )
        eligible = forward.valid_fraction >= MIN_VALID_FRACTION and backward.valid_fraction >= MIN_VALID_FRACTION
        if locked_cases is not None and not eligible:
            raise RuntimeError(f"compared method cannot score seed-0-locked Independent LRE case: {entry['case_id']}")
        records.append(
            {
                "case_id": entry["case_id"],
                "unit_id": "first_last",
                "value": 0.5 * (forward.normalized_residual + backward.normalized_residual) if eligible else 0.0,
                "forward_error": forward.normalized_residual,
                "backward_error": backward.normalized_residual,
                "forward_valid_fraction": forward.valid_fraction,
                "backward_valid_fraction": backward.valid_fraction,
                "forward_valid_pixels": forward.valid_pixels,
                "backward_valid_pixels": backward.valid_pixels,
                "eligible": eligible,
            }
        )

    runtime_traces.append(collect_loaded_source_files(source_root))
    runtime_trace = trace_payload(merge_load_traces(*runtime_traces))

    publish_component(
        args.output,
        METRIC_ID,
        records,
        {
            "core_evaluator_sha256": sha256_file(core_path),
            "pair": [0, 120],
            "geometry": "full pinned MegaSaM CVD artifact from all 121 decoded frames",
            "correspondence": "sealed local UFM endpoint flow and covisibility only",
            "upstream_cvd_updates": {"scale_shift": 100, "consistent_depth": 400},
            "coordinate_convention": "original/CVD half-pixel mapping; cam_c2w target_from_source=inverse(target)@source",
            "covisibility_threshold": COVISIBILITY_THRESHOLD,
            "minimum_valid_fraction_per_direction": MIN_VALID_FRACTION,
            "aggregation": "mean of eligible forward/backward normalized endpoint residuals",
            "independent_runtime_load_trace": runtime_trace,
            "torch_runtime_load_trace": current_locked_torch_load_trace(),
        },
    )


if __name__ == "__main__":
    main()
