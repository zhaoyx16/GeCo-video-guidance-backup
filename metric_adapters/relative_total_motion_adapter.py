#!/usr/bin/env python3
"""Compute the frozen RAFT adjacent-frame total-motion diagnostic offline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

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
from torchvision_runtime_guard import (
    current_locked_torchvision_load_trace,
    import_locked_torchvision,
)
from torch_runtime_guard import current_locked_torch_load_trace, import_locked_torch


METRIC_ID = "relative_total_motion_percent"
# Keep this distinct from the RAFT-Things asset shared by MegaSaM/VBench.
# The motion-retention metric is defined by TorchVision's RAFT-Large weight.
WEIGHT_ROLES = {"relative_motion_raft_large_checkpoint"}


def load_original_core(core_path: Path):
    spec = importlib.util.spec_from_file_location("locked_raft_motion_core", core_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the locked RAFT motion core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def baseline_motion_anchors(input_lock: Path, case_ids: set[str]) -> dict[str, float] | None:
    payload = read_json(require_readonly_regular(input_lock, "input lock"), "input lock")
    method_id = payload.get("method_id")
    if method_id in {"wan_unguided_seed0", "wan_lora_dpo_step64_base_traindev"}:
        return None
    binding = payload.get("baseline_metric_eligibility_lock")
    if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
        raise ValueError("compared-method motion scoring requires the frozen baseline eligibility lock")
    path = require_readonly_regular(Path(binding["path"]), "baseline eligibility lock")
    if sha256_file(path) != require_sha(binding.get("sha256"), "baseline eligibility lock SHA"):
        raise ValueError("baseline eligibility lock binding changed")
    eligibility = read_json(path, "baseline eligibility lock")
    anchors = eligibility.get("baseline_motion_anchors")
    expected_schema = (
        "geometry-selection-metric-traindev-eligibility-lock-v1"
        if method_id == "wan_lora_dpo_step64_adapted_traindev"
        else "geometry-selection-metric-eligibility-lock-v1"
    )
    if eligibility.get("schema") != expected_schema or not isinstance(anchors, dict) or not set(anchors).issubset(case_ids):
        raise ValueError("baseline eligibility lock does not provide valid motion anchors for this input lock")
    normalized: dict[str, float] = {}
    for case_id, value in anchors.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 1e-6:
            raise ValueError(f"invalid baseline motion anchor: {case_id}")
        normalized[case_id] = float(value)
    return normalized


def frames_from_locked_decode(core, decoder: Path, video: Path, workspace: Path):
    """Use the locked video decoder, then retain the original core's resize rule.

    The reference helper's ``load_video`` combines codec decoding and OpenCV
    preprocessing.  Video decoding must be bound to the evaluator lock, so this
    integration layer performs only the first part with locked FFmpeg and calls
    the original ``resized_shape`` plus the exact OpenCV RGB/INTER_AREA path for
    the metric's documented preprocessing.
    """
    import cv2
    import torch

    decoded = decode_all_frames(decoder, video, workspace)
    tensors = []
    for image_path in decoded:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"OpenCV could not read locked decoder frame: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out_h, out_w = core.resized_shape(*rgb.shape[:2], 512)
        rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
        tensors.append(torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0))
    if len(tensors) != 121:
        raise RuntimeError("locked decoder did not provide all 121 RAFT frames")
    return torch.stack(tensors)


def require_clean_dynamic_loader_environment() -> None:
    injected = sorted(name for name, value in os.environ.items() if name.startswith("LD_") and value)
    if injected:
        raise RuntimeError(f"dynamic-loader environment must be empty for sealed OpenCV execution: {injected}")


def native_dependency_inspector_identity() -> dict[str, str]:
    raw = json.loads(os.environ.get("GEOMETRY_EVAL_LOCKED_RUNTIME_IDENTITIES_JSON", "{}"))
    inspector = raw.get("opencv", {}).get("native_dependency_inspector") if isinstance(raw, dict) else None
    if not isinstance(inspector, dict) or not all(isinstance(inspector.get(key), str) and inspector[key] for key in ("path", "sha256", "interpreter_path", "interpreter_sha256")):
        raise RuntimeError("guard did not provide a sealed OpenCV dependency inspector identity")
    inspector_path = Path(inspector["path"]).resolve(strict=True)
    interpreter_path = Path(inspector["interpreter_path"]).resolve(strict=True)
    if sha256_file(inspector_path) != inspector["sha256"] or sha256_file(interpreter_path) != inspector["interpreter_sha256"]:
        raise RuntimeError("sealed OpenCV dependency inspector or interpreter changed")
    first_line = inspector_path.open("rb").readline().decode("utf-8", errors="strict").strip()
    if not first_line.startswith("#!") or Path(first_line[2:].split(maxsplit=1)[0]).resolve(strict=True) != interpreter_path:
        raise RuntimeError("sealed OpenCV dependency inspector does not use its sealed interpreter")
    return {"path": str(inspector_path), "sha256": inspector["sha256"], "interpreter_path": str(interpreter_path), "interpreter_sha256": inspector["interpreter_sha256"]}


def native_library_closure(native: Path, inspector: dict[str, str]) -> list[dict[str, str]]:
    require_clean_dynamic_loader_environment()
    result = subprocess.run(
        [inspector["path"], str(native)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env={"PATH": "/usr/bin:/bin"},
    )
    libraries: dict[str, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text or "linux-vdso" in text:
            continue
        if "not found" in text:
            raise RuntimeError(f"OpenCV native dependency is unresolved: {text}")
        tokens = text.replace("=>", " ").split()
        candidate = next((token for token in tokens if token.startswith("/")), None)
        if candidate is None:
            raise RuntimeError(f"could not resolve OpenCV native dependency: {text}")
        path = Path(candidate).resolve(strict=True)
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"OpenCV native dependency is not a sealed regular file: {path}")
        libraries[str(path)] = {"path": str(path), "sha256": sha256_file(path)}
    if not libraries:
        raise RuntimeError("ldd returned no sealed OpenCV native dependencies")
    return [libraries[path] for path in sorted(libraries)]


def opencv_identity() -> dict[str, object]:
    require_clean_dynamic_loader_environment()
    import cv2

    package_root = Path(cv2.__file__).resolve(strict=True).parent
    candidates = sorted(
        {path.resolve(strict=True) for pattern in ("*.so", "*.pyd", "*.dylib") for path in package_root.glob(pattern) if path.is_file()}
    )
    if len(candidates) != 1:
        raise RuntimeError(f"could not identify one native OpenCV implementation below {package_root}: {candidates}")
    binary = candidates[0]
    inspector = native_dependency_inspector_identity()
    return {
        "module_version": str(cv2.__version__),
        "native_extension_path": str(binary),
        "native_extension_sha256": sha256_file(binary),
        "build_information_sha256": hashlib.sha256(cv2.getBuildInformation().encode("utf-8")).hexdigest(),
        "native_library_closure": native_library_closure(binary, inspector),
        "native_dependency_inspector": inspector,
    }


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
    if schedule[METRIC_ID].get("adjacent_pairs") != [[index, index + 1] for index in range(120)]:
        raise ValueError("locked motion schedule must contain exactly all 120 adjacent frame pairs")
    entries = load_input_lock(args.input_lock, schedule_path)
    anchors = baseline_motion_anchors(args.input_lock, {entry["case_id"] for entry in entries})
    weights = load_weight_manifest(weight_manifest, WEIGHT_ROLES)

    import_locked_torchvision()
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    torchvision_trace = current_locked_torchvision_load_trace()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RAFT motion diagnostic must run on a CUDA device")
    core = load_original_core(core_path)
    expected_runtime = json.loads(os.environ.get("GEOMETRY_EVAL_LOCKED_RUNTIME_IDENTITIES_JSON", "{}"))
    if (
        not isinstance(expected_runtime, dict)
        or set(expected_runtime) != {"opencv", "torch", "torchvision"}
        or expected_runtime["opencv"] != opencv_identity()
        or expected_runtime["torch"] != current_locked_torch_load_trace()["runtime_identity"]
        or expected_runtime["torchvision"] != torchvision_trace["runtime_identity"]
    ):
        raise RuntimeError("OpenCV/TorchVision runtime differs from the evaluator-lock binding")
    checkpoint = torch.load(weights["relative_motion_raft_large_checkpoint"], map_location="cpu")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise RuntimeError("locked RAFT checkpoint does not contain a state dictionary")
    model = raft_large(weights=None, progress=False)
    result = model.load_state_dict({str(key).removeprefix("module."): value for key, value in checkpoint.items()}, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("locked RAFT checkpoint does not exactly match RAFT-Large")
    model = model.to(device).eval()
    transform = Raft_Large_Weights.C_T_SKHT_V2.transforms()

    records = []
    for entry in entries:
        if anchors is not None and entry["case_id"] not in anchors:
            continue
        with tempfile.TemporaryDirectory(prefix="raft-motion-frames-") as temporary:
            frames = frames_from_locked_decode(core, decoder, Path(entry["metric_video_path"]), Path(temporary))
            stats = core.motion_stats(model, transform, frames, device)
        if stats.get("n_pairs") != 120:
            raise RuntimeError("original RAFT motion core did not consume exactly 120 adjacent pairs")
        raw_total = float(stats["mean_flow_px"]) * 120.0
        ratio = 100.0 if anchors is None else 100.0 * raw_total / anchors[entry["case_id"]]
        records.append({
            "case_id": entry["case_id"],
            "unit_id": "total_motion",
            "value": ratio,
            "raw_total_motion": raw_total,
            "mean_flow_px": float(stats["mean_flow_px"]),
            "median_flow_px": float(stats["median_flow_px"]),
            "p90_flow_px": float(stats["p90_flow_px"]),
            "n_pairs": 120,
        })

    # RAFT model construction, transform construction, and inference can load
    # additional TorchVision modules after the initial guarded import.  Bind
    # the complete final module set, not the earlier setup-only snapshot.
    torchvision_trace = current_locked_torchvision_load_trace()

    publish_component(
        args.output,
        METRIC_ID,
        records,
        {
            "core_evaluator_sha256": sha256_file(core_path),
            "weights_enum": "Raft_Large_Weights.C_T_SKHT_V2",
            "long_side": 512,
            "decoder_integration": "locked FFmpeg -> lossless PNG; original resized_shape, OpenCV RGB/INTER_AREA preprocessing, and motion_stats",
            "opencv_runtime": expected_runtime["opencv"],
            "torchvision_runtime": torchvision_trace,
            "torch_runtime_load_trace": current_locked_torch_load_trace(),
            "ratio_anchor": "self=100 for Wan seed-0; frozen baseline eligibility anchors otherwise",
        },
    )


if __name__ == "__main__":
    main()
