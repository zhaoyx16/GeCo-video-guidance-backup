#!/usr/bin/env python3
"""Shared guarded-runtime utilities for the five frozen video metric adapters.

Adapters are deliberately small integration layers: they bind a fixed input lock,
schedule, decoder, core implementation and local weight manifest before calling
the pinned metric code.  They never choose samples, fetch weights, or publish a
partial denominator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


FORMAL_INPUT_SCHEMA = "geometry-selection-five-metric-hippasus-input-lock-v2"
TRAINDEV_INPUT_SCHEMA = "geometry-selection-five-metric-traindev-input-lock-v1"
TRAINDEV_METHOD_IDS = {
    "wan_lora_dpo_step64_base_traindev",
    "wan_lora_dpo_step64_adapted_traindev",
}
TRAINDEV_METHOD_LABELS = {
    "wan_lora_dpo_step64_base_traindev": "Wan LoRA-DPO step-64 paired base train-dev",
    "wan_lora_dpo_step64_adapted_traindev": "Full-Graph LoRA-DPO step-64 paired adapted train-dev",
}
COMPONENT_SCHEMA = "geometry-selection-metric-component-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    return value


def require_readonly_regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise ValueError(f"{label} is writable: {path}")
    return path.resolve(strict=True)


def require_guarded_runtime(adapter_path: Path) -> None:
    expected = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "GEOMETRY_EVAL_NETWORK": "disabled",
    }
    if any(os.environ.get(key) != value for key, value in expected.items()):
        raise RuntimeError("adapter was not launched by the locked offline worker guard")
    expected_path = os.environ.get("GEOMETRY_EVAL_LOCKED_ADAPTER_ENTRYPOINT_PATH")
    expected_sha = os.environ.get("GEOMETRY_EVAL_LOCKED_ADAPTER_ENTRYPOINT_SHA256")
    if expected_path != str(adapter_path.resolve(strict=True)) or expected_sha != sha256_file(adapter_path):
        raise RuntimeError("adapter path/SHA differs from the guarded evaluator lock")
    common_path = Path(__file__).resolve(strict=True)
    if os.environ.get("GEOMETRY_EVAL_LOCKED_ADAPTER_COMMON_PATH") != str(common_path) or os.environ.get("GEOMETRY_EVAL_LOCKED_ADAPTER_COMMON_SHA256") != sha256_file(common_path):
        raise RuntimeError("shared adapter runtime path/SHA differs from the guarded evaluator lock")


def locked_path(value: Path, env_path: str, env_sha: str, label: str) -> Path:
    expected_path = os.environ.get(env_path)
    expected_sha = os.environ.get(env_sha)
    resolved = require_readonly_regular(value, label)
    if expected_path != str(resolved) or expected_sha != sha256_file(resolved):
        raise RuntimeError(f"{label} differs from its guarded evaluator-lock binding")
    return resolved


def verify_core_and_dependencies(
    *,
    adapter_path: Path,
    core_evaluator: Path,
    weight_manifest: Path,
    schedule: Path,
    decoder: Path,
    output: Path,
) -> tuple[Path, Path, Path, Path]:
    require_guarded_runtime(adapter_path)
    core = locked_path(core_evaluator, "GEOMETRY_EVAL_LOCKED_CORE_EVALUATOR_PATH", "GEOMETRY_EVAL_LOCKED_CORE_EVALUATOR_SHA256", "core evaluator")
    weights = locked_path(weight_manifest, "GEOMETRY_EVAL_LOCKED_WEIGHT_MANIFEST_PATH", "GEOMETRY_EVAL_LOCKED_WEIGHT_MANIFEST_SHA256", "weight manifest")
    schedule_path = locked_path(schedule, "GEOMETRY_EVAL_LOCKED_SCHEDULE_PATH", "GEOMETRY_EVAL_LOCKED_SCHEDULE_SHA256", "schedule")
    decoder_path = locked_path(decoder, "GEOMETRY_EVAL_LOCKED_DECODER_BINARY", "GEOMETRY_EVAL_LOCKED_DECODER_SHA256", "decoder")
    expected_output = os.environ.get("GEOMETRY_EVAL_METRIC_OUTPUT_PATH")
    if expected_output != str(output.resolve(strict=False)):
        raise RuntimeError("adapter output does not match the guarded worker output path")
    return core, weights, schedule_path, decoder_path


def load_input_lock(input_lock: Path, schedule: Path) -> list[dict[str, Any]]:
    lock_path = locked_path(input_lock, "GEOMETRY_EVAL_LOCKED_INPUT_LOCK_PATH", "GEOMETRY_EVAL_LOCKED_INPUT_LOCK_SHA256", "input lock")
    payload = read_json(lock_path, "input lock")
    schedule_path = require_readonly_regular(schedule, "schedule")
    schema = payload.get("schema")
    train_dev = schema == TRAINDEV_INPUT_SCHEMA
    serialized_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if schema not in {FORMAL_INPUT_SCHEMA, TRAINDEV_INPUT_SCHEMA} or payload.get("evaluation_site") != "Hippasus":
        raise ValueError("unexpected Hippasus input-lock identity")
    if train_dev and (
        payload.get("scope") != "train_dev_evaluation"
        or payload.get("dataset_split") != "dev"
        or "formal_validation" in payload
        or payload.get("reserved_ids_disclosed") is not False
        or payload.get("method_id") not in TRAINDEV_METHOD_IDS
        or payload.get("method")
        != TRAINDEV_METHOD_LABELS.get(payload.get("method_id"))
        or any(
            token in serialized_payload
            for token in ("formal_validation", "validation_only", "wan_unguided_seed0")
        )
        or not isinstance(payload.get("traindev_reference_isolation_receipt_sha256"), str)
        or len(payload["traindev_reference_isolation_receipt_sha256"]) != 64
    ):
        raise ValueError("unexpected train-dev input-lock identity")
    if train_dev:
        evaluator_path = locked_path(
            Path(os.environ.get("GEOMETRY_EVAL_LOCKED_EVALUATOR_LOCK_PATH", "")),
            "GEOMETRY_EVAL_LOCKED_EVALUATOR_LOCK_PATH",
            "GEOMETRY_EVAL_LOCKED_EVALUATOR_LOCK_SHA256",
            "evaluator lock",
        )
        preflight_path = locked_path(
            Path(os.environ.get("GEOMETRY_EVAL_INPUT_PREFLIGHT_RECEIPT_PATH", "")),
            "GEOMETRY_EVAL_INPUT_PREFLIGHT_RECEIPT_PATH",
            "GEOMETRY_EVAL_INPUT_PREFLIGHT_RECEIPT_SHA256",
            "input preflight receipt",
        )
        evaluator = read_json(evaluator_path, "evaluator lock")
        preflight = read_json(preflight_path, "input preflight receipt")
        input_binding = {"path": str(lock_path), "sha256": sha256_file(lock_path)}
        evaluator_binding = {
            "path": str(evaluator_path),
            "sha256": sha256_file(evaluator_path),
        }
        if (
            evaluator.get("schema") != "geometry-selection-evaluator-lock-v2"
            or evaluator.get("scope") != "train_dev"
            or evaluator.get("input_manifest") != input_binding
            or preflight.get("schema")
            != "geometry-selection-metric-traindev-input-preflight-receipt-v1"
            or preflight.get("input_lock") != input_binding
            or preflight.get("evaluator_lock") != evaluator_binding
            or not isinstance(evaluator.get("traindev_provenance"), dict)
            or evaluator["traindev_provenance"].get("schema")
            != "geometry-selection-traindev-provenance-verification-v1"
            or evaluator["traindev_provenance"].get("status") != "verified"
            or evaluator["traindev_provenance"].get("case_count") != 100
            or evaluator["traindev_provenance"].get("record_count") != 800
            or evaluator["traindev_provenance"].get(
                "reference_isolation_receipt", {}
            ).get("sha256")
            != payload.get("traindev_reference_isolation_receipt_sha256")
            or evaluator["traindev_provenance"].get("generation_receipt", {}).get(
                "sha256"
            )
            != payload.get("source_generation_receipt_sha256")
            or (
                payload.get("method_id")
                == "wan_lora_dpo_step64_adapted_traindev"
                and evaluator["traindev_provenance"].get(
                    "expected_baseline_eligibility_sha256"
                )
                != payload.get("expected_baseline_eligibility_sha256")
            )
            or preflight.get("traindev_provenance")
            != evaluator.get("traindev_provenance")
        ):
            raise ValueError("train-dev adapter lacks a verified provenance preflight chain")
    if payload.get("metric_schedule_sha256") != sha256_file(schedule_path):
        raise ValueError("input lock does not bind the locked metric schedule")
    video_contract = payload.get("video_contract")
    if video_contract != {"frames": 121, "width": 1280, "height": 704, "fps": 24}:
        raise ValueError("input lock video contract differs from the frozen 121-frame Wan contract")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries or (train_dev and len(entries) != 100):
        raise ValueError("input lock needs nonempty entries")
    normalized: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    for order, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValueError("input lock entry must be an object")
        case_id = item.get("case_id")
        video_value = item.get("metric_video_path")
        video_sha = require_sha(item.get("video_sha256"), "input entry video SHA")
        if not isinstance(case_id, str) or not case_id or case_id in seen_cases or not isinstance(video_value, str) or not video_value:
            raise ValueError("input lock case/video identity is malformed")
        if train_dev and item.get("split_order") != order:
            raise ValueError("train-dev input lock case order changed")
        seen_cases.add(case_id)
        video = require_readonly_regular(Path(video_value), f"input video for {case_id}")
        if sha256_file(video) != video_sha:
            raise ValueError(f"input video SHA mismatch: {case_id}")
        if item.get("video_probe") != {"frames": 121, "width": 1280, "height": 704, "fps": 24}:
            raise ValueError(f"input lock video probe mismatch: {case_id}")
        normalized.append({**item, "metric_video_path": str(video)})
    return normalized if train_dev else sorted(normalized, key=lambda item: item["case_id"])


def load_weight_manifest(weight_manifest: Path, expected_roles: set[str]) -> dict[str, Path]:
    payload = read_json(require_readonly_regular(weight_manifest, "weight manifest"), "weight manifest")
    if payload.get("schema") != "geometry-selection-weight-content-manifest-v1" or payload.get("site") != "Hippasus" or payload.get("offline_ready") is not True or payload.get("load_closure_complete") is not True:
        raise ValueError("weight manifest is not an offline-ready Hippasus manifest")
    files = payload.get("files")
    if not isinstance(files, list):
        raise ValueError("weight manifest files are malformed")
    result: dict[str, Path] = {}
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("role"), str):
            raise ValueError("weight manifest role is malformed")
        role = item["role"]
        if role in result:
            raise ValueError(f"duplicate weight role: {role}")
        path_value = item.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"weight path missing for {role}")
        expected_sha = require_sha(item.get("sha256"), f"weight SHA for {role}")
        path = require_readonly_regular(Path(path_value), f"weight {role}")
        if sha256_file(path) != expected_sha:
            raise ValueError(f"weight SHA mismatch: {role}")
        result[role] = path
    if set(result) != expected_roles:
        raise ValueError(f"weight roles differ from frozen contract: expected {sorted(expected_roles)}, found {sorted(result)}")
    return result


def decode_all_frames(decoder: Path, video: Path, workspace: Path) -> list[Path]:
    output_dir = workspace / "frames"
    output_dir.mkdir(mode=0o700)
    output_pattern = output_dir / "%06d.png"
    subprocess.run(
        [str(decoder), "-v", "error", "-i", str(video), "-map", "0:v:0", "-vsync", "0", "-start_number", "0", str(output_pattern)],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    frames = sorted(output_dir.glob("*.png"))
    if len(frames) != 121 or [frame.name for frame in frames] != [f"{index:06d}.png" for index in range(121)]:
        raise RuntimeError(f"locked decoder did not yield the required 121 indexed frames: {video}")
    return frames


def require_schedule(schedule: Path, metric_id: str) -> dict[str, Any]:
    payload = read_json(require_readonly_regular(schedule, "schedule"), "schedule")
    if payload.get("schema") not in {
        "geometry-selection-metric-schedule-v2",
        "geometry-selection-metric-traindev-schedule-v1",
    } or payload.get("status") != "frozen" or not isinstance(payload.get(metric_id), dict):
        raise ValueError(f"schedule does not define {metric_id}")
    return payload


def finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return float(value)


def publish_component(output: Path, metric_id: str, records: Iterable[dict[str, Any]], details: dict[str, Any]) -> None:
    if output.exists() or output.is_symlink() or output.parent.is_symlink() or not output.parent.is_dir():
        raise FileExistsError(f"metric output must be a fresh path below an existing directory: {output}")
    normalized = list(records)
    identities: set[tuple[str, str]] = set()
    for record in normalized:
        if not isinstance(record, dict) or not isinstance(record.get("case_id"), str) or not record["case_id"] or not isinstance(record.get("unit_id"), str) or not record["unit_id"]:
            raise ValueError("metric record lacks a case/unit identity")
        identity = (record["case_id"], record["unit_id"])
        if identity in identities:
            raise ValueError(f"duplicate metric record: {identity}")
        identities.add(identity)
        record["value"] = finite(record.get("value"), f"metric value {identity}")
    payload = {"schema": COMPONENT_SCHEMA, "metric_id": metric_id, "records": sorted(normalized, key=lambda item: (item["case_id"], item["unit_id"])), "details": details}
    descriptor, temporary_name = tempfile.mkstemp(prefix=".metric-component-", suffix=".tmp", dir=str(output.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(payload) + b"\n")
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        os.link(temporary, output)
        directory = os.open(output.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
