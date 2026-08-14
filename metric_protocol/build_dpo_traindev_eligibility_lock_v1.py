#!/usr/bin/env python3
"""Train-dev baseline denominator builder with a distinct non-formal identity.

This is the train-dev counterpart of ``build_metric_eligibility_lock.py``.
It intentionally reuses the reviewed unit/runtime verification implementation,
but substitutes only the input schema/method identity gate and output schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import statistics
import tempfile
import types
from pathlib import Path
from typing import Any


def load_exact_sibling(path: Path, expected_sha256: str, module_name: str) -> types.ModuleType:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino):
            raise RuntimeError(f"{module_name} helper is not a stable regular file")
        digest, chunks = hashlib.sha256(), []
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
        named_after = os.stat(path, follow_symlinks=False)
        if digest.hexdigest() != expected_sha256 or (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns
        ) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ) or (after.st_dev, after.st_ino) != (named_after.st_dev, named_after.st_ino):
            raise RuntimeError(f"{module_name} helper differs from its reviewed SHA")
    finally:
        os.close(descriptor)
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    exec(compile(b"".join(chunks), str(path), "exec"), module.__dict__)
    return module


BASE = Path(__file__).resolve(strict=True).with_name("build_metric_eligibility_lock.py")
IMPL = load_exact_sibling(
    BASE,
    "21ffffc191601da651364c3aa738d8023072ae43738954e61cfc5229e5eb64e1",
    "locked_metric_eligibility_impl",
)

INPUT_SCHEMA = "geometry-selection-five-metric-traindev-input-lock-v1"
ELIGIBILITY_SCHEMA = "geometry-selection-metric-traindev-eligibility-lock-v1"
BASE_METHOD_ID = "wan_lora_dpo_step64_base_traindev"
BASE_METHOD_LABEL = "Wan LoRA-DPO step-64 paired base train-dev"
METRIC_NAMES = IMPL.METRIC_NAMES
VBENCH_UNITS = IMPL.VBENCH_UNITS


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def publish(path: Path, payload: dict[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing output: {path}")
    data = canonical_bytes(payload) + b"\n"
    descriptor, name = tempfile.mkstemp(prefix=".dpo-traindev-eligibility-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def base_lre_eligible(record: dict[str, Any]) -> bool:
    valid = (
        IMPL.finite(record.get("value"))
        and IMPL.finite(record.get("forward_valid_fraction"))
        and IMPL.finite(record.get("backward_valid_fraction"))
        and float(record["forward_valid_fraction"]) >= 0.2
        and float(record["backward_valid_fraction"]) >= 0.2
    )
    if record.get("eligible") is not valid:
        raise ValueError("base LRE eligibility flag differs from the frozen numeric threshold")
    return valid


def base_motion_anchor(record: dict[str, Any], key: tuple[str, str]) -> float | None:
    value_valid = IMPL.finite(record.get("value"))
    raw_valid = IMPL.finite(record.get("raw_total_motion"))
    if not value_valid or not raw_valid or float(record["raw_total_motion"]) <= 1e-6:
        return None
    value = float(record["value"])
    if abs(value - 100.0) > 1e-8:
        raise ValueError(f"train-dev base motion ratio must be exactly 100: {key}")
    return float(record["raw_total_motion"])


def build(
    *,
    input_path: Path,
    evaluator_path: Path,
    units_path: Path,
    derived_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    input_lock = IMPL.read_json(input_path)
    evaluator_lock = IMPL.read_json(evaluator_path)
    raw_units = IMPL.read_json(units_path)
    input_binding = IMPL.binding(input_path)
    evaluator_binding = IMPL.binding(evaluator_path)
    raw_binding = IMPL.binding(units_path)
    if (
        input_lock.get("schema") != INPUT_SCHEMA
        or input_lock.get("scope") != "train_dev_evaluation"
        or input_lock.get("evaluation_site") != "Hippasus"
        or input_lock.get("method") != BASE_METHOD_LABEL
        or input_lock.get("method_id") != BASE_METHOD_ID
        or input_lock.get("dataset_split") != "dev"
        or "formal_validation" in input_lock
        or input_lock.get("reserved_ids_disclosed") is not False
    ):
        raise ValueError("train-dev base input-lock identity/isolation mismatch")
    if (
        evaluator_lock.get("schema") != "geometry-selection-evaluator-lock-v2"
        or evaluator_lock.get("scope") != "train_dev"
        or evaluator_lock.get("site") != "Hippasus"
        or evaluator_lock.get("input_manifest") != input_binding
    ):
        raise ValueError("train-dev evaluator lock does not bind the exact base input")
    shared_identity = IMPL.require_sha(
        evaluator_lock.get("shared_evaluator_identity_sha256"), "shared evaluator identity"
    )
    if raw_units.get("schema") != "geometry-selection-metric-units-v1" or raw_units.get("method") != BASE_METHOD_LABEL:
        raise ValueError("train-dev base metric-unit identity mismatch")
    if raw_units.get("input_manifest_sha256") != input_binding["sha256"] or raw_units.get("evaluator_lock_sha256") != evaluator_binding["sha256"]:
        raise ValueError("train-dev base metric units bind the wrong input/evaluator")
    runtime_preflights = IMPL.verify_runtime_attestations(
        raw_units,
        evaluator_lock,
        evaluator_binding,
        input_binding,
        train_dev=True,
    )
    units = raw_units.get("units")
    entries = input_lock.get("entries")
    if not isinstance(units, dict) or set(units) != METRIC_NAMES or not isinstance(entries, list) or len(entries) != 100:
        raise ValueError("train-dev base requires exactly 100 cases and five metric unit sets")
    case_ids = [entry.get("case_id") for entry in entries if isinstance(entry, dict)]
    if len(case_ids) != 100 or len(set(case_ids)) != 100 or not all(isinstance(case, str) and case for case in case_ids):
        raise ValueError("train-dev base case denominator is malformed")
    expected_preflight = {
        (
            entry["case_id"],
            entry["video_sha256"],
            entry["metadata_sha256"],
            entry["complete_sha256"],
            entry["generation_lock_sha256"],
        )
        for entry in entries
    }
    for metric_name, evidence in runtime_preflights.items():
        actual = {
            (
                entry.get("case_id"),
                entry.get("video_sha256"),
                entry.get("metadata_sha256"),
                entry.get("complete_sha256"),
                entry.get("generation_lock_sha256"),
            )
            for entry in evidence["input_preflight"].get("case_artifacts", [])
            if isinstance(entry, dict)
        }
        if actual != expected_preflight:
            raise ValueError(f"{metric_name} preflight does not cover the exact base artifacts")
    cases = set(case_ids)
    expected = {
        "geco_fused": {(case, f"window_{index}") for case in cases for index in range(2)},
        "met3r": {(case, f"one_second_{index}") for case in cases for index in range(5)},
        "long_range_reprojection_error": {(case, "first_last") for case in cases},
        "relative_total_motion_percent": {(case, "total_motion") for case in cases},
        "vbench_quality": {(case, unit) for case in cases for unit in VBENCH_UNITS},
    }
    mapped = {
        metric: IMPL.exact_record_map(units[metric], expected[metric], metric)
        for metric in METRIC_NAMES
    }
    locked: dict[str, list[dict[str, str]]] = {metric: [] for metric in METRIC_NAMES}
    exclusions: dict[str, list[dict[str, str]]] = {metric: [] for metric in METRIC_NAMES}
    anchors: dict[str, float] = {}
    for key, record in mapped["geco_fused"].items():
        if not IMPL.finite(record.get("value")):
            raise ValueError(f"base GeCo must be finite: {key}")
        locked["geco_fused"].append({"case_id": key[0], "unit_id": key[1]})
    for key, record in mapped["met3r"].items():
        if not IMPL.finite(record.get("value")):
            raise ValueError(f"base MEt3R must be finite: {key}")
        locked["met3r"].append({"case_id": key[0], "unit_id": key[1]})
    for key, record in mapped["long_range_reprojection_error"].items():
        valid = base_lre_eligible(record)
        target = locked if valid else exclusions
        target["long_range_reprojection_error"].append(
            {"case_id": key[0], "unit_id": key[1], **({} if valid else {"reason": "base_lre_ineligible"})}
        )
    for key, record in mapped["relative_total_motion_percent"].items():
        raw = base_motion_anchor(record, key)
        valid = raw is not None
        target = locked if valid else exclusions
        target["relative_total_motion_percent"].append(
            {"case_id": key[0], "unit_id": key[1], **({} if valid else {"reason": "base_motion_anchor_nonpositive"})}
        )
        if valid:
            anchors[key[0]] = raw
    for key, record in mapped["vbench_quality"].items():
        if not IMPL.finite(record.get("value")):
            raise ValueError(f"base VBench must be finite: {key}")
        locked["vbench_quality"].append({"case_id": key[0], "unit_id": key[1]})
    for values in (*locked.values(), *exclusions.values()):
        values.sort(key=lambda item: (item["case_id"], item["unit_id"]))
    payload = {
        "schema": ELIGIBILITY_SCHEMA,
        "site": "Hippasus",
        "scope": "train_dev_evaluation",
        "reserved_ids_disclosed": False,
        "baseline_method_id": BASE_METHOD_ID,
        "baseline_input_lock": input_binding,
        "baseline_evaluator_lock": evaluator_binding,
        "shared_evaluator_identity_sha256": shared_identity,
        "baseline_units": raw_binding,
        "baseline_metric_worker_receipts": raw_units["metric_worker_receipts"],
        "baseline_metric_components": raw_units["metric_components"],
        "baseline_metric_runtime_evidence": {
            name: {
                "source_snapshot_rehash": evidence["source_snapshot_rehash"],
                "independent_lre_runtime_load_trace": evidence["independent_lre_runtime_load_trace"],
            }
            for name, evidence in sorted(runtime_preflights.items())
        },
        "case_ids": sorted(cases),
        "locked_units": locked,
        "baseline_exclusions": exclusions,
        "baseline_motion_anchors": dict(sorted(anchors.items())),
    }
    IMPL.safe_output(output_path, derived_root, [input_path, evaluator_path, units_path])
    return {"eligibility_lock_sha256": publish(output_path, payload), "output": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-input-lock", type=Path, required=True)
    parser.add_argument("--baseline-evaluator-lock", type=Path, required=True)
    parser.add_argument("--baseline-units", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(input_path=args.baseline_input_lock, evaluator_path=args.baseline_evaluator_lock, units_path=args.baseline_units, derived_root=args.derived_root, output_path=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
