#!/usr/bin/env python3
"""Publish the DPO-adapted train-dev input lock after base eligibility freezes.

The paired adapted entries are sealed together with the base mirror, but they
cannot become a metric input until the base run has fixed the Independent-LRE
denominator and the per-scene motion anchors.  This builder verifies that
closure and publishes a fresh read-only input lock below the evaluator locks
root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


INPUT_SCHEMA = "geometry-selection-five-metric-traindev-input-lock-v1"
BUNDLE_SCHEMA = "wan-lora-dpo-traindev-evaluator-bundle-v1"
ENTRIES_SCHEMA = "wan-lora-dpo-traindev-adapted-input-entries-v1"
ELIGIBILITY_SCHEMA = "geometry-selection-metric-traindev-eligibility-lock-v1"
BASE_METHOD_ID = "wan_lora_dpo_step64_base_traindev"
ADAPTED_METHOD_ID = "wan_lora_dpo_step64_adapted_traindev"
ADAPTED_METHOD_LABEL = "Full-Graph LoRA-DPO step-64 paired adapted train-dev"
METRICS = {
    "geco_fused",
    "met3r",
    "long_range_reprojection_error",
    "relative_total_motion_percent",
    "vbench_quality",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    return value


def read_readonly_json(path: Path, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
        raise ValueError(f"{label} must be a read-only regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload, {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)}


def verify_binding(value: Any, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} binding is malformed")
    expected = require_sha(value.get("sha256"), f"{label} SHA")
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label} path is malformed")
    payload, actual = read_readonly_json(Path(path_value), label)
    if actual["sha256"] != expected:
        raise ValueError(f"{label} SHA binding mismatch")
    return payload, actual


def require_readonly_below(root: Path, path_value: Any, expected_sha: Any, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label} path is malformed")
    expected = require_sha(expected_sha, f"{label} SHA")
    path = Path(path_value)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    resolved_root, resolved = root.resolve(strict=True), path.resolve(strict=True)
    if resolved_root not in resolved.parents:
        raise ValueError(f"{label} escapes the sealed mirror")
    for ancestor in (resolved_root, *resolved.parents):
        if ancestor == resolved_root or resolved_root in ancestor.parents:
            if ancestor.stat().st_mode & 0o222:
                raise ValueError(f"{label} has a writable mirror ancestor")
    if resolved.stat().st_mode & 0o222 or sha256_file(resolved) != expected:
        raise ValueError(f"{label} content or permissions changed")
    return resolved


def validate_locked_denominators(eligibility: dict[str, Any], case_ids: set[str]) -> None:
    locked = eligibility.get("locked_units")
    if not isinstance(locked, dict) or set(locked) != METRICS:
        raise ValueError("eligibility lock lacks the exact five metric denominators")
    for metric, units in locked.items():
        if not isinstance(units, list) or not units:
            raise ValueError(f"eligibility denominator is empty: {metric}")
        seen: set[tuple[str, str]] = set()
        for unit in units:
            if not isinstance(unit, dict):
                raise ValueError(f"eligibility unit is malformed: {metric}")
            key = (unit.get("case_id"), unit.get("unit_id"))
            if (
                not isinstance(key[0], str)
                or key[0] not in case_ids
                or not isinstance(key[1], str)
                or not key[1]
                or key in seen
            ):
                raise ValueError(f"eligibility unit identity is malformed: {metric}")
            seen.add(key)
    anchors = eligibility.get("baseline_motion_anchors")
    motion_cases = {unit["case_id"] for unit in locked["relative_total_motion_percent"]}
    if (
        not isinstance(anchors, dict)
        or set(anchors) != motion_cases
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or float(value) <= 1e-6
            for value in anchors.values()
        )
    ):
        raise ValueError("eligibility motion anchors differ from the locked motion denominator")


def safe_output(path: Path, derived_root: Path, inputs: list[Path]) -> None:
    if derived_root.is_symlink() or not derived_root.is_dir():
        raise ValueError("derived root must be an existing regular directory")
    derived = derived_root.resolve(strict=True)
    if tuple(derived.parts[-3:]) != ("outputs", "geometry-selection", "hippasus_evaluation"):
        raise ValueError("derived root must be the approved Hippasus evaluator root")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("output parent must be an existing regular directory")
    target = path.parent.resolve(strict=True) / path.name
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing existing output: {target}")
    if target == derived or derived not in target.parents or target.relative_to(derived).parts[0] != "locks":
        raise ValueError("adapted input lock must publish below the evaluator locks root")
    for source in inputs:
        resolved = source.resolve(strict=True)
        if target == resolved or target in resolved.parents or resolved in target.parents:
            raise ValueError("output must not overlap an input")


def publish(path: Path, payload: dict[str, Any]) -> str:
    data = canonical_bytes(payload) + b"\n"
    descriptor, name = tempfile.mkstemp(prefix=".dpo-traindev-adapted-lock-", suffix=".tmp", dir=path.parent)
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


def build_lock(
    *,
    bundle_ready_path: Path,
    base_input_lock_path: Path,
    eligibility_path: Path,
    expected_eligibility_sha256: str,
    output_path: Path,
    derived_root: Path,
) -> dict[str, Any]:
    ready, ready_binding = read_readonly_json(bundle_ready_path, "bundle READY")
    base, base_binding = read_readonly_json(base_input_lock_path, "base input lock")
    eligibility, eligibility_binding = read_readonly_json(eligibility_path, "base eligibility lock")
    expected_eligibility_sha256 = require_sha(
        expected_eligibility_sha256, "externally pinned base eligibility SHA"
    )
    if eligibility_binding["sha256"] != expected_eligibility_sha256:
        raise ValueError("base eligibility differs from its external completed-baseline SHA")
    if (
        ready.get("schema") != BUNDLE_SCHEMA
        or ready.get("status") != "READY"
        or ready.get("site") != "Hippasus"
        or ready.get("split") != "dev"
        or "formal_validation" in ready
        or ready.get("reserved_ids_disclosed") is not False
        or ready.get("case_count") != 100
        or ready.get("pair_count") != 100
        or ready.get("adapted_input_lock_status") != "blocked_until_base_eligibility_lock"
    ):
        raise ValueError("bundle READY identity/isolation mismatch")
    if ready.get("base_input_lock") != base_binding:
        raise ValueError("bundle READY does not bind this exact base input lock")
    adapted, adapted_binding = verify_binding(ready.get("adapted_input_entries"), "adapted input entries")
    isolation, isolation_binding = verify_binding(
        ready.get("traindev_reference_isolation_receipt"), "train-dev reference-isolation receipt"
    )
    protocol, protocol_binding = verify_binding(ready.get("metric_protocol"), "metric protocol")
    schedule, schedule_binding = verify_binding(ready.get("metric_schedule"), "metric schedule")
    if (
        base.get("schema") != INPUT_SCHEMA
        or base.get("scope") != "train_dev_evaluation"
        or base.get("evaluation_site") != "Hippasus"
        or base.get("method_id") != BASE_METHOD_ID
        or base.get("dataset_split") != "dev"
        or "formal_validation" in base
        or base.get("reserved_ids_disclosed") is not False
        or base.get("metric_protocol_sha256") != protocol_binding["sha256"]
        or base.get("metric_schedule_sha256") != schedule_binding["sha256"]
        or base.get("traindev_reference_isolation_receipt_sha256") != isolation_binding["sha256"]
        or base.get("bundle_reference_path") != ready.get("bundle_reference_path")
    ):
        raise ValueError("base input lock identity/isolation mismatch")
    if (
        isolation.get("schema") != "wan-lora-dpo-traindev-reference-isolation-receipt-v1"
        or isolation.get("status") != "verified_zero_overlap"
        or isolation.get("overlap_counts")
        != {"case_ids": 0, "scene_ids": 0, "transform_sources": 0}
        or isolation.get("ids_disclosed") is not False
    ):
        raise ValueError("train-dev reference-isolation receipt is not a zero-overlap proof")
    base_entries = base.get("entries")
    if not isinstance(base_entries, list) or len(base_entries) != 100:
        raise ValueError("base input lock must contain exactly 100 cases")
    if any(
        not isinstance(item, dict) or item.get("split_order") != index
        for index, item in enumerate(base_entries)
    ):
        raise ValueError("base input lock does not preserve exact train-dev case order")
    base_by_case = {
        item.get("case_id"): item for item in base_entries if isinstance(item, dict)
    }
    if len(base_by_case) != 100 or set(base_by_case) != set(eligibility.get("case_ids", [])):
        raise ValueError("base cases differ from the frozen eligibility denominator")
    if (
        eligibility.get("schema") != ELIGIBILITY_SCHEMA
        or eligibility.get("site") != "Hippasus"
        or eligibility.get("scope") != "train_dev_evaluation"
        or eligibility.get("baseline_input_lock") != base_binding
    ):
        raise ValueError("eligibility lock does not bind this exact train-dev base input")
    validate_locked_denominators(eligibility, set(base_by_case))
    if (
        adapted.get("schema") != ENTRIES_SCHEMA
        or adapted.get("site") != "Hippasus"
        or adapted.get("scope") != "train_dev_evaluation"
        or adapted.get("split") != "dev"
        or "formal_validation" in adapted
        or adapted.get("reserved_ids_disclosed") is not False
        or adapted.get("method") != ADAPTED_METHOD_LABEL
        or adapted.get("method_id") != ADAPTED_METHOD_ID
        or adapted.get("case_count") != 100
        or adapted.get("source_manifest_sha256") != base.get("source_manifest_sha256")
        or adapted.get("traindev_reference_isolation_receipt_sha256")
        != isolation_binding["sha256"]
        or adapted.get("source_generation_receipt_sha256") != base.get("source_generation_receipt_sha256")
        or adapted.get("metric_protocol_sha256") != protocol_binding["sha256"]
        or adapted.get("metric_schedule_sha256") != schedule_binding["sha256"]
        or adapted.get("mirror_root") != base.get("mirror_root")
        or adapted.get("mirror_index_sha256") != base.get("mirror_index_sha256")
        or adapted.get("bundle_reference_path") != base.get("bundle_reference_path")
    ):
        raise ValueError("adapted entry set differs from the sealed paired bundle")
    if (
        protocol.get("schema") != "geometry-selection-five-metric-traindev-protocol-v1"
        or protocol.get("status") != "frozen"
        or protocol.get("scope") != "train_dev_evaluation"
        or protocol.get("dataset", {}).get("split") != "dev"
        or protocol.get("dataset", {}).get("case_count") != 100
        or "formal_validation" in protocol.get("dataset", {})
        or protocol.get("dataset", {}).get("reserved_ids_disclosed") is not False
        or schedule.get("schema") != "geometry-selection-metric-traindev-schedule-v1"
        or schedule.get("status") != "frozen"
        or schedule.get("metric_protocol_sha256") != protocol_binding["sha256"]
    ):
        raise ValueError("derived train-dev protocol/schedule identity mismatch")
    budget = protocol.get("candidate_budget_policy", {}).get(ADAPTED_METHOD_ID)
    if not isinstance(budget, dict) or budget.get("candidate_count") != 1 or budget.get("selected_output_count") != 1:
        raise ValueError("adapted candidate budget is not the frozen paired one-output budget")
    adapted_entries = adapted.get("entries")
    if not isinstance(adapted_entries, list) or len(adapted_entries) != 100:
        raise ValueError("adapted entry set must contain exactly 100 cases")
    mirror_root = Path(base["mirror_root"])
    if mirror_root.is_symlink() or not mirror_root.is_dir() or mirror_root.stat().st_mode & 0o222:
        raise ValueError("paired evaluator mirror is not sealed")
    seen: set[str] = set()
    locked_entries: list[dict[str, Any]] = []
    for order, item in enumerate(adapted_entries):
        if not isinstance(item, dict):
            raise ValueError("adapted entry is malformed")
        case_id = item.get("case_id")
        base_item = base_by_case.get(case_id)
        if (
            not isinstance(case_id, str)
            or case_id in seen
            or not isinstance(base_item, dict)
            or item.get("split_order") != order
            or item.get("split_order") != base_item.get("split_order")
            or item.get("seed") != 0
            or item.get("lora_mode") != "adapted"
            or item.get("lora_step") != 64
            or item.get("pair_identity_sha256") != base_item.get("pair_identity_sha256")
            or item.get("conditioning_image_sha256") != base_item.get("conditioning_image_sha256")
            or item.get("prompt") != base_item.get("prompt")
            or item.get("video_probe") != base_item.get("video_probe")
        ):
            raise ValueError("adapted/base pair identity mismatch")
        seen.add(case_id)
        require_readonly_below(mirror_root, item.get("metric_video_path"), item.get("video_sha256"), f"{case_id}.video")
        require_readonly_below(mirror_root, item.get("metric_metadata_path"), item.get("metadata_sha256"), f"{case_id}.metadata")
        require_readonly_below(mirror_root, item.get("metric_complete_path"), item.get("complete_sha256"), f"{case_id}.COMPLETE")
        require_readonly_below(
            mirror_root,
            item.get("metric_generation_lock_path"),
            item.get("generation_lock_sha256"),
            f"{case_id}.generation_lock",
        )
        locked_entries.append(item)
    if seen != set(base_by_case):
        raise ValueError("adapted entry cases do not exactly match the base denominator")
    payload = {
        "schema": INPUT_SCHEMA,
        "scope": "train_dev_evaluation",
        "evaluation_site": "Hippasus",
        "method": ADAPTED_METHOD_LABEL,
        "method_id": ADAPTED_METHOD_ID,
        "candidate_budget": budget,
        "dataset_split": "dev",
        "reserved_ids_disclosed": False,
        "source_manifest_sha256": base["source_manifest_sha256"],
        "traindev_reference_isolation_receipt_sha256": isolation_binding["sha256"],
        "source_generation_receipt_sha256": base["source_generation_receipt_sha256"],
        "metric_protocol_sha256": protocol_binding["sha256"],
        "metric_schedule_sha256": schedule_binding["sha256"],
        "video_contract": base["video_contract"],
        "mirror_root": str(mirror_root.resolve(strict=True)),
        "mirror_index_sha256": base["mirror_index_sha256"],
        "bundle_reference_path": base["bundle_reference_path"],
        "source_bundle_ready": ready_binding,
        "source_adapted_entries": adapted_binding,
        "baseline_input_lock_sha256": base_binding["sha256"],
        "baseline_metric_eligibility_lock": eligibility_binding,
        "expected_baseline_eligibility_sha256": expected_eligibility_sha256,
        "selection_policy": "paired adapted mode; one seed-0 step-64 LoRA-DPO video per frozen train-dev scene",
        "entries": locked_entries,
    }
    safe_output(
        output_path,
        derived_root,
        [bundle_ready_path, base_input_lock_path, eligibility_path, Path(adapted_binding["path"])],
    )
    output_sha = publish(output_path, payload)
    return {"case_count": 100, "input_lock_sha256": output_sha, "output": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-ready", type=Path, required=True)
    parser.add_argument("--base-input-lock", type=Path, required=True)
    parser.add_argument("--baseline-eligibility-lock", type=Path, required=True)
    parser.add_argument("--expected-baseline-eligibility-sha256", required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_lock(
        bundle_ready_path=args.bundle_ready,
        base_input_lock_path=args.base_input_lock,
        eligibility_path=args.baseline_eligibility_lock,
        expected_eligibility_sha256=args.expected_baseline_eligibility_sha256,
        output_path=args.output,
        derived_root=args.derived_root,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
