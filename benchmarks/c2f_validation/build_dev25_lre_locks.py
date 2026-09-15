#!/usr/bin/env python3
"""Build the two-stage Independent LRE locks for paired C2F dev25 evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path


INPUT_SCHEMA = "geometry-selection-five-metric-hippasus-input-lock-v2"
ELIGIBILITY_SCHEMA = "geometry-selection-metric-eligibility-lock-v1"
SCOPE = "development_validation_disjoint_from_frozen_test_validation_debug"
METRIC = "long_range_reprojection_error"
EXPECTED_OVERLAP = {"test": 0, "validation": 0, "debug": 0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bound_json(path: Path, expected_sha: str, label: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(f"{label} SHA mismatch: expected {expected_sha}, found {actual_sha}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def atomic_readonly_json(path: Path, payload: dict) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def require_dev25_lock(payload: dict, method_id: str) -> None:
    entries = payload.get("entries")
    if (
        payload.get("schema") != INPUT_SCHEMA
        or payload.get("scope") != SCOPE
        or payload.get("reserved_overlap_counts") != EXPECTED_OVERLAP
        or payload.get("method_id") != method_id
        or not isinstance(entries, list)
        or len(entries) != 25
        or len({entry.get("case_id") for entry in entries}) != 25
    ):
        raise ValueError(f"unexpected dev25 input lock for {method_id}")


def build_baseline_alias(args: argparse.Namespace) -> None:
    source = read_bound_json(args.baseline_lock, args.baseline_lock_sha256, "baseline input lock")
    require_dev25_lock(source, "official_same_host_reference")
    alias = copy.deepcopy(source)
    alias["method_id"] = "wan_unguided_seed0"
    alias["lre_baseline_alias"] = {
        "source_method_id": "official_same_host_reference",
        "source_input_lock": {
            "path": str(args.baseline_lock.resolve()),
            "sha256": args.baseline_lock_sha256,
        },
        "reason": "The sealed LRE adapter reserves wan_unguided_seed0 for the method that establishes endpoint eligibility.",
    }
    output_sha = atomic_readonly_json(args.output, alias)
    print(json.dumps({"status": "ready", "path": str(args.output.resolve()), "sha256": output_sha}, indent=2))


def build_candidate_binding(args: argparse.Namespace) -> None:
    baseline_alias = read_bound_json(
        args.baseline_alias_lock, args.baseline_alias_lock_sha256, "baseline LRE alias lock"
    )
    if baseline_alias.get("method_id") != "wan_unguided_seed0":
        raise ValueError("baseline LRE alias does not carry the sealed adapter baseline identity")
    baseline_entries = baseline_alias.get("entries")
    if not isinstance(baseline_entries, list) or len(baseline_entries) != 25:
        raise ValueError("baseline LRE alias must contain all 25 cases")
    baseline_case_ids = {entry["case_id"] for entry in baseline_entries}

    component = read_bound_json(args.baseline_component, args.baseline_component_sha256, "baseline LRE component")
    records = component.get("records")
    if (
        component.get("schema") != "geometry-selection-metric-component-v1"
        or component.get("metric_id") != METRIC
        or not isinstance(records, list)
        or len(records) != 25
        or {record.get("case_id") for record in records} != baseline_case_ids
        or any(record.get("unit_id") != "first_last" for record in records)
    ):
        raise ValueError("baseline LRE component does not cover the frozen 25-case denominator")
    eligible = sorted(record["case_id"] for record in records if record.get("eligible") is True)
    if not eligible:
        raise ValueError("baseline produced no eligible Independent LRE cases")

    eligibility = {
        "schema": ELIGIBILITY_SCHEMA,
        "scope": SCOPE,
        "baseline_input_lock": {
            "path": str(args.baseline_alias_lock.resolve()),
            "sha256": args.baseline_alias_lock_sha256,
        },
        "baseline_component": {
            "path": str(args.baseline_component.resolve()),
            "sha256": args.baseline_component_sha256,
        },
        "locked_units": {
            METRIC: [{"case_id": case_id, "unit_id": "first_last"} for case_id in eligible],
        },
    }
    eligibility_sha = atomic_readonly_json(args.eligibility_output, eligibility)

    candidate = read_bound_json(args.candidate_lock, args.candidate_lock_sha256, "candidate input lock")
    require_dev25_lock(candidate, "c2f_k3_a0025")
    candidate_case_ids = {entry["case_id"] for entry in candidate["entries"]}
    if candidate_case_ids != baseline_case_ids:
        raise ValueError("candidate cases differ from the baseline LRE denominator")
    bound_candidate = copy.deepcopy(candidate)
    bound_candidate["baseline_metric_eligibility_lock"] = {
        "path": str(args.eligibility_output.resolve()),
        "sha256": eligibility_sha,
    }
    bound_candidate["lre_binding"] = {
        "baseline_eligible_case_count": len(eligible),
        "baseline_total_case_count": 25,
        "source_candidate_input_lock": {
            "path": str(args.candidate_lock.resolve()),
            "sha256": args.candidate_lock_sha256,
        },
    }
    candidate_sha = atomic_readonly_json(args.candidate_output, bound_candidate)
    print(
        json.dumps(
            {
                "status": "ready",
                "baseline_eligible_case_count": len(eligible),
                "eligibility_lock": {
                    "path": str(args.eligibility_output.resolve()),
                    "sha256": eligibility_sha,
                },
                "candidate_input_lock": {
                    "path": str(args.candidate_output.resolve()),
                    "sha256": candidate_sha,
                },
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline-alias")
    baseline.add_argument("--baseline-lock", type=Path, required=True)
    baseline.add_argument("--baseline-lock-sha256", required=True)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.set_defaults(handler=build_baseline_alias)

    candidate = subparsers.add_parser("candidate-binding")
    candidate.add_argument("--baseline-alias-lock", type=Path, required=True)
    candidate.add_argument("--baseline-alias-lock-sha256", required=True)
    candidate.add_argument("--baseline-component", type=Path, required=True)
    candidate.add_argument("--baseline-component-sha256", required=True)
    candidate.add_argument("--candidate-lock", type=Path, required=True)
    candidate.add_argument("--candidate-lock-sha256", required=True)
    candidate.add_argument("--eligibility-output", type=Path, required=True)
    candidate.add_argument("--candidate-output", type=Path, required=True)
    candidate.set_defaults(handler=build_candidate_binding)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
