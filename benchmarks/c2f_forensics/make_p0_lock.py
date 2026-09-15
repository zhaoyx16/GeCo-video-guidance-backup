#!/usr/bin/env python3
"""Build the immutable P0 outcome labels and balanced forensic subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path


EXPECTED = {
    "report": "c89fbcf672be1c392e8ddd35e0d0c9ac674fa5aecdb823d9d3db03335e3e8aaf",
    "manifest": "332b4183431868969ec7f9dfcf39207c393eb4efb7516fcec01ccea6cd5a4d67",
    "split": "d1eed2400547d755e265149185515c3e6fdf981f63635bbbfa152d33fa53205c",
    "config": "6f34f574306233b927ac13a69a23c536ccf86089c92dd0d97747f993cc775aca",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "report": args.report.resolve(),
        "manifest": args.manifest.resolve(),
        "split": args.split.resolve(),
        "config": args.config.resolve(),
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if actual != EXPECTED:
        raise RuntimeError(f"frozen input digest mismatch: {actual} != {EXPECTED}")

    report = read_json(paths["report"])
    manifest = read_json(paths["manifest"])
    if report.get("scope") != "development_only_not_final_validation":
        raise RuntimeError("paired report is not marked development-only")
    expected_overlap = {"test": 0, "validation": 0, "debug": 0}
    if report.get("reserved_overlap_counts") != expected_overlap:
        raise RuntimeError("paired report reserved-split audit failed")
    if manifest.get("_meta", {}).get("reserved_overlap_counts") != expected_overlap:
        raise RuntimeError("manifest reserved-split audit failed")

    manifest_case_ids = {case_id for case_id in manifest if not case_id.startswith("_")}
    records = report.get("per_case", [])
    report_case_ids = {record["case_id"] for record in records}
    if len(records) != 25 or manifest_case_ids != report_case_ids:
        raise RuntimeError("Dev25 report and manifest case sets differ")

    locked_cases = []
    for rank, record in enumerate(sorted(records, key=lambda item: item["met3r_one_second_delta"]), start=1):
        delta = float(record["met3r_one_second_delta"])
        locked_cases.append(
            {
                "rank_best_to_worst": rank,
                "case_id": record["case_id"],
                "motion_stratum": record["motion_stratum"],
                "met3r_one_second_baseline": float(record["met3r_one_second_baseline"]),
                "met3r_one_second_c2f": float(record["met3r_one_second_candidate"]),
                "met3r_one_second_delta": delta,
                "label": "winner" if delta < 0 else "loser",
                "near_tie": abs(delta) < 0.001,
            }
        )

    counterfactual_cases = []
    strata = sorted({record["motion_stratum"] for record in locked_cases})
    for stratum in strata:
        stratum_records = [record for record in locked_cases if record["motion_stratum"] == stratum]
        counterfactual_cases.extend(
            [
                {**min(stratum_records, key=lambda item: item["met3r_one_second_delta"]), "selection": "best_in_stratum"},
                {**max(stratum_records, key=lambda item: item["met3r_one_second_delta"]), "selection": "worst_in_stratum"},
            ]
        )

    winner_count = sum(record["label"] == "winner" for record in locked_cases)
    loser_count = len(locked_cases) - winner_count
    if (winner_count, loser_count) != (12, 13):
        raise RuntimeError(f"unexpected winner/loser counts: {(winner_count, loser_count)}")
    if len({record["case_id"] for record in counterfactual_cases}) != 10:
        raise RuntimeError("counterfactual subset is not ten unique cases")

    payload = {
        "schema": "c2f-p0-forensics-lock-v1",
        "purpose": "mechanism_discovery_only",
        "frozen_inputs": {
            name: {"path": str(paths[name]), "sha256": actual[name]} for name in sorted(paths)
        },
        "reserved_overlap_counts": expected_overlap,
        "primary_outcome": {
            "metric": "met3r_one_second",
            "delta_definition": "c2f_minus_baseline",
            "lower_is_better": True,
            "winner_rule": "delta < 0",
            "near_tie_sensitivity_threshold_abs": 0.001,
        },
        "winner_count": winner_count,
        "loser_count": loser_count,
        "cases_ranked_best_to_worst": locked_cases,
        "counterfactual_subset_rule": "within each motion stratum, minimum and maximum primary delta",
        "counterfactual_subset": counterfactual_cases,
        "successor_method_requirement": (
            "evaluate first on a new development holdout disjoint from this Dev25 and frozen test/validation/debug"
        ),
    }
    atomic_json(args.output.resolve(), payload)
    print(f"wrote {args.output.resolve()}")
    print(f"Dev25 labels: {winner_count} winners, {loser_count} losers")
    print("reserved overlap:", expected_overlap)
    print("counterfactual cases:", len(counterfactual_cases))


if __name__ == "__main__":
    main()
