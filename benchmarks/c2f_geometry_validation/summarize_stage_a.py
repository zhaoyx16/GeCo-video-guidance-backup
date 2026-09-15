#!/usr/bin/env python3
"""Aggregate completed Stage A case records without making a method claim."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import tempfile
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def resize_width(image: np.ndarray, width: int) -> np.ndarray:
    height = int(round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def make_overview(records: list[dict], root: Path, output: Path) -> None:
    rows = []
    for record in records:
        case_dir = root / record["case_id"]
        panels = []
        for target_time in (10, 25):
            path = case_dir / f"target_t{target_time:02d}_geometry_vs_c2f.png"
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"could not read {path}")
            panels.append(resize_width(image, 720))
        common_height = min(panel.shape[0] for panel in panels)
        panels = [cv2.resize(panel, (panel.shape[1], common_height), interpolation=cv2.INTER_AREA) for panel in panels]
        row = np.concatenate(panels, axis=1)
        label = f"{record['motion_stratum']} | {record['case_id']}"
        cv2.rectangle(row, (0, 0), (row.shape[1], 38), (0, 0, 0), -1)
        cv2.putText(row, label, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(row)
    width = min(row.shape[1] for row in rows)
    rows = [cv2.resize(row, (width, row.shape[0]), interpolation=cv2.INTER_AREA) for row in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.concatenate(rows, axis=0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=REPO_ROOT / "benchmarks/c2f_geometry_validation/STAGE_A_LOCK.json",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/vol/dissolve/yz10325/outputs/c2f_geometry_validation_0915/stage_a_v1"),
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock = read_json(args.lock.resolve())
    input_root = args.input_root.resolve()
    output_dir = (args.output_dir or input_root).resolve()
    records = []
    rows = []
    for case in lock["cases"]:
        case_dir = input_root / case["case_id"]
        complete_path = case_dir / "COMPLETE.json"
        record_path = case_dir / "STAGE_A_CASE.json"
        if not complete_path.is_file() or not record_path.is_file():
            raise FileNotFoundError(f"incomplete Stage A case: {case['case_id']}")
        record = read_json(record_path)
        if record["case_id"] != case["case_id"]:
            raise RuntimeError("case record identity mismatch")
        records.append(record)
        for target_time, summary in record["by_target_time"].items():
            rows.append(
                {
                    "case_id": record["case_id"],
                    "motion_stratum": record["motion_stratum"],
                    "target_time": int(target_time),
                    "sample_count": summary["sample_count"],
                    "evidence_count": summary["evidence_count"],
                    "evidence_coverage": summary["evidence_coverage"],
                    "accept_fraction_of_evidence": summary["accept_fraction_of_evidence"],
                    "median_reprojection_error_tokens": summary["reprojection_error_tokens_evidence"]["median"],
                    "p90_reprojection_error_tokens": summary["reprojection_error_tokens_evidence"]["p90"],
                    "geometry_forward_seconds": record["geometry_forward_seconds"],
                    "geometry_peak_allocated_mib": record["geometry_peak_allocated_mib"],
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "STAGE_A_SUMMARY.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    coverage = [record["all_targets"]["evidence_coverage"] for record in records]
    accept = [record["all_targets"]["accept_fraction_of_evidence"] for record in records]
    aggregate = {
        "schema": "c2f-external-geometry-stage-a-summary-v1",
        "scope": "signal_sanity_not_method_performance",
        "case_count": len(records),
        "target_set_count": len(rows),
        "motion_strata": [record["motion_stratum"] for record in records],
        "evidence_coverage_mean": statistics.fmean(coverage),
        "evidence_coverage_median": statistics.median(coverage),
        "accept_fraction_of_evidence_mean": statistics.fmean(accept),
        "accept_fraction_of_evidence_median": statistics.median(accept),
        "geometry_forward_seconds_mean": statistics.fmean(
            record["geometry_forward_seconds"] for record in records
        ),
        "geometry_peak_allocated_mib_max": max(
            record["geometry_peak_allocated_mib"] for record in records
        ),
        "visual_qa_status": "pending_manual_review",
        "cases": records,
    }
    atomic_json(output_dir / "STAGE_A_SUMMARY.json", aggregate)

    lines = [
        "# Stage A External Geometry Summary",
        "",
        "This is a coordinate/signal sanity check, not a video-generation result.",
        "",
        "| Stratum | Target t | Samples | Evidence coverage | Accepted among evidence | Median error (tokens) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        accepted = row["accept_fraction_of_evidence"]
        median_error = row["median_reprojection_error_tokens"]
        lines.append(
            f"| {row['motion_stratum']} | {row['target_time']} | {row['sample_count']} | "
            f"{row['evidence_coverage']:.1%} | "
            f"{accepted:.1%} | {median_error:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Mean evidence coverage: {aggregate['evidence_coverage_mean']:.1%}",
            f"Mean accepted among evidence: {aggregate['accept_fraction_of_evidence_mean']:.1%}",
            f"Mean geometry forward time: {aggregate['geometry_forward_seconds_mean']:.2f}s/case",
            f"Maximum allocated CUDA memory: {aggregate['geometry_peak_allocated_mib_max']:.1f} MiB",
            "",
            "Visual QA remains required before Stage B.",
        ]
    )
    atomic_text(output_dir / "STAGE_A_SUMMARY.md", "\n".join(lines) + "\n")
    make_overview(records, input_root, output_dir / "STAGE_A_OVERVIEW.png")
    print(f"wrote {output_dir / 'STAGE_A_SUMMARY.json'}")
    print(f"wrote {output_dir / 'STAGE_A_OVERVIEW.png'}")


if __name__ == "__main__":
    main()
