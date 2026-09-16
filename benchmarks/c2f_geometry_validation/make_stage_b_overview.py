#!/usr/bin/env python3
"""Assemble the ten locked Stage B B/C/G/U contact sheets into one overview."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1400)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    cases = lock.get("cases")
    if not isinstance(cases, list) or len(cases) != 10:
        raise ValueError("expected the frozen 10-case Stage B lock")
    panels = []
    records = []
    for case in cases:
        case_id = case["case_id"]
        report_path = args.visual_root / case_id / "visual_review.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("case_id") != case_id or set(report.get("comparator_locks", {})) != {"B", "C"}:
            raise ValueError(f"invalid formal visual record: {report_path}")
        image_path = Path(report["contact_sheet"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read {image_path}")
        height = round(image.shape[0] * args.width / image.shape[1])
        image = cv2.resize(image, (args.width, height), interpolation=cv2.INTER_AREA)
        title_height = 42
        titled = np.full((title_height + height, args.width, 3), 20, dtype=np.uint8)
        title = f"{case['selection_order']:02d} | {case['motion_stratum']} | {case_id}"
        cv2.putText(titled, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        titled[title_height:] = image
        panels.append(titled)
        records.append(
            {
                "case_id": case_id,
                "motion_stratum": case["motion_stratum"],
                "contact_sheet": str(image_path.resolve()),
                "contact_sheet_sha256": sha256_file(image_path),
                "visual_record": str(report_path.resolve()),
                "visual_record_sha256": sha256_file(report_path),
            }
        )
    separator = np.full((8, args.width, 3), 90, dtype=np.uint8)
    canvas_parts = []
    for index, panel in enumerate(panels):
        if index:
            canvas_parts.append(separator)
        canvas_parts.append(panel)
    canvas = np.vstack(canvas_parts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not cv2.imwrite(str(args.output), canvas):
        raise RuntimeError(f"failed to write {args.output}")
    receipt = {
        "schema": "wan-c2f-geometry-stage-b-formal-visual-overview-v1",
        "stage_b_lock": {"path": str(args.lock.resolve()), "sha256": sha256_file(args.lock)},
        "image": str(args.output.resolve()),
        "image_sha256": sha256_file(args.output),
        "records": records,
    }
    receipt_path = args.output.with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
