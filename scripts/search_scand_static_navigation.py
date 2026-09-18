#!/usr/bin/env python3
"""Sample VAMOS rows and materialize long-trajectory SCAND candidates."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-builder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ranges", default="0:60000:2000,330000:430000:2000")
    parser.add_argument("--min-path-length", type=float, default=10.0)
    parser.add_argument("--request-delay", type=float, default=0.0)
    return parser.parse_args()


def load_builder(path: Path):
    spec = importlib.util.spec_from_file_location("navigation_manifest_builder", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_rows(specification: str) -> list[int]:
    rows: set[int] = set()
    for item in specification.split(","):
        start, end, step = (int(value) for value in item.split(":"))
        rows.update(range(start, end + 1, step))
    return sorted(rows)


def download(url: str, output: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                output.write_bytes(response.read())
            return
        except Exception as error:
            last_error = error
            if attempt == 4:
                raise RuntimeError(f"Failed to download {url}") from last_error
            time.sleep(2 ** attempt)


def main() -> None:
    args = parse_args()
    builder = load_builder(Path(args.manifest_builder))
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    csv_path = output_dir / "candidates.csv"

    def save_candidates() -> None:
        (output_dir / "candidates.json").write_text(json.dumps(candidates, indent=2) + "\n")
        if not candidates:
            return
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
            writer.writeheader()
            writer.writerows(candidates)

    for ordinal, row_index in enumerate(parse_rows(args.ranges), start=1):
        try:
            row = builder.scand_row(row_index)
        except RuntimeError as error:
            print(f"[{ordinal}] skip row={row_index}: {error}", flush=True)
            time.sleep(max(args.request_delay, 30.0))
            continue
        points = np.asarray(row["shorter_trajectory_3d"] or row["trajectory_3d"], dtype=np.float64)
        segment = builder.scand_metrics(points)
        if segment.path_length < args.min_path_length:
            continue
        image_path = image_dir / f"row_{row_index:06d}-c{float(row.get('curvature', 0.0) or 0.0):.4f}.jpg"
        if not image_path.exists():
            download(row["image"]["src"], image_path)
        item = {
            "row_index": row_index,
            "image_path": str(image_path),
            "path_length_m": segment.path_length,
            "displacement_m": segment.displacement,
            "yaw_change_deg": segment.yaw_deg,
            "directness": segment.directness,
            "family": segment.family,
            "curvature": float(row.get("curvature", 0.0) or 0.0),
            "horizon": len(points),
        }
        candidates.append(item)
        save_candidates()
        print(
            f"[{ordinal}] row={row_index} path={segment.path_length:.2f} "
            f"yaw={segment.yaw_deg:+.1f} family={segment.family}",
            flush=True,
        )
        if args.request_delay > 0:
            time.sleep(args.request_delay)

    save_candidates()
    print(json.dumps({"sampled_candidates": len(candidates), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
