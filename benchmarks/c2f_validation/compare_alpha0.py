#!/usr/bin/env python3
"""Compare an encoded alpha=0 C2F video against its official Wan baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_videos(first: Path, second: Path) -> dict:
    captures = [cv2.VideoCapture(str(first)), cv2.VideoCapture(str(second))]
    if not all(capture.isOpened() for capture in captures):
        raise RuntimeError("failed to open one or both videos")
    frame_count = 0
    absolute_sum = 0.0
    squared_sum = 0.0
    element_count = 0
    max_absolute = 0
    try:
        while True:
            reads = [capture.read() for capture in captures]
            if reads[0][0] != reads[1][0]:
                raise RuntimeError("videos have different frame counts")
            if not reads[0][0]:
                break
            first_frame, second_frame = reads[0][1], reads[1][1]
            if first_frame.shape != second_frame.shape:
                raise RuntimeError(f"frame shape mismatch: {first_frame.shape} != {second_frame.shape}")
            difference = first_frame.astype(np.int16) - second_frame.astype(np.int16)
            absolute = np.abs(difference)
            absolute_sum += float(absolute.sum())
            squared_sum += float(np.square(difference.astype(np.float64)).sum())
            element_count += difference.size
            max_absolute = max(max_absolute, int(absolute.max()))
            frame_count += 1
    finally:
        for capture in captures:
            capture.release()
    mse = squared_sum / element_count
    return {
        "frame_count": frame_count,
        "mean_absolute_difference": absolute_sum / element_count,
        "max_absolute_difference": max_absolute,
        "mse": mse,
        "psnr_db": float("inf") if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    generated = args.metadata.parent / "video.mp4"
    baseline = Path(metadata["baseline_video"])
    comparison = compare_videos(generated, baseline)
    report = {
        "schema": "wan-c2f-alpha0-equivalence-v1",
        "generated_video": str(generated),
        "generated_sha256": sha256_file(generated),
        "baseline_video": str(baseline),
        "baseline_sha256": sha256_file(baseline),
        "encoded_file_exact_match": sha256_file(generated) == sha256_file(baseline),
        "decoded_comparison": comparison,
        "passed": comparison["max_absolute_difference"] == 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace existing report: {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
