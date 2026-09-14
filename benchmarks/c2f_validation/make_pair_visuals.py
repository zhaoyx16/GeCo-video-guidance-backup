#!/usr/bin/env python3
"""Create deterministic baseline/candidate visual-review artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"video contains no decoded frames: {path}")
    return frames, fps


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 34), (18, 18, 18), thickness=-1)
    cv2.putText(
        result,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def resize(frame: np.ndarray, width: int) -> np.ndarray:
    height = round(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-indices", help="Comma-separated 0-based indices; default is 7 uniform frames")
    parser.add_argument("--tile-width", type=int, default=320)
    args = parser.parse_args()

    baseline, baseline_fps = read_video(args.baseline)
    candidate, candidate_fps = read_video(args.candidate)
    if len(baseline) != len(candidate):
        raise RuntimeError(f"frame-count mismatch: {len(baseline)} != {len(candidate)}")
    if baseline[0].shape != candidate[0].shape:
        raise RuntimeError(f"frame-shape mismatch: {baseline[0].shape} != {candidate[0].shape}")
    if abs(baseline_fps - candidate_fps) > 1e-6:
        raise RuntimeError(f"fps mismatch: {baseline_fps} != {candidate_fps}")

    if args.frame_indices:
        indices = [int(value) for value in args.frame_indices.split(",")]
    else:
        indices = np.linspace(0, len(baseline) - 1, 7).round().astype(int).tolist()
    if any(index < 0 or index >= len(baseline) for index in indices):
        raise IndexError(f"frame indices outside [0, {len(baseline) - 1}]: {indices}")

    baseline_tiles = [resize(baseline[index], args.tile_width) for index in indices]
    candidate_tiles = [resize(candidate[index], args.tile_width) for index in indices]
    difference_tiles = [
        cv2.applyColorMap(
            np.clip(
                np.abs(base.astype(np.float32) - cand.astype(np.float32)).mean(axis=2) * 4.0,
                0,
                255,
            ).astype(np.uint8),
            cv2.COLORMAP_INFERNO,
        )
        for base, cand in zip(baseline_tiles, candidate_tiles)
    ]

    label_width = 120
    tile_height = baseline_tiles[0].shape[0]
    header_height = 34
    canvas = np.full(
        (header_height + 3 * tile_height, label_width + len(indices) * args.tile_width, 3),
        20,
        dtype=np.uint8,
    )
    row_names = ["Official", "C2F", "Abs diff x4"]
    rows = [baseline_tiles, candidate_tiles, difference_tiles]
    for column, index in enumerate(indices):
        x = label_width + column * args.tile_width
        cv2.putText(
            canvas,
            f"frame {index}",
            (x + 8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    for row, (name, tiles) in enumerate(zip(row_names, rows)):
        y = header_height + row * tile_height
        cv2.putText(
            canvas,
            name,
            (8, y + tile_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        for column, tile in enumerate(tiles):
            x = label_width + column * args.tile_width
            canvas[y : y + tile_height, x : x + args.tile_width] = tile

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet = args.output_dir / "contact_sheet.png"
    side_by_side = args.output_dir / "side_by_side.mp4"
    report_path = args.output_dir / "visual_review.json"
    if any(path.exists() for path in (contact_sheet, side_by_side, report_path)):
        raise FileExistsError(f"refusing to overwrite visual-review artifacts in {args.output_dir}")
    if not cv2.imwrite(str(contact_sheet), canvas):
        raise RuntimeError(f"failed to write {contact_sheet}")

    frame_height, frame_width = baseline[0].shape[:2]
    writer = cv2.VideoWriter(
        str(side_by_side),
        cv2.VideoWriter_fourcc(*"mp4v"),
        baseline_fps,
        (frame_width * 2, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {side_by_side}")
    per_frame_mad: list[float] = []
    try:
        for base, cand in zip(baseline, candidate):
            per_frame_mad.append(float(np.abs(base.astype(np.float32) - cand.astype(np.float32)).mean()))
            writer.write(np.hstack([add_label(base, "Official baseline"), add_label(cand, "C2F K3 alpha=0.025")]))
    finally:
        writer.release()

    report = {
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "frame_count": len(baseline),
        "fps": baseline_fps,
        "frame_shape_hwc": list(baseline[0].shape),
        "contact_sheet_indices": indices,
        "mean_absolute_pixel_difference": float(np.mean(per_frame_mad)),
        "per_frame_mean_absolute_pixel_difference": per_frame_mad,
        "contact_sheet": str(contact_sheet.resolve()),
        "side_by_side": str(side_by_side.resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
