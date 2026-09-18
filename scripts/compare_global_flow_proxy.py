#!/usr/bin/env python3
"""Compare robust frame-to-frame image motion against a baseline video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_grayscale_frames(path: Path, long_side: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        scale = long_side / max(height, width)
        resized = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))
    capture.release()
    if len(frames) < 2:
        raise RuntimeError(f"Need at least two frames: {path}")
    return frames


def robust_global_flow(frames: list[np.ndarray]) -> np.ndarray:
    vectors = []
    for source, target in zip(frames[:-1], frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            source,
            target,
            None,
            pyr_scale=0.5,
            levels=4,
            winsize=31,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )
        vectors.append(np.median(flow.reshape(-1, 2), axis=0))
    return np.asarray(vectors, dtype=np.float64)


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    length = min(len(reference), len(candidate))
    reference = reference[:length]
    candidate = candidate[:length]
    difference = candidate - reference
    return {
        "pair_count": int(length),
        "vector_rmse": float(np.sqrt(np.mean(np.sum(difference**2, axis=1)))),
        "magnitude_mae": float(
            np.mean(
                np.abs(
                    np.linalg.norm(candidate, axis=1)
                    - np.linalg.norm(reference, axis=1)
                )
            )
        ),
        "cumulative_vector_difference": float(
            np.linalg.norm(candidate.sum(axis=0) - reference.sum(axis=0))
        ),
        "reference_mean_magnitude": float(np.linalg.norm(reference, axis=1).mean()),
        "candidate_mean_magnitude": float(np.linalg.norm(candidate, axis=1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--long_side", type=int, default=320)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline_flow = robust_global_flow(read_grayscale_frames(args.baseline, args.long_side))
    report = {
        "baseline": str(args.baseline),
        "long_side": args.long_side,
        "candidates": {},
    }
    for specification in args.candidate:
        if "=" not in specification:
            raise ValueError("--candidate must use LABEL=/absolute/path.mp4")
        label, path_text = specification.split("=", 1)
        path = Path(path_text)
        candidate_flow = robust_global_flow(read_grayscale_frames(path, args.long_side))
        report["candidates"][label] = {
            "path": str(path),
            **compare(baseline_flow, candidate_flow),
        }

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
