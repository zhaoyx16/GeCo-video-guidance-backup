#!/usr/bin/env python3
"""Calibrate the independent epipolar diagnostic with known video controls."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from navigation_benchmark.epipolar import (
    EVALUATOR_NAME,
    EVALUATOR_VERSION,
    EpipolarConfig,
    evaluate_video,
    parse_float_list,
    parse_video_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_frames", required=True, help="Glob for ordered GT frames")
    parser.add_argument("--reference_fps", type=float, required=True)
    parser.add_argument("--reference_start_index", type=int)
    parser.add_argument("--reference_end_index", type=int)
    parser.add_argument("--reference_output_width", type=int)
    parser.add_argument("--reference_output_height", type=int)
    parser.add_argument(
        "--reference_resize_mode",
        choices=("none", "stretch", "center_crop"),
        default="none",
    )
    parser.add_argument("--video", action="append", default=[], help="LABEL=/path/video.mp4")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--control_video_dir", required=True)
    parser.add_argument("--lags_sec", default="0.5,1.0")
    parser.add_argument("--max_pairs", type=int, default=8)
    parser.add_argument("--max_side", type=int, default=960)
    parser.add_argument("--nonrigid_amplitude_px", type=float, default=24.0)
    return parser.parse_args()


def select_frame_paths(
    pattern: str, start_index: int | None, end_index: int | None
) -> list[Path]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"no frames matched: {pattern}")
    selected = []
    for value in paths:
        path = Path(value)
        if start_index is not None or end_index is not None:
            try:
                frame_index = int(path.stem)
            except ValueError as exc:
                raise ValueError(
                    "indexed selection requires numeric frame filenames"
                ) from exc
            if start_index is not None and frame_index < start_index:
                continue
            if end_index is not None and frame_index > end_index:
                continue
        selected.append(path)
    if not selected:
        raise ValueError("reference frame interval is empty")
    return selected


def load_frames(paths: list[Path]) -> list[np.ndarray]:
    frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    if any(frame is None for frame in frames):
        raise RuntimeError("one or more reference frames could not be read")
    return frames


def preprocess_reference(
    frames: list[np.ndarray],
    *,
    width: int | None,
    height: int | None,
    mode: str,
) -> list[np.ndarray]:
    if mode == "none":
        if width is not None or height is not None:
            raise ValueError("reference output size requires a resize mode")
        return frames
    if width is None or height is None or width <= 0 or height <= 0:
        raise ValueError("reference resize requires positive output width and height")
    result = []
    for frame in frames:
        if mode == "stretch":
            transformed = cv2.resize(
                frame, (width, height), interpolation=cv2.INTER_LANCZOS4
            )
        else:
            source_height, source_width = frame.shape[:2]
            scale = max(width / source_width, height / source_height)
            resized_width = max(width, int(round(source_width * scale)))
            resized_height = max(height, int(round(source_height * scale)))
            resized = cv2.resize(
                frame,
                (resized_width, resized_height),
                interpolation=cv2.INTER_LANCZOS4,
            )
            left = (resized_width - width) // 2
            top = (resized_height - height) // 2
            transformed = resized[top : top + height, left : left + width]
        result.append(transformed)
    return result


def nonrigid_warp(frames: list[np.ndarray], amplitude: float) -> list[np.ndarray]:
    result = []
    count = max(len(frames) - 1, 1)
    for index, frame in enumerate(frames):
        height, width = frame.shape[:2]
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        phase = 2.0 * np.pi * index / count
        envelope = np.sin(np.pi * index / count)
        shift = amplitude * envelope * np.sin(2.0 * np.pi * y / height + phase)
        result.append(
            cv2.remap(
                frame,
                x + shift.astype(np.float32),
                y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
        )
    return result


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=None,
    )
    for frame in frames:
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    writer.close()


def main() -> None:
    args = parse_args()
    reference_paths = select_frame_paths(
        args.reference_frames,
        args.reference_start_index,
        args.reference_end_index,
    )
    reference = preprocess_reference(
        load_frames(reference_paths),
        width=args.reference_output_width,
        height=args.reference_output_height,
        mode=args.reference_resize_mode,
    )
    config = EpipolarConfig(
        lags_sec=parse_float_list(args.lags_sec),
        max_pairs_per_lag=args.max_pairs,
        max_side=args.max_side,
    )
    variants = {
        "reference_gt": reference,
        "control_frozen": [reference[0].copy() for _ in reference],
        "control_nonrigid": nonrigid_warp(reference, args.nonrigid_amplitude_px),
    }
    output_dir = Path(args.control_video_dir)
    control_paths = {}
    for label, frames in variants.items():
        path = output_dir / f"{label}.mp4"
        write_video(path, frames, args.reference_fps)
        control_paths[label] = path
    reports = [
        evaluate_video(label, path, config=config)
        for label, path in control_paths.items()
    ]
    for spec in args.video:
        label, path = parse_video_spec(spec)
        reports.append(evaluate_video(label, path, config=config))

    payload = {
        "evaluator": {
            "name": EVALUATOR_NAME,
            "version": EVALUATOR_VERSION,
            "independence_policy": "independent",
        },
        "calibration": {
            "reference_frames": args.reference_frames,
            "reference_fps": args.reference_fps,
            "reference_start_index": args.reference_start_index,
            "reference_end_index": args.reference_end_index,
            "selected_frame_count": len(reference_paths),
            "selected_first_frame": str(reference_paths[0]),
            "selected_last_frame": str(reference_paths[-1]),
            "reference_output_width": args.reference_output_width,
            "reference_output_height": args.reference_output_height,
            "reference_resize_mode": args.reference_resize_mode,
            "nonrigid_amplitude_px": args.nonrigid_amplitude_px,
        },
        "config": config.__dict__,
        "videos": reports,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "lag_sec",
                "actual_lag_sec",
                "feature_detectable_fraction",
                "match_coverage_fraction",
                "parallax_observable_fraction",
                "low_motion_degenerate_fraction",
                "homography_degenerate_fraction",
                "geometry_model_success_fraction",
                "heldout_f_inlier_ratio_mean",
                "heldout_capped_sampson_px_mean",
                "heldout_capped_sampson_diagonal_ratio_mean",
                "failure_aware_capped_sampson_px_mean",
                "match_motion_ratio_median",
            ],
        )
        writer.writeheader()
        for report in reports:
            for lag in report["lags"].values():
                writer.writerow(
                    {
                        "label": report["label"],
                        "lag_sec": lag["lag_sec"],
                        "actual_lag_sec": lag["actual_lag_sec"],
                        **{
                            key: lag["summary"].get(key)
                            for key in writer.fieldnames
                            if key not in {"label", "lag_sec", "actual_lag_sec"}
                        },
                    }
                )
    print(f"saved: {output_json}")
    print(f"saved: {output_csv}")


if __name__ == "__main__":
    main()
