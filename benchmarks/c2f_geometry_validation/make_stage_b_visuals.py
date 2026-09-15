#!/usr/bin/env python3
"""Create deterministic B/C/G/U visual-review artifacts for one Stage B case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
METHOD_IDS = {
    "G": "c2f_geometry_hard_gate",
    "U": "c2f_geometry_uniform_norm_control",
}


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


def absolute_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    difference = np.abs(left.astype(np.float32) - right.astype(np.float32)).mean(axis=2)
    return cv2.applyColorMap(np.clip(difference * 4.0, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=REPO_ROOT / "benchmarks/c2f_geometry_validation/STAGE_B_LOCK.json",
    )
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--frame-indices", default="0,15,30,45,60,75,90,105,120")
    parser.add_argument("--tile-width", type=int, default=240)
    args = parser.parse_args()

    lock = json.loads(args.lock.resolve().read_text(encoding="utf-8"))
    cases = lock["cases"]
    if not 0 <= args.case_index < len(cases):
        parser.error(f"case-index must be in [0, {len(cases) - 1}]")
    case = cases[args.case_index]
    case_id = case["case_id"]
    video_paths = {
        "B": Path(case["baseline_video"]),
        "C": Path(case["c2f_video"]),
        **{
            label: args.generation_root / method_id / case_id / "seed_0/video.mp4"
            for label, method_id in METHOD_IDS.items()
        },
    }
    videos: dict[str, list[np.ndarray]] = {}
    fps_values: dict[str, float] = {}
    for label, path in video_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        videos[label], fps_values[label] = read_video(path)

    reference = videos["B"]
    reference_shape = reference[0].shape
    reference_fps = fps_values["B"]
    for label, frames in videos.items():
        if len(frames) != len(reference):
            raise RuntimeError(f"frame-count mismatch for {label}: {len(frames)} != {len(reference)}")
        if frames[0].shape != reference_shape:
            raise RuntimeError(f"frame-shape mismatch for {label}: {frames[0].shape} != {reference_shape}")
        if abs(fps_values[label] - reference_fps) > 1e-6:
            raise RuntimeError(f"fps mismatch for {label}: {fps_values[label]} != {reference_fps}")

    indices = [int(value) for value in args.frame_indices.split(",")]
    if any(index < 0 or index >= len(reference) for index in indices):
        raise IndexError(f"frame indices outside [0, {len(reference) - 1}]: {indices}")

    actual_rows = {
        label: [resize(videos[label][index], args.tile_width) for index in indices]
        for label in ("B", "C", "G", "U")
    }
    difference_rows = {
        "|C-B| x4": [absolute_difference(cand, base) for base, cand in zip(actual_rows["B"], actual_rows["C"])],
        "|G-B| x4": [absolute_difference(cand, base) for base, cand in zip(actual_rows["B"], actual_rows["G"])],
        "|G-U| x4": [absolute_difference(gated, uniform) for gated, uniform in zip(actual_rows["G"], actual_rows["U"])],
    }
    rows = {
        "B: Wan": actual_rows["B"],
        "C: Frozen C2F": actual_rows["C"],
        "G: Geometry gate": actual_rows["G"],
        "U: Uniform norm": actual_rows["U"],
        **difference_rows,
    }
    label_width = 190
    header_height = 34
    tile_height = next(iter(rows.values()))[0].shape[0]
    canvas = np.full(
        (header_height + len(rows) * tile_height, label_width + len(indices) * args.tile_width, 3),
        20,
        dtype=np.uint8,
    )
    for column, index in enumerate(indices):
        x = label_width + column * args.tile_width
        cv2.putText(canvas, f"frame {index}", (x + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    for row_index, (name, tiles) in enumerate(rows.items()):
        y = header_height + row_index * tile_height
        cv2.putText(canvas, name, (8, y + tile_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
        for column, tile in enumerate(tiles):
            x = label_width + column * args.tile_width
            canvas[y : y + tile_height, x : x + args.tile_width] = tile

    output_dir = args.output_root / case_id
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet = output_dir / "bcgu_contact_sheet.png"
    mosaic_video = output_dir / "bcgu_mosaic.mp4"
    report_path = output_dir / "visual_review.json"
    if any(path.exists() for path in (contact_sheet, mosaic_video, report_path)):
        raise FileExistsError(f"refusing to overwrite visual-review artifacts in {output_dir}")
    if not cv2.imwrite(str(contact_sheet), canvas):
        raise RuntimeError(f"failed to write {contact_sheet}")

    frame_height, frame_width = reference_shape[:2]
    half_width = frame_width // 2
    half_height = frame_height // 2
    writer = cv2.VideoWriter(
        str(mosaic_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        reference_fps,
        (half_width * 2, half_height * 2),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {mosaic_video}")
    try:
        for frame_index in range(len(reference)):
            tiles = []
            for label, title in (("B", "B: Wan"), ("C", "C: Frozen C2F"), ("G", "G: Geometry gate"), ("U", "U: Uniform norm")):
                tile = cv2.resize(videos[label][frame_index], (half_width, half_height), interpolation=cv2.INTER_AREA)
                tiles.append(add_label(tile, title))
            writer.write(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    finally:
        writer.release()

    mad = {}
    for left, right in (("B", "C"), ("B", "G"), ("B", "U"), ("G", "U")):
        values = [
            float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
            for a, b in zip(videos[left], videos[right])
        ]
        mad[f"{left}_{right}"] = {"mean": float(np.mean(values)), "per_frame": values}
    report = {
        "case_id": case_id,
        "video_paths": {label: str(path.resolve()) for label, path in video_paths.items()},
        "frame_count": len(reference),
        "fps": reference_fps,
        "frame_shape_hwc": list(reference_shape),
        "contact_sheet_indices": indices,
        "mean_absolute_pixel_difference": mad,
        "contact_sheet": str(contact_sheet.resolve()),
        "mosaic_video": str(mosaic_video.resolve()),
        "interpretation_warning": "Pixel MAD is a visual sanity check, not a geometry metric.",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
