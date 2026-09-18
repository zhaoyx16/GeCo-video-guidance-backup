#!/usr/bin/env python3
"""Create full-frame and tracked-ROI baseline/guided contact sheets."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--guided", type=Path, required=True)
    parser.add_argument("--geometry_map", type=Path, required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--crop_width", type=int, default=512)
    parser.add_argument("--crop_height", type=int, default=384)
    return parser.parse_args()


def read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames


def tracked_boxes(metadata: dict) -> dict[int, tuple[float, float, float, float]]:
    boxes = {
        int(metadata["seed_video_frame"]): tuple(
            float(v) for v in metadata["seed_roi_pixels_xyxy"]
        )
    }
    for item in metadata["active_by_target"]:
        boxes[int(item["target_video_frame"])] = tuple(
            float(v) for v in item["tracked_bbox_pixels_xyxy"]
        )
    return boxes


def interpolate_box(
    frame_index: int,
    boxes: dict[int, tuple[float, float, float, float]],
) -> tuple[int, int, int, int]:
    keys = sorted(boxes)
    if frame_index <= keys[0]:
        return tuple(round(v) for v in boxes[keys[0]])
    if frame_index >= keys[-1]:
        return tuple(round(v) for v in boxes[keys[-1]])
    right = next(key for key in keys if key >= frame_index)
    left = max(key for key in keys if key <= frame_index)
    if left == right:
        return tuple(round(v) for v in boxes[left])
    weight = (frame_index - left) / (right - left)
    return tuple(
        round((1.0 - weight) * a + weight * b)
        for a, b in zip(boxes[left], boxes[right])
    )


def crop_window(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    crop_width: int,
    crop_height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2
    left = min(max(center_x - crop_width // 2, 0), image_width - crop_width)
    top = min(max(center_y - crop_height // 2, 0), image_height - crop_height)
    return left, top, left + crop_width, top + crop_height


def labelled_panel(
    array: np.ndarray,
    label: str,
    box: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    panel = Image.fromarray(array)
    draw = ImageDraw.Draw(panel)
    if box is not None:
        draw.rectangle(box, outline=(255, 40, 40), width=5)
    font = ImageFont.load_default(size=22)
    label_box = draw.textbbox((0, 0), label, font=font)
    draw.rectangle(
        (0, 0, label_box[2] + 16, label_box[3] + 12),
        fill=(0, 0, 0),
    )
    draw.text((8, 6), label, fill=(255, 255, 255), font=font)
    return panel


def stack_rows(rows: list[list[Image.Image]], gap: int = 8) -> Image.Image:
    column_widths = [
        max(row[column].width for row in rows) for column in range(len(rows[0]))
    ]
    row_heights = [max(panel.height for panel in row) for row in rows]
    canvas = Image.new(
        "RGB",
        (
            sum(column_widths) + gap * (len(column_widths) - 1),
            sum(row_heights) + gap * (len(row_heights) - 1),
        ),
        (24, 24, 24),
    )
    y = 0
    for row, row_height in zip(rows, row_heights):
        x = 0
        for panel, column_width in zip(row, column_widths):
            canvas.paste(panel, (x, y))
            x += column_width + gap
        y += row_height + gap
    return canvas


def main() -> None:
    args = parse_args()
    frame_indices = [int(value) for value in args.frames.split(",")]
    baseline = read_video(args.baseline)
    guided = read_video(args.guided)
    if len(baseline) != len(guided):
        raise RuntimeError(
            f"Frame count mismatch: baseline={len(baseline)} guided={len(guided)}"
        )

    payload = torch.load(args.geometry_map, map_location="cpu", weights_only=False)
    boxes = tracked_boxes(payload["metadata"])
    height, width = baseline[0].shape[:2]

    full_rows = []
    crop_rows = []
    for frame_index in frame_indices:
        box = interpolate_box(frame_index, boxes)
        full_scale = 0.5
        scaled_box = tuple(round(value * full_scale) for value in box)
        baseline_full = cv2.resize(
            baseline[frame_index], None, fx=full_scale, fy=full_scale
        )
        guided_full = cv2.resize(
            guided[frame_index], None, fx=full_scale, fy=full_scale
        )
        full_rows.append(
            [
                labelled_panel(
                    baseline_full,
                    f"baseline f{frame_index:03d}",
                    scaled_box,
                ),
                labelled_panel(
                    guided_full,
                    f"guided f{frame_index:03d}",
                    scaled_box,
                ),
            ]
        )

        crop_box = crop_window(
            box,
            width,
            height,
            args.crop_width,
            args.crop_height,
        )
        left, top, right, bottom = crop_box
        local_box = (
            box[0] - left,
            box[1] - top,
            box[2] - left,
            box[3] - top,
        )
        crop_rows.append(
            [
                labelled_panel(
                    baseline[frame_index][top:bottom, left:right],
                    f"baseline ROI f{frame_index:03d}",
                    local_box,
                ),
                labelled_panel(
                    guided[frame_index][top:bottom, left:right],
                    f"guided ROI f{frame_index:03d}",
                    local_box,
                ),
            ]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.output_dir / "full_frame_comparison.png"
    crop_path = args.output_dir / "tracked_roi_comparison.png"
    stack_rows(full_rows).save(full_path)
    stack_rows(crop_rows).save(crop_path)
    print(f"saved: {full_path}")
    print(f"saved: {crop_path}")


if __name__ == "__main__":
    main()
