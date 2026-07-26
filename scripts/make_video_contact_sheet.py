#!/usr/bin/env python3
"""Render matched frames from one or more videos into a single review image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


def parse_video(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Each --video must be LABEL=/absolute/path/to/video.mp4")
    label, path = spec.split("=", 1)
    return label, Path(path)


def parse_indices(text: str) -> list[int]:
    indices = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not indices or min(indices) < 0:
        raise argparse.ArgumentTypeError("--indices must contain non-negative frame indices")
    return indices


def read_frame(path: Path, index: int) -> Image.Image:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {index} from {path}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="append", required=True, type=parse_video)
    parser.add_argument("--indices", default="0,8,16,24,32,40,48,56,64,72,80")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Video comparison")
    parser.add_argument("--thumb_width", type=int, default=220)
    args = parser.parse_args()

    indices = parse_indices(args.indices)
    videos: list[tuple[str, Path]] = args.video
    first_frame = read_frame(videos[0][1], indices[0])
    thumb_height = max(1, round(first_frame.height * args.thumb_width / first_frame.width))
    font = ImageFont.load_default()
    left = 94
    top = 48
    cell_label = 20
    gap = 8
    width = left + gap + len(indices) * (args.thumb_width + gap)
    height = top + len(videos) * (thumb_height + cell_label + gap) + gap
    sheet = Image.new("RGB", (width, height), (18, 19, 30))
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 10), args.title, fill=(242, 242, 247), font=font)

    for col, index in enumerate(indices):
        x = left + gap + col * (args.thumb_width + gap)
        draw.text((x, 30), f"frame {index}", fill=(220, 220, 230), font=font)

    for row, (label, path) in enumerate(videos):
        y = top + row * (thumb_height + cell_label + gap)
        draw.text((gap, y + thumb_height // 2), label, fill=(242, 242, 247), font=font)
        for col, index in enumerate(indices):
            x = left + gap + col * (args.thumb_width + gap)
            frame = read_frame(path, index).resize((args.thumb_width, thumb_height), Image.Resampling.LANCZOS)
            sheet.paste(frame, (x, y))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(output)


if __name__ == "__main__":
    main()
