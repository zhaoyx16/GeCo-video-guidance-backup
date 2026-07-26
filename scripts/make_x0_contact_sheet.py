#!/usr/bin/env python3
"""Create a labelled contact sheet from pipeline x0-debug PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def numeric_suffix(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="x0 predictions during denoising")
    parser.add_argument("--thumb_width", type=int, default=360)
    args = parser.parse_args()

    root = Path(args.input_dir)
    step_dirs = sorted((p for p in root.glob("step_*") if p.is_dir()), key=numeric_suffix)
    if not step_dirs:
        raise SystemExit(f"No step_* directories found in {root}")

    frames = sorted({p.name for step_dir in step_dirs for p in step_dir.glob("frame_*.png")}, key=lambda n: int(Path(n).stem.rsplit("_", 1)[-1]))
    if not frames:
        raise SystemExit(f"No frame_*.png files found in {root}")

    first = next(step_dir / frame for step_dir in step_dirs for frame in frames if (step_dir / frame).exists())
    with Image.open(first) as source:
        ratio = args.thumb_width / source.width
        thumb_height = max(1, round(source.height * ratio))

    label_height = 28
    title_height = 42
    row_label_width = 74
    padding = 8
    columns = len(step_dirs)
    rows = len(frames)
    canvas = Image.new(
        "RGB",
        (
            row_label_width + columns * (args.thumb_width + padding) + padding,
            title_height + label_height + rows * (thumb_height + label_height + padding) + padding,
        ),
        (18, 19, 30),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((padding, 12), args.title, fill=(240, 240, 245), font=font)

    for col, step_dir in enumerate(step_dirs):
        x = row_label_width + padding + col * (args.thumb_width + padding)
        draw.text((x, title_height + 8), f"step {numeric_suffix(step_dir)}", fill=(240, 240, 245), font=font)

    for row, frame_name in enumerate(frames):
        y = title_height + label_height + row * (thumb_height + label_height + padding)
        frame_index = int(Path(frame_name).stem.rsplit("_", 1)[-1])
        draw.text((padding, y + thumb_height // 2), f"frame {frame_index}", fill=(240, 240, 245), font=font)
        for col, step_dir in enumerate(step_dirs):
            x = row_label_width + padding + col * (args.thumb_width + padding)
            path = step_dir / frame_name
            if not path.exists():
                draw.rectangle((x, y, x + args.thumb_width, y + thumb_height), outline=(90, 90, 105))
                continue
            with Image.open(path) as image:
                image = image.convert("RGB").resize((args.thumb_width, thumb_height), Image.Resampling.LANCZOS)
                canvas.paste(image, (x, y))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(output)


if __name__ == "__main__":
    main()
