#!/usr/bin/env python3
"""Export frame-aligned, coordinate-gridded video comparisons for visual review."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


def parse_video(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("--video must be LABEL=/absolute/path/video.mp4")
    label, raw_path = spec.split("=", 1)
    path = Path(raw_path)
    if not label or not path.is_file():
        raise argparse.ArgumentTypeError(f"Invalid video specification: {spec}")
    return label, path


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def read_all_frames(path: Path, start: int, end: int | None) -> tuple[list[Image.Image], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    frames: list[Image.Image] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index >= start and (end is None or index <= end):
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if end is not None and index >= end:
            break
        index += 1
    capture.release()
    return frames, fps


def draw_coordinate_grid(
    frame: Image.Image,
    columns: int,
    rows: int,
    line_width: int,
) -> Image.Image:
    overlay = frame.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = load_font(max(14, round(frame.width / 42)))
    outline = max(1, line_width)

    for column in range(columns + 1):
        x = round(column * (frame.width - 1) / columns)
        draw.line((x, 0, x, frame.height - 1), fill=(255, 235, 30, 150), width=line_width)
    for row in range(rows + 1):
        y = round(row * (frame.height - 1) / rows)
        draw.line((0, y, frame.width - 1, y), fill=(255, 235, 30, 150), width=line_width)

    for column in range(columns):
        text = chr(ord("A") + column)
        x = round((column + 0.5) * frame.width / columns)
        draw.text(
            (x, 5),
            text,
            anchor="ma",
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=outline,
            stroke_fill=(0, 0, 0, 255),
        )
    for row in range(rows):
        text = str(row + 1)
        y = round((row + 0.5) * frame.height / rows)
        draw.text(
            (5, y),
            text,
            anchor="lm",
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=outline,
            stroke_fill=(0, 0, 0, 255),
        )
    return overlay.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="append", required=True, type=parse_video)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--title", default="Frame-aligned comparison")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--grid_columns", type=int, default=8)
    parser.add_argument("--grid_rows", type=int, default=4)
    parser.add_argument("--panel_width", type=int, default=640)
    parser.add_argument("--layout_columns", type=int, default=2)
    parser.add_argument("--index_stride", type=int, default=12)
    args = parser.parse_args()

    if args.start < 0 or args.end is not None and args.end < args.start:
        raise ValueError("Invalid --start/--end range")
    if args.grid_columns > 26:
        raise ValueError("--grid_columns must be <= 26")

    loaded: list[tuple[str, Path, list[Image.Image], float]] = []
    for label, path in args.video:
        frames, fps = read_all_frames(path, args.start, args.end)
        if not frames:
            raise RuntimeError(f"No frames read from {path}")
        loaded.append((label, path, frames, fps))

    frame_count = min(len(item[2]) for item in loaded)
    source_width, source_height = loaded[0][2][0].size
    panel_height = round(source_height * args.panel_width / source_width)
    header_height = 42
    gap = 10
    layout_rows = math.ceil(len(loaded) / args.layout_columns)
    canvas_width = args.layout_columns * args.panel_width + (args.layout_columns + 1) * gap
    canvas_height = 48 + layout_rows * (header_height + panel_height + gap) + gap
    title_font = load_font(22)
    label_font = load_font(17)

    frames_dir = args.output_dir / "frames"
    index_dir = args.output_dir / "index_pages"
    frames_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    rendered_paths: list[Path] = []

    for local_index in range(frame_count):
        source_index = args.start + local_index
        canvas = Image.new("RGB", (canvas_width, canvas_height), (16, 18, 25))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (gap, 10),
            f"{args.title} | frame {source_index:03d}",
            font=title_font,
            fill=(245, 245, 248),
        )
        for method_index, (label, _, frames, _) in enumerate(loaded):
            row, column = divmod(method_index, args.layout_columns)
            x = gap + column * (args.panel_width + gap)
            y = 48 + row * (header_height + panel_height + gap)
            draw.text((x, y + 8), label, font=label_font, fill=(240, 240, 244))
            panel = frames[local_index].resize(
                (args.panel_width, panel_height),
                Image.Resampling.LANCZOS,
            )
            panel = draw_coordinate_grid(
                panel,
                columns=args.grid_columns,
                rows=args.grid_rows,
                line_width=max(1, round(args.panel_width / 640)),
            )
            canvas.paste(panel, (x, y + header_height))
        output_path = frames_dir / f"frame_{source_index:03d}.png"
        canvas.save(output_path, compress_level=2)
        rendered_paths.append(output_path)

    thumb_width = 400
    thumb_height = round(canvas_height * thumb_width / canvas_width)
    page_columns = 3
    page_rows = 4
    per_page = page_columns * page_rows
    selected_paths = rendered_paths[:: max(1, args.index_stride)]
    for page_start in range(0, len(selected_paths), per_page):
        page_items = selected_paths[page_start : page_start + per_page]
        page = Image.new(
            "RGB",
            (
                page_columns * thumb_width + (page_columns + 1) * gap,
                page_rows * (thumb_height + 24) + (page_rows + 1) * gap,
            ),
            (16, 18, 25),
        )
        page_draw = ImageDraw.Draw(page)
        for item_index, item_path in enumerate(page_items):
            row, column = divmod(item_index, page_columns)
            x = gap + column * thumb_width
            y = gap + row * (thumb_height + 24)
            thumb = Image.open(item_path).resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
            page.paste(thumb, (x, y))
            page_draw.text((x, y + thumb_height + 4), item_path.stem, fill=(235, 235, 240))
        page.save(index_dir / f"index_page_{page_start // per_page:02d}.jpg", quality=92)

    manifest = {
        "title": args.title,
        "frame_range": [args.start, args.start + frame_count - 1],
        "frame_count": frame_count,
        "source_resolution": [source_width, source_height],
        "grid": {
            "columns": [chr(ord("A") + index) for index in range(args.grid_columns)],
            "rows": list(range(1, args.grid_rows + 1)),
            "cell_width_source_pixels": source_width / args.grid_columns,
            "cell_height_source_pixels": source_height / args.grid_rows,
            "report_format": "frame N, cell C2 (or C2-D2 for a region)",
        },
        "videos": [
            {"label": label, "path": str(path), "fps": fps, "available_frames": len(frames)}
            for label, path, frames, fps in loaded
        ],
        "frames_dir": str(frames_dir),
        "index_dir": str(index_dir),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
