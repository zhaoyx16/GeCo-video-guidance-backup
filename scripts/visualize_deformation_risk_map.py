#!/usr/bin/env python3
"""Overlay a draft-geometry risk map on the corresponding video frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--risk_map", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--frames",
        default="",
        help="Comma-separated zero-based video frames. Defaults to selected component frames.",
    )
    parser.add_argument("--panel_width", type=int, default=640)
    return parser.parse_args()


def read_video_frames(video_path: str, frame_ids: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    wanted = set(frame_ids)
    frames: dict[int, np.ndarray] = {}
    index = 0
    while wanted:
        ok, frame = cap.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = frame
            wanted.remove(index)
        index += 1
    cap.release()

    if wanted:
        raise RuntimeError(f"Video ended before frames were read: {sorted(wanted)}")
    return frames


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 34), (12, 12, 12), -1)
    cv2.putText(
        result,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    risk = torch.load(args.risk_map, map_location="cpu", weights_only=False)
    confidence = risk["confidence"].float().amax(dim=-1).numpy()
    metadata = risk.get("metadata", {})
    temporal_scale = int(metadata.get("temporal_scale", 4))

    if args.frames:
        frame_ids = [int(value) for value in args.frames.split(",") if value.strip()]
    else:
        frame_ids = list(metadata.get("component_selected_video_frames", []))
        if not frame_ids:
            frame_ids = list(metadata.get("target_video_frames", []))
    if not frame_ids:
        raise ValueError("No frames requested and no target frames found in map metadata")

    video_frames = read_video_frames(args.video, frame_ids)
    active = confidence[confidence > 0]
    heat_scale = float(np.percentile(active, 99)) if active.size else 1.0
    heat_scale = max(heat_scale, 1e-8)

    rows: list[np.ndarray] = []
    stats: list[str] = []
    for frame_id in frame_ids:
        latent_id = 0 if frame_id <= 0 else (frame_id - 1) // temporal_scale + 1
        map_time = min(max(latent_id - 1, 0), confidence.shape[0] - 1)
        token_conf = confidence[map_time]
        token_mask = token_conf > 0
        coverage = float(token_mask.mean())

        frame = video_frames[frame_id]
        height, width = frame.shape[:2]
        heat = cv2.resize(token_conf, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(
            token_mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        heat_u8 = np.clip(heat / heat_scale * 255.0, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
        overlay = frame.copy()
        overlay[mask] = cv2.addWeighted(frame, 0.35, color, 0.65, 0)[mask]

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

        label = (
            f"f{frame_id:03d} latent={latent_id:02d} map_t={map_time:02d} "
            f"support={coverage * 100:.2f}% max={token_conf.max():.3f}"
        )
        cv2.imwrite(str(output_dir / f"risk_overlay_f{frame_id:03d}.png"), add_label(overlay, label))

        resized_height = round(height * args.panel_width / width)
        original_small = cv2.resize(frame, (args.panel_width, resized_height), interpolation=cv2.INTER_AREA)
        overlay_small = cv2.resize(overlay, (args.panel_width, resized_height), interpolation=cv2.INTER_AREA)
        original_small = add_label(original_small, f"Original f{frame_id:03d}")
        overlay_small = add_label(overlay_small, label)
        rows.append(np.concatenate([original_small, overlay_small], axis=1))
        stats.append(label)

    sheet = np.concatenate(rows, axis=0)
    sheet_path = output_dir / "risk_overlay_contact_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)
    (output_dir / "risk_overlay_stats.txt").write_text("\n".join(stats) + "\n")
    print(f"saved_contact_sheet={sheet_path}")
    print(f"heat_scale_p99={heat_scale:.6f}")
    for line in stats:
        print(line)


if __name__ == "__main__":
    main()
