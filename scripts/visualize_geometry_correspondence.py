#!/usr/bin/env python3
"""Visualize the source pixels selected by a precomputed geometry map."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from external.guidance_wan.draft_geometry_map import frame_to_latent_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_path", required=True)
    parser.add_argument("--video_path", default=None)
    parser.add_argument("--target_frames", default=None)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read video frame {frame_index}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def add_label(image_rgb: np.ndarray, text: str) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image_bgr, (0, 0), (image_bgr.shape[1], 34), (15, 15, 15), -1)
    cv2.putText(
        image_bgr,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    args = parse_args()
    payload = torch.load(args.map_path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    video_path = args.video_path or metadata["draft_video"]
    _, token_h, token_w = metadata["token_grid"]
    temporal_scale = int(metadata["temporal_scale"])
    anchor_frames = [int(frame) for frame in metadata["anchor_video_frames"]]
    target_frames = metadata["target_video_frames"]
    if args.target_frames:
        target_frames = [int(item) for item in args.target_frames.split(",") if item]

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    source_tokens_by_time = {}
    output_h = output_w = None
    for source_frame in anchor_frames:
        source_rgb = read_frame(capture, source_frame)
        if output_h is None:
            output_h, output_w = source_rgb.shape[:2]
        source_time = frame_to_latent_index(
            source_frame,
            temporal_scale,
            int(metadata["token_grid"][0]),
        )
        source_tokens_by_time[source_time] = cv2.resize(
            source_rgb,
            (token_w, token_h),
            interpolation=cv2.INTER_AREA,
        ).reshape(-1, 3)

    target_row = []
    warp_row = []
    overlay_row = []
    conflict_confidence = payload.get("conflict_confidence")
    use_conflict_confidence = (
        conflict_confidence is not None
        and metadata.get("visibility_mode")
        in {
            "unsupported_foreground",
            "source_free_space_violation",
            "missing_source_surface",
        }
    )
    confidence_all = (
        conflict_confidence.unsqueeze(-1)
        if use_conflict_confidence
        else payload["confidence"]
    )
    source_index_all = payload["source_index"]
    source_time_all = payload["source_time"]

    for target_frame in target_frames:
        target_latent = frame_to_latent_index(
            target_frame,
            temporal_scale,
            int(metadata["token_grid"][0]),
        )
        if target_latent < 1:
            raise ValueError("Target frames must map after the conditioning latent")
        map_index = target_latent - 1
        confidence_slots = confidence_all[map_index]
        best_slot = confidence_slots.argmax(dim=-1)
        confidence = confidence_slots.gather(
            -1,
            best_slot.unsqueeze(-1),
        ).squeeze(-1).numpy()
        best_slot_flat = best_slot.reshape(-1, 1)
        source_index = source_index_all[map_index].reshape(
            token_h * token_w,
            -1,
        ).gather(1, best_slot_flat).squeeze(-1).numpy()
        source_time = source_time_all[map_index].gather(
            1,
            best_slot_flat,
        ).squeeze(-1).numpy()
        warp_flat = np.full((token_h * token_w, 3), 24, dtype=np.uint8)
        active_flat = confidence.reshape(-1) > 0
        for source_time_value in np.unique(source_time[active_flat]):
            source_time_value = int(source_time_value)
            if source_time_value not in source_tokens_by_time:
                raise ValueError(
                    f"No anchor frame found for source latent {source_time_value}"
                )
            from_this_anchor = active_flat & (source_time == source_time_value)
            warp_flat[from_this_anchor] = source_tokens_by_time[
                source_time_value
            ][source_index[from_this_anchor]]
        warp_small = warp_flat.reshape(token_h, token_w, 3)
        mask_small = confidence > 0

        warp = cv2.resize(
            warp_small,
            (output_w, output_h),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = cv2.resize(
            mask_small.astype(np.uint8),
            (output_w, output_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        target = read_frame(capture, target_frame)
        overlay = target.copy()
        overlay[mask] = (
            0.45 * target[mask].astype(np.float32)
            + 0.55 * warp[mask].astype(np.float32)
        ).astype(np.uint8)
        warp_masked = np.full_like(warp, 24)
        warp_masked[mask] = warp[mask]

        coverage = mask_small.mean()
        target_row.append(add_label(target, f"target f{target_frame}"))
        warp_row.append(
            add_label(
                warp_masked,
                (
                    "conflict source evidence"
                    if use_conflict_confidence
                    else "projected anchors"
                )
                + f", cov={coverage:.2f}",
            )
        )
        overlay_row.append(add_label(overlay, f"overlay f{target_frame}"))

    capture.release()
    cell_w = 320

    def resize_cell(image: np.ndarray) -> np.ndarray:
        cell_h = int(round(image.shape[0] * cell_w / image.shape[1]))
        return cv2.resize(image, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

    rows = [
        np.concatenate([resize_cell(image) for image in row], axis=1)
        for row in (target_row, warp_row, overlay_row)
    ]
    sheet = np.concatenate(rows, axis=0)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
