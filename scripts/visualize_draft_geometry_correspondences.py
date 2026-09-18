#!/usr/bin/env python3
"""Visualize target risk regions and their matched source-token locations."""

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
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", default="")
    parser.add_argument("--panel_width", type=int, default=480)
    return parser.parse_args()


def frame_to_latent(frame: int, temporal_scale: int) -> int:
    return 0 if frame <= 0 else (frame - 1) // temporal_scale + 1


def read_frames(video_path: str, frame_ids: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    wanted = set(frame_ids)
    result: dict[int, np.ndarray] = {}
    index = 0
    while wanted:
        ok, frame = cap.read()
        if not ok:
            break
        if index in wanted:
            result[index] = frame
            wanted.remove(index)
        index += 1
    cap.release()
    if wanted:
        raise RuntimeError(f"Missing video frames: {sorted(wanted)}")
    return result


def overlay_grid(
    frame: np.ndarray,
    grid: np.ndarray,
    color_map: int,
    outline: bool = True,
) -> np.ndarray:
    height, width = frame.shape[:2]
    active = grid > 0
    if not active.any():
        return frame.copy()
    scale = max(float(np.percentile(grid[active], 99)), 1e-8)
    heat = cv2.resize(grid, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(
        active.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    heat_u8 = np.clip(heat / scale * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(heat_u8, color_map)
    result = frame.copy()
    result[mask] = cv2.addWeighted(frame, 0.32, color, 0.68, 0)[mask]
    if outline:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(result, contours, -1, (0, 255, 255), 2)
    return result


def make_panel(frame: np.ndarray, label: str, width: int) -> np.ndarray:
    height = round(frame.shape[0] * width / frame.shape[1])
    panel = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (12, 12, 12), -1)
    cv2.putText(
        panel,
        label,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def scatter_source_grid(
    source_index: torch.Tensor,
    source_time: torch.Tensor,
    confidence: torch.Tensor,
    wanted_time: int,
    height: int,
    width: int,
) -> np.ndarray:
    result = torch.zeros(height * width, dtype=torch.float32)
    valid = (confidence > 0) & (source_time == wanted_time)
    if valid.any():
        indices = source_index[valid].long()
        values = confidence[valid].float()
        if hasattr(result, "scatter_reduce_"):
            result.scatter_reduce_(0, indices, values, reduce="amax", include_self=True)
        else:
            for index, value in zip(indices.tolist(), values.tolist()):
                result[index] = max(result[index], value)
    return result.reshape(height, width).numpy()


def project_source_colors(
    anchor_token_grids: dict[int, np.ndarray],
    source_index: torch.Tensor,
    source_time: torch.Tensor,
    confidence: torch.Tensor,
    height: int,
    width: int,
) -> np.ndarray:
    projected = torch.zeros(height * width, 3, dtype=torch.float32)
    weight_sum = confidence.float().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    weights = confidence.float() / weight_sum
    for slot in range(confidence.shape[-1]):
        slot_conf = confidence[:, slot]
        valid = slot_conf > 0
        if not valid.any():
            continue
        for latent_time, color_grid in anchor_token_grids.items():
            time_mask = valid & (source_time[:, slot] == latent_time)
            if not time_mask.any():
                continue
            source_ids = source_index[time_mask, slot].long().numpy()
            colors = torch.from_numpy(color_grid.reshape(-1, 3)[source_ids]).float()
            projected[time_mask] += colors * weights[time_mask, slot : slot + 1]
    return projected.reshape(height, width, 3).numpy()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.risk_map, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    temporal_scale = int(metadata.get("temporal_scale", 4))
    anchor_frames = [int(value) for value in metadata.get("anchor_video_frames", [])]
    if not anchor_frames:
        raise ValueError("risk map has no anchor_video_frames metadata")

    if args.frames:
        target_frames = [int(value) for value in args.frames.split(",") if value.strip()]
    else:
        target_frames = [int(value) for value in metadata.get("component_selected_video_frames", [])]
    if not target_frames:
        raise ValueError("No target frames requested")

    all_frames = read_frames(args.video, sorted(set(anchor_frames + target_frames)))
    confidence = payload["confidence"].float()
    source_time = payload["source_time"].long()
    source_index = payload["source_index"].long()
    grid_height, grid_width = confidence.shape[1:3]
    anchor_token_grids = {
        frame_to_latent(frame, temporal_scale): cv2.resize(
            all_frames[frame],
            (grid_width, grid_height),
            interpolation=cv2.INTER_AREA,
        )
        for frame in anchor_frames
    }

    rows: list[np.ndarray] = []
    for target_frame in target_frames:
        target_latent = frame_to_latent(target_frame, temporal_scale)
        map_time = target_latent - 1
        target_conf = confidence[map_time].amax(dim=-1).numpy()
        target_overlay = overlay_grid(
            all_frames[target_frame],
            target_conf,
            cv2.COLORMAP_TURBO,
        )
        coverage = float((target_conf > 0).mean()) * 100.0
        row = [
            make_panel(
                target_overlay,
                f"Target f{target_frame:03d}: risk {coverage:.2f}%",
                args.panel_width,
            )
        ]

        flat_conf = confidence[map_time].reshape(-1, confidence.shape[-1])
        projected_grid = project_source_colors(
            anchor_token_grids,
            source_index[map_time],
            source_time[map_time],
            flat_conf,
            grid_height,
            grid_width,
        )
        target_frame = all_frames[target_frame]
        target_height, target_width = target_frame.shape[:2]
        projected = cv2.resize(
            projected_grid,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)
        projected_mask = cv2.resize(
            (target_conf > 0).astype(np.uint8),
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        projected_overlay = target_frame.copy()
        projected_overlay[projected_mask] = projected[projected_mask]
        row.append(
            make_panel(
                projected_overlay,
                "Matched anchor RGB projection",
                args.panel_width,
            )
        )

        for anchor_frame in anchor_frames:
            anchor_latent = frame_to_latent(anchor_frame, temporal_scale)
            source_grid = scatter_source_grid(
                source_index[map_time],
                source_time[map_time],
                flat_conf,
                anchor_latent,
                grid_height,
                grid_width,
            )
            source_overlay = overlay_grid(
                all_frames[anchor_frame],
                source_grid,
                cv2.COLORMAP_VIRIDIS,
            )
            count = int((source_grid > 0).sum())
            row.append(
                make_panel(
                    source_overlay,
                    f"Matched source f{anchor_frame:03d}: {count} tokens",
                    args.panel_width,
                )
            )
        rows.append(np.concatenate(row, axis=1))

    sheet = np.concatenate(rows, axis=0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
