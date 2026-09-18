#!/usr/bin/env python3
"""Compare VGGT camera trajectories inferred from existing videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "vggt"))

from scripts.diagnose_vggt_geometry import frames_to_vggt_input, load_video_frame  # noqa: E402
from utils import vggt_infer  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402


def parse_indices(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item.strip()]
    if len(values) < 2:
        raise ValueError("Need at least two frame indices")
    return values


def rotation_angle_degrees(rotation: torch.Tensor) -> float:
    cosine = ((torch.trace(rotation) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cosine)).item())


def infer_video(model: VGGT, video: str, indices: list[int], device: torch.device) -> dict[str, object]:
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video}")
    frames = [load_video_frame(capture, index) for index in indices]
    capture.release()
    height, width = frames[0].shape[:2]
    inputs = frames_to_vggt_input(frames)
    compute_dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    with torch.inference_mode():
        geometry = vggt_infer(
            model,
            inputs,
            upsample_size=(height, width),
            point_prediction=False,
            compute_dtype=compute_dtype,
            device=device,
        )

    extrinsics = geometry["extrinsic"].detach().float().cpu()
    depth = geometry["depth_map"][..., 0].detach().float().cpu()
    rotations = extrinsics[:, :, :3]
    translations = extrinsics[:, :, 3]
    centers = -torch.bmm(rotations.transpose(1, 2), translations.unsqueeze(-1)).squeeze(-1)

    step_distances = torch.linalg.vector_norm(centers[1:] - centers[:-1], dim=-1)
    depth_values = depth[torch.isfinite(depth) & (depth > 0)]
    depth_scale = float(depth_values.median().item())
    path_length = float(step_distances.sum().item())
    endpoint_distance = float(torch.linalg.vector_norm(centers[-1] - centers[0]).item())
    incremental_rotations = [
        rotation_angle_degrees(rotations[index + 1] @ rotations[index].T)
        for index in range(len(indices) - 1)
    ]
    endpoint_rotation = rotation_angle_degrees(rotations[-1] @ rotations[0].T)

    return {
        "video": str(Path(video).resolve()),
        "frame_indices": indices,
        "depth_scale": depth_scale,
        "path_length": path_length,
        "path_over_depth": path_length / max(depth_scale, 1e-8),
        "endpoint_distance": endpoint_distance,
        "endpoint_over_depth": endpoint_distance / max(depth_scale, 1e-8),
        "cumulative_rotation_degrees": float(sum(incremental_rotations)),
        "endpoint_rotation_degrees": endpoint_rotation,
        "camera_centers": centers.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--frame_indices", default="0,12,24,36,48,60,72,84,96,108,120")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.video) != len(args.label):
        raise ValueError("--video and --label counts must match")

    device = torch.device(args.device)
    indices = parse_indices(args.frame_indices)
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    report = {
        label: infer_video(model, video, indices, device)
        for label, video in zip(args.label, args.video)
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    for label, values in report.items():
        print(
            f"{label}: path/depth={values['path_over_depth']:.4f} "
            f"end/depth={values['endpoint_over_depth']:.4f} "
            f"cum_rot={values['cumulative_rotation_degrees']:.2f}deg "
            f"end_rot={values['endpoint_rotation_degrees']:.2f}deg"
        )
    print(output)


if __name__ == "__main__":
    main()
