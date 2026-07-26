#!/usr/bin/env python3
"""Measure apparent adjacent-frame motion with the same RAFT model used for warp evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


def parse_video(value: str) -> tuple[str, Path]:
    name, filename = value.split("=", 1)
    path = Path(filename)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Missing video: {path}")
    return name, path


def resized_shape(height: int, width: int, long_side: int) -> tuple[int, int]:
    scale = long_side / float(max(height, width))
    return (
        max(8, int(round(height * scale / 8.0)) * 8),
        max(8, int(round(width * scale / 8.0)) * 8),
    )


def load_video(path: Path, long_side: int) -> tuple[torch.Tensor, float]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[torch.Tensor] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_h, out_w = resized_shape(*rgb.shape[:2], long_side)
        rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
        frames.append(torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0))
    capture.release()
    if len(frames) < 2 or fps <= 0:
        raise RuntimeError(f"Could not read a usable video from {path}")
    return torch.stack(frames), fps


@torch.inference_mode()
def motion_stats(model, transform, frames: torch.Tensor, device: torch.device) -> dict[str, float | int]:
    per_pair: list[float] = []
    for index in range(frames.shape[0] - 1):
        source, target = transform(
            frames[index].unsqueeze(0).to(device, non_blocking=True),
            frames[index + 1].unsqueeze(0).to(device, non_blocking=True),
        )
        flow = model(source, target)[-1]
        per_pair.append(float(torch.linalg.vector_norm(flow, dim=1).mean().item()))
    values = torch.tensor(per_pair)
    return {
        "n_pairs": len(per_pair),
        "mean_flow_px": float(values.mean().item()),
        "median_flow_px": float(values.median().item()),
        "p90_flow_px": float(torch.quantile(values, 0.9).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="append", type=parse_video, required=True, help="NAME=/absolute/path/video.mp4")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--long_side", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device)
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=True).to(device).eval()
    transform = weights.transforms()
    results: dict[str, object] = {
        "metric": "RAFT adjacent-frame flow magnitude (diagnostic only; not a quality score)",
        "long_side": args.long_side,
        "videos": {},
    }
    for name, path in args.video:
        frames, fps = load_video(path, args.long_side)
        stats = motion_stats(model, transform, frames, device)
        stats["fps"] = fps
        results["videos"][name] = stats
        print(f"{name}: {stats}", flush=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
