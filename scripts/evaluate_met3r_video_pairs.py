#!/usr/bin/env python3
"""Evaluate matched video frame pairs with the official MEt3R implementation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from met3r import MEt3R


class DINO16PatchFeatureMap(torch.nn.Module):
    """Expose DINO-S/16 patch tokens in the map shape expected by MEt3R."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.patch_size = 16

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.model.get_intermediate_layers(images, n=1)[0]
        patch_tokens = tokens[:, 1:, :]
        height = images.shape[-2] // self.patch_size
        width = images.shape[-1] // self.patch_size
        expected_tokens = height * width
        if patch_tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f"DINO patch-token mismatch: got {patch_tokens.shape[1]}, "
                f"expected {height}x{width}={expected_tokens}"
            )
        return (
            patch_tokens.reshape(patch_tokens.shape[0], height, width, patch_tokens.shape[-1])
            .permute(0, 3, 1, 2)
            .contiguous()
        )


class ImageNetNormalize(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.mean) / self.std


def parse_video_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--video must be LABEL=/absolute/path/video.mp4")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("video label must not be empty")
    return label, Path(path)


def parse_float_list(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("--lags_sec must contain positive comma-separated values")
    return result


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    capture.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from: {path}")
    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Invalid FPS {fps} for: {path}")
    return frames, fps


def resize_for_metric(frame: np.ndarray, short_side: int, multiple: int = 16) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = short_side / float(min(height, width))
    out_height = max(multiple, int(round(height * scale / multiple)) * multiple)
    out_width = max(multiple, int(round(width * scale / multiple)) * multiple)
    return cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)


def frame_to_tensor(frame: np.ndarray, short_side: int, device: torch.device) -> torch.Tensor:
    frame = resize_for_metric(frame, short_side)
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
    return tensor.to(device=device)


def evenly_spaced_starts(first: int, last: int, gap: int, max_pairs: int) -> list[int]:
    latest_start = last - gap
    if latest_start < first:
        return []
    count = min(max_pairs, latest_start - first + 1)
    starts = np.linspace(first, latest_start, num=count, dtype=np.int64)
    return sorted(set(int(item) for item in starts))


def evaluate_video(
    metric: MEt3R,
    label: str,
    path: Path,
    lags_sec: list[float],
    max_pairs: int,
    short_side: int,
    start_frame: int,
    end_frame: int | None,
    device: torch.device,
) -> dict:
    frames, fps = read_video(path)
    first = max(0, start_frame)
    last = len(frames) - 1 if end_frame is None else min(end_frame, len(frames) - 1)
    if first > last:
        raise ValueError(f"Empty frame range [{first}, {last}] for {label}")

    video_result = {
        "label": label,
        "path": str(path),
        "fps": fps,
        "num_frames": len(frames),
        "evaluated_range": [first, last],
        "lags": {},
    }

    for lag_sec in lags_sec:
        gap = max(1, int(round(lag_sec * fps)))
        starts = evenly_spaced_starts(first, last, gap, max_pairs)
        if not starts:
            raise ValueError(
                f"Range [{first}, {last}] is too short for lag={lag_sec}s ({gap} frames)"
            )

        pair_results = []
        for start in starts:
            end = start + gap
            image_a = frame_to_tensor(frames[start], short_side, device)
            image_b = frame_to_tensor(frames[end], short_side, device)
            images = torch.stack([image_a, image_b], dim=0).unsqueeze(0)

            with torch.inference_mode():
                score, *_ = metric(
                    images=images,
                    return_overlap_mask=False,
                    return_score_map=False,
                    return_projections=False,
                )
            value = float(score.mean().item())
            pair_results.append({"start": start, "end": end, "score": value})
            print(
                f"{label} lag={lag_sec:g}s pair={start:03d}->{end:03d} "
                f"score={value:.6f}",
                flush=True,
            )
            del images, image_a, image_b, score
            torch.cuda.empty_cache()

        values = [item["score"] for item in pair_results]
        video_result["lags"][str(lag_sec)] = {
            "gap_frames": gap,
            "pairs": pair_results,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
        }
        print(
            f"{label} lag={lag_sec:g}s mean={np.mean(values):.6f} "
            f"std={np.std(values):.6f} n={len(values)}",
            flush=True,
        )

    return video_result


def write_csv(results: list[dict], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["label", "video", "fps", "num_frames", "range", "lag_sec", "gap_frames", "n", "mean", "std", "median"]
        )
        for result in results:
            for lag_sec, lag_result in result["lags"].items():
                writer.writerow(
                    [
                        result["label"],
                        result["path"],
                        result["fps"],
                        result["num_frames"],
                        f"{result['evaluated_range'][0]}-{result['evaluated_range'][1]}",
                        lag_sec,
                        lag_result["gap_frames"],
                        len(lag_result["pairs"]),
                        lag_result["mean"],
                        lag_result["std"],
                        lag_result["median"],
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="append", type=parse_video_spec, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--lags_sec", type=parse_float_list, default=[0.5, 1.0])
    parser.add_argument("--max_pairs", type=int, default=8)
    parser.add_argument("--short_side", type=int, default=256)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int)
    parser.add_argument("--upsampler", choices=["featup", "bilinear"], default="featup")
    args = parser.parse_args()

    if args.max_pairs <= 0:
        parser.error("--max_pairs must be positive")
    if args.short_side <= 0:
        parser.error("--short_side must be positive")

    for label, video_path in args.video:
        if not video_path.is_file():
            parser.error(f"video does not exist: {video_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    print(f"device={device} torch={torch.__version__}", flush=True)
    feature_name = "dino16" if args.upsampler == "featup" else "dino_vits16"
    feature_weights = "mhamilton723/FeatUp" if args.upsampler == "featup" else "facebookresearch/dino:main"
    print(
        f"metric=MEt3R backbone=mast3r feature={feature_name} "
        f"upsampler={args.upsampler} distance=cosine img_size=None",
        flush=True,
    )

    metric = MEt3R(
        img_size=None,
        use_norm=True,
        backbone="mast3r",
        feature_backbone=feature_name,
        feature_backbone_weights=feature_weights,
        upsampler=args.upsampler,
        distance="cosine",
        freeze=True,
    )
    if args.upsampler == "bilinear":
        # This is the same token reshape and ImageNet normalization used by
        # FeatUp's DINOFeaturizer. Only the learned FeatUp upsampler is
        # replaced by MEt3R's supported bilinear interpolation mode.
        metric.feature_model = DINO16PatchFeatureMap(metric.feature_model)
        metric.norm = ImageNetNormalize()
    metric = metric.to(device).eval()

    results = []
    for label, video_path in args.video:
        results.append(
            evaluate_video(
                metric=metric,
                label=label,
                path=video_path,
                lags_sec=args.lags_sec,
                max_pairs=args.max_pairs,
                short_side=args.short_side,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                device=device,
            )
        )

    payload = {
        "metric": {
            "name": "MEt3R",
            "backbone": "mast3r",
            "feature_backbone": feature_name,
            "feature_backbone_weights": feature_weights,
            "upsampler": args.upsampler,
            "distance": "cosine",
            "img_size": None,
            "short_side": args.short_side,
        },
        "settings": {
            "lags_sec": args.lags_sec,
            "max_pairs": args.max_pairs,
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
        },
        "videos": results,
    }
    json_path = args.output_dir / "met3r_results.json"
    csv_path = args.output_dir / "met3r_summary.csv"
    json_path.write_text(json.dumps(payload, indent=2))
    write_csv(results, csv_path)
    print(f"saved_json={json_path}", flush=True)
    print(f"saved_csv={csv_path}", flush=True)


if __name__ == "__main__":
    main()
