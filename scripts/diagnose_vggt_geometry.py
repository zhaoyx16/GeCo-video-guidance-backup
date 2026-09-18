#!/usr/bin/env python3
"""Read-only geometry sanity check for an existing generated video.

This does not affect a diffusion pipeline.  It tests whether VGGT's inferred
depth, camera poses, and confidence maps give a coherent enough reprojection
map to justify geometry-aware attention transport later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "vggt"))

from utils import vggt_infer  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402


def parse_indices(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) < 2 or any(value < 0 for value in values):
        raise ValueError("--frame_indices needs at least two non-negative indices")
    return values


def parse_pairs(text: str, count: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in text.split(","):
        source, target = (int(x) for x in item.strip().split(":"))
        if not (0 <= source < count and 0 <= target < count and source != target):
            raise ValueError(f"Invalid selected-frame pair: {item}")
        pairs.append((source, target))
    return pairs


def load_video_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame_bgr = capture.read()
    if not ok:
        raise RuntimeError(f"Could not decode frame {index}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def frames_to_vggt_input(frames_rgb: list[np.ndarray], target_width: int = 518, patch_size: int = 14) -> torch.Tensor:
    """Same resize/crop convention used by demo_guidance_fast_failure.py."""
    array = np.stack(frames_rgb, axis=0)
    tensor = torch.from_numpy(array).float().div_(255.0).permute(0, 3, 1, 2).contiguous()
    _, _, height, width = tensor.shape
    scaled_height = max(patch_size, int(round((height * target_width / width) / patch_size) * patch_size))
    tensor = F.interpolate(tensor, size=(scaled_height, target_width), mode="bilinear", align_corners=False)
    if scaled_height > target_width:
        top = (scaled_height - target_width) // 2
        tensor = tensor[:, :, top : top + target_width, :]
    return tensor.unsqueeze(0)


def confidence_mask(confidence: torch.Tensor, percentile: float, floor: float) -> torch.Tensor:
    finite = torch.isfinite(confidence)
    if not finite.any():
        return torch.zeros_like(confidence, dtype=torch.bool)
    threshold = torch.maximum(
        torch.quantile(confidence[finite], percentile / 100.0),
        torch.tensor(floor, device=confidence.device, dtype=confidence.dtype),
    )
    return finite & (confidence >= threshold)


def project_source(
    depth_source: torch.Tensor,
    intrinsics_source: torch.Tensor,
    extrinsic_source: torch.Tensor,
    intrinsics_target: torch.Tensor,
    extrinsic_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project source pixels into target image using world-to-camera extrinsics."""
    height, width = depth_source.shape
    device = depth_source.device
    dtype = torch.float32
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    pixels = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).reshape(-1, 3).T
    depth = depth_source.float().reshape(1, -1)
    k_source = intrinsics_source.float()
    k_target = intrinsics_target.float()
    r_source, t_source = extrinsic_source[:, :3].float(), extrinsic_source[:, 3].float()
    r_target, t_target = extrinsic_target[:, :3].float(), extrinsic_target[:, 3].float()

    source_xyz = (torch.linalg.inv(k_source) @ pixels) * depth
    r_relative = r_target @ r_source.T
    t_relative = t_target - r_relative @ t_source
    target_xyz = r_relative @ source_xyz + t_relative[:, None]
    projected = k_target @ target_xyz
    target_depth = target_xyz[2].reshape(height, width)
    u = (projected[0] / projected[2].clamp_min(1e-6)).reshape(height, width)
    v = (projected[1] / projected[2].clamp_min(1e-6)).reshape(height, width)
    inside = (target_depth > 0) & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
    return u, v, target_depth, inside


def sample_target(map_hw: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    height, width = map_hw.shape
    grid = torch.stack(
        [2.0 * u / max(width - 1, 1) - 1.0, 2.0 * v / max(height - 1, 1) - 1.0], dim=-1
    ).unsqueeze(0)
    return F.grid_sample(
        map_hw.float().unsqueeze(0).unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )[0, 0]


def colour_depth(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = depth[valid & np.isfinite(depth)]
    if values.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    low, high = np.percentile(values, [2, 98])
    scaled = np.clip((depth - low) / max(high - low, 1e-6), 0.0, 1.0)
    image = (cm.turbo(scaled)[..., :3] * 255.0).astype(np.uint8)
    image[~valid] = 0
    return image


def resize_for_report(image: np.ndarray, width: int = 960) -> np.ndarray:
    height, old_width = image.shape[:2]
    if old_width <= width:
        return image
    return cv2.resize(image, (width, int(round(height * width / old_width))), interpolation=cv2.INTER_AREA)


def put_label(image: np.ndarray, text: str) -> np.ndarray:
    image = image.copy()
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 820), 42), (0, 0, 0), thickness=-1)
    cv2.putText(image, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def save_depth_sheet(frames: list[np.ndarray], depths: torch.Tensor, output: Path, indices: list[int]) -> None:
    panels: list[np.ndarray] = []
    for idx, (frame, depth) in enumerate(zip(frames, depths)):
        rgb = put_label(resize_for_report(frame), f"RGB frame {indices[idx]}")
        valid = np.isfinite(depth[..., 0].cpu().numpy()) & (depth[..., 0].cpu().numpy() > 0)
        color = put_label(resize_for_report(colour_depth(depth[..., 0].cpu().numpy(), valid)), f"VGGT depth frame {indices[idx]}")
        panels.extend([rgb, color])
    rows = [np.concatenate(panels[start : start + 2], axis=1) for start in range(0, len(panels), 2)]
    cv2.imwrite(str(output), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))


def save_projection_sheet(
    frames: list[np.ndarray],
    depths: torch.Tensor,
    confidence: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    source_idx: int,
    target_idx: int,
    selected_indices: list[int],
    output: Path,
    confidence_percentile: float,
    confidence_floor: float,
    depth_relative_threshold: float,
    grid_stride: int,
) -> dict[str, float]:
    source_depth = depths[source_idx, ..., 0]
    target_depth = depths[target_idx, ..., 0]
    source_conf = confidence[source_idx, ..., 0]
    target_conf = confidence[target_idx, ..., 0]
    u, v, projected_depth, inside = project_source(
        source_depth, intrinsics[source_idx], extrinsics[source_idx], intrinsics[target_idx], extrinsics[target_idx]
    )
    target_depth_at_projection = sample_target(target_depth, u, v)
    target_conf_at_projection = sample_target(target_conf, u, v)
    source_reliable = confidence_mask(source_conf, confidence_percentile, confidence_floor)
    target_reliable = confidence_mask(target_conf, confidence_percentile, confidence_floor)
    target_reliable_at_projection = sample_target(target_reliable.float(), u, v) > 0.5
    symmetric_depth_error = (target_depth_at_projection - projected_depth).abs() / (
        target_depth_at_projection.abs() + projected_depth.abs() + 1e-6
    )
    depth_consistent = symmetric_depth_error <= depth_relative_threshold
    visible = inside & source_reliable & target_reliable_at_projection & depth_consistent

    source = frames[source_idx].copy()
    target = frames[target_idx].copy()
    height, width = source.shape[:2]
    ys = torch.arange(grid_stride // 2, height, grid_stride, device=source_depth.device)
    xs = torch.arange(grid_stride // 2, width, grid_stride, device=source_depth.device)
    y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
    flat_y, flat_x = y_grid.flatten(), x_grid.flatten()
    colours = cm.hsv(np.linspace(0.0, 1.0, flat_x.numel(), endpoint=False))[..., :3]
    colours = (255.0 * colours).astype(np.uint8)

    for y_value, x_value, colour in zip(flat_y.tolist(), flat_x.tolist(), colours.tolist()):
        if not bool(inside[y_value, x_value]):
            continue
        colour_tuple = tuple(int(vv) for vv in colour)
        projected = (int(round(float(u[y_value, x_value]))), int(round(float(v[y_value, x_value]))))
        center = (int(x_value), int(y_value))
        if bool(visible[y_value, x_value]):
            cv2.circle(source, center, 3, colour_tuple, thickness=-1)
            cv2.circle(target, projected, 3, colour_tuple, thickness=-1)
        else:
            cv2.circle(source, center, 3, colour_tuple, thickness=1)
            cv2.drawMarker(target, projected, colour_tuple, markerType=cv2.MARKER_TILTED_CROSS, markerSize=7, thickness=1)

    source = put_label(resize_for_report(source), f"source f{selected_indices[source_idx]}: circles = visible projection")
    target = put_label(resize_for_report(target), f"target f{selected_indices[target_idx]}: circle visible; cross rejected")
    cv2.imwrite(str(output), cv2.cvtColor(np.concatenate([source, target], axis=1), cv2.COLOR_RGB2BGR))

    total = float(source_depth.numel())
    return {
        "source_frame": selected_indices[source_idx],
        "target_frame": selected_indices[target_idx],
        "source_confident_fraction": float(source_reliable.float().mean().item()),
        "in_bounds_fraction": float(inside.float().mean().item()),
        "depth_consistent_fraction_of_in_bounds": float(depth_consistent[inside].float().mean().item()) if inside.any() else 0.0,
        "visible_geometry_fraction": float(visible.float().mean().item()),
        "visible_geometry_count": int(visible.sum().item()),
        "pixels": int(total),
        "median_symmetric_depth_error_in_bounds": float(symmetric_depth_error[inside].median().item()) if inside.any() else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--frame_indices", default="0,40,80,96,120")
    parser.add_argument("--pairs", default="2:3,2:4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence_percentile", type=float, default=20.0)
    parser.add_argument("--confidence_floor", type=float, default=0.2)
    parser.add_argument("--depth_relative_threshold", type=float, default=0.15)
    parser.add_argument("--grid_stride", type=int, default=24)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    indices = parse_indices(args.frame_indices)
    pairs = parse_pairs(args.pairs, len(indices))
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    frames = [load_video_frame(capture, index) for index in indices]
    capture.release()
    height, width = frames[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in frames):
        raise RuntimeError("Video frames have inconsistent resolutions")

    device = torch.device(args.device)
    compute_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    print(f"Loading VGGT on {device}; frames={indices}; resolution={height}x{width}", flush=True)
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    inputs = frames_to_vggt_input(frames)
    with torch.inference_mode():
        geometry = vggt_infer(
            model, inputs, upsample_size=(height, width), point_prediction=False, compute_dtype=compute_dtype, device=device
        )

    depths = geometry["depth_map"].detach().float().cpu()
    confidence = geometry["vggt_conf"].detach().float().cpu()
    intrinsics = geometry["intrinsic"].detach().float().cpu()
    extrinsics = geometry["extrinsic"].detach().float().cpu()
    torch.save(
        {"frame_indices": indices, "depth": depths, "confidence": confidence, "intrinsic": intrinsics, "extrinsic": extrinsics},
        outdir / "vggt_geometry.pt",
    )

    save_depth_sheet(frames, depths, outdir / "depth_contact_sheet.png", indices)
    pair_stats = []
    for source_idx, target_idx in pairs:
        pair_name = f"project_f{indices[source_idx]:03d}_to_f{indices[target_idx]:03d}.png"
        pair_stats.append(
            save_projection_sheet(
                frames, depths, confidence, intrinsics, extrinsics, source_idx, target_idx, indices,
                outdir / pair_name, args.confidence_percentile, args.confidence_floor,
                args.depth_relative_threshold, args.grid_stride,
            )
        )

    report = {
        "video": str(Path(args.video).resolve()),
        "selected_video_frames": indices,
        "image_size": [height, width],
        "confidence_percentile": args.confidence_percentile,
        "confidence_floor": args.confidence_floor,
        "depth_relative_threshold": args.depth_relative_threshold,
        "pair_stats": pair_stats,
        "interpretation": (
            "Visible geometry fraction is the share of source pixels that are confident, project in-frame, and agree "
            "with target VGGT depth. This is a reliability diagnostic, not a generation score."
        ),
    }
    (outdir / "geometry_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
