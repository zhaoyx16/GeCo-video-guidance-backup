#!/usr/bin/env python3
"""Detect static-surface identity drift along persistent 3D tracks.

The geometry map supplies target-to-canonical token correspondences. DINOv2
patch descriptors then answer a complementary question to depth consistency:
does the re-observed surface still have the same visual identity? The output
uses the ordinary DraftGeometryMap format and can therefore drive the existing
attention-only guidance pipeline without a model or decoder in the second pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from external.guidance_wan.draft_geometry_map import (
    DraftGeometryMap,
    load_draft_geometry_map,
    save_draft_geometry_map,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--canonical_map", required=True)
    parser.add_argument("--local_map", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature_cache", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--similarity_start", type=float, default=0.55)
    parser.add_argument("--similarity_full", type=float, default=0.30)
    parser.add_argument("--drift_start", type=float, default=0.15)
    parser.add_argument("--drift_full", type=float, default=0.35)
    parser.add_argument("--min_local_similarity", type=float, default=0.55)
    parser.add_argument("--min_valid_sources", type=int, default=2)
    parser.add_argument("--risk_smooth_radius", type=int, default=1)
    parser.add_argument("--report_frames", default="12,16,20,24,28,32,36,40,44,48,52,56,60,64")
    parser.add_argument("--panel_width", type=int, default=420)
    return parser.parse_args()


def read_video(path: Path) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    capture = cv2.VideoCapture(str(path))
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"No frames were decoded from {path}")
    return frames


def extract_dino_features(
    frames: list[np.ndarray],
    latent_frames: int,
    temporal_scale: int,
    token_height: int,
    token_width: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitl14",
        pretrained=True,
    ).to(device).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).reshape(1, 3, 1, 1)
    selected = [
        frames[min(token_time * temporal_scale, len(frames) - 1)]
        for token_time in range(latent_frames)
    ]
    output: list[torch.Tensor] = []
    for start in range(0, len(selected), batch_size):
        batch_np = np.stack(selected[start : start + batch_size])
        batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
        batch = batch.permute(0, 3, 1, 2) / 255.0
        batch = F.interpolate(
            batch,
            size=(token_height * 14, token_width * 14),
            mode="bilinear",
            align_corners=False,
        )
        batch = (batch - mean) / std
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            patch = model.forward_features(batch)["x_norm_patchtokens"]
        expected = token_height * token_width
        if patch.shape[1] != expected:
            raise RuntimeError(
                f"DINO returned {patch.shape[1]} patches, expected {expected}"
            )
        patch = F.normalize(patch.float(), dim=-1)
        output.append(
            patch.reshape(-1, token_height, token_width, patch.shape[-1]).cpu()
        )
    return torch.cat(output, dim=0)


def colourize_risk(risk: np.ndarray) -> np.ndarray:
    risk_u8 = np.clip(risk * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(risk_u8, cv2.COLORMAP_TURBO)


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    canvas = cv2.copyMakeBorder(image, 38, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(
        canvas,
        label,
        (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def make_contact_sheet(
    frames: list[np.ndarray],
    risk: torch.Tensor,
    temporal_scale: int,
    report_frames: list[int],
    output_path: Path,
    panel_width: int,
) -> None:
    rows: list[np.ndarray] = []
    for frame_id in report_frames:
        if frame_id >= len(frames):
            continue
        token_time = min(frame_id // temporal_scale, risk.shape[0] - 1)
        image = frames[frame_id]
        risk_np = cv2.resize(
            risk[token_time].numpy(),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        heat_bgr = colourize_risk(risk_np)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mask = risk_np > 0
        overlay = image_bgr.copy()
        blended = cv2.addWeighted(image_bgr, 0.45, heat_bgr, 0.55, 0)
        overlay[mask] = blended[mask]
        height = round(image.shape[0] * panel_width / image.shape[1])
        original_small = cv2.resize(image_bgr, (panel_width, height))
        overlay_small = cv2.resize(overlay, (panel_width, height))
        support = float((risk[token_time] > 0).float().mean().item())
        row = np.concatenate(
            [
                add_label(original_small, f"original f{frame_id:03d}"),
                add_label(
                    overlay_small,
                    f"identity risk t={token_time:02d} support={support:.1%}",
                ),
            ],
            axis=1,
        )
        rows.append(row)
    if rows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), np.concatenate(rows, axis=0))


def main() -> None:
    args = parse_args()
    if not args.similarity_full < args.similarity_start:
        raise ValueError("similarity_full must be lower than similarity_start")
    if not args.drift_start < args.drift_full:
        raise ValueError("drift_start must be lower than drift_full")
    if args.batch_size < 1 or args.min_valid_sources < 1:
        raise ValueError("batch_size and min_valid_sources must be positive")
    if args.risk_smooth_radius < 0:
        raise ValueError("risk_smooth_radius must be non-negative")

    canonical = load_draft_geometry_map(args.canonical_map, "cpu")
    local = (
        load_draft_geometry_map(args.local_map, "cpu")
        if args.local_map
        else None
    )
    latent_frames, token_height, token_width = tuple(
        canonical.metadata["token_grid"]
    )
    temporal_scale = int(canonical.metadata.get("temporal_scale", 4))
    spatial_tokens = token_height * token_width
    frames = read_video(Path(args.video))
    cache_path = Path(args.feature_cache) if args.feature_cache else None
    if cache_path is not None and cache_path.exists():
        features = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"loaded_feature_cache={cache_path}", flush=True)
    else:
        features = extract_dino_features(
            frames,
            latent_frames,
            temporal_scale,
            token_height,
            token_width,
            torch.device(args.device),
            args.batch_size,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(features, cache_path)
            print(f"saved_feature_cache={cache_path}", flush=True)
    expected_shape = (latent_frames, token_height, token_width)
    if features.shape[:3] != expected_shape:
        raise ValueError(
            f"Feature cache shape {tuple(features.shape)} does not match "
            f"token grid {expected_shape}"
        )

    feature_flat = features.reshape(latent_frames, spatial_tokens, -1)
    source_time = canonical.source_time
    source_index = canonical.source_index
    geometry_confidence = canonical.confidence.reshape_as(source_time).float()
    if source_time.shape[0] != latent_frames - 1:
        raise ValueError("Canonical map has an incompatible time dimension")

    memory_slots = source_time.shape[-1]
    identity_confidence = torch.zeros_like(geometry_confidence)
    risk_grid = torch.zeros(
        latent_frames,
        token_height,
        token_width,
        dtype=torch.float32,
    )
    similarity_grid = torch.ones_like(risk_grid)
    report: list[dict[str, float | int]] = []

    for target_time in range(1, latent_frames):
        source_time_t = source_time[target_time - 1]
        source_index_t = source_index[target_time - 1]
        valid = (
            (geometry_confidence[target_time - 1] > 0)
            & (source_time_t >= 0)
            & (source_time_t < target_time)
            & (source_index_t >= 0)
            & (source_index_t < spatial_tokens)
        )
        target = feature_flat[target_time].unsqueeze(1)
        source = feature_flat[
            source_time_t.clamp(0, latent_frames - 1),
            source_index_t.clamp(0, spatial_tokens - 1),
        ]
        cosine = (target * source).sum(dim=-1)
        cosine = torch.where(valid, cosine, torch.full_like(cosine, -1.0))
        best_similarity = cosine.amax(dim=-1)
        valid_count = valid.sum(dim=-1)
        enough = valid_count >= args.min_valid_sources
        if local is None:
            local_best_similarity = torch.ones_like(best_similarity)
            drift_score = 1.0 - best_similarity
            risk = (
                (args.similarity_start - best_similarity)
                / (args.similarity_start - args.similarity_full)
            ).clamp(0.0, 1.0)
        else:
            local_time_t = local.source_time[target_time - 1]
            local_index_t = local.source_index[target_time - 1]
            local_confidence_t = local.confidence.reshape_as(
                local.source_time
            )[target_time - 1]
            local_valid = (
                (local_confidence_t > 0)
                & (local_time_t >= 0)
                & (local_time_t < target_time)
                & (local_index_t >= 0)
                & (local_index_t < spatial_tokens)
            )
            local_source = feature_flat[
                local_time_t.clamp(0, latent_frames - 1),
                local_index_t.clamp(0, spatial_tokens - 1),
            ]
            local_cosine = (target * local_source).sum(dim=-1)
            local_cosine = torch.where(
                local_valid,
                local_cosine,
                torch.full_like(local_cosine, -1.0),
            )
            local_best_similarity = local_cosine.amax(dim=-1)
            local_enough = (
                local_valid.sum(dim=-1) >= args.min_valid_sources
            )
            enough = enough & local_enough
            drift_score = local_best_similarity - best_similarity
            risk = (
                (drift_score - args.drift_start)
                / (args.drift_full - args.drift_start)
            ).clamp(0.0, 1.0)
            risk = risk * (
                local_best_similarity >= args.min_local_similarity
            ).float()
        risk = risk * enough.float()
        if args.risk_smooth_radius > 0:
            kernel = 2 * args.risk_smooth_radius + 1
            risk = F.avg_pool2d(
                risk.reshape(1, 1, token_height, token_width),
                kernel_size=kernel,
                stride=1,
                padding=args.risk_smooth_radius,
                count_include_pad=False,
            ).reshape(-1)
            risk = risk * enough.float()
        identity_confidence[target_time - 1] = (
            valid.float() * risk.unsqueeze(-1)
        )
        risk_grid[target_time] = risk.reshape(token_height, token_width)
        similarity_grid[target_time] = best_similarity.reshape(
            token_height, token_width
        )
        valid_similarity = best_similarity[enough]
        report.append(
            {
                "target_time": target_time,
                "video_frame": min(target_time * temporal_scale, len(frames) - 1),
                "valid_coverage": float(enough.float().mean().item()),
                "risk_coverage": float((risk > 0).float().mean().item()),
                "risk_mean": float(risk.mean().item()),
                "similarity_p10": (
                    float(torch.quantile(valid_similarity, 0.10).item())
                    if valid_similarity.numel()
                    else 0.0
                ),
                "similarity_median": (
                    float(valid_similarity.median().item())
                    if valid_similarity.numel()
                    else 0.0
                ),
                "local_similarity_median": (
                    float(local_best_similarity[enough].median().item())
                    if enough.any()
                    else 0.0
                ),
                "drift_p90": (
                    float(torch.quantile(drift_score[enough], 0.90).item())
                    if enough.any()
                    else 0.0
                ),
            }
        )

    active_similarity = similarity_grid[1:][risk_grid[1:] > 0]
    metadata = dict(canonical.metadata)
    metadata.update(
        {
            "format_version": 7,
            "map_type": "dino_identity_risk_on_persistent_geometry",
            "canonical_map": str(Path(args.canonical_map)),
            "local_map": str(Path(args.local_map)) if args.local_map else "",
            "video": str(Path(args.video)),
            "feature_cache": str(cache_path) if cache_path else "",
            "similarity_start": args.similarity_start,
            "similarity_full": args.similarity_full,
            "drift_start": args.drift_start,
            "drift_full": args.drift_full,
            "min_local_similarity": args.min_local_similarity,
            "min_valid_sources": args.min_valid_sources,
            "risk_smooth_radius": args.risk_smooth_radius,
        }
    )
    output = DraftGeometryMap(
        source_time=source_time.clone(),
        source_index=source_index.clone(),
        confidence=identity_confidence.reshape(
            latent_frames - 1,
            token_height,
            token_width,
            memory_slots,
        ),
        pair_stats=report,
        metadata=metadata,
    )
    output_path = Path(args.output)
    save_draft_geometry_map(output, output_path)
    report_path = output_path.with_suffix(".json")
    with report_path.open("w") as handle:
        json.dump(
            {
                "metadata": metadata,
                "summary": {
                    "mean_risk_coverage": float(
                        (risk_grid[1:] > 0).float().mean().item()
                    ),
                    "mean_risk": float(risk_grid[1:].mean().item()),
                    "active_similarity_median": (
                        float(active_similarity.median().item())
                        if active_similarity.numel()
                        else 0.0
                    ),
                },
                "per_time": report,
            },
            handle,
            indent=2,
        )
    report_frames = [
        int(value) for value in args.report_frames.split(",") if value.strip()
    ]
    contact_path = output_path.parent / "dino_identity_risk_contact_sheet.png"
    make_contact_sheet(
        frames,
        risk_grid,
        temporal_scale,
        report_frames,
        contact_path,
        args.panel_width,
    )
    print(f"saved_map={output_path}", flush=True)
    print(f"saved_report={report_path}", flush=True)
    print(f"saved_contact_sheet={contact_path}", flush=True)
    print(
        f"mean_risk_coverage={(risk_grid[1:] > 0).float().mean().item():.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
