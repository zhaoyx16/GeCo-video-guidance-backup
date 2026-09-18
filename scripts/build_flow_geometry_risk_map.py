#!/usr/bin/env python3
"""Build an offline deformation-risk map from observed and rigid optical flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "external" / "UFM"))

from external.guidance_wan.draft_geometry_map import (  # noqa: E402
    DraftGeometryMap,
    frame_to_latent_index,
    load_draft_geometry_map,
    save_draft_geometry_map,
)
from scripts.build_deformation_risk_map import (  # noqa: E402
    make_contact_sheet,
    overlay_risk,
    read_video_frames,
    token_centres,
)
from scripts.build_feature_identity_risk_map import (  # noqa: E402
    extract_oriented_edge_features,
    oriented_line_continuity,
)
from uniflowmatch.models.ufm import UniFlowMatchConfidence  # noqa: E402
from utils import (  # noqa: E402
    create_confidence_mask_torch,
    normalize_flow_to_unitless,
    rigid_flow_from_camera_motion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--link_map", required=True)
    parser.add_argument(
        "--fixed_anchor_frames",
        default="",
        help="Optional comma-separated persistent clean reference frames.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_frames", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--risk_mode",
        choices=("flow_geometry", "edge_birth"),
        default="flow_geometry",
    )
    parser.add_argument("--ufm_scale", type=float, default=0.25)
    parser.add_argument("--covis_threshold", type=float, default=0.5)
    parser.add_argument("--confidence_percentile", type=float, default=20.0)
    parser.add_argument("--confidence_floor", type=float, default=0.2)
    parser.add_argument("--min_anchor_agreement", type=int, default=2)
    parser.add_argument("--min_wrong_occ_agreement", type=int, default=2)
    parser.add_argument("--risk_quantile", type=float, default=0.90)
    parser.add_argument("--risk_full_quantile", type=float, default=0.99)
    parser.add_argument("--residual_floor", type=float, default=0.005)
    parser.add_argument("--wrong_occ_weight", type=float, default=0.5)
    parser.add_argument("--detection_scale", type=int, default=4)
    parser.add_argument("--edge_bins", type=int, default=8)
    parser.add_argument("--edge_birth_start", type=float, default=0.15)
    parser.add_argument("--edge_birth_full", type=float, default=0.55)
    parser.add_argument("--edge_presence", type=float, default=0.30)
    parser.add_argument("--edge_source_variation", type=float, default=0.40)
    parser.add_argument("--edge_continuity", type=int, default=7)
    return parser.parse_args()


@torch.inference_mode()
def predict_scaled_flow(
    model: UniFlowMatchConfidence,
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.from_numpy(source_rgb).to(device=device, dtype=torch.float32) / 255.0
    target = torch.from_numpy(target_rgb).to(device=device, dtype=torch.float32) / 255.0
    height, width = source.shape[:2]
    small_height = max(8, int(round(height * scale)))
    small_width = max(8, int(round(width * scale)))
    source_small = F.interpolate(
        source.permute(2, 0, 1).unsqueeze(0),
        size=(small_height, small_width),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    target_small = F.interpolate(
        target.permute(2, 0, 1).unsqueeze(0),
        size=(small_height, small_width),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    output = model.predict_correspondences_batched(
        source_image=source_small,
        target_image=target_small,
        data_norm_type="identity",
    )
    flow_small = output.flow.flow_output[0].unsqueeze(0)
    flow = F.interpolate(
        flow_small,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    flow[:, 0] *= width / float(small_width)
    flow[:, 1] *= height / float(small_height)
    covisibility = F.interpolate(
        output.covisibility.mask[0].reshape(1, 1, small_height, small_width),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return flow[0].permute(1, 2, 0), covisibility


def finite_quantile(values: torch.Tensor, quantile: float, fallback: float) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return fallback
    return float(torch.quantile(finite, quantile).item())


def flow_source_indices(
    flow: torch.Tensor,
    covisibility: torch.Tensor,
    target_confident: torch.Tensor,
    *,
    token_height: int,
    token_width: int,
    covis_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map target-grid cells to source-grid cells with target-to-source flow."""
    image_height, image_width = flow.shape[:2]
    pixel_y, pixel_x = token_centres(
        image_height,
        image_width,
        token_height,
        token_width,
        flow.device,
    )
    sampled_flow = flow[pixel_y, pixel_x]
    source_x_pixel = pixel_x.float() + sampled_flow[:, 0]
    source_y_pixel = pixel_y.float() + sampled_flow[:, 1]
    source_x = (
        source_x_pixel * token_width / image_width - 0.5
    ).round().long()
    source_y = (
        source_y_pixel * token_height / image_height - 0.5
    ).round().long()
    valid = (
        (source_x >= 0)
        & (source_x < token_width)
        & (source_y >= 0)
        & (source_y < token_height)
        & torch.isfinite(sampled_flow).all(dim=-1)
        & (covisibility[pixel_y, pixel_x] > covis_threshold)
        & target_confident[pixel_y, pixel_x]
    )
    source_index = (
        source_y.clamp(0, token_height - 1) * token_width
        + source_x.clamp(0, token_width - 1)
    )
    return source_index, valid


def main() -> None:
    args = parse_args()
    if not 0.0 < args.ufm_scale <= 1.0:
        raise ValueError("ufm_scale must lie in (0, 1]")
    if not 0.0 <= args.risk_quantile < args.risk_full_quantile <= 1.0:
        raise ValueError("Require 0 <= risk_quantile < risk_full_quantile <= 1")
    if args.detection_scale < 1:
        raise ValueError("detection_scale must be positive")
    if not 0.0 <= args.edge_birth_start < args.edge_birth_full <= 1.0:
        raise ValueError(
            "Require 0 <= edge_birth_start < edge_birth_full <= 1"
        )

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_frames = [
        int(item) for item in args.target_frames.split(",") if item.strip()
    ]
    fixed_anchor_frames = [
        int(item)
        for item in args.fixed_anchor_frames.split(",")
        if item.strip()
    ]

    geometry = torch.load(args.geometry, map_location="cpu", weights_only=False)
    links = load_draft_geometry_map(args.link_map, "cpu")
    latent_frames, token_height, token_width = tuple(links.metadata["token_grid"])
    temporal_scale = int(links.metadata.get("temporal_scale", 4))
    memory_slots = links.source_time.shape[-1]
    if len(fixed_anchor_frames) > memory_slots:
        raise ValueError(
            f"Got {len(fixed_anchor_frames)} fixed anchors for "
            f"{memory_slots} memory slots"
        )
    spatial_tokens = token_height * token_width
    detection_height = token_height * args.detection_scale
    detection_width = token_width * args.detection_scale
    detection_tokens = detection_height * detection_width

    frame_indices = [int(frame) for frame in geometry["frame_indices"]]
    frame_lookup = {frame: idx for idx, frame in enumerate(frame_indices)}
    required_frames = set(target_frames)
    required_frames.update(fixed_anchor_frames)
    for target_frame in target_frames:
        if fixed_anchor_frames:
            continue
        target_time = frame_to_latent_index(
            target_frame, temporal_scale, latent_frames
        )
        if target_time <= 0:
            continue
        row = target_time - 1
        for source_time in links.source_time[row].unique().tolist():
            if int(source_time) < target_time:
                required_frames.add(int(source_time) * temporal_scale)
    missing = sorted(required_frames.difference(frame_lookup))
    if missing:
        raise ValueError(f"Geometry file is missing required frames: {missing}")
    video_frames = read_video_frames(Path(args.video), sorted(required_frames))
    edge_features = None
    if args.risk_mode == "edge_birth":
        edge_features = extract_oriented_edge_features(
            video_frames,
            sorted(required_frames),
            token_height=detection_height,
            token_width=detection_width,
            bins=args.edge_bins,
        )

    depth = geometry["depth"].float().to(device)
    confidence = geometry["confidence"].float()[..., 0].to(device)
    intrinsic = geometry["intrinsic"].float().to(device)
    extrinsic = geometry["extrinsic"].float().to(device)
    image_height, image_width = depth.shape[1:3]
    pixel_y, pixel_x = token_centres(
        image_height, image_width, token_height, token_width, device
    )

    print("loading UFM...", flush=True)
    ufm = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base")
    ufm = ufm.to(device=device, dtype=torch.float32).eval()
    ufm.requires_grad_(False)

    output_confidence = torch.zeros_like(links.confidence)
    output_source_time = links.source_time.clone()
    output_source_index = links.source_index.clone()
    pair_stats: list[dict[str, float | int | str]] = []
    raw_report_frames: list[np.ndarray] = []
    overlay_report_frames: list[np.ndarray] = []

    for target_frame in target_frames:
        target_time = frame_to_latent_index(
            target_frame, temporal_scale, latent_frames
        )
        if target_time <= 0:
            continue
        row = target_time - 1
        target_sequence = frame_lookup[target_frame]
        target_confident = create_confidence_mask_torch(
            confidence[target_sequence],
            percentile_val=args.confidence_percentile,
            min_threshold=args.confidence_floor,
        )

        residual_slots = []
        visible_slots = []
        wrong_occ_slots = []
        edge_source_slots = []
        edge_valid_slots = []
        output_valid_slots = []
        source_labels = []
        for slot in range(memory_slots):
            if fixed_anchor_frames and slot < len(fixed_anchor_frames):
                source_frame = fixed_anchor_frames[slot]
                source_time = frame_to_latent_index(
                    source_frame, temporal_scale, latent_frames
                )
                if source_time >= target_time:
                    active_link = torch.zeros(
                        spatial_tokens, dtype=torch.bool
                    )
                else:
                    active_link = torch.ones(
                        spatial_tokens, dtype=torch.bool
                    )
                output_source_time[row, :, slot] = source_time
            elif fixed_anchor_frames:
                source_frame = -1
                source_time = -1
                active_link = torch.zeros(
                    spatial_tokens, dtype=torch.bool
                )
            else:
                source_times = links.source_time[row, :, slot]
                active_link = (
                    links.confidence[row, :, :, slot].reshape(-1) > 0
                )
                active_source_times = source_times[active_link]
                source_time = (
                    int(torch.mode(active_source_times).values.item())
                    if active_source_times.numel()
                    else -1
                )
                source_frame = source_time * temporal_scale
            if not active_link.any():
                residual_slots.append(
                    torch.full((spatial_tokens,), torch.nan, device=device)
                )
                visible_slots.append(
                    torch.zeros((spatial_tokens,), dtype=torch.bool, device=device)
                )
                wrong_occ_slots.append(
                    torch.zeros((spatial_tokens,), dtype=torch.bool, device=device)
                )
                edge_source_slots.append(
                    torch.zeros(
                        detection_tokens, args.edge_bins, dtype=torch.float32
                    )
                )
                edge_valid_slots.append(
                    torch.zeros(detection_tokens, dtype=torch.bool)
                )
                output_valid_slots.append(
                    torch.zeros(spatial_tokens, dtype=torch.bool)
                )
                source_labels.append(-1)
                continue
            source_labels.append(source_frame)
            source_sequence = frame_lookup[source_frame]

            observed_flow, covisibility = predict_scaled_flow(
                ufm,
                video_frames[target_frame],
                video_frames[source_frame],
                args.ufm_scale,
                device,
            )
            source_index_output, valid_output = flow_source_indices(
                observed_flow,
                covisibility,
                target_confident,
                token_height=token_height,
                token_width=token_width,
                covis_threshold=args.covis_threshold,
            )
            output_valid_slots.append(valid_output.cpu())
            output_source_index[row, :, slot] = source_index_output.cpu()
            if edge_features is not None:
                source_index_detection, valid_detection = flow_source_indices(
                    observed_flow,
                    covisibility,
                    target_confident,
                    token_height=detection_height,
                    token_width=detection_width,
                    covis_threshold=args.covis_threshold,
                )
                edge_source_slots.append(
                    edge_features[source_frame][source_index_detection.cpu()]
                )
                edge_valid_slots.append(valid_detection.cpu())
            rigid_flow, rigid_valid = rigid_flow_from_camera_motion(
                depth[target_sequence],
                intrinsic[[target_sequence, source_sequence]],
                extrinsic[[target_sequence, source_sequence]],
            )
            residual = normalize_flow_to_unitless(
                observed_flow - rigid_flow,
                intrinsic[target_sequence],
            ).norm(dim=-1)
            covisible = torch.isfinite(covisibility) & (
                covisibility > args.covis_threshold
            )
            valid_flow = rigid_valid & target_confident & covisible
            should_be_visible = rigid_valid & target_confident

            token_residual = residual[pixel_y, pixel_x]
            token_valid = valid_flow[pixel_y, pixel_x] & active_link.to(device)
            token_should_be_visible = (
                should_be_visible[pixel_y, pixel_x] & active_link.to(device)
            )
            token_covisible = covisible[pixel_y, pixel_x]
            residual_slots.append(
                torch.where(
                    token_valid,
                    token_residual,
                    torch.full_like(token_residual, torch.nan),
                )
            )
            visible_slots.append(token_valid)
            wrong_occ_slots.append(token_should_be_visible & ~token_covisible)

        visible_stack = torch.stack(visible_slots, dim=0)
        if args.risk_mode == "edge_birth":
            assert edge_features is not None
            source_edges = torch.stack(edge_source_slots, dim=0)
            edge_valid = torch.stack(edge_valid_slots, dim=0)
            valid_count_detection = edge_valid.sum(dim=0)
            enough_detection = (
                valid_count_detection >= args.min_anchor_agreement
            )
            source_edges = torch.where(
                edge_valid.unsqueeze(-1),
                source_edges,
                torch.full_like(source_edges, torch.nan),
            )
            median_source = torch.nanmedian(source_edges, dim=0).values
            source_deviation = torch.nanmedian(
                (source_edges - median_source.unsqueeze(0)).abs(),
                dim=0,
            ).values / median_source.clamp_min(0.10)
            stable_channels = (
                source_deviation <= args.edge_source_variation
            ) & torch.isfinite(median_source)
            target_edges = edge_features[target_frame]
            signed_birth = (
                target_edges - median_source
            ) / (target_edges + median_source + 1e-6)
            birth = (
                (signed_birth - args.edge_birth_start)
                / (args.edge_birth_full - args.edge_birth_start)
            ).clamp(0.0, 1.0)
            target_presence = (
                target_edges / max(args.edge_presence, 1e-6)
            ).clamp(0.0, 1.0)
            birth = (
                torch.nan_to_num(birth)
                * target_presence
                * stable_channels.float()
                * enough_detection.unsqueeze(-1).float()
            )
            continuity = oriented_line_continuity(
                birth,
                detection_height,
                detection_width,
                args.edge_continuity,
            )
            raw_gate_detection = (
                birth * continuity.sqrt()
            ).amax(dim=-1)
            candidate_values = raw_gate_detection[
                enough_detection & (raw_gate_detection > 0)
            ]
            risk_start = finite_quantile(
                candidate_values, args.risk_quantile, 0.0
            )
            risk_full = max(
                risk_start + 1e-6,
                finite_quantile(
                    candidate_values,
                    args.risk_full_quantile,
                    risk_start + 1e-6,
                ),
            )
            gate_detection = (
                (raw_gate_detection - risk_start)
                / (risk_full - risk_start)
            ).clamp(0.0, 1.0)
            gate = F.max_pool2d(
                gate_detection.reshape(
                    1, 1, detection_height, detection_width
                ),
                kernel_size=args.detection_scale,
                stride=args.detection_scale,
            )[0, 0].to(device).reshape(-1)
            valid_count = visible_stack.sum(dim=0)
            wrong_occ_stack = torch.stack(wrong_occ_slots, dim=0)
            for slot in range(memory_slots):
                slot_support = output_valid_slots[slot].to(device)
                output_confidence[row, :, :, slot] = (
                    gate.reshape(token_height, token_width)
                    * slot_support.reshape(token_height, token_width)
                ).cpu()
            wrong_gate = torch.zeros_like(gate)
            report_gate = gate_detection.reshape(
                detection_height, detection_width
            )
        else:
            residual_stack = torch.stack(residual_slots, dim=0)
            wrong_occ_stack = torch.stack(wrong_occ_slots, dim=0)
            valid_count = visible_stack.sum(dim=0)
            enough = valid_count >= args.min_anchor_agreement
            median_residual = torch.nanmedian(residual_stack, dim=0).values
            candidate_values = median_residual[enough]
            risk_start = max(
                args.residual_floor,
                finite_quantile(
                    candidate_values,
                    args.risk_quantile,
                    args.residual_floor,
                ),
            )
            risk_full = max(
                risk_start + 1e-6,
                finite_quantile(
                    candidate_values,
                    args.risk_full_quantile,
                    risk_start + 1e-6,
                ),
            )
            residual_gate = (
                (median_residual - risk_start) / (risk_full - risk_start)
            ).clamp(0.0, 1.0)
            residual_gate = torch.nan_to_num(residual_gate) * enough.float()
            wrong_count = wrong_occ_stack.sum(dim=0)
            wrong_gate = (
                wrong_count >= args.min_wrong_occ_agreement
            ).float() * args.wrong_occ_weight
            gate = torch.maximum(residual_gate, wrong_gate)
            for slot in range(memory_slots):
                original = links.confidence[row, :, :, slot].to(device)
                slot_support = (
                    visible_stack[slot] | wrong_occ_stack[slot]
                ).reshape(token_height, token_width)
                output_confidence[row, :, :, slot] = (
                    original
                    * gate.reshape(token_height, token_width)
                    * slot_support
                ).cpu()
            report_gate = gate.reshape(token_height, token_width)

        active = gate > 0
        stat = {
            "target_frame": target_frame,
            "target_latent_index": target_time,
            "source_frames": ",".join(str(frame) for frame in source_labels),
            "risk_coverage": float(active.float().mean().item()),
            "risk_mean_all": float(gate.mean().item()),
            "risk_mean_active": (
                float(gate[active].mean().item()) if active.any() else 0.0
            ),
            "residual_threshold": risk_start,
            "residual_full": risk_full,
            "mean_anchor_agreement": float(valid_count.float().mean().item()),
            "wrong_occ_fraction": float((wrong_gate > 0).float().mean().item()),
        }
        pair_stats.append(stat)
        print(json.dumps(stat, sort_keys=True), flush=True)
        frame = video_frames[target_frame]
        raw_report_frames.append(frame)
        overlay_report_frames.append(
            overlay_risk(
                frame,
                report_gate,
                report_gate.shape[0],
                report_gate.shape[1],
                (
                    f"f{target_frame} {args.risk_mode} coverage="
                    f"{stat['risk_coverage']:.3f} "
                    f"q={risk_start:.4f}"
                ),
            )
        )

    metadata = dict(links.metadata)
    metadata.update(
        {
            "format_version": 7,
            "map_type": (
                "ufm_edge_birth_risk"
                if args.risk_mode == "edge_birth"
                else "flow_geometry_deformation_risk"
            ),
            "video": str(Path(args.video)),
            "geometry": str(Path(args.geometry)),
            "link_map": str(Path(args.link_map)),
            "target_video_frames": target_frames,
            "fixed_anchor_video_frames": fixed_anchor_frames,
            "ufm_scale": args.ufm_scale,
            "covis_threshold": args.covis_threshold,
            "min_anchor_agreement": args.min_anchor_agreement,
            "min_wrong_occ_agreement": args.min_wrong_occ_agreement,
            "risk_quantile": args.risk_quantile,
            "risk_full_quantile": args.risk_full_quantile,
            "residual_floor": args.residual_floor,
            "wrong_occ_weight": args.wrong_occ_weight,
            "risk_mode": args.risk_mode,
            "detection_token_grid": [detection_height, detection_width],
            "edge_bins": args.edge_bins,
            "edge_birth_start": args.edge_birth_start,
            "edge_birth_full": args.edge_birth_full,
            "edge_presence": args.edge_presence,
            "edge_source_variation": args.edge_source_variation,
            "edge_continuity": args.edge_continuity,
        }
    )
    output = DraftGeometryMap(
        source_time=output_source_time,
        source_index=output_source_index,
        confidence=output_confidence,
        pair_stats=pair_stats,
        metadata=metadata,
    )
    output_stem = (
        "ufm_edge_birth" if args.risk_mode == "edge_birth" else "flow_geometry_risk"
    )
    output_path = output_dir / f"{output_stem}_map.pt"
    save_draft_geometry_map(output, output_path)
    report_path = output_dir / f"{output_stem}_report.json"
    report_path.write_text(
        json.dumps({"metadata": metadata, "target_stats": pair_stats}, indent=2)
    )
    make_contact_sheet(
        raw_report_frames,
        overlay_report_frames,
        output_dir / f"{output_stem}_contact_sheet.png",
    )
    print(f"saved_map={output_path}")
    print(f"saved_report={report_path}")


if __name__ == "__main__":
    main()
