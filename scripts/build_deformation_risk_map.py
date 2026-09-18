#!/usr/bin/env python3
"""Build an automatic, risk-gated transport map from frozen draft geometry.

The ordinary geometry map answers "where is an anchor correspondence valid?".
This builder additionally asks "where does the current draft disagree with a
multi-anchor static-world prediction?".  The saved map remains compatible with
the draft-geometry Wan pipeline: its confidence field is the product of
correspondence reliability and deformation risk.
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
    _confidence_mask,
    _project_token_centres,
    _sample_scalar,
    frame_to_latent_index,
    save_draft_geometry_map,
)


def parse_indices(value: str) -> list[int]:
    """Parse comma-separated indices and inclusive start:end:stride ranges."""
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            result.append(int(item))
            continue
        pieces = [int(piece) for piece in item.split(":")]
        if len(pieces) == 2:
            start, end = pieces
            stride = 1
        elif len(pieces) == 3:
            start, end, stride = pieces
        else:
            raise ValueError(f"Invalid frame range: {item}")
        if stride <= 0:
            raise ValueError("Frame-range stride must be positive")
        result.extend(range(start, end + 1, stride))
    return sorted(set(result))


def parse_grid(value: str) -> tuple[int, int, int]:
    pieces = tuple(int(piece) for piece in value.split(","))
    if len(pieces) != 3 or min(pieces) <= 0:
        raise ValueError("--token_grid must be latent_frames,token_height,token_width")
    return pieces


def read_video_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    requested = set(indices)
    frames: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(path))
    frame_index = 0
    while capture.isOpened() and requested:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        if frame_index in requested:
            frames[frame_index] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            requested.remove(frame_index)
        frame_index += 1
    capture.release()
    if requested:
        raise ValueError(f"Video ended before frames {sorted(requested)}")
    return frames


def token_centres(
    image_height: int,
    image_width: int,
    token_height: int,
    token_width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_y, token_x = torch.meshgrid(
        (torch.arange(token_height, device=device, dtype=torch.float32) + 0.5)
        * image_height
        / token_height,
        (torch.arange(token_width, device=device, dtype=torch.float32) + 0.5)
        * image_width
        / token_width,
        indexing="ij",
    )
    pixel_y = token_y.round().long().clamp(0, image_height - 1).reshape(-1)
    pixel_x = token_x.round().long().clamp(0, image_width - 1).reshape(-1)
    return pixel_y, pixel_x


def local_depth_cv(depth: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Coefficient of variation in a pixel neighbourhood around each depth."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Depth-patch size must be a positive odd integer")
    padding = kernel_size // 2
    depth_4d = depth.reshape(-1, 1, depth.shape[-2], depth.shape[-1]).float()
    mean = F.avg_pool2d(
        depth_4d, kernel_size, stride=1, padding=padding
    )
    mean_square = F.avg_pool2d(
        depth_4d.square(), kernel_size, stride=1, padding=padding
    )
    variance = (mean_square - mean.square()).clamp_min(0.0)
    return (
        variance.sqrt() / mean.abs().clamp_min(1e-6)
    ).reshape_as(depth)


def pair_projection(
    *,
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    source_depth_cv: torch.Tensor,
    target_depth_cv: torch.Tensor,
    source_confidence: torch.Tensor,
    target_confidence: torch.Tensor,
    source_intrinsic: torch.Tensor,
    target_intrinsic: torch.Tensor,
    source_extrinsic: torch.Tensor,
    target_extrinsic: torch.Tensor,
    token_height: int,
    token_width: int,
    confidence_percentile: float,
    confidence_floor: float,
    reverse_margin: float,
    max_depth_patch_cv: float,
) -> dict[str, torch.Tensor]:
    """Project one anchor to a target without rejecting depth disagreement."""
    spatial_tokens = token_height * token_width
    (
        source_token,
        source_pixel_x,
        source_pixel_y,
        target_u,
        target_v,
        expected_target_depth,
        inside,
    ) = _project_token_centres(
        source_depth,
        source_intrinsic,
        source_extrinsic,
        target_intrinsic,
        target_extrinsic,
        token_height,
        token_width,
    )
    source_confident = _confidence_mask(
        source_confidence, confidence_percentile, confidence_floor
    )
    target_confident = _confidence_mask(
        target_confidence, confidence_percentile, confidence_floor
    )
    source_token_confident = source_confident[source_pixel_y, source_pixel_x]
    target_confident_at_projection = (
        _sample_scalar(target_confident.float(), target_u, target_v) > 0.5
    )
    source_token_interior = (
        source_depth_cv[source_pixel_y, source_pixel_x] <= max_depth_patch_cv
    )
    target_interior_at_projection = (
        _sample_scalar(target_depth_cv, target_u, target_v)
        <= max_depth_patch_cv
    )
    valid = (
        inside
        & source_token_confident
        & target_confident_at_projection
        & source_token_interior
        & target_interior_at_projection
    )
    target_x = (
        target_u * token_width / source_depth.shape[1] - 0.5
    ).round().long()
    target_y = (
        target_v * token_height / source_depth.shape[0] - 0.5
    ).round().long()
    valid &= (
        (target_x >= 0)
        & (target_x < token_width)
        & (target_y >= 0)
        & (target_y < token_height)
    )
    target_token = (
        target_y.clamp(0, token_height - 1) * token_width
        + target_x.clamp(0, token_width - 1)
    )

    z_buffer = torch.full(
        (spatial_tokens,),
        float("inf"),
        device=source_depth.device,
        dtype=torch.float32,
    )
    if valid.any():
        z_buffer.scatter_reduce_(
            0,
            target_token[valid],
            expected_target_depth[valid],
            reduce="amin",
            include_self=True,
        )
    visible = valid & (
        expected_target_depth <= z_buffer[target_token] + 1e-5
    )
    source_for_target = torch.zeros(
        spatial_tokens, dtype=torch.long, device=source_depth.device
    )
    expected_for_target = torch.full(
        (spatial_tokens,),
        float("nan"),
        dtype=torch.float32,
        device=source_depth.device,
    )
    if visible.any():
        source_for_target[target_token[visible]] = source_token[visible]
        expected_for_target[target_token[visible]] = expected_target_depth[visible]

    # A target surface that projects in front of the observed anchor depth
    # should have been visible in that anchor.  If it was absent, a newly
    # generated closer surface is suspicious rather than a valid disocclusion.
    (
        target_token_identity,
        target_pixel_x,
        target_pixel_y,
        source_u,
        source_v,
        target_depth_in_source,
        reverse_inside,
    ) = _project_token_centres(
        target_depth,
        target_intrinsic,
        target_extrinsic,
        source_intrinsic,
        source_extrinsic,
        token_height,
        token_width,
    )
    source_depth_at_reverse = _sample_scalar(source_depth, source_u, source_v)
    source_confident_at_reverse = (
        _sample_scalar(source_confident.float(), source_u, source_v) > 0.5
    )
    source_interior_at_reverse = (
        _sample_scalar(source_depth_cv, source_u, source_v)
        <= max_depth_patch_cv
    )
    target_token_confident = target_confident[target_pixel_y, target_pixel_x]
    target_token_interior = (
        target_depth_cv[target_pixel_y, target_pixel_x]
        <= max_depth_patch_cv
    )
    reverse_error = (
        target_depth_in_source - source_depth_at_reverse
    ) / (
        target_depth_in_source.abs() + source_depth_at_reverse.abs() + 1e-6
    )
    reverse_should_be_visible = (
        reverse_inside
        & source_confident_at_reverse
        & target_token_confident
        & source_interior_at_reverse
        & target_token_interior
        & (reverse_error <= -reverse_margin)
    )
    if not torch.equal(
        target_token_identity,
        torch.arange(spatial_tokens, device=source_depth.device),
    ):
        raise RuntimeError("Reverse projection changed target token ordering")

    return {
        "source_index": source_for_target,
        "expected_depth": expected_for_target,
        "valid": torch.isfinite(expected_for_target),
        "reverse_should_be_visible": reverse_should_be_visible,
        "reverse_error": reverse_error,
    }


def build_target_risk(
    projections: list[dict[str, torch.Tensor]],
    target_depth_tokens: torch.Tensor,
    *,
    anchor_consensus_threshold: float,
    risk_start: float,
    risk_full: float,
    min_anchor_agreement: int,
    min_reverse_agreement: int,
    missing_surface_weight: float,
    suspicious_closer_weight: float,
) -> dict[str, torch.Tensor]:
    expected = torch.stack(
        [projection["expected_depth"] for projection in projections], dim=0
    )
    valid = torch.stack(
        [projection["valid"] for projection in projections], dim=0
    )
    median_expected = torch.nanmedian(expected, dim=0).values
    anchor_error = (expected - median_expected.unsqueeze(0)).abs() / (
        expected.abs() + median_expected.abs().unsqueeze(0) + 1e-6
    )
    consensus = valid & (anchor_error <= anchor_consensus_threshold)
    consensus_count = consensus.sum(dim=0)
    enough_anchors = consensus_count >= min_anchor_agreement

    signed_target_error = (
        median_expected - target_depth_tokens
    ) / (
        median_expected.abs() + target_depth_tokens.abs() + 1e-6
    )
    signed_target_error = torch.nan_to_num(
        signed_target_error, nan=0.0, posinf=0.0, neginf=0.0
    )
    risk = (
        (signed_target_error.abs() - risk_start)
        / max(risk_full - risk_start, 1e-6)
    ).clamp(0.0, 1.0)
    reverse_visible = torch.stack(
        [
            projection["reverse_should_be_visible"]
            for projection in projections
        ],
        dim=0,
    )
    reverse_agreement = (reverse_visible & consensus).sum(dim=0)

    # expected < target: an anchored foreground surface disappeared.
    missing_surface = signed_target_error <= -risk_start
    # expected > target: a closer target surface is only suspicious if it
    # should also have been visible, but was absent, in several anchors.
    suspicious_closer = (
        (signed_target_error >= risk_start)
        & (reverse_agreement >= min_reverse_agreement)
    )
    repairable = enough_anchors & (missing_surface | suspicious_closer)

    consensus_fraction = consensus_count.float() / max(len(projections), 1)
    finite_anchor_error = torch.where(
        consensus, anchor_error, torch.zeros_like(anchor_error)
    )
    mean_anchor_error = finite_anchor_error.sum(dim=0) / consensus_count.clamp_min(1)
    agreement_quality = (
        1.0 - mean_anchor_error / max(anchor_consensus_threshold, 1e-6)
    ).clamp(0.0, 1.0)
    direction_weight = (
        missing_surface.float() * missing_surface_weight
        + suspicious_closer.float() * suspicious_closer_weight
    )
    gate = (
        risk
        * consensus_fraction
        * agreement_quality
        * repairable.float()
        * direction_weight
    )
    candidate_confidence = consensus.float() * gate.unsqueeze(0)
    return {
        "gate": gate,
        "candidate_confidence": candidate_confidence,
        "signed_target_error": signed_target_error,
        "consensus_count": consensus_count,
        "reverse_agreement": reverse_agreement,
        "missing_surface": missing_surface & repairable,
        "suspicious_closer": suspicious_closer & repairable,
    }


def overlay_risk(
    frame_rgb: np.ndarray,
    gate: torch.Tensor,
    token_height: int,
    token_width: int,
    label: str,
) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    gate_image = F.interpolate(
        gate.reshape(1, 1, token_height, token_width),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()
    heat = cv2.applyColorMap(
        np.clip(gate_image * 255.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    alpha = np.clip(gate_image[..., None] * 0.85, 0.0, 0.75)
    output = (
        frame_rgb.astype(np.float32) * (1.0 - alpha)
        + heat.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)
    cv2.rectangle(output, (0, 0), (width, 36), (10, 10, 10), thickness=-1)
    cv2.putText(
        output,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def make_contact_sheet(
    raw_frames: list[np.ndarray],
    overlays: list[np.ndarray],
    output: Path,
    tile_width: int = 320,
) -> None:
    rows = []
    for frames in (raw_frames, overlays):
        tiles = []
        for frame in frames:
            height, width = frame.shape[:2]
            tile_height = round(height * tile_width / width)
            tiles.append(
                cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
            )
        rows.append(np.concatenate(tiles, axis=1))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(output), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--anchor_frames", default="")
    parser.add_argument(
        "--rolling_anchor_offsets",
        default="",
        help=(
            "Optional comma-separated positive frame offsets. For every target "
            "frame t, use t-offset as a past anchor. Fixed --anchor_frames can "
            "still be supplied and are merged with the rolling anchors."
        ),
    )
    parser.add_argument("--target_frames", required=True)
    parser.add_argument("--token_grid", default="31,22,40")
    parser.add_argument("--temporal_scale", type=int, default=4)
    parser.add_argument("--memory_slots", type=int, default=3)
    parser.add_argument("--confidence_percentile", type=float, default=20.0)
    parser.add_argument("--confidence_floor", type=float, default=0.2)
    parser.add_argument("--anchor_consensus_threshold", type=float, default=0.05)
    parser.add_argument("--risk_start", type=float, default=0.03)
    parser.add_argument("--risk_full", type=float, default=0.20)
    parser.add_argument("--reverse_margin", type=float, default=0.03)
    parser.add_argument("--depth_patch_size", type=int, default=31)
    parser.add_argument("--max_depth_patch_cv", type=float, default=0.08)
    parser.add_argument("--min_anchor_agreement", type=int, default=2)
    parser.add_argument("--min_reverse_agreement", type=int, default=2)
    parser.add_argument("--missing_surface_weight", type=float, default=1.0)
    parser.add_argument("--suspicious_closer_weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.missing_surface_weight < 0 or args.suspicious_closer_weight < 0:
        raise ValueError("Direction weights must be non-negative")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor_frames = parse_indices(args.anchor_frames)
    rolling_anchor_offsets = parse_indices(args.rolling_anchor_offsets)
    if any(offset <= 0 for offset in rolling_anchor_offsets):
        raise ValueError("Rolling anchor offsets must be positive")
    if not anchor_frames and not rolling_anchor_offsets:
        raise ValueError(
            "Supply --anchor_frames, --rolling_anchor_offsets, or both"
        )
    target_frames = parse_indices(args.target_frames)
    latent_frames, token_height, token_width = parse_grid(args.token_grid)
    spatial_tokens = token_height * token_width
    geometry = torch.load(args.geometry, map_location="cpu", weights_only=False)
    frame_indices = [int(index) for index in geometry["frame_indices"]]
    frame_lookup = {frame: index for index, frame in enumerate(frame_indices)}
    anchors_by_target: dict[int, list[int]] = {}
    for target_frame in target_frames:
        rolling_anchors = [
            target_frame - offset
            for offset in rolling_anchor_offsets
            if target_frame - offset >= 0
        ]
        anchors_by_target[target_frame] = sorted(
            set(
                frame
                for frame in anchor_frames + rolling_anchors
                if frame < target_frame
            )
        )
    required_frames = set(target_frames)
    for frames in anchors_by_target.values():
        required_frames.update(frames)
    missing = required_frames.difference(frame_lookup)
    if missing:
        raise ValueError(f"Geometry payload is missing frames {sorted(missing)}")
    if not any(
        len(frames) >= args.min_anchor_agreement
        for frames in anchors_by_target.values()
    ):
        raise ValueError("No target has enough anchors for min_anchor_agreement")

    depth = geometry["depth"].float()[..., 0]
    depth_cv = local_depth_cv(depth, args.depth_patch_size)
    confidence = geometry["confidence"].float()[..., 0]
    intrinsic = geometry["intrinsic"].float()
    extrinsic = geometry["extrinsic"].float()
    image_height, image_width = depth.shape[1:]
    pixel_y, pixel_x = token_centres(
        image_height, image_width, token_height, token_width, depth.device
    )
    video_frames = read_video_frames(
        Path(args.video), sorted(required_frames)
    )

    source_time = torch.zeros(
        (latent_frames - 1, spatial_tokens, args.memory_slots), dtype=torch.long
    )
    source_index = torch.zeros_like(source_time)
    transport_confidence = torch.zeros(
        (
            latent_frames - 1,
            token_height,
            token_width,
            args.memory_slots,
        ),
        dtype=torch.float32,
    )
    pair_stats: list[dict[str, float | int | str]] = []
    raw_report_frames: list[np.ndarray] = []
    overlay_report_frames: list[np.ndarray] = []
    used_anchor_frames: set[int] = set()

    for target_frame in target_frames:
        target_sequence = frame_lookup[target_frame]
        target_time = frame_to_latent_index(
            target_frame, args.temporal_scale, latent_frames
        )
        available_anchors = [
            frame
            for frame in anchors_by_target[target_frame]
            if frame_to_latent_index(
                frame, args.temporal_scale, latent_frames
            )
            < target_time
        ]
        if len(available_anchors) < args.min_anchor_agreement:
            continue
        available_anchors = available_anchors[-args.memory_slots :]
        projections = []
        for source_frame in available_anchors:
            source_sequence = frame_lookup[source_frame]
            projections.append(
                pair_projection(
                    source_depth=depth[source_sequence],
                    target_depth=depth[target_sequence],
                    source_depth_cv=depth_cv[source_sequence],
                    target_depth_cv=depth_cv[target_sequence],
                    source_confidence=confidence[source_sequence],
                    target_confidence=confidence[target_sequence],
                    source_intrinsic=intrinsic[source_sequence],
                    target_intrinsic=intrinsic[target_sequence],
                    source_extrinsic=extrinsic[source_sequence],
                    target_extrinsic=extrinsic[target_sequence],
                    token_height=token_height,
                    token_width=token_width,
                    confidence_percentile=args.confidence_percentile,
                    confidence_floor=args.confidence_floor,
                    reverse_margin=args.reverse_margin,
                    max_depth_patch_cv=args.max_depth_patch_cv,
                )
            )

        target_depth_tokens = depth[target_sequence, pixel_y, pixel_x]
        risk = build_target_risk(
            projections,
            target_depth_tokens,
            anchor_consensus_threshold=args.anchor_consensus_threshold,
            risk_start=args.risk_start,
            risk_full=args.risk_full,
            min_anchor_agreement=args.min_anchor_agreement,
            min_reverse_agreement=args.min_reverse_agreement,
            missing_surface_weight=args.missing_surface_weight,
            suspicious_closer_weight=args.suspicious_closer_weight,
        )
        gate = risk["gate"]
        for slot, (source_frame, projection) in enumerate(
            zip(available_anchors, projections)
        ):
            if risk["candidate_confidence"][slot].gt(0).any():
                used_anchor_frames.add(source_frame)
            source_time[target_time - 1, :, slot] = frame_to_latent_index(
                source_frame, args.temporal_scale, latent_frames
            )
            source_index[target_time - 1, :, slot] = projection["source_index"]
            transport_confidence[
                target_time - 1, :, :, slot
            ] = risk["candidate_confidence"][slot].reshape(
                token_height, token_width
            )

        active = gate > 0
        stat = {
            "target_frame": target_frame,
            "target_latent_index": target_time,
            "anchor_frames": ",".join(str(frame) for frame in available_anchors),
            "risk_coverage": float(active.float().mean().item()),
            "risk_mean_all": float(gate.mean().item()),
            "risk_mean_active": (
                float(gate[active].mean().item()) if active.any() else 0.0
            ),
            "missing_surface_fraction": float(
                risk["missing_surface"].float().mean().item()
            ),
            "suspicious_closer_fraction": float(
                risk["suspicious_closer"].float().mean().item()
            ),
            "mean_anchor_agreement": float(
                risk["consensus_count"].float().mean().item()
            ),
            "mean_reverse_agreement": float(
                risk["reverse_agreement"].float().mean().item()
            ),
        }
        pair_stats.append(stat)
        print(json.dumps(stat, sort_keys=True), flush=True)
        frame = video_frames[target_frame]
        raw_report_frames.append(frame)
        overlay_report_frames.append(
            overlay_risk(
                frame,
                gate,
                token_height,
                token_width,
                (
                    f"f{target_frame} risk coverage={stat['risk_coverage']:.3f} "
                    f"mean={stat['risk_mean_all']:.4f}"
                ),
            )
        )

    transport = DraftGeometryMap(
        source_time=source_time,
        source_index=source_index,
        confidence=transport_confidence,
        pair_stats=pair_stats,
        metadata={
            "format_version": 2,
            "map_type": "automatic_deformation_risk",
            "video": str(Path(args.video)),
            "geometry": str(Path(args.geometry)),
            "anchor_video_frames": sorted(used_anchor_frames),
            "fixed_anchor_video_frames": anchor_frames,
            "rolling_anchor_offsets": rolling_anchor_offsets,
            "target_video_frames": target_frames,
            "token_grid": list((latent_frames, token_height, token_width)),
            "temporal_scale": args.temporal_scale,
            "memory_slots": args.memory_slots,
            "confidence_percentile": args.confidence_percentile,
            "confidence_floor": args.confidence_floor,
            "anchor_consensus_threshold": args.anchor_consensus_threshold,
            "risk_start": args.risk_start,
            "risk_full": args.risk_full,
            "reverse_margin": args.reverse_margin,
            "depth_patch_size": args.depth_patch_size,
            "max_depth_patch_cv": args.max_depth_patch_cv,
            "min_anchor_agreement": args.min_anchor_agreement,
            "min_reverse_agreement": args.min_reverse_agreement,
            "missing_surface_weight": args.missing_surface_weight,
            "suspicious_closer_weight": args.suspicious_closer_weight,
        },
    )
    save_draft_geometry_map(transport, output_dir / "deformation_risk_map.pt")
    with (output_dir / "deformation_risk_report.json").open("w") as handle:
        json.dump(
            {"metadata": transport.metadata, "target_stats": pair_stats},
            handle,
            indent=2,
        )
    make_contact_sheet(
        raw_report_frames,
        overlay_report_frames,
        output_dir / "deformation_risk_contact_sheet.png",
    )
    print(f"saved_map={output_dir / 'deformation_risk_map.pt'}")
    print(
        "saved_contact_sheet="
        f"{output_dir / 'deformation_risk_contact_sheet.png'}"
    )


if __name__ == "__main__":
    main()
