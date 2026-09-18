#!/usr/bin/env python3
"""Keep spatially coherent deformation-risk regions that persist in time."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "external" / "guidance_wan"))

from draft_geometry_map import DraftGeometryMap, load_draft_geometry_map, save_draft_geometry_map


def persistent_runs(selected: list[bool], min_run: int) -> list[bool]:
    keep = [False] * len(selected)
    start = 0
    while start < len(selected):
        if not selected[start]:
            start += 1
            continue
        end = start + 1
        while end < len(selected) and selected[end]:
            end += 1
        if end - start >= min_run:
            keep[start:end] = [True] * (end - start)
        start = end
    return keep


def extend_correspondence_halo(
    source_time: torch.Tensor,
    source_index: torch.Tensor,
    confidence: torch.Tensor,
    target_index: int,
    radius: int,
    decay: float,
) -> float:
    """Extend a coherent component using its nearest local correspondence.

    The source displacement, rather than the absolute source pixel, is copied
    into the halo. This preserves a locally smooth correspondence field when
    the target position changes.
    """
    if radius <= 0:
        return float(confidence[target_index].amax(dim=-1).gt(0).float().mean())

    token_height, token_width = confidence.shape[1:3]
    confidence_grid = confidence[target_index]
    support = confidence_grid.amax(dim=-1).gt(0)
    support_y, support_x = torch.where(support)
    if support_y.numel() == 0:
        return 0.0

    grid_y, grid_x = torch.meshgrid(
        torch.arange(token_height),
        torch.arange(token_width),
        indexing="ij",
    )
    target_coordinates = torch.stack([grid_y, grid_x], dim=-1).reshape(-1, 2)
    support_coordinates = torch.stack([support_y, support_x], dim=-1)
    squared_distance = (
        target_coordinates[:, None].float() - support_coordinates[None].float()
    ).square().sum(dim=-1)
    nearest_distance_squared, nearest_support_index = squared_distance.min(dim=-1)
    nearest_distance = nearest_distance_squared.sqrt()
    halo = nearest_distance.le(float(radius))
    if not halo.any():
        return float(support.float().mean())

    target_coordinates = target_coordinates[halo]
    owner_coordinates = support_coordinates[nearest_support_index[halo]]
    owner_y, owner_x = owner_coordinates.unbind(dim=-1)
    target_y, target_x = target_coordinates.unbind(dim=-1)

    memory_slots = confidence.shape[-1]
    source_time_grid = source_time[target_index].reshape(
        token_height, token_width, memory_slots
    )
    source_index_grid = source_index[target_index].reshape(
        token_height, token_width, memory_slots
    )
    owner_confidence = confidence_grid[owner_y, owner_x]
    owner_source_time = source_time_grid[owner_y, owner_x]
    owner_source_index = source_index_grid[owner_y, owner_x]

    owner_source_y = torch.div(
        owner_source_index,
        token_width,
        rounding_mode="floor",
    )
    owner_source_x = owner_source_index.remainder(token_width)
    source_offset_y = owner_source_y - owner_y[:, None]
    source_offset_x = owner_source_x - owner_x[:, None]
    halo_source_y = (target_y[:, None] + source_offset_y).clamp(
        0, token_height - 1
    )
    halo_source_x = (target_x[:, None] + source_offset_x).clamp(
        0, token_width - 1
    )

    distance_weight = decay ** nearest_distance[halo]
    confidence_grid[target_y, target_x] = (
        owner_confidence * distance_weight[:, None]
    )
    source_time_grid[target_y, target_x] = owner_source_time
    source_index_grid[target_y, target_x] = (
        halo_source_y * token_width + halo_source_x
    )

    confidence[target_index] = confidence_grid
    source_time[target_index] = source_time_grid.reshape(
        token_height * token_width, memory_slots
    )
    source_index[target_index] = source_index_grid.reshape(
        token_height * token_width, memory_slots
    )
    return float(confidence_grid.amax(dim=-1).gt(0).float().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_map", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_component_fraction", type=float, default=0.02)
    parser.add_argument("--min_consecutive_targets", type=int, default=2)
    parser.add_argument("--min_mean_anchor_agreement", type=float, default=0.0)
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=4)
    parser.add_argument(
        "--selected_support_mode",
        choices=("largest_component", "full_risk"),
        default="largest_component",
        help=(
            "Use only the selected largest component, or restore every "
            "geometry-validated risk correspondence at selected timesteps."
        ),
    )
    parser.add_argument("--correspondence_halo_radius", type=int, default=0)
    parser.add_argument("--correspondence_halo_decay", type=float, default=0.85)
    args = parser.parse_args()

    if not 0.0 <= args.min_component_fraction <= 1.0:
        raise ValueError("--min_component_fraction must lie in [0, 1]")
    if args.min_consecutive_targets < 1:
        raise ValueError("--min_consecutive_targets must be positive")
    if args.min_mean_anchor_agreement < 0:
        raise ValueError("--min_mean_anchor_agreement must be non-negative")
    if args.correspondence_halo_radius < 0:
        raise ValueError("--correspondence_halo_radius must be non-negative")
    if not 0.0 < args.correspondence_halo_decay <= 1.0:
        raise ValueError("--correspondence_halo_decay must lie in (0, 1]")
    if args.selected_support_mode == "full_risk" and args.correspondence_halo_radius:
        raise ValueError(
            "--correspondence_halo_radius is only valid with "
            "--selected_support_mode=largest_component"
        )

    transport = load_draft_geometry_map(args.input_map, "cpu")
    confidence = transport.confidence.float()
    support = confidence.amax(dim=-1).gt(0)
    token_height, token_width = support.shape[-2:]
    spatial_tokens = token_height * token_width

    largest_masks: list[torch.Tensor] = []
    component_stats: list[dict[str, float | int | bool]] = []
    qualifies: list[bool] = []
    anchor_agreement_by_latent = {
        int(stat["target_latent_index"]): float(stat["mean_anchor_agreement"])
        for stat in transport.pair_stats
        if "target_latent_index" in stat and "mean_anchor_agreement" in stat
    }
    for target_index, target_support in enumerate(support):
        support_np = target_support.numpy().astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            support_np,
            connectivity=args.connectivity,
        )
        if num_labels <= 1:
            largest_label = 0
            largest_area = 0
            largest_mask = torch.zeros_like(target_support)
        else:
            component_areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = int(component_areas.argmax()) + 1
            largest_area = int(component_areas.max())
            largest_mask = torch.from_numpy(labels == largest_label)
        largest_fraction = largest_area / spatial_tokens
        target_latent_index = target_index + 1
        mean_anchor_agreement = anchor_agreement_by_latent.get(
            target_latent_index,
            float("inf"),
        )
        component_qualifies = (
            largest_fraction >= args.min_component_fraction
        )
        geometry_qualifies = (
            mean_anchor_agreement >= args.min_mean_anchor_agreement
        )
        target_qualifies = (
            component_qualifies
            and geometry_qualifies
        )
        largest_masks.append(largest_mask)
        qualifies.append(target_qualifies)
        component_stats.append(
            {
                "target_latent_index": target_latent_index,
                "support_fraction": float(target_support.float().mean()),
                "largest_component_tokens": largest_area,
                "largest_component_fraction": largest_fraction,
                "mean_anchor_agreement": mean_anchor_agreement,
                "geometry_qualifies": geometry_qualifies,
                "spatially_qualifies": component_qualifies,
                "eligible_before_temporal_filter": target_qualifies,
            }
        )

    keep_targets = persistent_runs(qualifies, args.min_consecutive_targets)
    filtered_source_time = transport.source_time.clone()
    filtered_source_index = transport.source_index.clone()
    filtered_confidence = torch.zeros_like(confidence)
    for target_index, keep in enumerate(keep_targets):
        if keep:
            if args.selected_support_mode == "full_risk":
                filtered_confidence[target_index] = confidence[target_index]
            else:
                filtered_confidence[target_index] = (
                    confidence[target_index]
                    * largest_masks[target_index].unsqueeze(-1)
                )
            halo_fraction = extend_correspondence_halo(
                filtered_source_time,
                filtered_source_index,
                filtered_confidence,
                target_index,
                args.correspondence_halo_radius,
                args.correspondence_halo_decay,
            )
        else:
            halo_fraction = 0.0
        component_stats[target_index]["temporally_qualifies"] = keep
        component_stats[target_index]["selected_support_fraction"] = float(
            filtered_confidence[target_index]
            .amax(dim=-1)
            .gt(0)
            .float()
            .mean()
        )
        component_stats[target_index]["halo_support_fraction"] = halo_fraction

    metadata = dict(transport.metadata)
    temporal_scale = int(metadata.get("temporal_scale", 4))
    selected_video_frames = [
        (index + 1) * temporal_scale
        for index, keep in enumerate(keep_targets)
        if keep
    ]
    metadata.update(
        {
            "spatial_filter": (
                "persistent_component_temporal_gate_full_risk_support"
                if args.selected_support_mode == "full_risk"
                else "largest_persistent_component"
            ),
            "component_min_fraction": args.min_component_fraction,
            "component_min_consecutive_targets": args.min_consecutive_targets,
            "component_min_mean_anchor_agreement": args.min_mean_anchor_agreement,
            "component_connectivity": args.connectivity,
            "component_selected_support_mode": args.selected_support_mode,
            "component_correspondence_halo_radius": args.correspondence_halo_radius,
            "component_correspondence_halo_decay": args.correspondence_halo_decay,
            "component_selected_video_frames": selected_video_frames,
            "unfiltered_map": str(Path(args.input_map)),
        }
    )

    pair_stats = []
    for stat in transport.pair_stats:
        stat_copy = dict(stat)
        latent_index = int(stat_copy.get("target_latent_index", -1))
        stat_copy["component_selected"] = (
            1 <= latent_index <= len(keep_targets)
            and keep_targets[latent_index - 1]
        )
        pair_stats.append(stat_copy)

    filtered = DraftGeometryMap(
        source_time=filtered_source_time,
        source_index=filtered_source_index,
        confidence=filtered_confidence,
        pair_stats=pair_stats,
        metadata=metadata,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_map = output_dir / "deformation_risk_map.pt"
    save_draft_geometry_map(filtered, output_map)

    report = {
        "input_map": str(Path(args.input_map)),
        "output_map": str(output_map),
        "settings": {
            "min_component_fraction": args.min_component_fraction,
            "min_consecutive_targets": args.min_consecutive_targets,
            "min_mean_anchor_agreement": args.min_mean_anchor_agreement,
            "connectivity": args.connectivity,
            "selected_support_mode": args.selected_support_mode,
            "correspondence_halo_radius": args.correspondence_halo_radius,
            "correspondence_halo_decay": args.correspondence_halo_decay,
        },
        "selected_video_frames": selected_video_frames,
        "targets": component_stats,
    }
    report_path = output_dir / "component_filter_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
