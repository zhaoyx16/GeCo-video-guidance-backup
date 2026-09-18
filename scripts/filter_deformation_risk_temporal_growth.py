#!/usr/bin/env python3
"""Keep deformation-risk bursts that persist and grow over time.

Transient depth/pose errors often spike for one or two frames around a turn or
disocclusion. A structural deformation is more likely to accumulate across
several consecutive target frames. This post-process keeps only windows whose
mean risk is approximately monotonic and grows by a minimum ratio.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "external" / "guidance_wan"))

from draft_geometry_map import DraftGeometryMap, load_draft_geometry_map, save_draft_geometry_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_map", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--min_growth_ratio", type=float, default=2.0)
    parser.add_argument(
        "--max_step_drop",
        type=float,
        default=0.0,
        help="Maximum fractional drop allowed between consecutive risk values.",
    )
    parser.add_argument("--min_final_severity", type=float, default=0.0)
    args = parser.parse_args()

    if args.window < 2:
        raise ValueError("--window must be at least 2")
    if args.min_growth_ratio < 1.0:
        raise ValueError("--min_growth_ratio must be at least 1")
    if not 0.0 <= args.max_step_drop < 1.0:
        raise ValueError("--max_step_drop must lie in [0, 1)")
    if args.min_final_severity < 0.0:
        raise ValueError("--min_final_severity must be non-negative")

    transport = load_draft_geometry_map(args.input_map, "cpu")
    confidence = transport.confidence.float()
    per_target = confidence.amax(dim=-1)
    severity = per_target.mean(dim=(1, 2))
    coverage = per_target.gt(0).float().mean(dim=(1, 2))
    selected = torch.zeros_like(severity, dtype=torch.bool)
    qualifying_windows: list[dict[str, object]] = []

    for end in range(args.window - 1, len(severity)):
        start = end - args.window + 1
        values = severity[start : end + 1]
        positive = bool(values[0] > 0)
        persistent = bool(
            torch.all(values[1:] >= values[:-1] * (1.0 - args.max_step_drop))
        )
        grows_enough = bool(values[-1] >= values[0] * args.min_growth_ratio)
        strong_enough = bool(values[-1] >= args.min_final_severity)
        if positive and persistent and grows_enough and strong_enough:
            selected[start : end + 1] = True
            qualifying_windows.append(
                {
                    "start_index": start,
                    "end_index": end,
                    "severity": [float(value) for value in values],
                }
            )

    filtered_confidence = confidence.clone()
    filtered_confidence[~selected] = 0
    metadata = dict(transport.metadata)
    temporal_scale = int(metadata.get("temporal_scale", 4))
    selected_video_frames = [
        (index + 1) * temporal_scale
        for index, keep in enumerate(selected.tolist())
        if keep
    ]
    metadata.update(
        {
            "temporal_filter": "persistent_growth",
            "temporal_growth_window": args.window,
            "temporal_growth_min_ratio": args.min_growth_ratio,
            "temporal_growth_max_step_drop": args.max_step_drop,
            "temporal_growth_min_final_severity": args.min_final_severity,
            "temporal_growth_selected_video_frames": selected_video_frames,
            "unfiltered_map": str(Path(args.input_map)),
        }
    )

    pair_stats = []
    for stat in transport.pair_stats:
        stat_copy = dict(stat)
        latent_index = int(stat_copy.get("target_latent_index", -1))
        stat_copy["temporal_growth_selected"] = (
            1 <= latent_index <= len(selected) and bool(selected[latent_index - 1])
        )
        pair_stats.append(stat_copy)

    filtered = DraftGeometryMap(
        source_time=transport.source_time,
        source_index=transport.source_index,
        confidence=filtered_confidence,
        pair_stats=pair_stats,
        metadata=metadata,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_map = output_dir / "deformation_risk_map.pt"
    save_draft_geometry_map(filtered, output_map)

    timeline = []
    for index, (severity_value, coverage_value, keep) in enumerate(
        zip(severity.tolist(), coverage.tolist(), selected.tolist())
    ):
        timeline.append(
            {
                "target_latent_index": index + 1,
                "target_video_frame": (index + 1) * temporal_scale,
                "severity_mean_all": severity_value,
                "coverage": coverage_value,
                "selected": keep,
            }
        )
    report = {
        "input_map": str(Path(args.input_map)),
        "output_map": str(output_map),
        "settings": {
            "window": args.window,
            "min_growth_ratio": args.min_growth_ratio,
            "max_step_drop": args.max_step_drop,
            "min_final_severity": args.min_final_severity,
        },
        "selected_video_frames": selected_video_frames,
        "qualifying_windows": qualifying_windows,
        "timeline": timeline,
    }
    report_path = output_dir / "temporal_growth_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
