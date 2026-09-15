#!/usr/bin/env python3
"""Render the preregistered and exploratory C2F P0 forensic evidence."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MOTION_ORDER = ["forward", "forward_left", "forward_right", "lateral_left", "lateral_right"]
MOTION_COLORS = {
    "forward": "#2563eb",
    "forward_left": "#0f766e",
    "forward_right": "#ca8a04",
    "lateral_left": "#9333ea",
    "lateral_right": "#dc2626",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    return float(np.corrcoef(rankdata(np.asarray(x)), rankdata(np.asarray(y)))[0, 1])


def history_margin_q90(sample_path: Path) -> dict[str, float]:
    values = defaultdict(list)
    with gzip.open(sample_path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["replay_mode"] == "baseline_observe":
                values[row["case_id"]].append(float(row["history_confidence_margin"]))
    return {case_id: float(np.quantile(case_values, 0.9)) for case_id, case_values in values.items()}


def aggregate_step(rows: list[dict[str, str]], predictor: str, step: int) -> tuple[list[float], list[float]]:
    grouped = defaultdict(list)
    outcomes = {}
    for row in rows:
        if row["replay_mode"] != "baseline_observe" or int(row["step"]) != step:
            continue
        grouped[row["case_id"]].append(float(row[predictor]))
        outcomes[row["case_id"]] = float(row["met3r_one_second_delta"])
    case_ids = sorted(grouped)
    return [float(np.mean(grouped[case_id])) for case_id in case_ids], [outcomes[case_id] for case_id in case_ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    analysis_dir = args.analysis_dir.resolve()
    output = (args.output or analysis_dir / "P0_FORENSICS_OVERVIEW.png").resolve()
    report = json.loads((analysis_dir / "P0_REPORT.json").read_text(encoding="utf-8"))
    scenes = read_csv(analysis_dir / "SCENE_FEATURES.csv")
    layer_steps = read_csv(analysis_dir / "LAYER_STEP_FEATURES.csv")
    source_bins = read_csv(analysis_dir / "SOURCE_RISK_BINS_BY_SCENE.csv")
    q90_by_case = history_margin_q90(analysis_dir / "INTERVENTION_SAMPLES.csv.gz")

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # A: preregistered effects plus the explicitly post-hoc candidate.
    ax = axes[0, 0]
    names = report["primary_predictors"] + ["history_confidence_margin_mean"]
    labels = [
        "Source retrieval failure",
        "Path-cycle error",
        "Multi-lag velocity disagreement",
        "Long-lag fraction",
        "History confidence margin*",
    ]
    y_positions = np.arange(len(names))[::-1]
    for y_position, name in zip(y_positions, names):
        item = report["associations"][name]
        rho = item["spearman_rho"]
        ci80 = item["bootstrap_ci80"]
        ci95 = item["bootstrap_ci95"]
        color = "#d97706" if name == "history_confidence_margin_mean" else "#1d4ed8"
        ax.plot(ci95, [y_position, y_position], color=color, alpha=0.3, linewidth=2)
        ax.plot(ci80, [y_position, y_position], color=color, linewidth=5)
        ax.scatter([rho], [y_position], color=color, s=36, zorder=3)
    ax.axvline(0, color="#525252", linewidth=1)
    ax.set_yticks(y_positions, labels)
    ax.set_xlim(-0.8, 0.8)
    ax.set_xlabel("Spearman rho with MEt3R delta (positive = harmful)")
    ax.set_title("A. Preregistered predictors do not explain C2F outcomes")
    ax.grid(axis="x", alpha=0.2)
    ax.text(0.01, -0.17, "* Secondary mean statistic; tail result in panel B is post-hoc.", transform=ax.transAxes)

    # B: post-hoc q90 history-margin candidate, with every scene visible.
    ax = axes[0, 1]
    x = np.asarray([q90_by_case[row["case_id"]] for row in scenes])
    y = np.asarray([float(row["met3r_one_second_delta"]) for row in scenes])
    for row, x_value, y_value in zip(scenes, x, y):
        motion = row["motion_stratum"]
        ax.scatter(x_value, y_value, color=MOTION_COLORS[motion], s=42, alpha=0.9)
    order = np.argsort(x)
    linear = np.polyfit(x, y, 1)
    ax.plot(x[order], np.polyval(linear, x[order]), color="#111827", linestyle="--", linewidth=1)
    ax.axhline(0, color="#525252", linewidth=1)
    ax.set_xlabel("Scene q90 of top1-top2 history confidence margin")
    ax.set_ylabel("C2F - baseline MEt3R-1s")
    ax.set_title(f"B. Post-hoc candidate: decisive history retrieval (rho={spearman(x, y):.3f})")
    ax.grid(alpha=0.2)
    for motion in MOTION_ORDER:
        ax.scatter([], [], color=MOTION_COLORS[motion], label=motion)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    # C: association over denoising steps, always retaining scenes as units.
    ax = axes[1, 0]
    profiles = {
        "History margin": "history_confidence_margin_mean",
        "Source retrieval failure": "source_unreliable_active_fraction",
        "Path-cycle error": "path_cycle_error_mean",
        "Velocity disagreement": "velocity_rms_mean",
    }
    colors = ["#d97706", "#2563eb", "#0f766e", "#9333ea"]
    steps = list(range(20, 30))
    for (label, predictor), color in zip(profiles.items(), colors):
        rhos = []
        for step in steps:
            step_x, step_y = aggregate_step(layer_steps, predictor, step)
            rhos.append(spearman(step_x, step_y))
        ax.plot(steps, rhos, marker="o", markersize=3, color=color, label=label)
    ax.axhline(0, color="#525252", linewidth=1)
    ax.set_xticks(steps)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Scene-level Spearman rho with MEt3R delta")
    ax.set_title("C. Only history-margin association strengthens over time")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)

    # D: the locked source-risk bins show no monotonic winner/loser separation.
    ax = axes[1, 1]
    baseline_bins = [row for row in source_bins if row["replay_mode"] == "baseline_observe"]
    bin_order = list(dict.fromkeys(row["source_risk_bin"] for row in baseline_bins))
    winner_means = []
    loser_means = []
    for bin_name in bin_order:
        winner_means.append(
            np.mean([
                float(row["sample_fraction"])
                for row in baseline_bins
                if row["source_risk_bin"] == bin_name and row["label"] == "winner"
            ])
        )
        loser_means.append(
            np.mean([
                float(row["sample_fraction"])
                for row in baseline_bins
                if row["source_risk_bin"] == bin_name and row["label"] == "loser"
            ])
        )
    positions = np.arange(len(bin_order))
    width = 0.38
    ax.bar(positions - width / 2, winner_means, width, label="Winner", color="#15803d")
    ax.bar(positions + width / 2, loser_means, width, label="Loser", color="#b91c1c")
    compact_labels = [
        "condition",
        "zero",
        "(0,.2]",
        "(.2,.4]",
        "(.4,.6]",
        "(.6,.8]",
        "(.8,1]",
    ]
    ax.set_xticks(positions, compact_labels, rotation=25, ha="right")
    ax.set_ylabel("Mean fraction of sampled interventions")
    ax.set_title("D. Source-reliability bins are non-monotonic")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)

    fig.suptitle("C2F P0 failure forensics on seen Dev25 (mechanism discovery only)", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
