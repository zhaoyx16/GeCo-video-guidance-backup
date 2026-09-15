#!/usr/bin/env python3
"""Analyze preregistered C2F P0 scene- and intervention-level diagnostics."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


PRIMARY_PREDICTORS = [
    "source_unreliable_active_fraction",
    "path_cycle_error_mean",
    "velocity_rms_mean",
    "lag_gt1_fraction",
]

SECONDARY_PREDICTORS = [
    "active_fraction",
    "confidence_mean",
    "similarity_mean",
    "fine_margin_mean",
    "coarse_margin_mean",
    "history_confidence_margin_mean",
    "source_previous_confidence_mean",
    "raw_candidate_count_mean",
    "gated_candidate_count_mean",
    "relative_value_residual_mean",
    "source_to_context_norm_ratio_mean",
    "spatial_displacement_mean",
    "conditioning_source_active_fraction",
    "candidate_raw_active_fraction",
    "candidate_gated_active_fraction",
    "candidate_reciprocal_fraction",
    "candidate_coarse_cycle_error_mean",
    "candidate_path_cycle_error_lag23_mean",
]

SELECTED_METRICS = {
    "confidence": "confidence_mean",
    "similarity": "similarity_mean",
    "fine_margin": "fine_margin_mean",
    "coarse_margin": "coarse_margin_mean",
    "path_cycle_error": "path_cycle_error_mean",
    "velocity_rms": "velocity_rms_mean",
    "history_confidence_margin": "history_confidence_margin_mean",
    "source_previous_confidence": "source_previous_confidence_mean",
    "raw_candidate_count": "raw_candidate_count_mean",
    "gated_candidate_count": "gated_candidate_count_mean",
    "relative_value_residual": "relative_value_residual_mean",
    "relative_update": "relative_update_mean",
    "source_to_context_norm_ratio": "source_to_context_norm_ratio_mean",
    "spatial_displacement": "spatial_displacement_mean",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def weighted_mean(pairs: list[tuple[float, int]]) -> float:
    denominator = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / denominator if denominator else float("nan")


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


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else float("nan")


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def correlation_summary(
    x: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    near_tie: np.ndarray,
    seed: int,
    permutations: int,
    bootstraps: int,
) -> dict:
    observed = spearman(x, y)
    rng = np.random.default_rng(seed)
    permutation_values = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        permutation_values[index] = spearman(x, y[rng.permutation(len(y))])
    permutation_p = (
        float((np.count_nonzero(np.abs(permutation_values) >= abs(observed)) + 1) / (permutations + 1))
        if math.isfinite(observed)
        else float("nan")
    )

    bootstrap_values = []
    for _ in range(bootstraps):
        sample = rng.integers(0, len(y), size=len(y))
        value = spearman(x[sample], y[sample])
        if math.isfinite(value):
            bootstrap_values.append(value)
    bootstrap_array = np.asarray(bootstrap_values, dtype=np.float64)

    loo = []
    for index in range(len(y)):
        keep = np.arange(len(y)) != index
        value = spearman(x[keep], y[keep])
        if math.isfinite(value):
            loo.append(value)
    same_sign = (
        float(np.mean(np.sign(loo) == np.sign(observed)))
        if loo and observed != 0 and math.isfinite(observed)
        else float("nan")
    )

    x_demeaned = x.copy()
    y_demeaned = y.copy()
    for stratum in np.unique(strata):
        indices = strata == stratum
        x_demeaned[indices] -= x[indices].mean()
        y_demeaned[indices] -= y[indices].mean()

    no_ties = ~near_tie
    no_largest = np.ones(len(y), dtype=bool)
    no_largest[np.argmax(np.abs(y))] = False
    return {
        "spearman_rho": finite_or_none(observed),
        "permutation_two_sided_p": finite_or_none(permutation_p),
        "bootstrap_ci80": [
            finite_or_none(percentile(bootstrap_array, 0.1)),
            finite_or_none(percentile(bootstrap_array, 0.9)),
        ],
        "bootstrap_ci95": [
            finite_or_none(percentile(bootstrap_array, 0.025)),
            finite_or_none(percentile(bootstrap_array, 0.975)),
        ],
        "leave_one_out_same_sign_fraction": finite_or_none(same_sign),
        "motion_stratum_demeaned_spearman": finite_or_none(spearman(x_demeaned, y_demeaned)),
        "exclude_near_ties_spearman": finite_or_none(spearman(x[no_ties], y[no_ties])),
        "exclude_largest_abs_outcome_spearman": finite_or_none(spearman(x[no_largest], y[no_largest])),
    }


def replay_path(root: Path, mode: str, case_id: str, seed: int) -> Path:
    return root / mode / "diagnostic" / "through_step_29" / case_id / f"seed_{seed}" / "replay.json"


def preintervention_record(record: dict) -> dict:
    matches = [
        item for item in record["diagnostics"]["records"]
        if item["step"] == 20 and item["layer"] == 10
    ]
    if len(matches) != 1:
        raise RuntimeError(f"missing unique step-20/layer-10 record for {record['case_id']}")
    cleaned = json.loads(json.dumps(matches[0]))
    cleaned["selected"].pop("relative_update", None)
    cleaned["samples"].pop("relative_update", None)
    return cleaned


def aggregate_scene(record: dict) -> tuple[dict, list[dict], list[dict]]:
    diagnostics = record["diagnostics"]
    layer_step_records = diagnostics["records"]
    expected = {(step, layer) for step in range(20, 30) for layer in (10, 15, 20)}
    actual = {(item["step"], item["layer"]) for item in layer_step_records}
    if actual != expected or len(layer_step_records) != len(expected):
        raise RuntimeError(f"incomplete diagnostic grid for {record['case_id']}")

    scene = {
        "case_id": record["case_id"],
        "motion_stratum": record["primary_outcome"]["motion_stratum"],
        "label": record["primary_outcome"]["label"],
        "near_tie": bool(record["primary_outcome"]["near_tie"]),
        "met3r_one_second_delta": float(record["primary_outcome"]["met3r_one_second_delta"]),
        "wall_seconds": float(record["wall_seconds"]),
        "peak_memory_mib": float(record["peak_memory_mib"]),
    }

    active_count = sum(item["active_count"] for item in layer_step_records)
    target_count = sum(item["target_token_count"] for item in layer_step_records)
    scene["active_fraction"] = active_count / target_count
    scene["source_unreliable_active_fraction"] = sum(
        item["source_unreliable_active_fraction"] * item["active_count"] for item in layer_step_records
    ) / max(active_count, 1)
    scene["conditioning_source_active_fraction"] = sum(
        item["conditioning_source_active_fraction"] * item["active_count"] for item in layer_step_records
    ) / max(active_count, 1)

    lag_counts = defaultdict(int)
    for item in layer_step_records:
        for lag, count in item["selected_lag_histogram"].items():
            lag_counts[int(lag)] += int(count)
    scene["lag1_fraction"] = lag_counts[1] / max(active_count, 1)
    scene["lag_gt1_fraction"] = (lag_counts[2] + lag_counts[3]) / max(active_count, 1)
    scene["lag3_fraction"] = lag_counts[3] / max(active_count, 1)

    for source_name, output_name in SELECTED_METRICS.items():
        pairs = []
        for item in layer_step_records:
            stats = item["selected"][source_name]
            if stats["mean"] is not None:
                pairs.append((float(stats["mean"]), int(stats["count"])))
        scene[output_name] = weighted_mean(pairs)

    candidate_valid_count = 0
    candidate_raw_active_count = 0.0
    candidate_gated_active_count = 0.0
    candidate_reciprocal_count = 0.0
    candidate_coarse_cycle_pairs = []
    candidate_path_cycle_lag23_pairs = []
    for item in layer_step_records:
        for lag_record in item["candidate_by_lag"]:
            valid_count = int(lag_record["valid_count"])
            candidate_valid_count += valid_count
            candidate_raw_active_count += lag_record["raw_active_fraction"] * valid_count
            candidate_gated_active_count += lag_record["gated_active_fraction"] * valid_count
            candidate_reciprocal_count += lag_record["reciprocal_fraction"] * valid_count
            coarse_cycle = lag_record["coarse_cycle_error"]
            if coarse_cycle["mean"] is not None:
                candidate_coarse_cycle_pairs.append((float(coarse_cycle["mean"]), int(coarse_cycle["count"])))
            path_cycle = lag_record["path_cycle_error"]
            if lag_record["lag"] > 1 and path_cycle["mean"] is not None:
                candidate_path_cycle_lag23_pairs.append(
                    (float(path_cycle["mean"]), int(path_cycle["count"]))
                )
    scene["candidate_raw_active_fraction"] = candidate_raw_active_count / max(candidate_valid_count, 1)
    scene["candidate_gated_active_fraction"] = candidate_gated_active_count / max(candidate_valid_count, 1)
    scene["candidate_reciprocal_fraction"] = candidate_reciprocal_count / max(candidate_valid_count, 1)
    scene["candidate_coarse_cycle_error_mean"] = weighted_mean(candidate_coarse_cycle_pairs)
    scene["candidate_path_cycle_error_lag23_mean"] = weighted_mean(candidate_path_cycle_lag23_pairs)

    flattened_records = []
    samples = []
    for item in layer_step_records:
        flat = {
            "case_id": record["case_id"],
            "replay_mode": record["replay_mode"],
            "motion_stratum": scene["motion_stratum"],
            "label": scene["label"],
            "met3r_one_second_delta": scene["met3r_one_second_delta"],
            "step": item["step"],
            "layer": item["layer"],
            "active_fraction": item["active_fraction"],
            "source_unreliable_active_fraction": item["source_unreliable_active_fraction"],
            "conditioning_source_active_fraction": item["conditioning_source_active_fraction"],
        }
        for source_name, output_name in SELECTED_METRICS.items():
            flat[output_name] = item["selected"][source_name]["mean"]
        for lag_record in item["candidate_by_lag"]:
            prefix = f"candidate_lag{lag_record['lag']}"
            flat[f"{prefix}_raw_active_fraction"] = lag_record["raw_active_fraction"]
            flat[f"{prefix}_gated_active_fraction"] = lag_record["gated_active_fraction"]
            flat[f"{prefix}_reciprocal_fraction"] = lag_record["reciprocal_fraction"]
            flat[f"{prefix}_coarse_cycle_error_mean"] = lag_record["coarse_cycle_error"]["mean"]
            flat[f"{prefix}_path_cycle_error_mean"] = lag_record["path_cycle_error"]["mean"]
        flattened_records.append(flat)

        columns = item.get("samples", {})
        sample_count = len(columns.get("target_time", []))
        for sample_index in range(sample_count):
            sample = {
                "case_id": record["case_id"],
                "replay_mode": record["replay_mode"],
                "motion_stratum": scene["motion_stratum"],
                "label": scene["label"],
                "met3r_one_second_delta": scene["met3r_one_second_delta"],
                "step": item["step"],
                "layer": item["layer"],
            }
            for name, values in columns.items():
                sample[name] = values[sample_index]
            samples.append(sample)
    return scene, flattened_records, samples


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def source_risk_bin(sample: dict) -> str:
    if sample["source_is_conditioning"]:
        return "conditioning"
    confidence = float(sample["source_previous_confidence"])
    if confidence <= 0.0:
        return "generated_unreliable_zero"
    if confidence <= 0.2:
        return "generated_(0,0.2]"
    if confidence <= 0.4:
        return "generated_(0.2,0.4]"
    if confidence <= 0.6:
        return "generated_(0.4,0.6]"
    if confidence <= 0.8:
        return "generated_(0.6,0.8]"
    return "generated_(0.8,1.0]"


def build_source_risk_bins(samples: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for sample in samples:
        grouped[(sample["case_id"], sample["replay_mode"])].append(sample)
    rows = []
    ordered_bins = [
        "conditioning",
        "generated_unreliable_zero",
        "generated_(0,0.2]",
        "generated_(0.2,0.4]",
        "generated_(0.4,0.6]",
        "generated_(0.6,0.8]",
        "generated_(0.8,1.0]",
    ]
    for (case_id, replay_mode), case_samples in grouped.items():
        by_bin = defaultdict(list)
        for sample in case_samples:
            by_bin[source_risk_bin(sample)].append(sample)
        for bin_name in ordered_bins:
            bin_samples = by_bin[bin_name]
            template = case_samples[0]
            rows.append(
                {
                    "case_id": case_id,
                    "replay_mode": replay_mode,
                    "motion_stratum": template["motion_stratum"],
                    "label": template["label"],
                    "met3r_one_second_delta": template["met3r_one_second_delta"],
                    "source_risk_bin": bin_name,
                    "sample_count": len(bin_samples),
                    "sample_fraction": len(bin_samples) / len(case_samples),
                    "path_cycle_error_mean": (
                        float(np.mean([sample["path_cycle_error"] for sample in bin_samples]))
                        if bin_samples
                        else None
                    ),
                    "velocity_rms_mean": (
                        float(np.mean([sample["velocity_rms"] for sample in bin_samples]))
                        if bin_samples
                        else None
                    ),
                    "relative_value_residual_mean": (
                        float(np.mean([sample["relative_value_residual"] for sample in bin_samples]))
                        if bin_samples
                        else None
                    ),
                    "selected_lag_mean": (
                        float(np.mean([sample["selected_lag"] for sample in bin_samples]))
                        if bin_samples
                        else None
                    ),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--bootstraps", type=int, default=20000)
    args = parser.parse_args()

    lock_path = args.lock.resolve()
    lock = read_json(lock_path)
    if lock.get("schema") != "c2f-p0-forensics-lock-v1":
        raise RuntimeError("unexpected P0 lock schema")

    all_scenes: dict[str, list[dict]] = {"baseline_observe": [], "c2f": []}
    all_layer_steps = []
    all_samples = []
    replay_hashes = {}
    raw_replays: dict[str, dict[str, dict]] = {"baseline_observe": {}, "c2f": {}}
    for mode in ("baseline_observe", "c2f"):
        for outcome in lock["cases_ranked_best_to_worst"]:
            path = replay_path(args.replay_root.resolve(), mode, outcome["case_id"], args.seed)
            if not path.is_file():
                raise FileNotFoundError(path)
            replay_hashes[str(path)] = sha256_file(path)
            record = read_json(path)
            if record["primary_outcome"] != outcome:
                raise RuntimeError(f"outcome drift in {path}")
            raw_replays[mode][outcome["case_id"]] = record
            scene, layer_steps, samples = aggregate_scene(record)
            all_scenes[mode].append(scene)
            all_layer_steps.extend(layer_steps)
            all_samples.extend(samples)

    baseline_by_case = {row["case_id"]: row for row in all_scenes["baseline_observe"]}
    c2f_by_case = {row["case_id"]: row for row in all_scenes["c2f"]}
    if set(baseline_by_case) != set(c2f_by_case):
        raise RuntimeError("baseline-observe and C2F replay case sets differ")
    preintervention_parity = {
        case_id: preintervention_record(raw_replays["baseline_observe"][case_id])
        == preintervention_record(raw_replays["c2f"][case_id])
        for case_id in baseline_by_case
    }
    if not all(preintervention_parity.values()):
        failures = [case_id for case_id, passed in preintervention_parity.items() if not passed]
        raise RuntimeError(f"baseline/C2F pre-intervention diagnostic mismatch: {failures}")

    scene_rows = []
    for outcome in lock["cases_ranked_best_to_worst"]:
        case_id = outcome["case_id"]
        row = dict(baseline_by_case[case_id])
        for name, value in c2f_by_case[case_id].items():
            if name in {"case_id", "motion_stratum", "label", "near_tie", "met3r_one_second_delta"}:
                continue
            row[f"c2f_{name}"] = value
            if isinstance(value, (int, float)) and isinstance(row.get(name), (int, float)):
                row[f"trajectory_delta_{name}"] = value - row[name]
        scene_rows.append(row)

    y = np.asarray([row["met3r_one_second_delta"] for row in scene_rows], dtype=np.float64)
    labels = np.asarray([row["label"] for row in scene_rows])
    strata = np.asarray([row["motion_stratum"] for row in scene_rows])
    near_tie = np.asarray([row["near_tie"] for row in scene_rows], dtype=bool)
    associations = {}
    association_rows = []
    for predictor_index, predictor in enumerate(PRIMARY_PREDICTORS + SECONDARY_PREDICTORS):
        x = np.asarray([row[predictor] for row in scene_rows], dtype=np.float64)
        if not np.all(np.isfinite(x)):
            raise RuntimeError(f"non-finite predictor: {predictor}")
        winners = x[labels == "winner"]
        losers = x[labels == "loser"]
        summary = correlation_summary(
            x,
            y,
            strata,
            near_tie,
            seed=20260915 + predictor_index,
            permutations=args.permutations,
            bootstraps=args.bootstraps,
        )
        summary.update(
            family="primary" if predictor in PRIMARY_PREDICTORS else "secondary",
            winner_median=percentile(winners, 0.5),
            winner_iqr=[percentile(winners, 0.25), percentile(winners, 0.75)],
            loser_median=percentile(losers, 0.5),
            loser_iqr=[percentile(losers, 0.25), percentile(losers, 0.75)],
            loser_minus_winner_median=percentile(losers, 0.5) - percentile(winners, 0.5),
        )
        associations[predictor] = summary
        association_rows.append({"predictor": predictor, **summary})

    output_dir = args.output_dir.resolve()
    write_csv(output_dir / "SCENE_FEATURES.csv", scene_rows)
    write_csv(output_dir / "LAYER_STEP_FEATURES.csv", all_layer_steps)
    write_csv(output_dir / "ASSOCIATIONS.csv", association_rows)
    write_csv(output_dir / "SOURCE_RISK_BINS_BY_SCENE.csv", build_source_risk_bins(all_samples))
    write_gzip_csv(output_dir / "INTERVENTION_SAMPLES.csv.gz", all_samples)

    report = {
        "schema": "c2f-p0-forensics-report-v1",
        "scope": "mechanism_discovery_only",
        "analysis_script_path": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "analysis_parameters": {
            "seed_rule": "20260915 + predictor_index",
            "permutations_per_predictor": args.permutations,
            "bootstraps_per_predictor": args.bootstraps,
        },
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "replay_root": str(args.replay_root.resolve()),
        "replay_record_sha256": replay_hashes,
        "case_count": len(scene_rows),
        "winner_count": int(np.count_nonzero(labels == "winner")),
        "loser_count": int(np.count_nonzero(labels == "loser")),
        "primary_outcome_sign": "positive_is_harmful",
        "preintervention_step20_layer10_parity": preintervention_parity,
        "primary_predictors": PRIMARY_PREDICTORS,
        "secondary_predictors": SECONDARY_PREDICTORS,
        "associations": associations,
        "counterfactual_subset": lock["counterfactual_subset"],
        "interpretation_status": "exploratory_pending_counterfactual",
    }
    atomic_json(output_dir / "P0_REPORT.json", report)

    lines = [
        "# C2F P0 Forensics",
        "",
        "This is a mechanism-discovery analysis on the seen Dev25, not a generalization claim.",
        "Positive MEt3R-1s delta is harmful; therefore a positive predictor correlation means higher predictor values accompany worse C2F outcomes.",
        "",
        "| Predictor | Family | Spearman rho | 80% bootstrap CI | Motion-demeaned rho | LOO sign | Winner median | Loser median |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for predictor in PRIMARY_PREDICTORS + SECONDARY_PREDICTORS:
        item = associations[predictor]
        ci = item["bootstrap_ci80"]
        lines.append(
            f"| {predictor} | {item['family']} | {item['spearman_rho']!s} | "
            f"[{ci[0]!s}, {ci[1]!s}] | {item['motion_stratum_demeaned_spearman']!s} | "
            f"{item['leave_one_out_same_sign_fraction']!s} | {item['winner_median']:.6f} | "
            f"{item['loser_median']:.6f} |"
        )
    lines.extend(
        [
            "",
            "No factor is promoted to a method component from this table alone. The locked counterfactual is required.",
        ]
    )
    (output_dir / "P0_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote P0 analysis to {output_dir}")


if __name__ == "__main__":
    main()
