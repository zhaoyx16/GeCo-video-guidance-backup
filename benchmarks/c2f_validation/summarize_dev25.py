#!/usr/bin/env python3
"""Create paired scene-level statistics and apply the preregistered dev gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_cases(manifest: dict) -> list[tuple[str, dict]]:
    records = [(case_id, case) for case_id, case in manifest.items() if not case_id.startswith("_")]
    return sorted(records, key=lambda item: item[1]["c2f_dev_selection"]["selection_order"])


def component_records(root: Path, name: str, expected_metric: str) -> list[dict]:
    path = root / "components" / f"{name}.json"
    payload = read_json(path)
    if payload.get("schema") != "geometry-selection-metric-component-v1":
        raise ValueError(f"unexpected component schema: {path}")
    if payload.get("metric_id") != expected_metric:
        raise ValueError(f"unexpected metric ID in {path}: {payload.get('metric_id')}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"component records missing: {path}")
    return records


def unit_map(records: list[dict]) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in records:
        case_id, unit_id = record.get("case_id"), record.get("unit_id")
        if not isinstance(case_id, str) or not isinstance(unit_id, str) or unit_id in result[case_id]:
            raise ValueError("duplicate or malformed metric record")
        result[case_id][unit_id] = record
    return dict(result)


def aggregate_prefix(
    records: dict[str, dict[str, dict]], case_ids: list[str], prefix: str, expected_units: int
) -> dict[str, float]:
    values = {}
    for case_id in case_ids:
        selected = [record for unit, record in records.get(case_id, {}).items() if unit.startswith(prefix)]
        if len(selected) != expected_units:
            raise ValueError(f"{case_id}: expected {expected_units} units with prefix {prefix}, found {len(selected)}")
        numbers = [float(record["value"]) for record in selected]
        if not all(math.isfinite(number) for number in numbers):
            raise ValueError(f"non-finite metric value for {case_id}/{prefix}")
        values[case_id] = statistics.fmean(numbers)
    return values


def aggregate_lre(
    baseline: dict[str, dict[str, dict]], candidate: dict[str, dict[str, dict]], case_ids: list[str]
) -> tuple[dict[str, float], dict[str, float], dict]:
    baseline_values, candidate_values = {}, {}
    eligibility = []
    for case_id in case_ids:
        base = baseline.get(case_id, {}).get("first_last")
        cand = candidate.get(case_id, {}).get("first_last")
        if base is None or cand is None:
            raise ValueError(f"missing LRE first-last record for {case_id}")
        base_ok, cand_ok = base.get("eligible") is True, cand.get("eligible") is True
        eligibility.append(
            {
                "case_id": case_id,
                "baseline_eligible": base_ok,
                "candidate_eligible": cand_ok,
                "joint_eligible": base_ok and cand_ok,
            }
        )
        if base_ok and cand_ok:
            baseline_values[case_id] = float(base["value"])
            candidate_values[case_id] = float(cand["value"])
    return baseline_values, candidate_values, {
        "joint_eligible_count": len(baseline_values),
        "baseline_eligible_count": sum(item["baseline_eligible"] for item in eligibility),
        "candidate_eligible_count": sum(item["candidate_eligible"] for item in eligibility),
        "per_case": eligibility,
    }


def quantile_sorted(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(probability * (len(ordered) - 1))))
    return ordered[index]


def paired_summary(
    baseline: dict[str, float],
    candidate: dict[str, float],
    direction: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict:
    if set(baseline) != set(candidate) or not baseline:
        raise ValueError("paired metric case sets must be identical and nonempty")
    case_ids = sorted(baseline)
    base_values = [baseline[case_id] for case_id in case_ids]
    cand_values = [candidate[case_id] for case_id in case_ids]
    sign = 1.0 if direction == "lower" else -1.0
    directional = [sign * (base - cand) for base, cand in zip(base_values, cand_values)]
    baseline_mean = statistics.fmean(base_values)
    candidate_mean = statistics.fmean(cand_values)
    relative_improvement = 100.0 * statistics.fmean(directional) / max(abs(baseline_mean), 1e-12)

    generator = random.Random(bootstrap_seed)
    bootstrap_absolute, bootstrap_relative = [], []
    for _ in range(bootstrap_resamples):
        indices = [generator.randrange(len(case_ids)) for _ in case_ids]
        sampled_base = statistics.fmean(base_values[index] for index in indices)
        sampled_directional = statistics.fmean(directional[index] for index in indices)
        bootstrap_absolute.append(sampled_directional)
        bootstrap_relative.append(100.0 * sampled_directional / max(abs(sampled_base), 1e-12))

    per_case = []
    for case_id, base, cand, improvement in zip(case_ids, base_values, cand_values, directional):
        per_case.append(
            {
                "case_id": case_id,
                "baseline": base,
                "candidate": cand,
                "candidate_minus_baseline": cand - base,
                "directional_improvement": improvement,
                "win": improvement > 0.0,
            }
        )
    return {
        "direction": direction,
        "scene_count": len(case_ids),
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "candidate_minus_baseline": candidate_mean - baseline_mean,
        "mean_directional_improvement": statistics.fmean(directional),
        "median_directional_improvement": statistics.median(directional),
        "mean_relative_improvement_percent": relative_improvement,
        "win_rate": sum(value > 0.0 for value in directional) / len(directional),
        "tie_rate": sum(value == 0.0 for value in directional) / len(directional),
        "bootstrap": {
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "absolute_ci80": [quantile_sorted(bootstrap_absolute, 0.10), quantile_sorted(bootstrap_absolute, 0.90)],
            "absolute_ci95": [quantile_sorted(bootstrap_absolute, 0.025), quantile_sorted(bootstrap_absolute, 0.975)],
            "relative_percent_ci80": [quantile_sorted(bootstrap_relative, 0.10), quantile_sorted(bootstrap_relative, 0.90)],
            "relative_percent_ci95": [quantile_sorted(bootstrap_relative, 0.025), quantile_sorted(bootstrap_relative, 0.975)],
        },
        "per_case": per_case,
    }


def generation_stats(root: Path, method_id: str, case_ids: list[str]) -> dict:
    wall, peak = {}, {}
    for case_id in case_ids:
        metadata_path = root / method_id / case_id / "seed_0" / "metadata.json"
        metadata = read_json(metadata_path)
        wall[case_id] = float(metadata["wall_seconds"])
        peak[case_id] = float(metadata["peak_memory_mib"])
    return {"wall_seconds": wall, "peak_memory_mib": peak}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--visual-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    args = parser.parse_args()

    manifest = read_json(args.selection_manifest)
    expected_overlap = {"test": 0, "validation": 0, "debug": 0}
    if manifest.get("_meta", {}).get("reserved_overlap_counts") != expected_overlap:
        raise RuntimeError("selection manifest does not certify zero reserved-split overlap")
    cases = ordered_cases(manifest)
    case_ids = [case_id for case_id, _ in cases]
    if len(case_ids) != 25:
        raise RuntimeError("the preregistered gate requires exactly 25 scenes")

    visual_gate = read_json(args.visual_gate)
    if visual_gate.get("status") != "complete_before_metrics":
        raise RuntimeError("visual gate must be completed before metric summary")
    visual_records = visual_gate.get("records")
    visual_case_ids = [item.get("case_id") for item in visual_records] if isinstance(visual_records, list) else []
    if (
        len(visual_case_ids) != len(case_ids)
        or len(set(visual_case_ids)) != len(case_ids)
        or set(visual_case_ids) != set(case_ids)
    ):
        raise RuntimeError("visual gate must cover every development case exactly once")

    baseline_components = {
        "met3r": unit_map(component_records(args.baseline_metrics, "met3r_multiscale", "met3r_multiscale")),
        "geco": unit_map(component_records(args.baseline_metrics, "geco_fused", "geco_fused")),
        "lre": unit_map(
            component_records(args.baseline_metrics, "long_range_reprojection_error", "long_range_reprojection_error")
        ),
        "motion": unit_map(
            component_records(args.baseline_metrics, "relative_total_motion_raw", "relative_total_motion_raw")
        ),
        "vbench": unit_map(component_records(args.baseline_metrics, "vbench_quality", "vbench_quality")),
    }
    candidate_components = {
        "met3r": unit_map(component_records(args.candidate_metrics, "met3r_multiscale", "met3r_multiscale")),
        "geco": unit_map(component_records(args.candidate_metrics, "geco_fused", "geco_fused")),
        "lre": unit_map(
            component_records(args.candidate_metrics, "long_range_reprojection_error", "long_range_reprojection_error")
        ),
        "motion": unit_map(
            component_records(args.candidate_metrics, "relative_total_motion_raw", "relative_total_motion_raw")
        ),
        "vbench": unit_map(component_records(args.candidate_metrics, "vbench_quality", "vbench_quality")),
    }

    definitions = {
        "met3r_half_second": ("met3r", "half_second_", 10, "lower"),
        "met3r_one_second": ("met3r", "one_second_", 5, "lower"),
        "met3r_first_last": ("met3r", "first_last_", 1, "lower"),
        "geco_fused": ("geco", "window_", 2, "lower"),
        "vbench_quality": ("vbench", "official_quality", 1, "higher"),
    }
    summaries = {}
    scene_values = {}
    for name, (component, prefix, count, direction) in definitions.items():
        base_values = aggregate_prefix(baseline_components[component], case_ids, prefix, count)
        cand_values = aggregate_prefix(candidate_components[component], case_ids, prefix, count)
        summaries[name] = paired_summary(
            base_values, cand_values, direction, args.bootstrap_resamples, args.bootstrap_seed
        )
        scene_values[name] = {"baseline": base_values, "candidate": cand_values}

    base_lre, cand_lre, lre_eligibility = aggregate_lre(
        baseline_components["lre"], candidate_components["lre"], case_ids
    )
    if base_lre:
        summaries["independent_lre"] = paired_summary(
            base_lre, cand_lre, "lower", args.bootstrap_resamples, args.bootstrap_seed
        )
        scene_values["independent_lre"] = {"baseline": base_lre, "candidate": cand_lre}
    else:
        summaries["independent_lre"] = {"scene_count": 0, "status": "no_joint_eligible_scenes"}

    base_motion = aggregate_prefix(baseline_components["motion"], case_ids, "total_motion", 1)
    cand_motion = aggregate_prefix(candidate_components["motion"], case_ids, "total_motion", 1)
    motion_ratios = {case_id: 100.0 * cand_motion[case_id] / base_motion[case_id] for case_id in case_ids}
    motion_summary = {
        "aggregate_candidate_over_baseline_percent": 100.0 * sum(cand_motion.values()) / sum(base_motion.values()),
        "mean_scene_ratio_percent": statistics.fmean(motion_ratios.values()),
        "median_scene_ratio_percent": statistics.median(motion_ratios.values()),
        "min_scene_ratio_percent": min(motion_ratios.values()),
        "max_scene_ratio_percent": max(motion_ratios.values()),
        "per_case": motion_ratios,
    }

    baseline_generation = generation_stats(args.generation_root, "official_same_host_reference", case_ids)
    candidate_generation = generation_stats(args.generation_root, "c2f_k3_a0025", case_ids)
    runtime_ratio = 100.0 * statistics.fmean(candidate_generation["wall_seconds"].values()) / statistics.fmean(
        baseline_generation["wall_seconds"].values()
    )
    peak_ratio = 100.0 * max(candidate_generation["peak_memory_mib"].values()) / max(
        baseline_generation["peak_memory_mib"].values()
    )
    efficiency = {
        "baseline_mean_wall_seconds": statistics.fmean(baseline_generation["wall_seconds"].values()),
        "candidate_mean_wall_seconds": statistics.fmean(candidate_generation["wall_seconds"].values()),
        "runtime_candidate_over_baseline_percent": runtime_ratio,
        "baseline_max_peak_memory_mib": max(baseline_generation["peak_memory_mib"].values()),
        "candidate_max_peak_memory_mib": max(candidate_generation["peak_memory_mib"].values()),
        "peak_memory_candidate_over_baseline_percent": peak_ratio,
    }

    primary = summaries["met3r_one_second"]
    primary_checks = {
        "mean_relative_improvement_at_least_2pct": primary["mean_relative_improvement_percent"] >= 2.0,
        "median_directional_improvement_positive": primary["median_directional_improvement"] > 0.0,
        "win_rate_at_least_60pct": primary["win_rate"] >= 0.60,
        "bootstrap_relative_ci80_lower_positive": primary["bootstrap"]["relative_percent_ci80"][0] > 0.0,
    }
    secondary_names = ["met3r_half_second", "met3r_first_last", "independent_lre"]
    secondary_improvements = {
        name: summaries[name].get("mean_relative_improvement_percent") for name in secondary_names
    }
    secondary_checks = {
        "at_least_two_mean_improvements": sum(
            value is not None and value > 0.0 for value in secondary_improvements.values()
        )
        >= 2,
        "none_worse_than_5pct": all(
            value is not None and value >= -5.0 for value in secondary_improvements.values()
        ),
    }
    visual_pass = all(
        item.get("baseline_usable") is True
        and item.get("candidate_usable") is True
        and item.get("candidate_no_new_severe_artifact") is True
        for item in visual_records
    )
    guardrails = {
        "visual_all_pairs_pass": visual_pass,
        "motion_between_90_and_110pct": 90.0 <= motion_summary["aggregate_candidate_over_baseline_percent"] <= 110.0,
        "vbench_quality_not_worse_than_1pct": summaries["vbench_quality"]["mean_relative_improvement_percent"] >= -1.0,
        "runtime_overhead_at_most_10pct": runtime_ratio <= 110.0,
        "peak_memory_overhead_at_most_10pct": peak_ratio <= 110.0,
    }
    all_primary = all(primary_checks.values())
    all_secondary = all(secondary_checks.values())
    all_guardrails = all(guardrails.values())
    if all_primary and all_secondary and all_guardrails:
        decision = "advance"
    elif primary["mean_relative_improvement_percent"] <= 0.0 or primary["win_rate"] <= 0.50 or not visual_pass:
        decision = "stop_or_falsify_current_c2f"
    else:
        decision = "borderline"

    per_case_rows = []
    strata = {case_id: case["c2f_dev_selection"]["motion_stratum"] for case_id, case in cases}
    for case_id in case_ids:
        row = {"case_id": case_id, "motion_stratum": strata[case_id]}
        for name, values in scene_values.items():
            if case_id in values["baseline"]:
                row[f"{name}_baseline"] = values["baseline"][case_id]
                row[f"{name}_candidate"] = values["candidate"][case_id]
                row[f"{name}_delta"] = values["candidate"][case_id] - values["baseline"][case_id]
        row["motion_ratio_percent"] = motion_ratios[case_id]
        per_case_rows.append(row)

    # Strata are diagnostic only. The preregistered decision remains scene-level
    # over all 25 cases, so small strata cannot silently replace the main test.
    stratum_summaries = {}
    for stratum in sorted(set(strata.values())):
        stratum_ids = [case_id for case_id in case_ids if strata[case_id] == stratum]
        metric_summaries = {}
        for name, values in scene_values.items():
            available = [case_id for case_id in stratum_ids if case_id in values["baseline"]]
            if not available:
                metric_summaries[name] = {"scene_count": 0, "status": "no_joint_eligible_scenes"}
                continue
            metric_summaries[name] = paired_summary(
                {case_id: values["baseline"][case_id] for case_id in available},
                {case_id: values["candidate"][case_id] for case_id in available},
                summaries[name]["direction"],
                args.bootstrap_resamples,
                args.bootstrap_seed,
            )
        stratum_motion = {case_id: motion_ratios[case_id] for case_id in stratum_ids}
        stratum_summaries[stratum] = {
            "scene_count": len(stratum_ids),
            "case_ids": stratum_ids,
            "metrics": metric_summaries,
            "motion_ratio_percent": {
                "mean": statistics.fmean(stratum_motion.values()),
                "median": statistics.median(stratum_motion.values()),
            },
        }

    output = {
        "schema": "wan-c2f-disjoint-dev25-paired-report-v1",
        "status": "complete",
        "scope": "development_only_not_final_validation",
        "selection_manifest": {"path": str(args.selection_manifest.resolve()), "sha256": sha256_file(args.selection_manifest)},
        "reserved_overlap_counts": expected_overlap,
        "baseline_method": "official_same_host_reference",
        "candidate_method": "c2f_k3_a0025",
        "visual_gate": {"path": str(args.visual_gate.resolve()), "sha256": sha256_file(args.visual_gate)},
        "metrics": summaries,
        "lre_eligibility": lre_eligibility,
        "motion_retention": motion_summary,
        "efficiency": efficiency,
        "diagnostic_motion_strata": stratum_summaries,
        "preregistered_gate": {
            "primary_checks": primary_checks,
            "secondary_relative_improvements_percent": secondary_improvements,
            "secondary_checks": secondary_checks,
            "guardrails": guardrails,
            "decision": decision,
        },
        "per_case": per_case_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / "PAIRED_DEV25_REPORT.json"
    report_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output_dir / "PER_CASE.csv"
    fieldnames = sorted({key for row in per_case_rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_case_rows)
    print(json.dumps({"decision": decision, "report": str(report_path), "per_case": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
