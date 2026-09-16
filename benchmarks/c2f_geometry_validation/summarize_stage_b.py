#!/usr/bin/env python3
"""Summarize the locked four-way Stage B geometry-gated C2F experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


METHODS = {
    "B": "official_same_host_reference",
    "C": "c2f_k3_a0025",
    "G": "c2f_geometry_hard_gate",
    "U": "c2f_geometry_uniform_norm_control",
}
DEFINITIONS = {
    "met3r_half_second": ("met3r_multiscale", "half_second_", 10, "lower"),
    "met3r_one_second": ("met3r_multiscale", "one_second_", 5, "lower"),
    "met3r_first_last": ("met3r_multiscale", "first_last_", 1, "lower"),
    "geco_fused": ("geco_fused", "window_", 2, "lower"),
    "vbench_quality": ("vbench_quality", "official_quality", 1, "higher"),
    "relative_motion": ("relative_total_motion_raw", "total_motion", 1, "higher"),
}


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


def component(root: Path, metric_id: str) -> dict[str, dict[str, dict]]:
    path = root / "components" / f"{metric_id}.json"
    payload = read_json(path)
    if payload.get("schema") != "geometry-selection-metric-component-v1" or payload.get("metric_id") != metric_id:
        raise ValueError(f"invalid metric component: {path}")
    result: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in payload.get("records", []):
        case_id, unit_id = record.get("case_id"), record.get("unit_id")
        if not isinstance(case_id, str) or not isinstance(unit_id, str) or unit_id in result[case_id]:
            raise ValueError(f"duplicate or malformed record in {path}")
        result[case_id][unit_id] = record
    return dict(result)


def aggregate(
    records: dict[str, dict[str, dict]], case_ids: list[str], prefix: str, expected_units: int
) -> dict[str, float]:
    values = {}
    for case_id in case_ids:
        selected = [record for unit_id, record in records.get(case_id, {}).items() if unit_id.startswith(prefix)]
        if len(selected) != expected_units:
            raise ValueError(f"{case_id}: expected {expected_units} {prefix} units, found {len(selected)}")
        numbers = [float(record["value"]) for record in selected]
        if not all(math.isfinite(number) for number in numbers):
            raise ValueError(f"non-finite value for {case_id}/{prefix}")
        values[case_id] = statistics.fmean(numbers)
    return values


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(probability * (len(ordered) - 1))))
    return ordered[index]


def exact_sign_flip_p(differences: list[float]) -> dict:
    observed = statistics.fmean(differences)
    null = [
        statistics.fmean(sign * value for sign, value in zip(signs, differences))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    tolerance = 1e-15
    return {
        "assumption": "paired differences are exchangeable under sign flips",
        "permutations": len(null),
        "one_sided_positive_p": sum(value >= observed - tolerance for value in null) / len(null),
        "two_sided_p": sum(abs(value) >= abs(observed) - tolerance for value in null) / len(null),
    }


def paired_summary(
    reference: dict[str, float],
    candidate: dict[str, float],
    direction: str,
    resamples: int,
    seed: int,
) -> dict:
    if set(reference) != set(candidate) or not reference:
        raise ValueError("paired case sets must be identical and nonempty")
    case_ids = sorted(reference)
    reference_values = [reference[case_id] for case_id in case_ids]
    candidate_values = [candidate[case_id] for case_id in case_ids]
    sign = 1.0 if direction == "lower" else -1.0
    improvements = [
        sign * (reference_value - candidate_value)
        for reference_value, candidate_value in zip(reference_values, candidate_values)
    ]
    reference_mean = statistics.fmean(reference_values)
    candidate_mean = statistics.fmean(candidate_values)
    generator = random.Random(seed)
    boot_absolute, boot_relative = [], []
    for _ in range(resamples):
        indices = [generator.randrange(len(case_ids)) for _ in case_ids]
        sampled_reference = statistics.fmean(reference_values[index] for index in indices)
        sampled_improvement = statistics.fmean(improvements[index] for index in indices)
        boot_absolute.append(sampled_improvement)
        boot_relative.append(100.0 * sampled_improvement / max(abs(sampled_reference), 1e-12))
    per_case = []
    for case_id, reference_value, candidate_value, improvement in zip(
        case_ids, reference_values, candidate_values, improvements
    ):
        per_case.append(
            {
                "case_id": case_id,
                "reference": reference_value,
                "candidate": candidate_value,
                "candidate_minus_reference": candidate_value - reference_value,
                "directional_improvement": improvement,
                "win": improvement > 0.0,
            }
        )
    return {
        "direction": direction,
        "scene_count": len(case_ids),
        "reference_mean": reference_mean,
        "candidate_mean": candidate_mean,
        "candidate_minus_reference": candidate_mean - reference_mean,
        "mean_directional_improvement": statistics.fmean(improvements),
        "median_directional_improvement": statistics.median(improvements),
        "mean_relative_improvement_percent": 100.0
        * statistics.fmean(improvements)
        / max(abs(reference_mean), 1e-12),
        "win_rate": sum(value > 0.0 for value in improvements) / len(improvements),
        "tie_rate": sum(value == 0.0 for value in improvements) / len(improvements),
        "bootstrap": {
            "resamples": resamples,
            "seed": seed,
            "absolute_ci80": [quantile(boot_absolute, 0.10), quantile(boot_absolute, 0.90)],
            "absolute_ci95": [quantile(boot_absolute, 0.025), quantile(boot_absolute, 0.975)],
            "relative_percent_ci80": [quantile(boot_relative, 0.10), quantile(boot_relative, 0.90)],
            "relative_percent_ci95": [quantile(boot_relative, 0.025), quantile(boot_relative, 0.975)],
        },
        "exact_sign_flip": exact_sign_flip_p(improvements),
        "per_case": per_case,
    }


def generation_stats(root: Path, method_id: str, case_ids: list[str]) -> dict:
    wall, peak = {}, {}
    for case_id in case_ids:
        metadata = read_json(root / method_id / case_id / "seed_0/metadata.json")
        wall[case_id] = float(metadata["wall_seconds"])
        peak[case_id] = float(metadata["peak_memory_mib"])
    return {
        "mean_wall_seconds": statistics.fmean(wall.values()),
        "median_wall_seconds": statistics.median(wall.values()),
        "max_peak_memory_mib": max(peak.values()),
        "per_case_wall_seconds": wall,
        "per_case_peak_memory_mib": peak,
    }


def geometry_stats(root: Path, case_ids: list[str]) -> dict:
    forward, shared_load, peak = {}, {}, {}
    for case_id in case_ids:
        metadata = read_json(root / case_id / "GEOMETRY_METADATA.json")
        forward[case_id] = float(metadata["geometry_forward_seconds"])
        shared_load[case_id] = float(metadata["model_load_seconds_shared"])
        peak[case_id] = float(metadata["geometry_peak_allocated_mib"])
    return {
        "mean_forward_seconds_per_scene": statistics.fmean(forward.values()),
        "one_time_shared_model_load_seconds": max(shared_load.values()),
        "max_peak_allocated_mib": max(peak.values()),
        "per_case_forward_seconds": forward,
    }


def intervention_stats(root: Path, method_id: str, case_ids: list[str]) -> dict:
    per_case = {}
    all_records = []
    for case_id in case_ids:
        metadata = read_json(root / method_id / case_id / "seed_0/metadata.json")
        diagnostics = metadata.get("geometry_diagnostics", {})
        records = diagnostics.get("records", [])
        if len(records) != 30:
            raise ValueError(f"expected 30 intervention records for {method_id}/{case_id}")
        all_records.extend(records)
        active = sum(int(record["active_correspondences"]) for record in records)
        accepted = sum(int(record["accepted_correspondences"]) for record in records)
        per_case[case_id] = {
            "active_correspondences": active,
            "accepted_correspondences": accepted,
            "survival_fraction": accepted / active,
            "mean_uniform_scale": statistics.fmean(float(record["uniform_scale"]) for record in records),
            "max_norm_match_relative_error": max(
                float(record["norm_match_relative_error"]) for record in records
            ),
        }
    total_active = sum(item["active_correspondences"] for item in per_case.values())
    total_accepted = sum(item["accepted_correspondences"] for item in per_case.values())
    return {
        "event_count": len(all_records),
        "weighted_survival_fraction": total_accepted / total_active,
        "mean_case_survival_fraction": statistics.fmean(
            item["survival_fraction"] for item in per_case.values()
        ),
        "max_norm_match_relative_error": max(
            item["max_norm_match_relative_error"] for item in per_case.values()
        ),
        "per_case": per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-b-lock", type=Path, required=True)
    parser.add_argument("--visual-gate", type=Path, required=True)
    parser.add_argument("--baseline-main", type=Path, required=True)
    parser.add_argument("--baseline-aux", type=Path, required=True)
    parser.add_argument("--c2f-main", type=Path, required=True)
    parser.add_argument("--c2f-aux", type=Path, required=True)
    parser.add_argument("--geometry-metrics", type=Path, required=True)
    parser.add_argument("--uniform-metrics", type=Path, required=True)
    parser.add_argument("--baseline-generation-root", type=Path, required=True)
    parser.add_argument("--c2f-generation-root", type=Path, required=True)
    parser.add_argument("--new-generation-root", type=Path, required=True)
    parser.add_argument("--geometry-precompute-root", type=Path, required=True)
    parser.add_argument("--lre-status", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260916)
    args = parser.parse_args()

    stage_b = read_json(args.stage_b_lock)
    cases = stage_b.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) != 10
        or stage_b.get("reserved_overlap_counts") != {"test": 0, "validation": 0, "debug": 0}
    ):
        raise ValueError("invalid frozen Stage B lock")
    case_ids = [case["case_id"] for case in cases]
    strata = {case["case_id"]: case["motion_stratum"] for case in cases}
    visual_gate = read_json(args.visual_gate)
    visual_records = visual_gate.get("records", [])
    if (
        visual_gate.get("status") != "complete_before_metrics"
        or {record.get("case_id") for record in visual_records} != set(case_ids)
        or not all(
            all(record.get(field) is True for field in ("B_usable", "C_usable", "G_usable", "U_usable"))
            and record.get("G_no_new_severe_artifact") is True
            and record.get("U_no_new_severe_artifact") is True
            for record in visual_records
        )
    ):
        raise ValueError("formal Stage B visual gate is incomplete")
    lre_status = read_json(args.lre_status)
    if lre_status.get("stage_b_baseline_eligible_count") != 0:
        raise ValueError("this summary expects the frozen Stage B LRE denominator to be empty")

    roots = {
        "B": {"main": args.baseline_main, "aux": args.baseline_aux},
        "C": {"main": args.c2f_main, "aux": args.c2f_aux},
        "G": {"main": args.geometry_metrics, "aux": args.geometry_metrics},
        "U": {"main": args.uniform_metrics, "aux": args.uniform_metrics},
    }
    component_location = {
        "met3r_multiscale": "main",
        "geco_fused": "main",
        "vbench_quality": "aux",
        "relative_total_motion_raw": "aux",
    }
    components = {
        method: {
            metric_id: component(method_roots[location], metric_id)
            for metric_id, location in component_location.items()
        }
        for method, method_roots in roots.items()
    }
    values = {
        method: {
            metric_name: aggregate(
                components[method][component_id], case_ids, prefix, expected_units
            )
            for metric_name, (component_id, prefix, expected_units, _direction) in DEFINITIONS.items()
        }
        for method in METHODS
    }

    method_means = {
        method: {
            metric_name: statistics.fmean(metric_values.values())
            for metric_name, metric_values in method_values.items()
        }
        for method, method_values in values.items()
    }
    comparison_specs = {
        "G_vs_B": ("B", "G"),
        "G_vs_C": ("C", "G"),
        "G_vs_U": ("U", "G"),
        "C_vs_B": ("B", "C"),
        "U_vs_B": ("B", "U"),
    }
    comparisons = {}
    for comparison_name, (reference, candidate) in comparison_specs.items():
        comparisons[comparison_name] = {
            metric_name: paired_summary(
                values[reference][metric_name],
                values[candidate][metric_name],
                definition[3],
                args.bootstrap_resamples,
                args.bootstrap_seed,
            )
            for metric_name, definition in DEFINITIONS.items()
            if metric_name != "relative_motion"
        }

    motion_retention = {}
    for method in ("C", "G", "U"):
        ratios = {
            case_id: 100.0 * values[method]["relative_motion"][case_id] / values["B"]["relative_motion"][case_id]
            for case_id in case_ids
        }
        motion_retention[method] = {
            "aggregate_candidate_over_baseline_percent": 100.0
            * sum(values[method]["relative_motion"].values())
            / sum(values["B"]["relative_motion"].values()),
            "mean_scene_ratio_percent": statistics.fmean(ratios.values()),
            "median_scene_ratio_percent": statistics.median(ratios.values()),
            "per_case": ratios,
        }

    generation = {
        "B": generation_stats(args.baseline_generation_root, METHODS["B"], case_ids),
        "C": generation_stats(args.c2f_generation_root, METHODS["C"], case_ids),
        "G": generation_stats(args.new_generation_root, METHODS["G"], case_ids),
        "U": generation_stats(args.new_generation_root, METHODS["U"], case_ids),
    }
    efficiency = {
        "sampling": generation,
        "geometry_precompute": geometry_stats(args.geometry_precompute_root, case_ids),
        "interpretation_warning": (
            "G cases 4-10 ran concurrently with an unrelated GPU0 workload; raw G wall time is not a clean "
            "method-overhead estimate. The first three uncontended G runs and all U runs are the usable timing evidence."
        ),
    }
    intervention = {
        "G": intervention_stats(args.new_generation_root, METHODS["G"], case_ids),
        "U": intervention_stats(args.new_generation_root, METHODS["U"], case_ids),
    }

    primary = {name: comparisons[name]["met3r_one_second"] for name in ("G_vs_B", "G_vs_C", "G_vs_U")}
    geometry_selection_supported = all(
        primary[name]["mean_directional_improvement"] > 0.0 for name in primary
    )
    if geometry_selection_supported:
        decision = "descriptive_support_requires_dev25_confirmation"
    elif primary["G_vs_U"]["mean_directional_improvement"] <= 0.0:
        decision = "geometry_selection_not_supported_uniform_weakening_is_as_good_or_better"
    else:
        decision = "geometry_gate_reduces_c2f_harm_but_does_not_improve_over_baseline"

    per_case_rows = []
    for case_id in case_ids:
        row: dict[str, object] = {"case_id": case_id, "motion_stratum": strata[case_id]}
        for metric_name in DEFINITIONS:
            for method in METHODS:
                row[f"{metric_name}_{method}"] = values[method][metric_name][case_id]
            for comparison_name, (reference, candidate) in comparison_specs.items():
                direction = DEFINITIONS[metric_name][3]
                sign = 1.0 if direction == "lower" else -1.0
                row[f"{metric_name}_{comparison_name}_directional_improvement"] = sign * (
                    values[reference][metric_name][case_id] - values[candidate][metric_name][case_id]
                )
        row["geometry_survival_fraction"] = intervention["G"]["per_case"][case_id]["survival_fraction"]
        per_case_rows.append(row)

    report = {
        "schema": "wan-c2f-geometry-stage-b-four-way-report-v1",
        "status": "complete",
        "scope": "10-case development experiment_not_validation_or_test",
        "stage_b_lock": {"path": str(args.stage_b_lock.resolve()), "sha256": sha256_file(args.stage_b_lock)},
        "visual_gate": {"path": str(args.visual_gate.resolve()), "sha256": sha256_file(args.visual_gate)},
        "methods": METHODS,
        "case_count": 10,
        "method_means": method_means,
        "comparisons": comparisons,
        "lre": {
            "status": "not_evaluated_no_baseline_eligible_cases",
            "eligibility_lock": {"path": str(args.lre_status.resolve()), "sha256": sha256_file(args.lre_status)},
            "baseline_eligible_count": 0,
        },
        "motion_retention": motion_retention,
        "intervention": intervention,
        "efficiency": efficiency,
        "primary_question": {
            "metric": "met3r_one_second",
            "required_comparisons": ["G_vs_B", "G_vs_C", "G_vs_U"],
            "descriptive_all_three_mean_improvements_positive": geometry_selection_supported,
            "decision": decision,
            "small_sample_warning": (
                "N=10 is a development mechanism test. Bootstrap intervals and exact sign-flip tests are descriptive; "
                "no generalization claim is permitted without the frozen larger split."
            ),
        },
        "per_case": per_case_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / "STAGE_B_FOUR_WAY_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output_dir / "STAGE_B_PER_CASE.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in per_case_rows for key in row}))
        writer.writeheader()
        writer.writerows(per_case_rows)
    print(json.dumps({"decision": decision, "report": str(report_path), "per_case": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
