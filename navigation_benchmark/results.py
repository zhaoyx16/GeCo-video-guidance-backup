"""Fail-closed paired aggregation for the navigation benchmark protocol."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import (
    GENERATION_RUN_RECORD_TYPE,
    METRIC_RESULT_RECORD_TYPE,
    ManifestValidationError,
    ValidationIssue,
    canonical_json,
    statistical_unit_id,
    validate_manifest,
)


@dataclass(frozen=True)
class PairedMetricSummary:
    """A cluster-aware paired comparison for one metric and two named arms."""

    metric_name: str
    direction: str
    baseline_method: str
    candidate_method: str
    pair_count: int
    cluster_count: int
    independent_unit: str
    baseline_mean: float
    candidate_mean: float
    raw_delta_mean: float
    pair_improvement_mean: float
    improvement_mean: float
    improvement_median: float
    fraction_improved: float
    bootstrap_ci95: tuple[float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "direction": self.direction,
            "baseline_method": self.baseline_method,
            "candidate_method": self.candidate_method,
            "pair_count": self.pair_count,
            "cluster_count": self.cluster_count,
            "independent_unit": self.independent_unit,
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "raw_delta_mean": self.raw_delta_mean,
            "pair_improvement_mean": self.pair_improvement_mean,
            "improvement_mean": self.improvement_mean,
            "improvement_median": self.improvement_median,
            "fraction_improved": self.fraction_improved,
            "bootstrap_ci95": list(self.bootstrap_ci95),
        }


def aggregate_paired_metric(
    run_records: Sequence[Mapping[str, Any]],
    metric_records: Sequence[Mapping[str, Any]],
    *,
    baseline_method: str,
    candidate_method: str,
    metric_name: str,
    expected_methods: Sequence[str] | None = None,
    bootstrap_samples: int = 2_000,
    random_seed: int = 0,
    artifact_root: str | Path | None = None,
) -> PairedMetricSummary:
    """Aggregate a metric only when every expected arm is complete and bound.

    This function is intentionally fail-closed.  It validates the complete
    manifest, rejects planned/failed/missing arms and unbound metrics, then
    computes a cluster bootstrap over declared scene/sequence units.  It never
    drops an inconvenient pair from a final benchmark statistic.
    """

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least one")
    expected = tuple(expected_methods or (baseline_method, candidate_method))
    if baseline_method not in expected or candidate_method not in expected:
        raise ValueError("expected_methods must include baseline_method and candidate_method")
    if len(set(expected)) != len(expected):
        raise ValueError("expected_methods must not contain duplicates")

    records = list(run_records) + list(metric_records)
    issues = validate_manifest(
        records,
        expected_methods=expected,
        require_completed=True,
        artifact_root=artifact_root,
    )
    if issues:
        raise ManifestValidationError(issues)

    runs = [record for record in run_records if record.get("record_type") == GENERATION_RUN_RECORD_TYPE]
    metrics = [record for record in metric_records if record.get("record_type") == METRIC_RESULT_RECORD_TYPE]
    methods_by_pair = _methods_by_pair(runs)
    metric_by_run = _required_metric_by_run(metrics, metric_name)

    selected_metric_records: list[Mapping[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for pair_id, methods in sorted(methods_by_pair.items()):
        missing_methods = [method for method in expected if method not in methods]
        if missing_methods:
            raise ValueError(f"{pair_id}: missing expected methods {missing_methods}")
        arm_metrics: dict[str, Mapping[str, Any]] = {}
        for method in expected:
            run = methods[method]
            run_id = str(run["run_id"])
            metric = metric_by_run.get(run_id)
            if metric is None:
                raise ValueError(f"{pair_id}: missing {metric_name} result for expected arm {method}")
            arm_metrics[method] = metric
            selected_metric_records.append(metric)

        baseline_metric = arm_metrics[baseline_method]
        candidate_metric = arm_metrics[candidate_method]
        direction = str(baseline_metric["direction"])
        if candidate_metric.get("direction") != direction:
            raise ValueError(f"{pair_id}: baseline and candidate metric directions differ")
        baseline_value = float(baseline_metric["value"])
        candidate_value = float(candidate_metric["value"])
        raw_delta = candidate_value - baseline_value
        improvement = -raw_delta if direction == "lower_is_better" else raw_delta
        condition = methods[baseline_method]["condition"]
        scene = condition["scene"]
        source_clip = condition["source_clip"]
        statistical_unit = scene["statistical_unit"]
        cluster_level = statistical_unit["level"]
        cluster_id = statistical_unit_id(
            dataset_id=str(scene["dataset_id"]),
            scene_id=str(scene["scene_id"]),
            sequence_id=str(source_clip["sequence_id"]),
            level=str(cluster_level),
        )
        rows.append(
            {
                "pair_id": pair_id,
                "cluster_id": str(cluster_id),
                "cluster_level": str(cluster_level),
                "baseline": baseline_value,
                "candidate": candidate_value,
                "raw_delta": raw_delta,
                "improvement": improvement,
                "direction": direction,
            }
        )

    _assert_evaluator_equivalence(selected_metric_records)
    directions = {row["direction"] for row in rows}
    if len(directions) != 1:
        raise ValueError(f"metric '{metric_name}' has inconsistent directions: {sorted(directions)}")
    unit_levels = {row["cluster_level"] for row in rows}
    if len(unit_levels) != 1:
        raise ValueError("all rows in one aggregate must declare the same independent-unit level")

    by_cluster: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_cluster.setdefault(str(row["cluster_id"]), []).append(row)
    if len(by_cluster) < 2:
        raise ValueError("cluster bootstrap requires at least two independent scene/sequence clusters")
    cluster_rows = [
        {
            "baseline": statistics.fmean(row["baseline"] for row in values),
            "candidate": statistics.fmean(row["candidate"] for row in values),
            "raw_delta": statistics.fmean(row["raw_delta"] for row in values),
            "improvement": statistics.fmean(row["improvement"] for row in values),
        }
        for _, values in sorted(by_cluster.items())
    ]
    cluster_improvements = [float(row["improvement"]) for row in cluster_rows]
    ci_low, ci_high = _cluster_bootstrap_mean_ci(cluster_improvements, bootstrap_samples, random_seed)
    pair_improvements = [float(row["improvement"]) for row in rows]
    return PairedMetricSummary(
        metric_name=metric_name,
        direction=directions.pop(),
        baseline_method=baseline_method,
        candidate_method=candidate_method,
        pair_count=len(rows),
        cluster_count=len(cluster_rows),
        independent_unit=unit_levels.pop(),
        baseline_mean=statistics.fmean(float(row["baseline"]) for row in cluster_rows),
        candidate_mean=statistics.fmean(float(row["candidate"]) for row in cluster_rows),
        raw_delta_mean=statistics.fmean(float(row["raw_delta"]) for row in cluster_rows),
        pair_improvement_mean=statistics.fmean(pair_improvements),
        improvement_mean=statistics.fmean(cluster_improvements),
        improvement_median=statistics.median(cluster_improvements),
        fraction_improved=sum(value > 0 for value in cluster_improvements) / len(cluster_improvements),
        bootstrap_ci95=(ci_low, ci_high),
    )


def _methods_by_pair(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    result: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        pair_id = str(record["pair_id"])
        method = record["method"]
        result.setdefault(pair_id, {})[str(method["name"])] = record
    return result


def _required_metric_by_run(
    metric_records: Sequence[Mapping[str, Any]], metric_name: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    for record in metric_records:
        if record.get("metric_name") != metric_name:
            continue
        run_id = str(record.get("run_id"))
        if run_id in result:
            duplicates.append(run_id)
        result[run_id] = record
    if duplicates:
        raise ManifestValidationError(
            [ValidationIssue(run_id, "metric_name", f"duplicate '{metric_name}' metric result") for run_id in duplicates]
        )
    return result


def _assert_evaluator_equivalence(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise ValueError("no selected metric records")
    reference = records[0].get("evaluator")
    if not isinstance(reference, Mapping):
        raise ValueError("selected metric has no evaluator")
    reference_serialized = canonical_json(reference)
    for record in records[1:]:
        evaluator = record.get("evaluator")
        if not isinstance(evaluator, Mapping) or canonical_json(evaluator) != reference_serialized:
            raise ValueError("all selected metric records must use identical evaluator fingerprint and config")


def _cluster_bootstrap_mean_ci(
    cluster_means: Sequence[float], bootstrap_samples: int, random_seed: int
) -> tuple[float, float]:
    rng = random.Random(random_seed)
    count = len(cluster_means)
    means = [
        math.fsum(cluster_means[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(bootstrap_samples)
    ]
    means.sort()
    lower_index = max(0, int(math.floor(0.025 * (bootstrap_samples - 1))))
    upper_index = min(bootstrap_samples - 1, int(math.ceil(0.975 * (bootstrap_samples - 1))))
    return means[lower_index], means[upper_index]
