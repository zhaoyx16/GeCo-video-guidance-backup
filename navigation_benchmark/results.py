"""Paired metric aggregation for the navigation benchmark protocol."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .manifest import (
    GENERATION_RUN_RECORD_TYPE,
    METRIC_RESULT_RECORD_TYPE,
    ManifestValidationError,
    ValidationIssue,
    validate_metric_result,
    validate_paired_conditions,
)


@dataclass(frozen=True)
class PairedMetricSummary:
    """Aggregate paired improvement for one candidate method and one metric."""

    metric_name: str
    direction: str
    baseline_method: str
    candidate_method: str
    pair_count: int
    baseline_mean: float
    candidate_mean: float
    raw_delta_mean: float
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
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "raw_delta_mean": self.raw_delta_mean,
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
    bootstrap_samples: int = 2_000,
    random_seed: int = 0,
) -> PairedMetricSummary:
    """Aggregate candidate-vs-baseline results using matched pair identifiers.

    Improvement values are always positive when the candidate is better,
    regardless of whether the underlying metric is higher-is-better or
    lower-is-better.
    """

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least one")

    runs = [
        record
        for record in run_records
        if record.get("record_type") == GENERATION_RUN_RECORD_TYPE
    ]
    pairing_issues = validate_paired_conditions(runs)
    if pairing_issues:
        raise ManifestValidationError(pairing_issues)

    method_by_pair: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in runs:
        pair_id = record.get("pair_id")
        method = record.get("method")
        method_name = method.get("name") if isinstance(method, Mapping) else None
        if isinstance(pair_id, str) and isinstance(method_name, str):
            method_by_pair.setdefault(pair_id, {})[method_name] = record

    metric_issues: list[ValidationIssue] = []
    selected_values: dict[str, tuple[float, str]] = {}
    for metric in metric_records:
        if metric.get("record_type") != METRIC_RESULT_RECORD_TYPE:
            continue
        metric_issues.extend(validate_metric_result(metric))
        if metric.get("metric_name") != metric_name:
            continue
        run_id = metric.get("run_id")
        if not isinstance(run_id, str):
            continue
        if run_id in selected_values:
            metric_issues.append(
                ValidationIssue(run_id, "metric_name", f"duplicate '{metric_name}' metric result")
            )
            continue
        selected_values[run_id] = (float(metric["value"]), str(metric["direction"]))
    if metric_issues:
        raise ManifestValidationError(metric_issues)

    baseline_values: list[float] = []
    candidate_values: list[float] = []
    improvements: list[float] = []
    raw_deltas: list[float] = []
    directions: set[str] = set()
    missing: list[str] = []

    for pair_id, methods in sorted(method_by_pair.items()):
        baseline = methods.get(baseline_method)
        candidate = methods.get(candidate_method)
        if baseline is None or candidate is None:
            continue
        baseline_id = str(baseline["run_id"])
        candidate_id = str(candidate["run_id"])
        if baseline_id not in selected_values or candidate_id not in selected_values:
            missing.append(pair_id)
            continue
        baseline_value, baseline_direction = selected_values[baseline_id]
        candidate_value, candidate_direction = selected_values[candidate_id]
        if baseline_direction != candidate_direction:
            raise ValueError(f"{pair_id}: metric directions differ between paired methods")
        directions.add(baseline_direction)
        raw_delta = candidate_value - baseline_value
        improvement = -raw_delta if baseline_direction == "lower_is_better" else raw_delta
        baseline_values.append(baseline_value)
        candidate_values.append(candidate_value)
        raw_deltas.append(raw_delta)
        improvements.append(improvement)

    if missing:
        raise ValueError(
            f"missing '{metric_name}' results for paired runs: {', '.join(sorted(missing))}"
        )
    if not improvements:
        raise ValueError(
            f"no complete pairs found for baseline='{baseline_method}', "
            f"candidate='{candidate_method}', metric='{metric_name}'"
        )
    if len(directions) != 1:
        raise ValueError(f"metric '{metric_name}' has inconsistent directions: {sorted(directions)}")

    ci_low, ci_high = _bootstrap_mean_ci(improvements, bootstrap_samples, random_seed)
    return PairedMetricSummary(
        metric_name=metric_name,
        direction=directions.pop(),
        baseline_method=baseline_method,
        candidate_method=candidate_method,
        pair_count=len(improvements),
        baseline_mean=statistics.fmean(baseline_values),
        candidate_mean=statistics.fmean(candidate_values),
        raw_delta_mean=statistics.fmean(raw_deltas),
        improvement_mean=statistics.fmean(improvements),
        improvement_median=statistics.median(improvements),
        fraction_improved=sum(value > 0 for value in improvements) / len(improvements),
        bootstrap_ci95=(ci_low, ci_high),
    )


def _bootstrap_mean_ci(
    values: Sequence[float],
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[float, float]:
    rng = random.Random(random_seed)
    n = len(values)
    means = [
        math.fsum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(bootstrap_samples)
    ]
    means.sort()
    lower_index = max(0, int(math.floor(0.025 * (bootstrap_samples - 1))))
    upper_index = min(bootstrap_samples - 1, int(math.ceil(0.975 * (bootstrap_samples - 1))))
    return means[lower_index], means[upper_index]
