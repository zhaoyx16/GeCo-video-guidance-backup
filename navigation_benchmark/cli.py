"""Command-line entry points for manifest validation and paired reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .manifest import (
    ManifestValidationError,
    load_records,
    validate_manifest,
    write_records,
)
from .results import aggregate_paired_metric
from .trajectory import evaluate_anchor_trajectory, load_pose_series, trajectory_metric_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Protocol tooling for controlled navigation video benchmarks."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a run/metric manifest")
    validate.add_argument("--manifest", required=True, help="JSON or JSONL manifest path")
    validate.add_argument(
        "--expected-method",
        action="append",
        default=[],
        help="Method that every completed pair must contain; repeat for each arm.",
    )
    validate.add_argument(
        "--require-static-scene",
        action="store_true",
        help="Reject records not marked eligible by the static-scene criteria.",
    )

    trajectory = subparsers.add_parser(
        "trajectory", help="compare externally supplied predicted poses against GT anchors"
    )
    trajectory.add_argument("--gt-poses", required=True, help="JSON GT anchor pose series")
    trajectory.add_argument("--predicted-poses", required=True, help="JSON predicted anchor pose series")
    trajectory.add_argument(
        "--scale-alignment",
        choices=("none", "least_squares"),
        default="none",
        help="Use least_squares only when the pose estimator has unknown global scale.",
    )
    trajectory.add_argument("--run-id", help="Emit metric-result records for this run")
    trajectory.add_argument(
        "--metric-output",
        help="JSON/JSONL destination for trajectory metric records; requires --run-id",
    )
    trajectory.add_argument("--evaluator-name", default="external_pose_estimator")
    trajectory.add_argument("--evaluator-version", default="unspecified")
    trajectory.add_argument("--evaluator-model-id", default="external_pose_estimator")
    trajectory.add_argument("--evaluator-checkpoint-revision", default="unspecified")
    trajectory.add_argument(
        "--evaluator-config-json",
        default='{"pose_convention":"unspecified"}',
        help="JSON object describing pose-estimator configuration.",
    )

    aggregate = subparsers.add_parser(
        "aggregate", help="aggregate a metric over matched baseline/candidate pairs"
    )
    aggregate.add_argument("--manifest", required=True, help="run manifest JSON or JSONL")
    aggregate.add_argument("--metrics", required=True, help="metric result JSON or JSONL")
    aggregate.add_argument("--baseline-method", required=True)
    aggregate.add_argument("--candidate-method", required=True)
    aggregate.add_argument("--metric", required=True, dest="metric_name")
    aggregate.add_argument("--bootstrap-samples", type=int, default=2_000)
    aggregate.add_argument("--random-seed", type=int, default=0)
    aggregate.add_argument("--output", help="optional JSON summary destination")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        issues = validate_manifest(
            load_records(args.manifest),
            expected_methods=args.expected_method,
            require_static_scene=args.require_static_scene,
        )
        print(json.dumps({"valid": not issues, "issues": [issue.as_dict() for issue in issues]}, indent=2))
        return 0 if not issues else 1

    if args.command == "trajectory":
        if bool(args.run_id) != bool(args.metric_output):
            raise SystemExit("--run-id and --metric-output must be supplied together")
        report = evaluate_anchor_trajectory(
            load_pose_series(args.gt_poses),
            load_pose_series(args.predicted_poses),
            scale_alignment=args.scale_alignment,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.run_id:
            try:
                evaluator_config = json.loads(args.evaluator_config_json)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"--evaluator-config-json is invalid JSON: {exc.msg}") from exc
            if not isinstance(evaluator_config, dict):
                raise SystemExit("--evaluator-config-json must decode to an object")
            records = trajectory_metric_records(
                args.run_id,
                report,
                evaluator_name=args.evaluator_name,
                evaluator_version=args.evaluator_version,
                evaluator_model_id=args.evaluator_model_id,
                evaluator_checkpoint_revision=args.evaluator_checkpoint_revision,
                evaluator_config=evaluator_config,
            )
            write_records(args.metric_output, records)
        return 0

    if args.command == "aggregate":
        try:
            summary = aggregate_paired_metric(
                load_records(args.manifest),
                load_records(args.metrics),
                baseline_method=args.baseline_method,
                candidate_method=args.candidate_method,
                metric_name=args.metric_name,
                bootstrap_samples=args.bootstrap_samples,
                random_seed=args.random_seed,
            )
        except ManifestValidationError as exc:
            raise SystemExit(str(exc)) from exc
        payload = summary.as_dict()
        print(json.dumps(payload, indent=2, sort_keys=True))
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
