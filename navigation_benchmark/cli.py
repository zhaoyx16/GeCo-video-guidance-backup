"""Command-line entry points for strict navigation benchmark protocol tooling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .manifest import ManifestValidationError, load_records, validate_manifest, write_records
from .results import aggregate_paired_metric
from .trajectory import evaluate_bound_anchor_trajectory, load_pose_artifact, trajectory_metric_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Protocol tooling for controlled navigation video benchmarks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a frozen run/metric manifest")
    validate.add_argument("--manifest", required=True, help="JSON or JSONL manifest path")
    validate.add_argument("--expected-method", action="append", default=[], help="Expected arm; repeat for every final pair arm.")
    validate.add_argument("--require-static-scene", action="store_true", help="Reject non-static-scene records.")
    validate.add_argument("--require-completed", action="store_true", help="Reject planned or failed arms in every pair.")

    trajectory = subparsers.add_parser("trajectory", help="evaluate bound external pose artifacts for one completed run")
    trajectory.add_argument("--manifest", required=True, help="Frozen run manifest JSON or JSONL")
    trajectory.add_argument("--run-id", required=True, help="Completed run to evaluate")
    trajectory.add_argument("--reference-pose-artifact", required=True, help="JSON pose artifact bound to source poses")
    trajectory.add_argument("--predicted-pose-artifact", required=True, help="JSON pose artifact bound to generated output")
    trajectory.add_argument("--metric-artifact-uri", required=True, help="Immutable URI of the evaluator output artifact")
    trajectory.add_argument("--metric-artifact-sha256", required=True, help="SHA-256 of the evaluator output artifact")
    trajectory.add_argument("--metric-output", required=True, help="JSON or JSONL destination for metric records")
    trajectory.add_argument("--scale-alignment", choices=("none", "least_squares"), default="none")
    trajectory.add_argument("--evaluator-name", required=True)
    trajectory.add_argument("--evaluator-version", required=True)
    trajectory.add_argument("--evaluator-model-id", required=True)
    trajectory.add_argument("--evaluator-checkpoint-revision", required=True)
    trajectory.add_argument("--evaluator-config-json", required=True, help="Full evaluator configuration JSON object")
    trajectory.add_argument("--evaluator-independence-policy", choices=("independent",), default="independent")

    aggregate = subparsers.add_parser("aggregate", help="fail-closed cluster-bootstrap paired aggregation")
    aggregate.add_argument("--manifest", required=True, help="Split manifest plus run records")
    aggregate.add_argument("--metrics", required=True, help="Metric result JSON or JSONL")
    aggregate.add_argument("--baseline-method", required=True)
    aggregate.add_argument("--candidate-method", required=True)
    aggregate.add_argument("--expected-method", action="append", required=True, help="Every expected arm; repeat for all arms.")
    aggregate.add_argument("--metric", required=True, dest="metric_name")
    aggregate.add_argument("--bootstrap-samples", type=int, default=2_000)
    aggregate.add_argument("--random-seed", type=int, default=0)
    aggregate.add_argument("--output", help="Optional JSON summary destination")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        issues = validate_manifest(
            load_records(args.manifest),
            expected_methods=args.expected_method,
            require_static_scene=args.require_static_scene,
            require_completed=args.require_completed,
            artifact_root=Path(args.manifest).parent,
        )
        print(json.dumps({"valid": not issues, "issues": [issue.as_dict() for issue in issues]}, indent=2))
        return 0 if not issues else 1

    if args.command == "trajectory":
        manifest_records = load_records(args.manifest)
        issues = validate_manifest(manifest_records, artifact_root=Path(args.manifest).parent)
        if issues:
            raise SystemExit("\n".join(str(issue) for issue in issues))
        run = _find_run(manifest_records, args.run_id)
        evaluator_config = _parse_object(args.evaluator_config_json, "--evaluator-config-json")
        evaluator = {
            "name": args.evaluator_name,
            "version": args.evaluator_version,
            "model_id": args.evaluator_model_id,
            "checkpoint_revision": args.evaluator_checkpoint_revision,
            "config": evaluator_config,
            "independence_policy": args.evaluator_independence_policy,
        }
        report = evaluate_bound_anchor_trajectory(
            run,
            load_pose_artifact(args.reference_pose_artifact),
            load_pose_artifact(args.predicted_pose_artifact),
            evaluator=evaluator,
            scale_alignment=args.scale_alignment,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        records = trajectory_metric_records(
            run,
            report,
            evaluator=evaluator,
            metric_artifact={"uri": args.metric_artifact_uri, "sha256": args.metric_artifact_sha256},
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
                expected_methods=args.expected_method,
                bootstrap_samples=args.bootstrap_samples,
                random_seed=args.random_seed,
                artifact_root=Path(args.manifest).parent,
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


def _find_run(records: Sequence[dict[str, object]], run_id: str) -> dict[str, object]:
    matches = [record for record in records if record.get("record_type") == "generation_run" and record.get("run_id") == run_id]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one generation run with run_id={run_id!r}")
    return matches[0]


def _parse_object(value: str, option_name: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{option_name} is invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"{option_name} must decode to an object")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
