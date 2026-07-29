"""Command-line entry points for strict navigation benchmark protocol tooling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .epipolar import (
    EpipolarConfig,
    build_report as build_epipolar_report,
    parse_float_list,
    parse_video_spec,
    write_report as write_epipolar_report,
)
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

    epipolar = subparsers.add_parser(
        "epipolar",
        help="run independent SIFT/MAGSAC epipolar and motion-observability diagnostics",
    )
    epipolar.add_argument(
        "--video",
        action="append",
        required=True,
        type=parse_video_spec,
        help="LABEL=/absolute/path/video.mp4; repeat for each video",
    )
    epipolar.add_argument("--output", required=True, help="Destination JSON report")
    epipolar.add_argument("--lags-sec", default="0.5,1.0", type=parse_float_list)
    epipolar.add_argument("--max-pairs", type=int, default=8)
    epipolar.add_argument("--max-lag-error-sec", type=float, default=0.001)
    epipolar.add_argument("--max-side", type=int, default=960)
    epipolar.add_argument("--sift-features", type=int, default=4096)
    epipolar.add_argument("--ratio-threshold", type=float, default=0.75)
    epipolar.add_argument("--min-matches", type=int, default=24)
    epipolar.add_argument("--min-motion-ratio", type=float, default=0.002)
    epipolar.add_argument("--ransac-threshold-px", type=float, default=1.0)
    epipolar.add_argument("--heldout-fraction", type=float, default=0.3)
    epipolar.add_argument("--heldout-inlier-threshold-px", type=float, default=1.5)
    epipolar.add_argument("--capped-error-px", type=float, default=5.0)
    epipolar.add_argument("--homography-inlier-threshold-px", type=float, default=2.0)
    epipolar.add_argument("--homography-dominance-margin", type=float, default=0.05)
    epipolar.add_argument("--start-frame", type=int, default=0)
    epipolar.add_argument("--end-frame", type=int)
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

    if args.command == "epipolar":
        config = EpipolarConfig(
            lags_sec=args.lags_sec,
            max_pairs_per_lag=args.max_pairs,
            max_lag_error_sec=args.max_lag_error_sec,
            max_side=args.max_side,
            sift_features=args.sift_features,
            ratio_threshold=args.ratio_threshold,
            min_matches=args.min_matches,
            min_motion_ratio=args.min_motion_ratio,
            ransac_threshold_px=args.ransac_threshold_px,
            heldout_fraction=args.heldout_fraction,
            heldout_inlier_threshold_px=args.heldout_inlier_threshold_px,
            capped_error_px=args.capped_error_px,
            homography_inlier_threshold_px=args.homography_inlier_threshold_px,
            homography_dominance_margin=args.homography_dominance_margin,
        )
        report = build_epipolar_report(
            args.video,
            config=config,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
        write_epipolar_report(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
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
