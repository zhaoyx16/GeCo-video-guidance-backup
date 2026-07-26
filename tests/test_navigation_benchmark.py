from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path

from navigation_benchmark.manifest import (
    METRIC_RESULT_RECORD_TYPE,
    SCHEMA_VERSION,
    load_records,
    validate_manifest,
    with_condition_hash,
    with_record_hash,
    write_records,
)
from navigation_benchmark.results import aggregate_paired_metric
from navigation_benchmark.trajectory import evaluate_anchor_trajectory, trajectory_metric_records


HASH = "a" * 64


def valid_condition(seed: int = 7) -> dict:
    return {
        "protocol": {"id": "controlled-navigation-frame-guidance", "version": "v1"},
        "scene": {
            "scene_id": "synthetic-scene",
            "dataset_id": "synthetic-dataset",
            "split": "test",
            "static_scene_eligibility": {
                "eligible": True,
                "criteria_version": "static-navigation-v1",
                "rationale": "Synthetic rigid scene.",
            },
        },
        "source_clip": {
            "source_uri": "dataset://synthetic/sequence-00/clip-01",
            "source_sha256": HASH,
            "sequence_id": "sequence-00",
            "clip_id": "clip-01",
            "start_frame": 0,
            "end_frame": 8,
            "source_fps": 4.0,
            "time_origin": "clip_relative",
            "intrinsics_ref": {
                "uri": "dataset://synthetic/sequence-00/intrinsics.json",
                "format": "json",
                "sha256": HASH,
            },
            "poses_ref": {
                "uri": "dataset://synthetic/sequence-00/poses.json",
                "format": "json",
                "sha256": HASH,
            },
            "anchors": [
                {
                    "role": "first",
                    "frame_index": 0,
                    "timestamp_sec": 0.0,
                    "frame_uri": "dataset://synthetic/frame-000.png",
                    "sha256": HASH,
                },
                {
                    "role": "middle",
                    "frame_index": 4,
                    "timestamp_sec": 1.0,
                    "frame_uri": "dataset://synthetic/frame-004.png",
                    "sha256": HASH,
                },
                {
                    "role": "last",
                    "frame_index": 8,
                    "timestamp_sec": 2.0,
                    "frame_uri": "dataset://synthetic/frame-008.png",
                    "sha256": HASH,
                },
            ],
        },
        "frame_guidance": {
            "enabled": True,
            "anchor_roles": ["first", "middle", "last"],
        },
        "prompt": "A static corridor observed by a moving camera.",
        "seed": seed,
        "model": {"model_id": "synthetic-vdm", "checkpoint_revision": "test-revision"},
        "sampling": {
            "height": 256,
            "width": 448,
            "num_frames": 9,
            "fps": 4.0,
            "num_inference_steps": 5,
            "scheduler": {"name": "synthetic-flow-match"},
        },
    }


def valid_run(
    *,
    run_id: str,
    pair_id: str,
    method_name: str,
    schedule_state: str,
    seed: int = 7,
    completed: bool = False,
) -> dict:
    run = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "generation_run",
        "run_id": run_id,
        "pair_id": pair_id,
        "status": "completed" if completed else "planned",
        "method": {
            "name": method_name,
            "version": "test-method-v1",
            "parameters": {"guidance_lr": 0.0 if schedule_state != "active_guidance" else 1.0},
            "mechanism": {
                "id": "synthetic-geometry-guidance",
                "version": "test-mechanism-v1",
                "time_travel": "absent",
                "temporal_vae_context": "past_only_approximate",
                "guidance_schedule_state": schedule_state,
            },
        },
        "condition": valid_condition(seed),
        "output": {"video_uri": f"outputs://synthetic/{run_id}.mp4"},
        "execution": None,
    }
    if completed:
        run["execution"] = {
            "git_commit": "deadbeef",
            "runtime_sec": 12.5,
            "devices": {"video_diffusion": "cuda:0", "vae": "cuda:1"},
            "peak_vram_mib": {
                "cuda:0": {"allocated": 1024.0, "reserved": 1536.0},
                "cuda:1": {"allocated": 512.0, "reserved": 768.0},
            },
        }
    run = with_condition_hash(run)
    return with_record_hash(run) if completed else run


def metric(run_id: str, value: float, *, direction: str = "lower_is_better") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": METRIC_RESULT_RECORD_TYPE,
        "run_id": run_id,
        "metric_name": "geco_fused",
        "metric_role": "guidance_aligned",
        "direction": direction,
        "value": value,
        "evaluator": {
            "name": "synthetic-geco-eval",
            "version": "v1",
            "model_id": "synthetic-vggt-ufm",
            "checkpoint_revision": "test-revision",
            "config": {"window_sec": 1.0},
        },
    }


def pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ]


class NavigationBenchmarkManifestTests(unittest.TestCase):
    def test_valid_three_arm_pair(self) -> None:
        runs = [
            valid_run(
                run_id="fg-only",
                pair_id="scene-seed-7",
                method_name="fg_only",
                schedule_state="baseline_all_zero_schedule",
            ),
            valid_run(
                run_id="fg-rgb",
                pair_id="scene-seed-7",
                method_name="fg_rgb_geco",
                schedule_state="active_guidance",
            ),
            valid_run(
                run_id="fg-latent",
                pair_id="scene-seed-7",
                method_name="fg_latent_geometry",
                schedule_state="active_guidance",
            ),
        ]
        issues = validate_manifest(
            runs,
            expected_methods=("fg_only", "fg_rgb_geco", "fg_latent_geometry"),
            require_static_scene=True,
        )
        self.assertEqual([], issues)

    def test_pair_validation_detects_non_method_difference(self) -> None:
        baseline = valid_run(
            run_id="baseline",
            pair_id="scene-seed-7",
            method_name="fg_only",
            schedule_state="baseline_all_zero_schedule",
        )
        candidate = valid_run(
            run_id="candidate",
            pair_id="scene-seed-7",
            method_name="fg_rgb_geco",
            schedule_state="active_guidance",
        )
        candidate["condition"]["sampling"]["num_frames"] = 11
        candidate = with_condition_hash(candidate)
        issues = validate_manifest([baseline, candidate])
        self.assertTrue(any(issue.field == "condition" for issue in issues))

    def test_schedule_state_is_not_optional(self) -> None:
        run = valid_run(
            run_id="zero-lr",
            pair_id="scene-seed-7",
            method_name="fg_rgb_geco",
            schedule_state="positive_schedule_zero_lr",
        )
        del run["method"]["mechanism"]["guidance_schedule_state"]
        run = with_condition_hash(run)
        issues = validate_manifest([run])
        self.assertTrue(
            any(issue.field == "method.mechanism.guidance_schedule_state" for issue in issues)
        )

    def test_completed_run_requires_detailed_execution_provenance(self) -> None:
        run = valid_run(
            run_id="complete",
            pair_id="scene-seed-7",
            method_name="fg_only",
            schedule_state="baseline_all_zero_schedule",
            completed=True,
        )
        self.assertEqual([], validate_manifest([run]))
        invalid = copy.deepcopy(run)
        del invalid["execution"]["peak_vram_mib"]["cuda:0"]["reserved"]
        invalid = with_record_hash(invalid)
        issues = validate_manifest([invalid])
        self.assertTrue(
            any(issue.field == "execution.peak_vram_mib.cuda:0.reserved" for issue in issues)
        )

    def test_clip_relative_anchor_timestamps_handle_nonzero_start(self) -> None:
        run = valid_run(
            run_id="nonzero-start",
            pair_id="scene-seed-7",
            method_name="fg_only",
            schedule_state="baseline_all_zero_schedule",
        )
        source = run["condition"]["source_clip"]
        source["start_frame"] = 100
        source["end_frame"] = 108
        for anchor, frame_index in zip(source["anchors"], (100, 104, 108)):
            anchor["frame_index"] = frame_index
            anchor["timestamp_sec"] = (frame_index - 100) / 4.0
        run = with_condition_hash(run)
        self.assertEqual([], validate_manifest([run]))

    def test_jsonl_round_trip(self) -> None:
        records = [
            valid_run(
                run_id="round-trip",
                pair_id="scene-seed-7",
                method_name="fg_only",
                schedule_state="baseline_all_zero_schedule",
            )
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "runs.jsonl"
            write_records(path, records)
            self.assertEqual(records, load_records(path))


class TrajectoryTests(unittest.TestCase):
    def test_exact_trajectory_has_zero_error(self) -> None:
        gt = [pose(), pose(1.0), pose(2.0)]
        report = evaluate_anchor_trajectory(gt, gt)
        self.assertAlmostEqual(0.0, float(report["translation_rmse"]))
        self.assertAlmostEqual(0.0, float(report["rotation_deg_mean"]))
        self.assertAlmostEqual(1.0, float(report["motion_ratio"]))

    def test_least_squares_scale_alignment(self) -> None:
        gt = [pose(), pose(1.0), pose(2.0)]
        predicted = [pose(), pose(0.5), pose(1.0)]
        report = evaluate_anchor_trajectory(gt, predicted, scale_alignment="least_squares")
        self.assertAlmostEqual(2.0, float(report["translation_scale_factor"]))
        self.assertAlmostEqual(0.0, float(report["translation_rmse"]))
        metric_records = trajectory_metric_records(
            "run-id",
            report,
            evaluator_name="synthetic-pose",
            evaluator_version="v1",
            evaluator_model_id="synthetic-model",
            evaluator_checkpoint_revision="r1",
            evaluator_config={"pose_convention": "world_from_camera"},
        )
        self.assertEqual([], validate_manifest(metric_records))


class PairedAggregationTests(unittest.TestCase):
    def test_lower_is_better_aggregation(self) -> None:
        runs = [
            valid_run(
                run_id="baseline-a",
                pair_id="pair-a",
                method_name="fg_only",
                schedule_state="baseline_all_zero_schedule",
            ),
            valid_run(
                run_id="candidate-a",
                pair_id="pair-a",
                method_name="fg_rgb_geco",
                schedule_state="active_guidance",
            ),
            valid_run(
                run_id="baseline-b",
                pair_id="pair-b",
                method_name="fg_only",
                schedule_state="baseline_all_zero_schedule",
                seed=8,
            ),
            valid_run(
                run_id="candidate-b",
                pair_id="pair-b",
                method_name="fg_rgb_geco",
                schedule_state="active_guidance",
                seed=8,
            ),
        ]
        metrics = [
            metric("baseline-a", 1.0),
            metric("candidate-a", 0.5),
            metric("baseline-b", 2.0),
            metric("candidate-b", 1.5),
        ]
        summary = aggregate_paired_metric(
            runs,
            metrics,
            baseline_method="fg_only",
            candidate_method="fg_rgb_geco",
            metric_name="geco_fused",
            bootstrap_samples=100,
            random_seed=3,
        )
        self.assertEqual(2, summary.pair_count)
        self.assertAlmostEqual(0.5, summary.improvement_mean)
        self.assertAlmostEqual(1.0, summary.fraction_improved)
        self.assertTrue(math.isfinite(summary.bootstrap_ci95[0]))


if __name__ == "__main__":
    unittest.main()
