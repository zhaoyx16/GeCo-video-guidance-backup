from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from navigation_benchmark.manifest import (
    METRIC_RESULT_RECORD_TYPE,
    SCHEMA_VERSION,
    ManifestValidationError,
    load_records,
    statistical_unit_id,
    validate_manifest,
    with_condition_hash,
    with_evaluator_fingerprint,
    with_record_hash,
    with_split_manifest_hash,
    write_records,
)
from navigation_benchmark.results import aggregate_paired_metric
from navigation_benchmark.trajectory import evaluate_bound_anchor_trajectory, trajectory_metric_records


_TEST_ARTIFACT_DIRECTORY = Path(tempfile.mkdtemp(prefix="geco-navigation-protocol-"))


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def split_assignment(scene_id: str = "synthetic-scene", sequence_id: str = "sequence-00", split: str = "test") -> dict:
    return {
        "dataset_id": "synthetic-dataset",
        "scene_id": scene_id,
        "sequence_id": sequence_id,
        "split": split,
    }


def frozen_split_source(manifest_id: str, assignments: list[dict]) -> dict:
    registry_id = f"synthetic-registry-{manifest_id}"
    version = "2026-07-26"
    payload = {"registry_id": registry_id, "version": version, "assignments": assignments}
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    path = _TEST_ARTIFACT_DIRECTORY / f"{digest(serialized)}.json"
    path.write_text(serialized, encoding="utf-8")
    return {
        "uri": f"registry://synthetic/splits/{manifest_id}.json",
        "version": version,
        "registry_id": registry_id,
        "artifact_path": str(path),
        "format": "json",
        "expected_sha256": digest(serialized),
    }


def valid_split_manifest(*assignments: dict, manifest_id: str = "synthetic-split-v1") -> dict:
    assignment_list = list(assignments or (split_assignment(),))
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "split_manifest",
        "split_manifest_id": manifest_id,
        "status": "frozen",
        "frozen_source": frozen_split_source(manifest_id, assignment_list),
        "assignments": assignment_list,
    }
    return with_split_manifest_hash(record)


def valid_condition(
    split_manifest: dict,
    *,
    seed: int = 7,
    scene_id: str = "synthetic-scene",
    sequence_id: str = "sequence-00",
    split: str = "test",
    start_frame: int = 0,
    end_frame: int = 8,
) -> dict:
    middle = start_frame + (end_frame - start_frame) // 2
    source_fps = 4.0
    source_time = lambda frame: (frame - start_frame) / source_fps
    generated_frames = 9
    generated_fps = 4.0
    generated_middle = (generated_frames - 1) // 2
    anchors = []
    generated_anchor_map = []
    for role, source_frame, generated_frame in (
        ("first", start_frame, 0),
        ("middle", middle, generated_middle),
        ("last", end_frame, generated_frames - 1),
    ):
        anchors.append(
            {
                "role": role,
                "frame_index": source_frame,
                "timestamp_sec": source_time(source_frame),
                "frame_uri": f"dataset://synthetic/{sequence_id}/frame-{source_frame:05d}.png",
                "sha256": digest(f"anchor-{sequence_id}-{source_frame}"),
            }
        )
        generated_anchor_map.append(
            {
                "role": role,
                "source_frame_index": source_frame,
                "source_timestamp_sec": source_time(source_frame),
                "generated_frame_index": generated_frame,
                "generated_timestamp_sec": generated_frame / generated_fps,
            }
        )
    return {
        "protocol": {"id": "controlled-navigation-frame-guidance", "version": "v1"},
        "split_manifest": {
            "id": split_manifest["split_manifest_id"],
            "sha256": split_manifest["split_manifest_hash"],
            "frozen_source_expected_sha256": split_manifest["frozen_source"]["expected_sha256"],
        },
        "scene": {
            "scene_id": scene_id,
            "dataset_id": "synthetic-dataset",
            "split": split,
            "statistical_unit": {
                "cluster_id": statistical_unit_id(
                    dataset_id="synthetic-dataset",
                    scene_id=scene_id,
                    sequence_id=sequence_id,
                    level="sequence",
                ),
                "level": "sequence",
            },
            "static_scene_eligibility": {
                "eligible": True,
                "criteria_version": "static-navigation-v1",
                "rationale": "Synthetic rigid scene.",
            },
        },
        "source_clip": {
            "source_uri": f"dataset://synthetic/{sequence_id}/clip-01",
            "source_sha256": digest(f"source-{sequence_id}"),
            "sequence_id": sequence_id,
            "clip_id": "clip-01",
            "start_frame": start_frame,
            "end_frame": end_frame,
            "source_fps": source_fps,
            "time_origin": "clip_relative",
            "anchor_policy": {"id": "first_middle_last_floor_v1", "middle_rule": "floor_midpoint"},
            "intrinsics_ref": {
                "uri": f"dataset://synthetic/{sequence_id}/intrinsics.json",
                "format": "json",
                "sha256": digest(f"intrinsics-{sequence_id}"),
            },
            "poses_ref": {
                "uri": f"dataset://synthetic/{sequence_id}/poses.json",
                "format": "json",
                "sha256": digest(f"poses-{sequence_id}"),
                "pose_convention": "C2W",
                "translation_unit": "meters",
            },
            "anchors": anchors,
        },
        "frame_guidance": {
            "enabled": True,
            "anchor_roles": ["first", "middle", "last"],
            "generated_timing": {
                "mapping_policy": "first_middle_last_index_v1",
                "generated_frame_count": generated_frames,
                "generated_fps": generated_fps,
                "anchor_map": generated_anchor_map,
            },
        },
        "prompt": "A static corridor observed by a moving camera.",
        "seed": seed,
        "model": {
            "model_id": "synthetic-vdm",
            "checkpoint_revision": "test-revision",
            "config": {"variant": "synthetic"},
        },
        "sampling": {
            "height": 256,
            "width": 448,
            "num_frames": generated_frames,
            "fps": generated_fps,
            "num_inference_steps": 5,
            "scheduler": {"name": "synthetic-flow-match", "config": {"shift": 1.0}},
        },
    }


def valid_run(
    split_manifest: dict,
    *,
    run_id: str,
    pair_id: str,
    method_name: str,
    schedule_state: str,
    completed: bool = False,
    seed: int = 7,
    scene_id: str = "synthetic-scene",
    sequence_id: str = "sequence-00",
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
        "condition": valid_condition(
            split_manifest,
            seed=seed,
            scene_id=scene_id,
            sequence_id=sequence_id,
        ),
        "output": {"video_uri": f"outputs://synthetic/{run_id}.mp4"},
        "execution": None,
    }
    if completed:
        run["output"]["sha256"] = digest(f"video-{run_id}")
        run["execution"] = {
            "git_commit": "deadbeef",
            "runtime_sec": 12.5,
            "devices": {"video_diffusion": "cuda:0", "vae": "cuda:1", "metric": "cuda:2"},
            "peak_vram_mib": {
                "cuda:0": {"allocated": 1024.0, "reserved": 1536.0},
                "cuda:1": {"allocated": 512.0, "reserved": 768.0},
                "cuda:2": {"allocated": 256.0, "reserved": 384.0},
            },
        }
    run = with_condition_hash(run)
    return with_record_hash(run) if completed else run


def rehash_run(run: dict) -> dict:
    run = with_condition_hash(run)
    return with_record_hash(run) if run["status"] == "completed" else run


def default_evaluator(*, policy: str = "guidance_aligned", config: dict | None = None) -> dict:
    return with_evaluator_fingerprint(
        {
            "name": "synthetic-evaluator",
            "version": "v1",
            "model_id": "synthetic-model",
            "checkpoint_revision": "test-revision",
            "config": config or {"window_sec": 1.0},
            "independence_policy": policy,
        }
    )


def metric(
    run: dict,
    value: float,
    *,
    metric_name: str = "geco_fused",
    metric_role: str = "guidance_aligned",
    evaluator: dict | None = None,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": METRIC_RESULT_RECORD_TYPE,
        "run_id": run["run_id"],
        "run_record_hash": run["record_hash"],
        "evaluated_output_sha256": run["output"]["sha256"],
        "metric_name": metric_name,
        "metric_role": metric_role,
        "direction": "lower_is_better",
        "value": value,
        "evaluator": evaluator or default_evaluator(),
        "metric_artifact": {"uri": f"metrics://synthetic/{run['run_id']}/{metric_name}.json", "sha256": digest(f"metric-{run['run_id']}-{metric_name}")},
    }


def pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> list[list[float]]:
    return [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]]


def reference_pose_artifact(run: dict, poses: list[list[list[float]]]) -> dict:
    source = run["condition"]["source_clip"]
    return {
        "uri": "dataset://synthetic/poses.json",
        "sha256": source["poses_ref"]["sha256"],
        "pose_convention": "C2W",
        "translation_unit": "meters",
        "anchor_poses": [
            {
                "role": anchor["role"],
                "source_frame_index": anchor["frame_index"],
                "source_timestamp_sec": anchor["timestamp_sec"],
                "matrix": matrix,
            }
            for anchor, matrix in zip(source["anchors"], poses)
        ],
    }


def predicted_pose_artifact(run: dict, poses: list[list[list[float]]]) -> dict:
    timing = run["condition"]["frame_guidance"]["generated_timing"]
    return {
        "uri": f"metrics://synthetic/{run['run_id']}/predicted-poses.json",
        "sha256": digest(f"predicted-poses-{run['run_id']}"),
        "input_video_sha256": run["output"]["sha256"],
        "pose_convention": "C2W",
        "translation_unit": "meters",
        "anchor_poses": [
            {
                "role": mapping["role"],
                "generated_frame_index": mapping["generated_frame_index"],
                "generated_timestamp_sec": mapping["generated_timestamp_sec"],
                "matrix": matrix,
            }
            for mapping, matrix in zip(timing["anchor_map"], poses)
        ],
    }


class NavigationBenchmarkManifestTests(unittest.TestCase):
    def test_valid_three_arm_pair_with_frozen_split(self) -> None:
        split = valid_split_manifest()
        runs = [
            valid_run(split, run_id="fg-only", pair_id="scene-seed-7", method_name="fg_only", schedule_state="baseline_all_zero_schedule"),
            valid_run(split, run_id="fg-rgb", pair_id="scene-seed-7", method_name="fg_rgb_geco", schedule_state="active_guidance"),
            valid_run(split, run_id="fg-latent", pair_id="scene-seed-7", method_name="fg_latent_geometry", schedule_state="active_guidance"),
        ]
        self.assertEqual([], validate_manifest([split, *runs], expected_methods=("fg_only", "fg_rgb_geco", "fg_latent_geometry"), require_static_scene=True))

    def test_global_sequence_disjointness_across_splits(self) -> None:
        test_split = valid_split_manifest(split_assignment(split="test"), manifest_id="test-split")
        train_split = valid_split_manifest(split_assignment(split="train"), manifest_id="train-split")
        issues = validate_manifest([test_split, train_split])
        self.assertTrue(any("globally assigned" in issue.message for issue in issues))

    def test_run_rejects_nonmatching_frozen_split_hash(self) -> None:
        split = valid_split_manifest()
        run = valid_run(split, run_id="run", pair_id="pair", method_name="fg_only", schedule_state="baseline_all_zero_schedule")
        run["condition"]["split_manifest"]["sha256"] = digest("wrong")
        run = rehash_run(run)
        issues = validate_manifest([split, run])
        self.assertTrue(any(issue.field == "condition.split_manifest.sha256" for issue in issues))

    def test_split_requires_external_frozen_registry_pin(self) -> None:
        split = valid_split_manifest()
        del split["frozen_source"]
        issues = validate_manifest([split])
        self.assertTrue(any(issue.field == "frozen_source" for issue in issues))

        split = valid_split_manifest(manifest_id="tampered-split")
        Path(split["frozen_source"]["artifact_path"]).write_text("{}\n", encoding="utf-8")
        issues = validate_manifest([split])
        self.assertTrue(any(issue.field == "frozen_source.expected_sha256" for issue in issues))

        split = valid_split_manifest(manifest_id="assignment-mismatch")
        payload = {
            "registry_id": split["frozen_source"]["registry_id"],
            "version": split["frozen_source"]["version"],
            "assignments": [],
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        Path(split["frozen_source"]["artifact_path"]).write_text(serialized, encoding="utf-8")
        split["frozen_source"]["expected_sha256"] = digest(serialized)
        split = with_split_manifest_hash(split)
        issues = validate_manifest([split])
        self.assertTrue(any(issue.field == "assignments" and "frozen split artifact" in issue.message for issue in issues))

        split = valid_split_manifest()
        run = valid_run(split, run_id="run", pair_id="pair", method_name="fg_only", schedule_state="baseline_all_zero_schedule")
        del run["condition"]["split_manifest"]["frozen_source_expected_sha256"]
        run = rehash_run(run)
        issues = validate_manifest([split, run])
        self.assertTrue(any(issue.field == "condition.split_manifest.frozen_source_expected_sha256" for issue in issues))

    def test_relative_split_artifact_uses_manifest_artifact_root(self) -> None:
        split = valid_split_manifest(manifest_id="relative-source")
        artifact_path = Path(split["frozen_source"]["artifact_path"])
        split["frozen_source"]["artifact_path"] = artifact_path.name
        split = with_split_manifest_hash(split)
        issues = validate_manifest([split])
        self.assertTrue(any(issue.field == "frozen_source.artifact_path" for issue in issues))
        self.assertEqual([], validate_manifest([split], artifact_root=artifact_path.parent))

    def test_statistical_unit_must_be_derived_from_source_identity(self) -> None:
        split = valid_split_manifest()
        run = valid_run(split, run_id="cluster", pair_id="pair", method_name="fg_only", schedule_state="baseline_all_zero_schedule")
        run["condition"]["scene"]["statistical_unit"]["cluster_id"] = "free-form-cluster"
        run = rehash_run(run)
        issues = validate_manifest([split, run])
        self.assertTrue(any(issue.field == "condition.scene.statistical_unit.cluster_id" for issue in issues))

    def test_pair_validation_detects_nonmethod_difference(self) -> None:
        split = valid_split_manifest()
        baseline = valid_run(split, run_id="baseline", pair_id="pair", method_name="fg_only", schedule_state="baseline_all_zero_schedule")
        candidate = valid_run(split, run_id="candidate", pair_id="pair", method_name="fg_rgb_geco", schedule_state="active_guidance")
        candidate["condition"]["sampling"]["num_inference_steps"] = 8
        candidate = rehash_run(candidate)
        issues = validate_manifest([split, baseline, candidate])
        self.assertTrue(any(issue.field == "condition" for issue in issues))

    def test_contractual_anchors_reject_nonendpoint_or_nondeterministic_middle(self) -> None:
        split = valid_split_manifest()
        run = valid_run(split, run_id="anchors", pair_id="pair", method_name="fg_only", schedule_state="baseline_all_zero_schedule")
        run["condition"]["source_clip"]["anchors"][0]["frame_index"] = 1
        run["condition"]["source_clip"]["anchors"][1]["frame_index"] = 3
        run = rehash_run(run)
        issues = validate_manifest([split, run])
        fields = {issue.field for issue in issues}
        self.assertIn("condition.source_clip.anchors[0].frame_index", fields)
        self.assertIn("condition.source_clip.anchors[1].frame_index", fields)

    def test_contractual_generated_mapping_rejects_wrong_middle_frame(self) -> None:
        split = valid_split_manifest()
        run = valid_run(split, run_id="mapping", pair_id="pair", method_name="fg_only", schedule_state="baseline_all_zero_schedule")
        run["condition"]["frame_guidance"]["generated_timing"]["anchor_map"][1]["generated_frame_index"] = 3
        run = rehash_run(run)
        issues = validate_manifest([split, run])
        self.assertTrue(any("generated_frame_index" in issue.field for issue in issues))

    def test_frame_guidance_requires_three_distinct_generated_frames(self) -> None:
        split = valid_split_manifest()
        run = valid_run(split, run_id="short", pair_id="pair", method_name="fg_only", schedule_state="baseline_all_zero_schedule")
        run["condition"]["sampling"]["num_frames"] = 1
        timing = run["condition"]["frame_guidance"]["generated_timing"]
        timing["generated_frame_count"] = 1
        for mapping in timing["anchor_map"]:
            mapping["generated_frame_index"] = 0
            mapping["generated_timestamp_sec"] = 0.0
        run = rehash_run(run)
        issues = validate_manifest([split, run])
        fields = {issue.field for issue in issues}
        self.assertIn("condition.sampling.num_frames", fields)
        self.assertIn("condition.frame_guidance.generated_timing.anchor_map", fields)

    def test_completed_run_requires_hash_valid_output_and_provenance(self) -> None:
        split = valid_split_manifest()
        run = valid_run(split, run_id="complete", pair_id="pair", method_name="fg_only", schedule_state="baseline_all_zero_schedule", completed=True)
        self.assertEqual([], validate_manifest([split, run]))
        invalid = copy.deepcopy(run)
        del invalid["output"]["sha256"]
        invalid = with_record_hash(invalid)
        issues = validate_manifest([split, invalid])
        self.assertTrue(any(issue.field == "output.sha256" for issue in issues))

    def test_independent_metric_role_rejects_guidance_aligned_evaluator(self) -> None:
        split = valid_split_manifest()
        run = valid_run(split, run_id="metric-policy", pair_id="pair", method_name="fg_only", schedule_state="baseline_all_zero_schedule", completed=True)
        result = metric(
            run,
            1.0,
            metric_role="independent_geometry",
            evaluator=default_evaluator(policy="guidance_aligned"),
        )
        issues = validate_manifest([split, run, result])
        self.assertTrue(any(issue.field == "evaluator.independence_policy" for issue in issues))

    def test_jsonl_round_trip(self) -> None:
        split = valid_split_manifest()
        run = valid_run(split, run_id="round", pair_id="pair", method_name="fg_only", schedule_state="baseline_all_zero_schedule")
        records = [split, run]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "runs.jsonl"
            write_records(path, records)
            self.assertEqual(records, load_records(path))

    def test_schema_declares_extensions_as_only_open_ended_policy(self) -> None:
        schema_path = Path(__file__).parents[1] / "schemas" / "navigation_benchmark_v1.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertFalse(schema["$defs"]["generationRun"]["additionalProperties"])
        self.assertTrue(schema["$defs"]["extensions"]["additionalProperties"])
        self.assertIn("splitManifest", schema["$defs"])


class TrajectoryBindingTests(unittest.TestCase):
    def test_bound_exact_trajectory_has_zero_error_and_hashes(self) -> None:
        split = valid_split_manifest()
        run = valid_run(split, run_id="trajectory", pair_id="pair", method_name="fg_only", schedule_state="baseline_all_zero_schedule", completed=True)
        poses = [pose(), pose(1.0), pose(2.0)]
        evaluator = default_evaluator(policy="independent", config={"backend": "synthetic-pose"})
        report = evaluate_bound_anchor_trajectory(run, reference_pose_artifact(run, poses), predicted_pose_artifact(run, poses), evaluator=evaluator)
        self.assertAlmostEqual(0.0, float(report["translation_rmse"]))
        self.assertAlmostEqual(1.0, float(report["motion_ratio"]))
        metrics = trajectory_metric_records(run, report, evaluator=evaluator, metric_artifact={"uri": "metrics://trajectory/report.json", "sha256": digest("trajectory-report")})
        self.assertEqual([], validate_manifest([split, run, *metrics]))

    def test_bound_trajectory_rejects_unbound_output_or_reference(self) -> None:
        split = valid_split_manifest()
        run = valid_run(split, run_id="trajectory", pair_id="pair", method_name="fg_only", schedule_state="baseline_all_zero_schedule", completed=True)
        poses = [pose(), pose(1.0), pose(2.0)]
        predicted = predicted_pose_artifact(run, poses)
        predicted["input_video_sha256"] = digest("other-video")
        with self.assertRaisesRegex(ValueError, "not bound to the completed output"):
            evaluate_bound_anchor_trajectory(run, reference_pose_artifact(run, poses), predicted, evaluator=default_evaluator(policy="independent"))
        reference = reference_pose_artifact(run, poses)
        reference["sha256"] = digest("wrong-reference")
        with self.assertRaisesRegex(ValueError, "does not match source_clip.poses_ref"):
            evaluate_bound_anchor_trajectory(run, reference, predicted_pose_artifact(run, poses), evaluator=default_evaluator(policy="independent"))

    def test_trajectory_rejects_guidance_aligned_evaluator(self) -> None:
        split = valid_split_manifest()
        run = valid_run(split, run_id="trajectory", pair_id="pair", method_name="fg_only", schedule_state="baseline_all_zero_schedule", completed=True)
        poses = [pose(), pose(1.0), pose(2.0)]
        with self.assertRaisesRegex(ValueError, "independence_policy=independent"):
            evaluate_bound_anchor_trajectory(run, reference_pose_artifact(run, poses), predicted_pose_artifact(run, poses), evaluator=default_evaluator(policy="guidance_aligned"))

    def test_trajectory_metric_rejects_binding_not_matching_its_run(self) -> None:
        split = valid_split_manifest()
        run = valid_run(split, run_id="trajectory-binding", pair_id="pair", method_name="fg_only", schedule_state="baseline_all_zero_schedule", completed=True)
        poses = [pose(), pose(1.0), pose(2.0)]
        evaluator = default_evaluator(policy="independent", config={"backend": "synthetic-pose"})
        report = evaluate_bound_anchor_trajectory(run, reference_pose_artifact(run, poses), predicted_pose_artifact(run, poses), evaluator=evaluator)
        metrics = trajectory_metric_records(run, report, evaluator=evaluator, metric_artifact={"uri": "metrics://trajectory/report.json", "sha256": digest("trajectory-report")})
        metrics[0]["trajectory_binding"]["predicted_input_video_sha256"] = digest("other-output")
        issues = validate_manifest([split, run, *metrics])
        self.assertTrue(any(issue.field == "trajectory_binding.predicted_input_video_sha256" for issue in issues))


class PairedAggregationTests(unittest.TestCase):
    def _two_cluster_runs(self) -> tuple[dict, list[dict]]:
        split = valid_split_manifest(
            split_assignment("scene-a", "sequence-a"),
            split_assignment("scene-b", "sequence-b"),
        )
        runs = [
            valid_run(split, run_id="baseline-a", pair_id="pair-a", method_name="fg_only", schedule_state="baseline_all_zero_schedule", completed=True, scene_id="scene-a", sequence_id="sequence-a"),
            valid_run(split, run_id="candidate-a", pair_id="pair-a", method_name="fg_rgb_geco", schedule_state="active_guidance", completed=True, scene_id="scene-a", sequence_id="sequence-a"),
            valid_run(split, run_id="baseline-b", pair_id="pair-b", method_name="fg_only", schedule_state="baseline_all_zero_schedule", completed=True, seed=8, scene_id="scene-b", sequence_id="sequence-b"),
            valid_run(split, run_id="candidate-b", pair_id="pair-b", method_name="fg_rgb_geco", schedule_state="active_guidance", completed=True, seed=8, scene_id="scene-b", sequence_id="sequence-b"),
        ]
        return split, runs

    def test_cluster_bootstrap_aggregation_is_fail_closed(self) -> None:
        split, runs = self._two_cluster_runs()
        metrics = [metric(runs[0], 1.0), metric(runs[1], 0.5), metric(runs[2], 2.0), metric(runs[3], 1.5)]
        summary = aggregate_paired_metric([split, *runs], metrics, baseline_method="fg_only", candidate_method="fg_rgb_geco", expected_methods=("fg_only", "fg_rgb_geco"), metric_name="geco_fused", bootstrap_samples=100, random_seed=3)
        self.assertEqual(2, summary.pair_count)
        self.assertEqual(2, summary.cluster_count)
        self.assertAlmostEqual(0.5, summary.improvement_mean)
        self.assertTrue(math.isfinite(summary.bootstrap_ci95[0]))

    def test_aggregation_rejects_missing_or_failed_expected_arm(self) -> None:
        split = valid_split_manifest()
        baseline = valid_run(split, run_id="baseline", pair_id="pair", method_name="fg_only", schedule_state="baseline_all_zero_schedule", completed=True)
        failed_candidate = valid_run(split, run_id="candidate", pair_id="pair", method_name="fg_rgb_geco", schedule_state="active_guidance", completed=False)
        with self.assertRaises(ManifestValidationError):
            aggregate_paired_metric([split, baseline, failed_candidate], [metric(baseline, 1.0)], baseline_method="fg_only", candidate_method="fg_rgb_geco", expected_methods=("fg_only", "fg_rgb_geco"), metric_name="geco_fused")

    def test_aggregation_rejects_metric_binding_or_evaluator_mismatch(self) -> None:
        split, runs = self._two_cluster_runs()
        metrics = [metric(runs[0], 1.0), metric(runs[1], 0.5), metric(runs[2], 2.0), metric(runs[3], 1.5)]
        metrics[1]["evaluated_output_sha256"] = digest("wrong-output")
        with self.assertRaises(ManifestValidationError):
            aggregate_paired_metric([split, *runs], metrics, baseline_method="fg_only", candidate_method="fg_rgb_geco", expected_methods=("fg_only", "fg_rgb_geco"), metric_name="geco_fused")
        metrics = [metric(runs[0], 1.0), metric(runs[1], 0.5, evaluator=default_evaluator(config={"window_sec": 2.0})), metric(runs[2], 2.0), metric(runs[3], 1.5)]
        with self.assertRaisesRegex(ValueError, "identical evaluator"):
            aggregate_paired_metric([split, *runs], metrics, baseline_method="fg_only", candidate_method="fg_rgb_geco", expected_methods=("fg_only", "fg_rgb_geco"), metric_name="geco_fused")
        missing_artifact_hash = metric(runs[0], 1.0)
        del missing_artifact_hash["metric_artifact"]["sha256"]
        issues = validate_manifest([split, runs[0], missing_artifact_hash])
        self.assertTrue(any(issue.field == "metric_artifact.sha256" for issue in issues))

    def test_aggregation_rejects_ci_without_two_independent_clusters(self) -> None:
        split = valid_split_manifest()
        runs = [
            valid_run(split, run_id="baseline", pair_id="pair", method_name="fg_only", schedule_state="baseline_all_zero_schedule", completed=True),
            valid_run(split, run_id="candidate", pair_id="pair", method_name="fg_rgb_geco", schedule_state="active_guidance", completed=True),
        ]
        with self.assertRaisesRegex(ValueError, "at least two independent"):
            aggregate_paired_metric([split, *runs], [metric(runs[0], 1.0), metric(runs[1], 0.5)], baseline_method="fg_only", candidate_method="fg_rgb_geco", expected_methods=("fg_only", "fg_rgb_geco"), metric_name="geco_fused")


if __name__ == "__main__":
    unittest.main()
