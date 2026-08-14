from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import build_dpo_traindev_adapted_input_lock_v1 as adapted
import build_dpo_traindev_evaluator_bundle_v1 as bundle
import build_dpo_traindev_eligibility_lock_v1 as eligibility
import build_evaluator_lock
import build_metric_eligibility_lock as formal_eligibility
import build_metric_receipt
from metric_adapters import metric_adapter_common
import run_metric_worker_guarded
import traindev_provenance_v1 as provenance
import verify_evaluator_lock
import verify_metric_input_lock as preflight


def dump_readonly(path: Path, value: dict) -> dict[str, str]:
    path.write_bytes(bundle.canonical_bytes(value))
    path.chmod(0o444)
    return {"path": str(path.resolve()), "sha256": bundle.sha256_file(path)}


def synthetic_generation_receipt(source_root: Path) -> tuple[dict, list[tuple[str, dict]]]:
    cases = [(f"case-{index:03d}", {}) for index in range(100)]
    tasks = []
    pairs = []
    for case_index, (case_id, _) in enumerate(cases):
        pair_sha = hashlib.sha256(f"pair-{case_index}".encode()).hexdigest()
        videos = {}
        for mode_index, mode in enumerate(bundle.MODES):
            task_index = case_index * 2 + mode_index
            video_sha = hashlib.sha256(f"video-{task_index}".encode()).hexdigest()
            videos[mode] = video_sha
            tasks.append({
                "task_index": task_index,
                "case_index": case_index,
                "case_id": case_id,
                "mode": mode,
                "run_id": f"{case_index:03d}-{mode}",
                "metadata_sha256": hashlib.sha256(
                    bundle.canonical_bytes({"task_index": task_index})
                ).hexdigest(),
                "video_sha256": video_sha,
                "pair_identity_sha256": pair_sha,
                "stored_video_probe": bundle.EXPECTED_STORED_VIDEO_PROBE,
                "independent_video_probe": bundle.EXPECTED_INDEPENDENT_VIDEO_PROBE,
            })
        pairs.append({
            "case_index": case_index,
            "case_id": case_id,
            "pair_identity_sha256": pair_sha,
            "base_video_sha256": videos["base"],
            "adapted_video_sha256": videos["adapted"],
        })
    receipt = {
        "schema": bundle.GENERATION_RECEIPT_SCHEMA,
        "status": "COMPLETE",
        "case_count": 100,
        "task_count": 200,
        "pair_count": 100,
        "frozen_video_spec": {
            "width": 1280,
            "height": 704,
            "fps_numerator": 24,
            "fps_denominator": 1,
            "nb_frames": 121,
            "full_decode_backends": ["imageio-ffmpeg-full-decode", "ffmpeg-full-decode"],
        },
        "contract": {
            "output_root": str(source_root.resolve()),
            "manifest_sha256": bundle.EXPECTED_MANIFEST_SHA256,
            "case_count": 100,
            "seed": 0,
            "steps": 50,
            "frames": 121,
            "height": 704,
            "width": 1280,
            "fps": 24,
            "guidance_scale": 5.0,
        },
        "generation_repo": {
            "commit": bundle.EXPECTED_GENERATION_COMMIT,
            "clean": True,
            "runner_sha256": bundle.EXPECTED_RUNNER_SHA256,
        },
        "reviewed_sources": {
            "controller_sha256": bundle.EXPECTED_CONTROLLER_SHA256,
            "tests_sha256": bundle.EXPECTED_CONTROLLER_TESTS_SHA256,
        },
        "independent_approval": {"sha256": bundle.EXPECTED_APPROVAL_SHA256},
        "model": {"identity_sha256": bundle.EXPECTED_MODEL_IDENTITY_SHA256},
        "lora": {"checkpoint_receipt_sha256": bundle.EXPECTED_LORA_RECEIPT_SHA256},
        "lora_weight_sha256": bundle.EXPECTED_LORA_WEIGHT_SHA256,
        "tasks": tasks,
        "pairs": pairs,
    }
    return receipt, cases


def synthetic_sealed_bundle(root: Path) -> tuple[Path, dict[str, str], str, str]:
    """Build a tiny-byte, full 100-pair/800-record provenance closure."""
    root.mkdir(parents=True)
    receipt, cases = synthetic_generation_receipt(root)
    entries_by_mode = {"base": [], "adapted": []}
    records = []
    for case_index, (case_id, _) in enumerate(cases):
        pair_sha = receipt["pairs"][case_index]["pair_identity_sha256"]
        for mode_index, mode in enumerate(bundle.MODES):
            task_index = case_index * 2 + mode_index
            run = root / "mirror" / mode / f"{case_index:03d}_{case_id}"
            run.mkdir(parents=True)
            contents = {
                "video.mp4": f"video-{task_index}".encode(),
                "metadata.json": bundle.canonical_bytes({"task_index": task_index}),
                "COMPLETE": f"complete-{task_index}".encode(),
                ".generation.lock": f"lock-{task_index}".encode(),
            }
            bindings = {}
            for name, data in contents.items():
                path = run / name
                path.write_bytes(data)
                bindings[name] = {
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
            task = receipt["tasks"][task_index]
            self_metadata_sha = bindings["metadata.json"]["sha256"]
            if task["video_sha256"] != bindings["video.mp4"]["sha256"] or task["metadata_sha256"] != self_metadata_sha:
                raise AssertionError("synthetic task bytes differ from synthetic receipt")
            entry = {
                "case_id": case_id,
                "split_order": case_index,
                "seed": 0,
                "run_id": task["run_id"],
                "metric_video_path": bindings["video.mp4"]["path"],
                "metric_metadata_path": bindings["metadata.json"]["path"],
                "metric_complete_path": bindings["COMPLETE"]["path"],
                "metric_generation_lock_path": bindings[".generation.lock"]["path"],
                "video_sha256": bindings["video.mp4"]["sha256"],
                "metadata_sha256": self_metadata_sha,
                "complete_sha256": bindings["COMPLETE"]["sha256"],
                "generation_lock_sha256": bindings[".generation.lock"]["sha256"],
                "pair_identity_sha256": pair_sha,
                "lora_mode": mode,
                "lora_step": 64,
                "video_probe": bundle.EXPECTED_VIDEO_PROBE,
            }
            entries_by_mode[mode].append(entry)
            for kind, name in (
                ("video", "video.mp4"),
                ("metadata", "metadata.json"),
                ("COMPLETE", "COMPLETE"),
                ("generation_lock", ".generation.lock"),
            ):
                records.append({
                    "mode": mode,
                    "case_id": case_id,
                    "kind": kind,
                    **bindings[name],
                })
    provenance_dir = root / "provenance"
    provenance_dir.mkdir()
    generation_binding = dump_readonly(
        provenance_dir / "PAIRED_GENERATION_RECEIPT.json", receipt
    )
    manifest_binding = dump_readonly(
        provenance_dir / "dev100_manifest_960p_v2.json", {"synthetic": True}
    )
    manifest_sha = manifest_binding["sha256"]
    receipt["contract"]["manifest_sha256"] = manifest_sha
    generation_binding = dump_readonly(
        provenance_dir / "PAIRED_GENERATION_RECEIPT.rewritten.json", receipt
    )
    (provenance_dir / "PAIRED_GENERATION_RECEIPT.json").chmod(0o644)
    (provenance_dir / "PAIRED_GENERATION_RECEIPT.json").unlink()
    (provenance_dir / "PAIRED_GENERATION_RECEIPT.rewritten.json").rename(
        provenance_dir / "PAIRED_GENERATION_RECEIPT.json"
    )
    generation_binding = {
        "path": str((provenance_dir / "PAIRED_GENERATION_RECEIPT.json").resolve()),
        "sha256": bundle.sha256_file(provenance_dir / "PAIRED_GENERATION_RECEIPT.json"),
    }
    isolation = {
        "schema": provenance.ISOLATION_SCHEMA,
        "status": "verified_zero_overlap",
        "train_dev": {
            "manifest_sha256": manifest_sha,
            "case_count": 100,
            "commitments": {key: "1" * 64 for key in ("case_ids", "scene_ids", "transform_sources")},
        },
        "excluded_reference": {
            "manifest_sha256": provenance.EXPECTED_REFERENCE_MANIFEST_SHA256,
            "source_manifest_sha256": provenance.EXPECTED_REFERENCE_SOURCE_MANIFEST_SHA256,
            "case_count": 100,
            "commitments": {key: "2" * 64 for key in ("case_ids", "scene_ids", "transform_sources")},
        },
        "overlap_counts": {"case_ids": 0, "scene_ids": 0, "transform_sources": 0},
        "ids_disclosed": False,
    }
    isolation_binding = dump_readonly(
        root / "TRAINDEV_REFERENCE_ISOLATION_RECEIPT.json", isolation
    )
    mirror_index = {
        "schema": provenance.MIRROR_INDEX_SCHEMA,
        "site": "Hippasus",
        "split": "dev",
        "case_count": 100,
        "method_count": 2,
        "record_count": 800,
        "generation_receipt_sha256": generation_binding["sha256"],
        "manifest_sha256": manifest_sha,
        "records": records,
    }
    mirror_index_binding = dump_readonly(root / "MIRROR_INDEX.json", mirror_index)
    base_input = {
        "schema": provenance.INPUT_SCHEMA,
        "scope": "train_dev_evaluation",
        "evaluation_site": "Hippasus",
        "method": bundle.BASE_METHOD_LABEL,
        "method_id": bundle.BASE_METHOD_ID,
        "candidate_budget": {"candidate_count": 1, "selected_output_count": 1},
        "dataset_split": "dev",
        "reserved_ids_disclosed": False,
        "source_manifest_sha256": manifest_sha,
        "traindev_reference_isolation_receipt_sha256": isolation_binding["sha256"],
        "source_generation_receipt_sha256": generation_binding["sha256"],
        "mirror_index_sha256": mirror_index_binding["sha256"],
        "mirror_root": str(root.resolve()),
        "entries": entries_by_mode["base"],
    }
    base_binding = dump_readonly(root / "BASE_INPUT_LOCK.json", base_input)
    adapted_entries = {
        "schema": provenance.ADAPTED_ENTRIES_SCHEMA,
        "site": "Hippasus",
        "scope": "train_dev_evaluation",
        "entries": entries_by_mode["adapted"],
    }
    adapted_binding = dump_readonly(root / "ADAPTED_INPUT_ENTRIES.json", adapted_entries)
    ready = {
        "schema": provenance.BUNDLE_SCHEMA,
        "status": "READY",
        "site": "Hippasus",
        "split": "dev",
        "reserved_ids_disclosed": False,
        "case_count": 100,
        "task_count": 200,
        "pair_count": 100,
        "mirror_file_count": 800,
        "generation_receipt": generation_binding,
        "manifest": manifest_binding,
        "traindev_reference_isolation_receipt": isolation_binding,
        "mirror_index": mirror_index_binding,
        "base_input_lock": base_binding,
        "adapted_input_entries": adapted_binding,
    }
    ready_binding = dump_readonly(root / "BUNDLE_READY.json", ready)
    bundle.seal_tree(root)
    return root / "BASE_INPUT_LOCK.json", ready_binding, generation_binding["sha256"], manifest_sha


class TrainDevEvaluationContractTest(unittest.TestCase):
    def test_external_manifest_and_generation_commitments_are_full_sha256(self) -> None:
        for value in (
            bundle.EXPECTED_MANIFEST_SHA256,
            bundle.EXPECTED_FORMAL_MANIFEST_SHA256,
            bundle.EXPECTED_FORMAL_SOURCE_MANIFEST_SHA256,
            provenance.EXPECTED_MANIFEST_SHA256,
            provenance.EXPECTED_REFERENCE_MANIFEST_SHA256,
            provenance.EXPECTED_REFERENCE_SOURCE_MANIFEST_SHA256,
            bundle.PARENT_PROTOCOL_SHA256,
            bundle.PARENT_SCHEDULE_SHA256,
        ):
            self.assertTrue(bundle.is_sha256(value))
        self.assertEqual(
            provenance.EXPECTED_REFERENCE_MANIFEST_SHA256,
            bundle.EXPECTED_FORMAL_MANIFEST_SHA256,
        )
        self.assertEqual(
            provenance.EXPECTED_REFERENCE_SOURCE_MANIFEST_SHA256,
            bundle.EXPECTED_FORMAL_SOURCE_MANIFEST_SHA256,
        )
        self.assertEqual(
            provenance.EXPECTED_STORED_VIDEO_PROBE,
            bundle.EXPECTED_STORED_VIDEO_PROBE,
        )
        self.assertEqual(
            provenance.EXPECTED_INDEPENDENT_VIDEO_PROBE,
            bundle.EXPECTED_INDEPENDENT_VIDEO_PROBE,
        )
        self.assertEqual(bundle.EXPECTED_STORED_VIDEO_PROBE["avg_frame_rate"], "24.0")
        self.assertEqual(bundle.EXPECTED_INDEPENDENT_VIDEO_PROBE["avg_frame_rate"], "24/1")

    def test_exact_helper_loaders_reject_replacement_symlink_and_one_byte_change(self) -> None:
        modules = (
            eligibility,
            formal_eligibility,
            build_evaluator_lock,
            build_metric_receipt,
            preflight,
            run_metric_worker_guarded,
            verify_evaluator_lock,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            helper = root / "helper.py"
            helper.write_text("VALUE = 1\n", encoding="utf-8")
            expected = hashlib.sha256(helper.read_bytes()).hexdigest()
            for module in modules:
                loaded = module.load_exact_sibling(helper, expected, "test_helper")
                self.assertEqual(loaded.VALUE, 1)
            helper.write_text("VALUE = 2\n", encoding="utf-8")
            for module in modules:
                with self.assertRaises(RuntimeError):
                    module.load_exact_sibling(helper, expected, "test_helper")
            target = root / "target.py"
            target.write_text("VALUE = 1\n", encoding="utf-8")
            helper.unlink()
            helper.symlink_to(target)
            for module in modules:
                with self.assertRaises(OSError):
                    module.load_exact_sibling(helper, expected, "test_helper")

    def test_publication_fault_exposes_no_writable_final_path(self) -> None:
        publishers = (
            (eligibility, "publish"),
            (adapted, "publish"),
            (build_evaluator_lock, "atomic_publish"),
            (preflight, "publish"),
            (build_metric_receipt, "publish"),
            (formal_eligibility, "publish"),
            (run_metric_worker_guarded, "publish"),
            (verify_evaluator_lock, "publish"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (module, name) in enumerate(publishers):
                output = root / f"output-{index}.json"
                real_link = os.link

                def fail_after_check(source: os.PathLike, target: os.PathLike) -> None:
                    self.assertEqual(Path(source).stat().st_mode & 0o222, 0)
                    self.assertFalse(Path(target).exists())
                    raise RuntimeError("fault before publication")

                with mock.patch.object(module.os, "link", side_effect=fail_after_check):
                    with self.assertRaises(RuntimeError):
                        getattr(module, name)(output, {"schema": "test"})
                self.assertFalse(output.exists())

    def test_arbitrary_isolation_hash_fails_adapter_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule_path = root / "schedule.json"
            schedule_binding = dump_readonly(schedule_path, {"schema": "schedule"})
            input_path = root / "input.json"
            input_payload = {
                "schema": metric_adapter_common.TRAINDEV_INPUT_SCHEMA,
                "scope": "train_dev_evaluation",
                "evaluation_site": "Hippasus",
                "dataset_split": "dev",
                "reserved_ids_disclosed": False,
                "method_id": bundle.BASE_METHOD_ID,
                "traindev_reference_isolation_receipt_sha256": "b" * 64,
                "source_generation_receipt_sha256": "c" * 64,
                "metric_schedule_sha256": schedule_binding["sha256"],
                "entries": [],
            }
            input_binding = dump_readonly(input_path, input_payload)
            provenance_payload = {
                "schema": "geometry-selection-traindev-provenance-verification-v1",
                "status": "verified",
                "case_count": 100,
                "record_count": 800,
                "reference_isolation_receipt": {"sha256": "a" * 64},
                "generation_receipt": {"sha256": "c" * 64},
            }
            evaluator_path = root / "evaluator.json"
            evaluator_binding = dump_readonly(evaluator_path, {
                "schema": "geometry-selection-evaluator-lock-v2",
                "scope": "train_dev",
                "input_manifest": input_binding,
                "traindev_provenance": provenance_payload,
            })
            preflight_path = root / "preflight.json"
            preflight_binding = dump_readonly(preflight_path, {
                "schema": "geometry-selection-metric-traindev-input-preflight-receipt-v1",
                "input_lock": input_binding,
                "evaluator_lock": evaluator_binding,
                "traindev_provenance": provenance_payload,
            })
            environment = {
                "GEOMETRY_EVAL_LOCKED_INPUT_LOCK_PATH": input_binding["path"],
                "GEOMETRY_EVAL_LOCKED_INPUT_LOCK_SHA256": input_binding["sha256"],
                "GEOMETRY_EVAL_LOCKED_EVALUATOR_LOCK_PATH": evaluator_binding["path"],
                "GEOMETRY_EVAL_LOCKED_EVALUATOR_LOCK_SHA256": evaluator_binding["sha256"],
                "GEOMETRY_EVAL_INPUT_PREFLIGHT_RECEIPT_PATH": preflight_binding["path"],
                "GEOMETRY_EVAL_INPUT_PREFLIGHT_RECEIPT_SHA256": preflight_binding["sha256"],
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                with self.assertRaises(ValueError):
                    metric_adapter_common.load_input_lock(input_path, schedule_path)

    def test_relabelled_or_self_declared_base_input_fails_bundle_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path, ready_binding, generation_sha, manifest_sha = synthetic_sealed_bundle(
                root / "bundle"
            )
            input_payload = json.loads(input_path.read_text())
            input_binding = {
                "path": str(input_path.resolve()),
                "sha256": bundle.sha256_file(input_path),
            }
            with mock.patch.object(provenance, "EXPECTED_MANIFEST_SHA256", manifest_sha):
                verified = provenance.verify_train_dev_provenance(
                    input_payload=input_payload,
                    input_binding=input_binding,
                    source_bundle_ready=ready_binding,
                    expected_generation_receipt_sha256=generation_sha,
                )
            self.assertEqual(verified["status"], "verified")
            forged = copy.deepcopy(input_payload)
            forged["traindev_reference_isolation_receipt_sha256"] = "b" * 64
            forged_path = root / "forged-input.json"
            forged_binding = dump_readonly(forged_path, forged)
            with mock.patch.object(provenance, "EXPECTED_MANIFEST_SHA256", manifest_sha):
                with self.assertRaises(ValueError):
                    provenance.verify_train_dev_provenance(
                        input_payload=forged,
                        input_binding=forged_binding,
                        source_bundle_ready=ready_binding,
                        expected_generation_receipt_sha256=generation_sha,
                    )
            for index, relabelled in enumerate((
                {**input_payload, "method": "Wan unguided seed-0 incumbent"},
                {**input_payload, "self_declared_scope": "validation_only"},
            )):
                relabelled_path = root / f"relabelled-{index}.json"
                relabelled_binding = dump_readonly(relabelled_path, relabelled)
                with mock.patch.object(provenance, "EXPECTED_MANIFEST_SHA256", manifest_sha):
                    with self.assertRaises(ValueError):
                        provenance.verify_train_dev_provenance(
                            input_payload=relabelled,
                            input_binding=relabelled_binding,
                            source_bundle_ready=ready_binding,
                            expected_generation_receipt_sha256=generation_sha,
                        )

    def test_derived_protocol_preserves_metric_definitions_and_schedule(self) -> None:
        root = Path(__file__).parent
        protocol = json.loads((root / "five_metric_protocol_v2.json").read_text())
        schedule = json.loads((root / "metric_schedule_v2.json").read_text())
        original_metrics = copy.deepcopy(protocol["metrics"])
        original_references = copy.deepcopy(protocol["reference_evaluator_identities"])
        original_weights = copy.deepcopy(protocol["weight_source_requirements"])
        original_video = copy.deepcopy(protocol["video_contract"])
        original_schedule = copy.deepcopy(schedule)

        derived_protocol, derived_schedule = bundle.derive_protocol_and_schedule(
            protocol,
            schedule,
            bundle.EXPECTED_MANIFEST_SHA256,
            "a" * 64,
        )

        self.assertEqual(derived_protocol["metrics"], original_metrics)
        self.assertEqual(derived_protocol["reference_evaluator_identities"], original_references)
        self.assertEqual(derived_protocol["weight_source_requirements"], original_weights)
        self.assertEqual(derived_protocol["video_contract"], original_video)
        self.assertEqual(derived_protocol["schema"], bundle.TRAINDEV_PROTOCOL_SCHEMA)
        serialized = json.dumps(derived_protocol, sort_keys=True)
        for prohibited in ("formal_validation", "validation_only", "wan_unguided_seed0"):
            self.assertNotIn(prohibited, serialized)
        self.assertEqual(derived_protocol["dataset"], {
            "name": "DL3DV-1K independent train-dev",
            "split": "dev",
            "case_count": 100,
            "source_manifest_sha256": bundle.EXPECTED_MANIFEST_SHA256,
            "reserved_ids_disclosed": False,
        })
        self.assertEqual(derived_protocol["candidate_budget_policy"][bundle.BASE_METHOD_ID]["candidate_count"], 1)
        self.assertEqual(derived_protocol["candidate_budget_policy"][bundle.ADAPTED_METHOD_ID]["candidate_count"], 1)
        expected_schedule = copy.deepcopy(original_schedule)
        expected_schedule["schema"] = bundle.TRAINDEV_SCHEDULE_SCHEMA
        expected_schedule["metric_protocol_sha256"] = hashlib.sha256(
            bundle.canonical_bytes(derived_protocol)
        ).hexdigest()
        self.assertEqual(derived_schedule, expected_schedule)

        altered = copy.deepcopy(derived_protocol)
        altered["metrics"]["long_range_reprojection_error"]["minimum_valid_fraction_per_direction"] = 0.1
        with self.assertRaises(bundle.ContractError):
            bundle.verify_derived_contract(
                protocol,
                schedule,
                altered,
                derived_schedule,
                bundle.EXPECTED_MANIFEST_SHA256,
                "a" * 64,
            )
        altered_schedule = copy.deepcopy(derived_schedule)
        altered_schedule["long_range_reprojection_error"]["pairs"] = [[0, 119]]
        with self.assertRaises(bundle.ContractError):
            bundle.verify_derived_contract(
                protocol,
                schedule,
                derived_protocol,
                altered_schedule,
                bundle.EXPECTED_MANIFEST_SHA256,
                "a" * 64,
            )


    def test_pair_normalization_removes_only_mode(self) -> None:
        base = {"seed": 0, "lora_dpo": {"mode": "base", "step": 64, "weight": "same"}}
        adapted_config = copy.deepcopy(base)
        adapted_config["lora_dpo"]["mode"] = "adapted"
        self.assertEqual(bundle.normalized_pair_config(base), bundle.normalized_pair_config(adapted_config))
        changed = copy.deepcopy(adapted_config)
        changed["seed"] = 1
        self.assertNotEqual(bundle.normalized_pair_config(base), bundle.normalized_pair_config(changed))

    def test_isolation_receipt_proves_zero_overlap_without_disclosing_ids(self) -> None:
        def manifest(prefix: str) -> dict:
            return {
                f"{prefix}-case-{index:03d}": {
                    "scene_id": hashlib.sha256(f"{prefix}-scene-{index}".encode()).hexdigest(),
                    "transforms_sha256": hashlib.sha256(
                        f"{prefix}-transform-{index}".encode()
                    ).hexdigest(),
                }
                for index in range(100)
            }

        train = manifest("train")
        reference = manifest("reference")
        receipt = bundle.build_isolation_receipt(train, reference, "c" * 64)
        self.assertEqual(
            receipt["schema"],
            "wan-lora-dpo-traindev-reference-isolation-receipt-v1",
        )
        self.assertEqual(
            receipt["overlap_counts"],
            {"case_ids": 0, "scene_ids": 0, "transform_sources": 0},
        )
        self.assertFalse(receipt["ids_disclosed"])
        serialized = json.dumps(receipt, sort_keys=True)
        self.assertNotIn(next(iter(train)), serialized)
        overlap = copy.deepcopy(reference)
        first_reference = next(iter(overlap.values()))
        first_train = next(iter(train.values()))
        first_reference["scene_id"] = first_train["scene_id"]
        with self.assertRaises(bundle.ContractError):
            bundle.build_isolation_receipt(train, overlap, "c" * 64)

    def test_path_traversal_is_rejected_lexically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with self.assertRaises(bundle.ContractError):
                bundle.relative_source_path(
                    str(root / ".." / "escape" / "video.mp4"),
                    root,
                    "video",
                )


    def test_open_directory_below_rejects_symlink_component(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            real = tmp_path / "real"
            real.mkdir()
            (real / "run_x").mkdir()
            (tmp_path / "alias").symlink_to(real, target_is_directory=True)
            with self.assertRaises(bundle.ContractError):
                bundle.open_directory_below(tmp_path, Path("alias/run_x"), "run")

    def test_source_ancestor_replacement_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "parent" / "run").mkdir(parents=True)
            run_fd = bundle.open_directory_below(root, Path("parent/run"), "run")
            (root / "parent").rename(root / "old-parent")
            (root / "parent" / "run").mkdir(parents=True)
            with self.assertRaises(bundle.ContractError):
                bundle.verify_directory_below_identity(
                    root, Path("parent/run"), run_fd, "run"
                )
            os.close(run_fd)

    def test_generation_lock_replacement_race_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / ".generation.lock"
            lock.write_bytes(b"locked\n")
            expected = hashlib.sha256(b"locked\n").hexdigest()
            directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            descriptor, identity = bundle.acquire_generation_lock_at(
                directory_fd, expected, "generation lock"
            )
            (root / ".generation.lock.old").write_bytes(b"placeholder")
            os.replace(lock, root / ".generation.lock.old")
            lock.write_bytes(b"replacement\n")
            with self.assertRaises(bundle.ContractError):
                bundle.release_generation_lock_at(
                    directory_fd, descriptor, identity, "generation lock"
                )
            os.close(directory_fd)

    def test_externally_supplied_receipt_sha_is_mandatory(self) -> None:
        payload = b'{"status":"COMPLETE"}\n'
        with self.assertRaises(bundle.ContractError):
            bundle.verify_external_payload_sha(payload, "0" * 64, "generation receipt")
        self.assertEqual(
            bundle.verify_external_payload_sha(
                payload, hashlib.sha256(payload).hexdigest(), "generation receipt"
            ),
            hashlib.sha256(payload).hexdigest(),
        )

    def test_mixed_method_and_missing_case_receipts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt, cases = synthetic_generation_receipt(root)
            bundle.validate_generation_receipt(
                receipt, root, bundle.EXPECTED_MANIFEST_SHA256, cases
            )
            mixed = copy.deepcopy(receipt)
            mixed["tasks"][1]["mode"] = "base"
            with self.assertRaises(bundle.ContractError):
                bundle.validate_generation_receipt(
                    mixed, root, bundle.EXPECTED_MANIFEST_SHA256, cases
                )
            missing = copy.deepcopy(receipt)
            missing["pairs"].pop()
            with self.assertRaises(bundle.ContractError):
                bundle.validate_generation_receipt(
                    missing, root, bundle.EXPECTED_MANIFEST_SHA256, cases
                )
            duplicate = copy.deepcopy(receipt)
            duplicate["tasks"][1] = copy.deepcopy(duplicate["tasks"][0])
            with self.assertRaises(bundle.ContractError):
                bundle.validate_generation_receipt(
                    duplicate, root, bundle.EXPECTED_MANIFEST_SHA256, cases
                )
            reordered = copy.deepcopy(receipt)
            reordered["tasks"][0], reordered["tasks"][1] = (
                reordered["tasks"][1],
                reordered["tasks"][0],
            )
            with self.assertRaises(bundle.ContractError):
                bundle.validate_generation_receipt(
                    reordered, root, bundle.EXPECTED_MANIFEST_SHA256, cases
                )

    def test_lre_threshold_and_motion_anchor_are_numeric_and_frozen(self) -> None:
        self.assertTrue(eligibility.base_lre_eligible({
            "value": 0.1,
            "forward_valid_fraction": 0.2,
            "backward_valid_fraction": 0.2,
            "eligible": True,
        }))
        self.assertFalse(eligibility.base_lre_eligible({
            "value": 0.1,
            "forward_valid_fraction": 0.19,
            "backward_valid_fraction": 0.2,
            "eligible": False,
        }))
        self.assertEqual(
            eligibility.base_motion_anchor(
                {"value": 100.0, "raw_total_motion": 2.5},
                ("case", "total_motion"),
            ),
            2.5,
        )
        with self.assertRaises(ValueError):
            eligibility.base_motion_anchor(
                {"value": 99.0, "raw_total_motion": 2.5},
                ("case", "total_motion"),
            )
        self.assertIsNone(eligibility.base_motion_anchor(
            {"value": 100.0, "raw_total_motion": True},
            ("case", "total_motion"),
        ))

    def test_train_dev_eligibility_build_uses_train_dev_attestation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = []
            units = {metric: [] for metric in eligibility.METRIC_NAMES}
            for index in range(100):
                case_id = f"case-{index:03d}"
                entry = {
                    "case_id": case_id,
                    "split_order": index,
                    "video_sha256": hashlib.sha256(f"video-{index}".encode()).hexdigest(),
                    "metadata_sha256": hashlib.sha256(f"metadata-{index}".encode()).hexdigest(),
                    "complete_sha256": hashlib.sha256(f"complete-{index}".encode()).hexdigest(),
                    "generation_lock_sha256": hashlib.sha256(f"lock-{index}".encode()).hexdigest(),
                }
                entries.append(entry)
                units["geco_fused"].extend(
                    {"case_id": case_id, "unit_id": f"window_{unit}", "value": 1.0}
                    for unit in range(2)
                )
                units["met3r"].extend(
                    {"case_id": case_id, "unit_id": f"one_second_{unit}", "value": 1.0}
                    for unit in range(5)
                )
                units["long_range_reprojection_error"].append({
                    "case_id": case_id,
                    "unit_id": "first_last",
                    "value": 0.1,
                    "forward_valid_fraction": 1.0,
                    "backward_valid_fraction": 1.0,
                    "eligible": True,
                })
                units["relative_total_motion_percent"].append({
                    "case_id": case_id,
                    "unit_id": "total_motion",
                    "value": 100.0,
                    "raw_total_motion": 1.0,
                })
                units["vbench_quality"].extend(
                    {"case_id": case_id, "unit_id": unit, "value": 1.0}
                    for unit in eligibility.VBENCH_UNITS
                )
            input_path = root / "base-input.json"
            input_binding = dump_readonly(input_path, {
                "schema": eligibility.INPUT_SCHEMA,
                "scope": "train_dev_evaluation",
                "evaluation_site": "Hippasus",
                "method": eligibility.BASE_METHOD_LABEL,
                "method_id": eligibility.BASE_METHOD_ID,
                "dataset_split": "dev",
                "reserved_ids_disclosed": False,
                "entries": entries,
            })
            evaluator_path = root / "evaluator.json"
            evaluator_binding = dump_readonly(evaluator_path, {
                "schema": "geometry-selection-evaluator-lock-v2",
                "scope": "train_dev",
                "site": "Hippasus",
                "input_manifest": input_binding,
                "shared_evaluator_identity_sha256": "a" * 64,
            })
            units_path = root / "units.json"
            dump_readonly(units_path, {
                "schema": "geometry-selection-metric-units-v1",
                "method": eligibility.BASE_METHOD_LABEL,
                "input_manifest_sha256": input_binding["sha256"],
                "evaluator_lock_sha256": evaluator_binding["sha256"],
                "metric_worker_receipts": {},
                "metric_components": {},
                "units": units,
            })
            preflight_cases = [
                {
                    key: entry[key]
                    for key in (
                        "case_id",
                        "video_sha256",
                        "metadata_sha256",
                        "complete_sha256",
                        "generation_lock_sha256",
                    )
                }
                for entry in entries
            ]
            runtime = {
                metric: {
                    "input_preflight": {"case_artifacts": preflight_cases},
                    "source_snapshot_rehash": {},
                    "independent_lre_runtime_load_trace": {},
                }
                for metric in eligibility.METRIC_NAMES
            }
            with mock.patch.object(
                eligibility.IMPL,
                "verify_runtime_attestations",
                return_value=runtime,
            ) as verify, mock.patch.object(
                eligibility.IMPL, "safe_output"
            ), mock.patch.object(
                eligibility, "publish", return_value="b" * 64
            ):
                result = eligibility.build(
                    input_path=input_path,
                    evaluator_path=evaluator_path,
                    units_path=units_path,
                    derived_root=root,
                    output_path=root / "eligibility.json",
                )
            self.assertEqual(result["eligibility_lock_sha256"], "b" * 64)
            self.assertTrue(verify.call_args.kwargs["train_dev"])

    def test_shared_evaluator_identity_is_method_neutral(self) -> None:
        neutral = {
            "site": "Hippasus",
            "protocol_binding": {"path": "/protocol", "sha256": "a" * 64},
            "environment_binding": {"path": "/environment", "sha256": "b" * 64},
            "source_snapshot_closure": {"snapshot": {"sha256": "c" * 64}},
            "decoder": {"path": "/decoder", "sha256": "d" * 64},
            "decoder_trusted_root": {"path": "/trusted", "sha256": "e" * 64},
            "schedule_binding": {"path": "/schedule", "sha256": "f" * 64},
            "verified_metrics": {"geco_fused": {"sha256": "1" * 64}},
        }
        base_identity, base_sha = build_evaluator_lock.build_shared_evaluator_identity(**neutral)
        adapted_identity, adapted_sha = build_evaluator_lock.build_shared_evaluator_identity(**neutral)
        self.assertEqual(base_identity, adapted_identity)
        self.assertEqual(base_sha, adapted_sha)
        self.assertNotIn("traindev_provenance", base_identity)

    def test_adapted_input_rejects_forged_external_eligibility_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = root / "ready.json"
            base = root / "base.json"
            frozen = root / "eligibility.json"
            dump_readonly(ready, {})
            dump_readonly(base, {})
            actual = dump_readonly(frozen, {"schema": adapted.ELIGIBILITY_SCHEMA})
            self.assertNotEqual(actual["sha256"], "a" * 64)
            with self.assertRaises(ValueError):
                adapted.build_lock(
                    bundle_ready_path=ready,
                    base_input_lock_path=base,
                    eligibility_path=frozen,
                    expected_eligibility_sha256="a" * 64,
                    output_path=root / "adapted.json",
                    derived_root=root,
                )

    def test_baseline_extras_are_filtered_but_adapted_extras_fail(self) -> None:
        records = [
            {"case_id": "locked", "unit_id": "first_last", "value": 0.1},
            {"case_id": "excluded", "unit_id": "first_last", "value": 0.0},
        ]
        mapped = build_metric_receipt.record_map(
            records,
            {("locked", "first_last")},
            "long_range_reprojection_error",
            allowed_extras={("excluded", "first_last")},
        )
        self.assertEqual(set(mapped), {("locked", "first_last")})
        with self.assertRaises(ValueError):
            build_metric_receipt.record_map(
                records,
                {("locked", "first_last")},
                "long_range_reprojection_error",
            )


    def test_evaluator_input_scope_separates_train_dev_from_formal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            entries = [
                {"case_id": f"case-{index}", "split_order": index}
                for index in range(100)
            ]
            isolation_binding = dump_readonly(
                tmp_path / "isolation.json",
                {
                    "schema": provenance.ISOLATION_SCHEMA,
                    "status": "verified_zero_overlap",
                    "overlap_counts": {
                        "case_ids": 0,
                        "scene_ids": 0,
                        "transform_sources": 0,
                    },
                    "ids_disclosed": False,
                },
            )
            input_lock = {
                "schema": build_evaluator_lock.TRAINDEV_INPUT_SCHEMA,
                "scope": "train_dev_evaluation",
                "evaluation_site": "Hippasus",
                "method": bundle.BASE_METHOD_LABEL,
                "method_id": bundle.BASE_METHOD_ID,
                "dataset_split": "dev",
                "reserved_ids_disclosed": False,
                "traindev_reference_isolation_receipt_sha256": isolation_binding["sha256"],
                "entries": entries,
            }
            binding = dump_readonly(tmp_path / "input.json", input_lock)
            _, verified = build_evaluator_lock.verify_input(binding, "train_dev", "Hippasus")
            self.assertEqual(verified, input_lock)
            with self.assertRaises(ValueError):
                build_evaluator_lock.verify_input(binding, "formal", "Hippasus")
            modified = copy.deepcopy(input_lock)
            modified["formal_validation"] = True
            binding2 = dump_readonly(tmp_path / "bad.json", modified)
            with self.assertRaises(ValueError):
                build_evaluator_lock.verify_input(binding2, "train_dev", "Hippasus")
            wrong_order = copy.deepcopy(input_lock)
            wrong_order["entries"][1]["split_order"] = 9
            binding3 = dump_readonly(tmp_path / "wrong-order.json", wrong_order)
            with self.assertRaises(ValueError):
                build_evaluator_lock.verify_input(binding3, "train_dev", "Hippasus")

    def test_adapted_input_cannot_publish_before_baseline_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = root / "ready.json"
            base = root / "base.json"
            dump_readonly(ready, {})
            dump_readonly(base, {})
            output = root / "adapted.json"
            with self.assertRaises(ValueError):
                adapted.build_lock(
                    bundle_ready_path=ready,
                    base_input_lock_path=base,
                    eligibility_path=root / "missing-eligibility.json",
                    expected_eligibility_sha256="a" * 64,
                    output_path=output,
                    derived_root=root,
                )
            self.assertFalse(output.exists())

    def test_failed_staging_is_preserved_readonly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary) / "attempt"
            staging.mkdir()
            artifact = staging / "partial.bin"
            artifact.write_bytes(b"partial")
            bundle.preserve_failed_staging(staging)
            self.assertTrue(staging.is_dir())
            self.assertTrue(artifact.is_file())
            self.assertEqual(artifact.stat().st_mode & 0o222, 0)
            self.assertEqual(staging.stat().st_mode & 0o222, 0)

    def test_extra_mirror_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary) / "staging"
            output = Path(temporary) / "final"
            mirror = staging / "mirror"
            mirror.mkdir(parents=True)
            records = []
            empty_sha = hashlib.sha256(b"").hexdigest()
            for index in range(800):
                path = mirror / f"artifact-{index:03d}"
                path.write_bytes(b"")
                records.append({
                    "path": str(output / "mirror" / path.name),
                    "sha256": empty_sha,
                    "bytes": 0,
                })
            bundle.validate_mirror_closure(staging, output, records)
            (mirror / "extra").write_bytes(b"")
            with self.assertRaises(bundle.ContractError):
                bundle.validate_mirror_closure(staging, output, records)

    def test_train_dev_preflight_verifies_generation_locks_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            derived = Path(temporary) / "outputs" / "geometry-selection" / "hippasus_evaluation"
            lock_dir = derived / "locks"
            receipts = derived / "receipts"
            for path in (derived / "mirrors", lock_dir, receipts):
                path.mkdir(parents=True, exist_ok=True)
            input_path, ready_binding, generation_sha, manifest_sha = synthetic_sealed_bundle(
                derived / "mirrors" / "bundle"
            )
            input_payload = json.loads(input_path.read_text())
            input_binding = {
                "path": str(input_path.resolve()),
                "sha256": bundle.sha256_file(input_path),
            }
            with mock.patch.object(
                preflight._provenance_module,
                "EXPECTED_MANIFEST_SHA256",
                manifest_sha,
            ):
                verified_provenance = preflight.verify_train_dev_provenance(
                    input_payload=input_payload,
                    input_binding=input_binding,
                    source_bundle_ready=ready_binding,
                    expected_generation_receipt_sha256=generation_sha,
                )
            evaluator_path = lock_dir / "evaluator.json"
            evaluator_binding = dump_readonly(evaluator_path, {
                "schema": "geometry-selection-evaluator-lock-v2",
                "scope": "train_dev",
                "site": "Hippasus",
                "input_manifest": input_binding,
                "traindev_provenance": verified_provenance,
            })
            output = receipts / "preflight.json"
            with mock.patch.object(
                preflight._provenance_module,
                "EXPECTED_MANIFEST_SHA256",
                manifest_sha,
            ), mock.patch.object(sys, "argv", [
                    "verify_metric_input_lock.py",
                    "--input-lock", str(input_path),
                    "--evaluator-lock", str(evaluator_path),
                    "--derived-root", str(derived),
                    "--output", str(output),
                ]):
                    preflight.main()
            payload = json.loads(output.read_text())
            self.assertEqual(
                payload["schema"],
                "geometry-selection-metric-traindev-input-preflight-receipt-v1",
            )
            self.assertEqual(
                [item["split_order"] for item in payload["case_artifacts"]],
                list(range(100)),
            )
            self.assertTrue(all("generation_lock_sha256" in item for item in payload["case_artifacts"]))
            self.assertEqual(payload["evaluator_lock"], evaluator_binding)
            self.assertEqual(payload["traindev_provenance"], verified_provenance)


    def test_locked_denominator_requires_exact_motion_anchors(self) -> None:
        cases = {"case-a", "case-b"}
        locked = {
            "geco_fused": [{"case_id": case, "unit_id": "window_0"} for case in cases],
            "met3r": [{"case_id": case, "unit_id": "one_second_0"} for case in cases],
            "long_range_reprojection_error": [{"case_id": case, "unit_id": "first_last"} for case in cases],
            "relative_total_motion_percent": [{"case_id": case, "unit_id": "total_motion"} for case in cases],
            "vbench_quality": [{"case_id": case, "unit_id": "official_quality"} for case in cases],
        }
        eligibility = {"locked_units": locked, "baseline_motion_anchors": {"case-a": 1.0, "case-b": 2.0}}
        adapted.validate_locked_denominators(eligibility, cases)
        missing = copy.deepcopy(eligibility)
        missing["baseline_motion_anchors"].pop("case-b")
        with self.assertRaises(ValueError):
            adapted.validate_locked_denominators(missing, cases)


    def test_readonly_artifact_verification_rejects_writable_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mirror"
            root.mkdir()
            artifact = root / "video.mp4"
            artifact.write_bytes(b"video")
            expected = hashlib.sha256(b"video").hexdigest()
            root.chmod(0o555)
            artifact.chmod(0o444)
            self.assertEqual(
                adapted.require_readonly_below(root, str(artifact), expected, "video").resolve(),
                artifact.resolve(),
            )
            artifact.chmod(0o644)
            with self.assertRaises(ValueError):
                adapted.require_readonly_below(root, str(artifact), expected, "video")
            artifact.chmod(0o444)
            root.chmod(0o755)
            alias = root / "alias.mp4"
            alias.symlink_to(artifact)
            root.chmod(0o555)
            with self.assertRaises(ValueError):
                adapted.require_readonly_below(root, str(alias), expected, "video")
            root.chmod(0o755)


if __name__ == "__main__":
    unittest.main()
