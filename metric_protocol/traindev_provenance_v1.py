#!/usr/bin/env python3
"""Strict, non-disclosing provenance verification for DPO train-dev scoring."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


BUNDLE_SCHEMA = "wan-lora-dpo-traindev-evaluator-bundle-v1"
INPUT_SCHEMA = "geometry-selection-five-metric-traindev-input-lock-v1"
ISOLATION_SCHEMA = "wan-lora-dpo-traindev-reference-isolation-receipt-v1"
MIRROR_INDEX_SCHEMA = "wan-lora-dpo-traindev-evaluator-mirror-index-v1"
GENERATION_SCHEMA = "wan-lora-dpo-traindev-paired-generation-receipt-v1"
ADAPTED_ENTRIES_SCHEMA = "wan-lora-dpo-traindev-adapted-input-entries-v1"
ELIGIBILITY_SCHEMA = "geometry-selection-metric-traindev-eligibility-lock-v1"
BASE_METHOD_ID = "wan_lora_dpo_step64_base_traindev"
ADAPTED_METHOD_ID = "wan_lora_dpo_step64_adapted_traindev"
METHOD_LABELS = {
    BASE_METHOD_ID: "Wan LoRA-DPO step-64 paired base train-dev",
    ADAPTED_METHOD_ID: "Full-Graph LoRA-DPO step-64 paired adapted train-dev",
}
EXPECTED_MANIFEST_SHA256 = "c48bf6ab56359bd22c08b905b4be1cf2822dd7f315ad6b21d693693d5d53f579"
EXPECTED_REFERENCE_MANIFEST_SHA256 = "24a4e47a42e576f3947f37e98aadfb3959447f6248d06652ca37e2981cb4d89"
EXPECTED_REFERENCE_SOURCE_MANIFEST_SHA256 = "da4c05c0ec8f6f8fd08daf3a69482c631d221f84fcb7a5d251c1e3fb1509dd9d"
EXPECTED_GENERATION_COMMIT = "efa13ac0ce45f3e24ddd3701fc2c7ea1e1fc85a3"
EXPECTED_RUNNER_SHA256 = "c9bb9f7c200cbb8bb4a64466326fa21a2542b2381d3f0c08a0aa680eb081fdf8"
EXPECTED_CONTROLLER_SHA256 = "402c6e8ac720c9166d4c6d82d982d416189470a3061442f27daf0131e1e4b62b"
EXPECTED_CONTROLLER_TESTS_SHA256 = "87461746097bc7ef56d38c80d8ca1c667bb6016b935f4bdde2657b71381c3ef6"
EXPECTED_APPROVAL_SHA256 = "cb078d3299ff0526340d893f849e5a1c1fd1f68cfc5539bc6f64b739ab947a18"
EXPECTED_MODEL_IDENTITY_SHA256 = "f0235d0491e5911382b65055775c859ecc5f2a3b4301a68f985f47eb7092a569"
EXPECTED_LORA_RECEIPT_SHA256 = "5d9509e36d5d9d9861df254bc64bdc3342eba79980d39316375b6795e8773ef0"
EXPECTED_LORA_WEIGHT_SHA256 = "043331e8c9cc67cbc168d60256aed7cfa3f6f65a0b19ee6fb4bafd374d5a5ee3"
MODES = ("base", "adapted")
KINDS = ("video", "metadata", "COMPLETE", "generation_lock")


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA256 hex string")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    return value


def hash_fd(descriptor: int) -> tuple[str, int]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest, size = hashlib.sha256(), 0
    while True:
        block = os.read(descriptor, 8 * 1024 * 1024)
        if not block:
            break
        digest.update(block)
        size += len(block)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest(), size


def read_bound_json(binding: Any, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
        raise ValueError(f"{label} binding is malformed")
    expected = require_sha(binding.get("sha256"), f"{label} SHA")
    path = Path(binding["path"])
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_mode & 0o222
            or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ValueError(f"{label} must be a sealed regular file")
        actual, size = hash_fd(descriptor)
        after = os.fstat(descriptor)
        named_after = os.stat(path, follow_symlinks=False)
        if (
            actual != expected
            or size != before.st_size
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or (after.st_dev, after.st_ino) != (named_after.st_dev, named_after.st_ino)
        ):
            raise ValueError(f"{label} changed or differs from its binding")
        os.lseek(descriptor, 0, os.SEEK_SET)
        data = b""
        while len(data) < size:
            block = os.read(descriptor, min(8 * 1024 * 1024, size - len(data)))
            if not block:
                break
            data += block
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload, {"path": str(path.resolve(strict=True)), "sha256": actual}


def require_sealed_below(root: Path, path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
        raise ValueError(f"{label} must be a sealed regular file")
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if resolved_root not in resolved.parents:
        raise ValueError(f"{label} escapes the sealed bundle root")
    relative = resolved.relative_to(resolved_root)
    current = resolved_root
    root_info = os.lstat(current)
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode) or root_info.st_mode & 0o222:
        raise ValueError("train-dev bundle root is not sealed")
    for component in relative.parts[:-1]:
        current = current / component
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o222:
            raise ValueError(f"{label} has an unsealed ancestor")
    return resolved


def require_exact_path(
    binding: dict[str, str], expected: Path, label: str, root: Path
) -> None:
    actual = Path(binding["path"])
    if actual.absolute() != expected.absolute() or actual.resolve(strict=True) != expected.resolve(strict=True):
        raise ValueError(f"{label} path differs from the sealed bundle closure")
    require_sealed_below(root, actual, label)


def validate_isolation(payload: dict[str, Any], input_payload: dict[str, Any]) -> None:
    train = payload.get("train_dev")
    reference = payload.get("excluded_reference")
    if (
        payload.get("schema") != ISOLATION_SCHEMA
        or payload.get("status") != "verified_zero_overlap"
        or payload.get("overlap_counts")
        != {"case_ids": 0, "scene_ids": 0, "transform_sources": 0}
        or payload.get("ids_disclosed") is not False
        or not isinstance(train, dict)
        or not isinstance(reference, dict)
        or train.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or train.get("manifest_sha256") != input_payload.get("source_manifest_sha256")
        or train.get("case_count") != 100
        or reference.get("manifest_sha256") != EXPECTED_REFERENCE_MANIFEST_SHA256
        or reference.get("source_manifest_sha256") != EXPECTED_REFERENCE_SOURCE_MANIFEST_SHA256
        or reference.get("case_count") != 100
    ):
        raise ValueError("train-dev reference-isolation proof is malformed")
    for side in (train, reference):
        commitments = side.get("commitments")
        if not isinstance(commitments, dict) or set(commitments) != {
            "case_ids", "scene_ids", "transform_sources"
        }:
            raise ValueError("train-dev isolation commitments are incomplete")
        for label, value in commitments.items():
            require_sha(value, f"{label} set commitment")


def validate_generation(payload: dict[str, Any], input_payload: dict[str, Any]) -> list[dict[str, Any]]:
    contract = payload.get("contract")
    repo = payload.get("generation_repo")
    reviewed = payload.get("reviewed_sources")
    approval = payload.get("independent_approval")
    model = payload.get("model")
    lora = payload.get("lora")
    if (
        payload.get("schema") != GENERATION_SCHEMA
        or payload.get("status") != "COMPLETE"
        or payload.get("case_count") != 100
        or payload.get("task_count") != 200
        or payload.get("pair_count") != 100
        or payload.get("frozen_video_spec")
        != {
            "width": 1280,
            "height": 704,
            "fps_numerator": 24,
            "fps_denominator": 1,
            "nb_frames": 121,
            "full_decode_backends": ["imageio-ffmpeg-full-decode", "ffmpeg-full-decode"],
        }
        or not all(isinstance(item, dict) for item in (contract, repo, reviewed, approval, model, lora))
        or contract.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or contract.get("case_count") != 100
        or contract.get("seed") != 0
        or contract.get("steps") != 50
        or contract.get("frames") != 121
        or contract.get("height") != 704
        or contract.get("width") != 1280
        or contract.get("fps") != 24
        or contract.get("guidance_scale") != 5.0
        or repo.get("commit") != EXPECTED_GENERATION_COMMIT
        or repo.get("clean") is not True
        or repo.get("runner_sha256") != EXPECTED_RUNNER_SHA256
        or reviewed.get("controller_sha256") != EXPECTED_CONTROLLER_SHA256
        or reviewed.get("tests_sha256") != EXPECTED_CONTROLLER_TESTS_SHA256
        or approval.get("sha256") != EXPECTED_APPROVAL_SHA256
        or model.get("identity_sha256") != EXPECTED_MODEL_IDENTITY_SHA256
        or lora.get("checkpoint_receipt_sha256") != EXPECTED_LORA_RECEIPT_SHA256
        or payload.get("lora_weight_sha256") != EXPECTED_LORA_WEIGHT_SHA256
    ):
        raise ValueError("paired train-dev generation receipt identity is malformed")
    entries = input_payload.get("entries")
    tasks, pairs = payload.get("tasks"), payload.get("pairs")
    if not isinstance(entries, list) or len(entries) != 100 or not isinstance(tasks, list) or len(tasks) != 200 or not isinstance(pairs, list) or len(pairs) != 100:
        raise ValueError("paired train-dev generation receipt has the wrong denominator")
    case_ids = [entry.get("case_id") if isinstance(entry, dict) else None for entry in entries]
    for index, task in enumerate(tasks):
        case_index, mode_index = divmod(index, 2)
        if (
            not isinstance(task, dict)
            or task.get("task_index") != index
            or task.get("case_index") != case_index
            or task.get("case_id") != case_ids[case_index]
            or task.get("mode") != MODES[mode_index]
            or not isinstance(task.get("run_id"), str)
            or not task["run_id"]
            or task.get("stored_video_probe")
            != {
                "width": 1280,
                "height": 704,
                "avg_frame_rate": "24.0",
                "nb_frames": 121,
                "backend": "imageio-ffmpeg-full-decode",
            }
            or task.get("independent_video_probe")
            != {
                "width": 1280,
                "height": 704,
                "avg_frame_rate": "24.0",
                "nb_frames": 121,
                "backend": "ffmpeg-full-decode",
            }
        ):
            raise ValueError(f"paired generation task order differs at index {index}")
        for key in ("video_sha256", "metadata_sha256", "pair_identity_sha256"):
            require_sha(task.get(key), f"generation task {index} {key}")
    for index, pair in enumerate(pairs):
        base, adapted = tasks[index * 2 : index * 2 + 2]
        if (
            not isinstance(pair, dict)
            or pair.get("case_index") != index
            or pair.get("case_id") != case_ids[index]
            or pair.get("pair_identity_sha256") != base["pair_identity_sha256"]
            or pair.get("pair_identity_sha256") != adapted["pair_identity_sha256"]
            or pair.get("base_video_sha256") != base["video_sha256"]
            or pair.get("adapted_video_sha256") != adapted["video_sha256"]
        ):
            raise ValueError(f"paired generation pair differs at index {index}")
    return tasks


def validate_mirror_index(
    payload: dict[str, Any], root: Path, generation_sha: str, manifest_sha: str
) -> dict[tuple[str, str, str], dict[str, Any]]:
    records = payload.get("records")
    if (
        payload.get("schema") != MIRROR_INDEX_SCHEMA
        or payload.get("site") != "Hippasus"
        or payload.get("split") != "dev"
        or payload.get("case_count") != 100
        or payload.get("method_count") != 2
        or payload.get("record_count") != 800
        or payload.get("generation_receipt_sha256") != generation_sha
        or payload.get("manifest_sha256") != manifest_sha
        or not isinstance(records, list)
        or len(records) != 800
    ):
        raise ValueError("train-dev mirror index identity is malformed")
    mapped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("train-dev mirror record is malformed")
        mode, case_id, kind = record.get("mode"), record.get("case_id"), record.get("kind")
        key = (mode, case_id, kind)
        if mode not in MODES or not isinstance(case_id, str) or kind not in KINDS or key in mapped:
            raise ValueError("train-dev mirror record key is malformed or duplicate")
        expected_sha = require_sha(record.get("sha256"), "train-dev mirror record SHA")
        expected_size = record.get("bytes")
        raw_path = record.get("path")
        if not isinstance(expected_size, int) or expected_size < 0 or not isinstance(raw_path, str):
            raise ValueError("train-dev mirror record payload is malformed")
        path = Path(raw_path)
        resolved = require_sealed_below(root, path, "train-dev mirror artifact")
        if root not in resolved.parents or resolved.parent.parent.parent != root / "mirror":
            raise ValueError("train-dev mirror record path escapes the sealed mode closure")
        if resolved.parent.parent.name != mode or resolved.name != {
            "video": "video.mp4",
            "metadata": "metadata.json",
            "COMPLETE": "COMPLETE",
            "generation_lock": ".generation.lock",
        }[kind]:
            raise ValueError("train-dev mirror record path/mode/kind mismatch")
        actual_payload, actual_binding = read_bound_json(
            {"path": str(resolved), "sha256": expected_sha},
            "train-dev mirror JSON artifact",
        ) if kind == "metadata" else ({}, {"path": str(resolved), "sha256": expected_sha})
        if kind != "metadata":
            descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222:
                    raise ValueError("train-dev mirror artifact is not sealed")
                actual_sha, actual_size = hash_fd(descriptor)
            finally:
                os.close(descriptor)
            if actual_sha != expected_sha or actual_size != expected_size:
                raise ValueError("train-dev mirror artifact differs from its index")
        else:
            if Path(actual_binding["path"]).stat().st_size != expected_size or not isinstance(actual_payload, dict):
                raise ValueError("train-dev mirror metadata differs from its indexed size")
        mapped[key] = record
    return mapped


def verify_train_dev_provenance(
    *,
    input_payload: dict[str, Any],
    input_binding: dict[str, str],
    source_bundle_ready: Any,
    expected_generation_receipt_sha256: Any,
    expected_baseline_eligibility_sha256: Any = None,
) -> dict[str, Any]:
    if input_payload.get("schema") != INPUT_SCHEMA or input_payload.get("scope") != "train_dev_evaluation":
        raise ValueError("train-dev provenance verifier received a cross-scope input")
    method_id = input_payload.get("method_id")
    serialized_input = json.dumps(input_payload, sort_keys=True, separators=(",", ":"))
    if (
        method_id not in METHOD_LABELS
        or input_payload.get("method") != METHOD_LABELS[method_id]
        or any(
            token in serialized_input
            for token in ("formal_validation", "validation_only", "wan_unguided_seed0")
        )
    ):
        raise ValueError("train-dev input method label or scope identity is prohibited")
    expected_generation_sha = require_sha(
        expected_generation_receipt_sha256, "external generation receipt SHA"
    )
    root = Path(input_payload.get("mirror_root", ""))
    if not root.is_absolute() or root.is_symlink() or not root.is_dir() or root.stat().st_mode & 0o222:
        raise ValueError("train-dev mirror root must be an absolute sealed directory")
    root = root.resolve(strict=True)
    ready, ready_binding = read_bound_json(source_bundle_ready, "source bundle READY")
    require_exact_path(ready_binding, root / "BUNDLE_READY.json", "source bundle READY", root)
    if (
        ready.get("schema") != BUNDLE_SCHEMA
        or ready.get("status") != "READY"
        or ready.get("site") != "Hippasus"
        or ready.get("split") != "dev"
        or ready.get("reserved_ids_disclosed") is not False
        or ready.get("case_count") != 100
        or ready.get("task_count") != 200
        or ready.get("pair_count") != 100
        or ready.get("mirror_file_count") != 800
    ):
        raise ValueError("source bundle READY identity is malformed")
    generation, generation_binding = read_bound_json(ready.get("generation_receipt"), "source generation receipt")
    manifest, manifest_binding = read_bound_json(ready.get("manifest"), "train-dev manifest")
    isolation, isolation_binding = read_bound_json(
        ready.get("traindev_reference_isolation_receipt"), "train-dev isolation receipt"
    )
    mirror_index, mirror_index_binding = read_bound_json(ready.get("mirror_index"), "train-dev mirror index")
    base_input, base_binding = read_bound_json(ready.get("base_input_lock"), "bundle base input lock")
    adapted_entries, adapted_binding = read_bound_json(ready.get("adapted_input_entries"), "bundle adapted entries")
    for binding, relative, label in (
        (generation_binding, Path("provenance/PAIRED_GENERATION_RECEIPT.json"), "generation receipt"),
        (manifest_binding, Path("provenance/dev100_manifest_960p_v2.json"), "manifest"),
        (isolation_binding, Path("TRAINDEV_REFERENCE_ISOLATION_RECEIPT.json"), "isolation receipt"),
        (mirror_index_binding, Path("MIRROR_INDEX.json"), "mirror index"),
        (base_binding, Path("BASE_INPUT_LOCK.json"), "base input lock"),
        (adapted_binding, Path("ADAPTED_INPUT_ENTRIES.json"), "adapted entries"),
    ):
        require_exact_path(binding, root / relative, label, root)
    if generation_binding["sha256"] != expected_generation_sha:
        raise ValueError("source generation receipt differs from its external SHA")
    if manifest_binding["sha256"] != EXPECTED_MANIFEST_SHA256 or input_payload.get("source_manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("train-dev source manifest differs from its frozen SHA")
    validate_isolation(isolation, input_payload)
    tasks = validate_generation(generation, input_payload)
    records = validate_mirror_index(mirror_index, root, generation_binding["sha256"], manifest_binding["sha256"])
    if (
        input_payload.get("traindev_reference_isolation_receipt_sha256")
        != isolation_binding["sha256"]
        or input_payload.get("source_generation_receipt_sha256")
        != generation_binding["sha256"]
        or input_payload.get("mirror_index_sha256") != mirror_index_binding["sha256"]
    ):
        raise ValueError("train-dev input does not bind the exact bundle provenance files")
    if method_id == BASE_METHOD_ID:
        mode = "base"
        if expected_baseline_eligibility_sha256 is not None:
            raise ValueError("base provenance must not receive an adapted eligibility SHA")
        if ready.get("base_input_lock") != input_binding or base_binding != input_binding or base_input != input_payload:
            raise ValueError("base input is not the exact input sealed by BUNDLE_READY")
    elif method_id == ADAPTED_METHOD_ID:
        mode = "adapted"
        expected_eligibility_sha = require_sha(
            expected_baseline_eligibility_sha256,
            "external completed-baseline eligibility SHA",
        )
        if input_payload.get("source_bundle_ready") != ready_binding or input_payload.get("source_adapted_entries") != adapted_binding:
            raise ValueError("adapted input does not bind the exact paired bundle/entry set")
        if adapted_entries.get("schema") != ADAPTED_ENTRIES_SCHEMA or adapted_entries.get("entries") != input_payload.get("entries"):
            raise ValueError("adapted input differs from the exact sealed adapted entries")
        eligibility, eligibility_binding = read_bound_json(
            input_payload.get("baseline_metric_eligibility_lock"), "baseline eligibility lock"
        )
        if (
            eligibility.get("schema") != ELIGIBILITY_SCHEMA
            or eligibility.get("scope") != "train_dev_evaluation"
            or eligibility.get("baseline_method_id") != BASE_METHOD_ID
            or eligibility.get("baseline_input_lock") != base_binding
            or input_payload.get("baseline_input_lock_sha256") != base_binding["sha256"]
            or eligibility_binding["sha256"] != expected_eligibility_sha
            or input_payload.get("expected_baseline_eligibility_sha256")
            != expected_eligibility_sha
            or set(eligibility.get("case_ids", []))
            != {entry.get("case_id") for entry in input_payload.get("entries", []) if isinstance(entry, dict)}
        ):
            raise ValueError("adapted input lacks the exact base-first eligibility chain")
    else:
        raise ValueError("unknown train-dev method identity")
    entries = input_payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 100:
        raise ValueError("train-dev input must contain exactly 100 ordered entries")
    for order, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("split_order") != order or entry.get("lora_mode") != mode:
            raise ValueError("train-dev input order/mode differs from the sealed pair grid")
        case_id = entry.get("case_id")
        task = tasks[order * 2 + MODES.index(mode)]
        if task.get("case_id") != case_id or task.get("video_sha256") != entry.get("video_sha256") or task.get("metadata_sha256") != entry.get("metadata_sha256") or task.get("pair_identity_sha256") != entry.get("pair_identity_sha256"):
            raise ValueError("train-dev input differs from its paired generation receipt")
        for kind, path_key, sha_key in (
            ("video", "metric_video_path", "video_sha256"),
            ("metadata", "metric_metadata_path", "metadata_sha256"),
            ("COMPLETE", "metric_complete_path", "complete_sha256"),
            ("generation_lock", "metric_generation_lock_path", "generation_lock_sha256"),
        ):
            record = records.get((mode, case_id, kind))
            if record is None or record.get("path") != entry.get(path_key) or record.get("sha256") != entry.get(sha_key):
                raise ValueError("train-dev input entry differs from the exact mirror index closure")
    return {
        "schema": "geometry-selection-traindev-provenance-verification-v1",
        "status": "verified",
        "source_bundle_ready": ready_binding,
        "generation_receipt": generation_binding,
        "manifest": manifest_binding,
        "reference_isolation_receipt": isolation_binding,
        "mirror_index": mirror_index_binding,
        "base_input_lock": base_binding,
        "adapted_entries": adapted_binding,
        "expected_generation_receipt_sha256": expected_generation_sha,
        "expected_baseline_eligibility_sha256": (
            expected_eligibility_sha if mode == "adapted" else None
        ),
        "mode": mode,
        "case_count": 100,
        "record_count": 800,
    }
