"""Validation helpers binding generated artifacts to the frozen DL3DV protocol."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


DL3DV_PROTOCOL_SCHEMA = "dl3dv-geometry-selection-v1"
FORMAL_SPLIT_COUNTS = {"debug": 3, "validation": 100, "test": 100}


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("_meta", {}).get("schema") != DL3DV_PROTOCOL_SCHEMA:
        raise ValueError(f"protocol schema must be {DL3DV_PROTOCOL_SCHEMA}")
    return payload


def validate_formal_protocol(path: Path) -> dict[str, Any]:
    payload = load_protocol(path)
    metadata = payload["_meta"]
    if metadata.get("formal_protocol") is not True:
        raise ValueError("protocol is not marked formal")
    if metadata.get("split_counts") != FORMAL_SPLIT_COUNTS:
        raise ValueError(f"formal split counts must be {FORMAL_SPLIT_COUNTS}")
    cases = [value for key, value in payload.items() if not key.startswith("_")]
    actual_counts = {
        split: sum(case.get("split") == split for case in cases)
        for split in FORMAL_SPLIT_COUNTS
    }
    if actual_counts != FORMAL_SPLIT_COUNTS:
        raise ValueError(f"actual split counts do not match metadata: {actual_counts}")
    scene_uids = [case["scene_uid"] for case in cases]
    image_hashes = [case["image_sha256"] for case in cases]
    if len(set(scene_uids)) != len(scene_uids) or len(set(image_hashes)) != len(image_hashes):
        raise ValueError("formal protocol is not scene/image disjoint")
    for split, count in FORMAL_SPLIT_COUNTS.items():
        orders = sorted(case["split_order"] for case in cases if case["split"] == split)
        if orders != list(range(count)):
            raise ValueError(f"split_order is incomplete or duplicated for {split}")
    policy = metadata.get("candidate_seed_policy", {})
    if policy.get("candidate_seeds") != [0, 1, 2, 3] or policy.get("incumbent_seed") != 0:
        raise ValueError("formal candidate seed policy must be fixed seeds 0--3, incumbent 0")
    return payload


def formal_split_cases(payload: dict[str, Any], split: str) -> dict[str, dict[str, Any]]:
    if split not in FORMAL_SPLIT_COUNTS:
        raise ValueError(f"unknown formal split: {split}")
    return {
        key: value
        for key, value in payload.items()
        if not key.startswith("_") and value.get("split") == split
    }


def resolve_protocol_image(case: dict[str, Any], dataset_root: Path) -> Path:
    root = dataset_root.resolve()
    relative = Path(case["dataset_relative_image"])
    if relative.is_absolute():
        raise ValueError("dataset_relative_image must be relative")
    image = (root / relative).resolve()
    try:
        image.relative_to(root)
    except ValueError as error:
        raise ValueError("dataset_relative_image escapes dataset_root") from error
    if not image.is_file():
        raise FileNotFoundError(image)
    if file_sha256(image) != case["image_sha256"]:
        raise ValueError(f"frozen conditioning image hash mismatch: {image}")
    return image


def resolve_protocol_transforms(case: dict[str, Any], dataset_root: Path) -> Path:
    root = dataset_root.resolve()
    relative = Path(case["dataset_relative_transforms"])
    if relative.is_absolute():
        raise ValueError("dataset_relative_transforms must be relative")
    transforms = (root / relative).resolve()
    try:
        transforms.relative_to(root)
    except ValueError as error:
        raise ValueError("dataset_relative_transforms escapes dataset_root") from error
    if not transforms.is_file():
        raise FileNotFoundError(transforms)
    if file_sha256(transforms) != case["transforms_sha256"]:
        raise ValueError(f"frozen transforms hash mismatch: {transforms}")
    return transforms


def validate_committed_test_release(
    release_path: Path,
    repo_root: Path,
    expected_commit: str,
    expected_fields: dict[str, Any],
) -> dict[str, Any]:
    committed = validate_committed_file(release_path, repo_root, expected_commit)
    release = json.loads(committed)
    mismatches = {
        key: (release.get(key), expected)
        for key, expected in expected_fields.items()
        if release.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"test release mismatch: {mismatches}")
    return release


def validate_committed_file(
    path: Path,
    repo_root: Path,
    expected_commit: str,
) -> bytes:
    repo_root = repo_root.resolve()
    path = path.resolve()
    try:
        relative = path.relative_to(repo_root)
    except ValueError as error:
        raise ValueError("formal artifact must be inside the frozen repository") from error
    committed = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{expected_commit}:{relative.as_posix()}"],
        check=True,
        capture_output=True,
    ).stdout
    if committed != path.read_bytes():
        raise ValueError("formal artifact differs from the file committed in the frozen revision")
    return committed


def _validate_pool_cases_against_protocol(
    pool: dict[str, Any],
    protocol: dict[str, Any],
    dataset_root: Path,
    *,
    formal: bool,
) -> None:
    split = pool["cases"][0]["split"]
    frozen_cases = {
        key: value
        for key, value in protocol.items()
        if not key.startswith("_") and value.get("split") == split
    }
    pool_ids = {case["case_id"] for case in pool["cases"]}
    if formal and pool_ids != set(frozen_cases):
        raise ValueError("formal candidate pool must contain the complete frozen split")
    seed_policy = protocol["_meta"].get("candidate_seed_policy", {})
    expected_backbones = set(protocol["_meta"].get("generation_profiles", {}))
    for case in pool["cases"]:
        case_id = case["case_id"]
        frozen = frozen_cases.get(case_id)
        if frozen is None:
            raise ValueError(f"candidate pool case is absent from protocol: {case_id}")
        expected = (frozen["scene_uid"], frozen["text_prompt"], frozen["image_sha256"])
        actual = (case["scene_uid"], case["prompt"], case["conditioning_image_sha256"])
        if actual != expected:
            raise ValueError(f"candidate pool case differs from protocol: {case_id}")
        if formal and case["backbone"] not in expected_backbones:
            raise ValueError(f"formal candidate backbone is not frozen: {case['backbone']}")
        if formal and case["generation"] != protocol["_meta"]["generation_profiles"][case["backbone"]]:
            raise ValueError(f"candidate pool generation profile differs: {case_id}")
        resolve_protocol_image(frozen, dataset_root)
        resolve_protocol_transforms(frozen, dataset_root)
        if formal:
            seeds = sorted(candidate["seed"] for candidate in case["candidates"])
            if seeds != seed_policy["candidate_seeds"]:
                raise ValueError(f"candidate pool seeds differ from protocol: {case_id}")
            for candidate in case["candidates"]:
                expected_incumbent = candidate["seed"] == seed_policy["incumbent_seed"]
                if candidate["is_incumbent"] is not expected_incumbent:
                    raise ValueError(f"candidate pool incumbent differs from protocol: {case_id}")


def validate_candidate_pool_against_protocol(
    pool: dict[str, Any],
    protocol_path: Path,
    dataset_root: Path,
    *,
    formal: bool = True,
) -> None:
    protocol_path = protocol_path.resolve()
    if file_sha256(protocol_path) != pool["protocol_manifest_sha256"]:
        raise ValueError("candidate pool does not match the protocol digest")
    protocol = validate_formal_protocol(protocol_path) if formal else load_protocol(protocol_path)
    _validate_pool_cases_against_protocol(pool, protocol, dataset_root, formal=formal)


def validate_candidate_spec_against_protocol(
    spec: dict[str, Any],
    protocol_path: Path,
    dataset_root: Path,
    *,
    formal: bool = True,
    expected_git_commit: str | None = None,
) -> None:
    protocol_path = protocol_path.resolve()
    if file_sha256(protocol_path) != spec["protocol_manifest_sha256"]:
        raise ValueError("candidate spec does not match the frozen protocol digest")
    protocol = validate_formal_protocol(protocol_path) if formal else load_protocol(protocol_path)
    frozen_cases = formal_split_cases(protocol, spec["split"]) if formal else {
        key: value
        for key, value in protocol.items()
        if not key.startswith("_") and value["split"] == spec["split"]
    }
    if formal and {case["case_id"] for case in spec["cases"]} != set(frozen_cases):
        raise ValueError("formal candidate spec must contain the complete frozen split")
    seed_policy = protocol["_meta"].get("candidate_seed_policy", {})
    expected_backbones = set(protocol["_meta"].get("generation_profiles", {}))
    if formal and spec["backbone"] not in expected_backbones:
        raise ValueError(f"formal candidate backbone is not frozen: {spec['backbone']}")
    if formal and spec["generation"] != protocol["_meta"]["generation_profiles"][spec["backbone"]]:
        raise ValueError("formal candidate generation profile differs from protocol")
    if formal and spec["candidate_count"] != len(seed_policy["candidate_seeds"]):
        raise ValueError("candidate count differs from the frozen seed policy")
    for source_case in spec["cases"]:
        case_id = source_case["case_id"]
        if case_id not in protocol or case_id.startswith("_"):
            raise ValueError(f"candidate case is absent from frozen protocol: {case_id}")
        frozen = protocol[case_id]
        expected = {
            "scene_uid": frozen["scene_uid"],
            "split": frozen["split"],
            "prompt": frozen["text_prompt"],
        }
        actual = {
            "scene_uid": source_case["scene_uid"],
            "split": spec["split"],
            "prompt": source_case["prompt"],
        }
        if actual != expected:
            raise ValueError(f"candidate case differs from frozen protocol: {case_id}")
        frozen_image = resolve_protocol_image(frozen, dataset_root)
        resolve_protocol_transforms(frozen, dataset_root)
        candidate_image = Path(source_case["conditioning_image"]).resolve()
        if file_sha256(candidate_image) != file_sha256(frozen_image):
            raise ValueError(f"candidate conditioning image differs from protocol: {case_id}")
        if formal:
            seeds = sorted(candidate["seed"] for candidate in source_case["candidates"])
            if seeds != seed_policy["candidate_seeds"]:
                raise ValueError(f"candidate seeds differ from protocol: {case_id}")
            for candidate in source_case["candidates"]:
                expected_incumbent = candidate["seed"] == seed_policy["incumbent_seed"]
                if candidate["is_incumbent"] is not expected_incumbent:
                    raise ValueError(f"incumbent assignment differs from protocol: {case_id}")
                metadata_path = Path(candidate.get("generation_metadata", "")).resolve()
                if not metadata_path.is_file():
                    raise FileNotFoundError(
                        f"formal candidate requires generation_metadata: {case_id}"
                    )
                generation = json.loads(metadata_path.read_text(encoding="utf-8"))
                video = Path(candidate["video"]).resolve()
                expected_generation = {
                    "case_id": case_id,
                    "manifest_sha256": spec["protocol_manifest_sha256"],
                    "image_sha256": frozen["image_sha256"],
                    "prompt": frozen["text_prompt"],
                    "method": "baseline",
                    "seed": candidate["seed"],
                    "video_sha256": file_sha256(video),
                    "backbone": "wan" if spec["backbone"].startswith("Wan") else "cosmos",
                }
                if expected_git_commit is not None:
                    expected_generation["code_identity"] = {
                        "commit": expected_git_commit,
                        "dirty": False,
                    }
                mismatches = {
                    key: (generation.get(key), expected)
                    for key, expected in expected_generation.items()
                    if generation.get(key) != expected
                }
                if mismatches:
                    raise ValueError(
                        f"generation sidecar differs from candidate/protocol for {case_id}: "
                        f"{mismatches}"
                    )
                protocol_record = generation.get("protocol", {})
                expected_protocol = {
                    "mode": "frozen",
                    "split": spec["split"],
                    "expected_split": spec["split"],
                }
                protocol_mismatches = {
                    key: (protocol_record.get(key), expected)
                    for key, expected in expected_protocol.items()
                    if protocol_record.get(key) != expected
                }
                if protocol_mismatches:
                    raise ValueError(
                        f"generation sidecar protocol mismatch for {case_id}: "
                        f"{protocol_mismatches}"
                    )
                generation_profile = generation.get("generation", {})
                profile_mismatches = {
                    key: (generation_profile.get(key), expected)
                    for key, expected in spec["generation"].items()
                    if generation_profile.get(key) != expected
                }
                if profile_mismatches:
                    raise ValueError(
                        f"generation profile sidecar mismatch for {case_id}: "
                        f"{profile_mismatches}"
                    )
                if Path(generation["video"]).resolve() != video:
                    raise ValueError(f"generation sidecar video path mismatch: {case_id}")
