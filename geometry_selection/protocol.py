"""Validation helpers binding generated artifacts to the frozen DL3DV protocol."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


DL3DV_PROTOCOL_SCHEMA = "dl3dv-geometry-selection-v1"
FORMAL_SPLIT_COUNTS = {"debug": 3, "validation": 100, "test": 100}
IMPLEMENTATION_ROOTS = (
    "geometry_selection",
    "benchmarks/dl3dv_geco",
    "scripts",
    "external/guidance_wan",
    "external/guidance_cosmos",
)


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_tree_sha256(repo_root: Path) -> str:
    """Hash executable project sources without including experiment artifacts."""
    root = repo_root.resolve()
    files: list[Path] = []
    for relative_root in IMPLEMENTATION_ROOTS:
        source_root = (root / relative_root).resolve()
        try:
            source_root.relative_to(root)
        except ValueError as error:
            raise ValueError(f"implementation root escapes repository: {source_root}") from error
        if source_root.is_dir():
            files.extend(path for path in source_root.rglob("*.py") if path.is_file())
    if not files:
        raise ValueError("implementation source set is empty")
    digest = hashlib.sha256()
    for path in sorted(set(files), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
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
    model_lock_sha256 = metadata.get("model_lock_sha256")
    if not isinstance(model_lock_sha256, str) or len(model_lock_sha256) != 64:
        raise ValueError("formal protocol must bind model_lock_sha256")
    if metadata.get("split_counts") != FORMAL_SPLIT_COUNTS:
        raise ValueError(f"formal split counts must be {FORMAL_SPLIT_COUNTS}")
    cases = [value for key, value in payload.items() if not key.startswith("_")]
    if len(cases) != sum(FORMAL_SPLIT_COUNTS.values()):
        raise ValueError(
            f"formal protocol must contain exactly {sum(FORMAL_SPLIT_COUNTS.values())} cases"
        )
    unknown_splits = sorted(
        {case.get("split") for case in cases} - set(FORMAL_SPLIT_COUNTS),
        key=str,
    )
    if unknown_splits:
        raise ValueError(f"formal protocol contains unknown splits: {unknown_splits}")
    actual_counts = {
        split: sum(case.get("split") == split for case in cases)
        for split in FORMAL_SPLIT_COUNTS
    }
    if actual_counts != FORMAL_SPLIT_COUNTS:
        raise ValueError(f"actual split counts do not match metadata: {actual_counts}")
    scene_uids = [case["scene_uid"] for case in cases]
    canonical_scene_uids = [
        f"dl3dv:{Path(case['dataset_relative_transforms']).parent.as_posix()}"
        for case in cases
    ]
    mismatched_scene_uids = [
        (case["scene_uid"], canonical)
        for case, canonical in zip(cases, canonical_scene_uids)
        if case["scene_uid"] != canonical
    ]
    if mismatched_scene_uids:
        raise ValueError(
            "formal protocol scene_uid must be derived from transforms parent: "
            f"{mismatched_scene_uids[:3]}"
        )
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


def validate_experiment_lock(
    lock_path: Path,
    repo_root: Path,
    expected_commit: str,
    *,
    protocol_manifest_sha256: str,
    model_lock_sha256: str,
    split: str,
    backbone: str,
    candidate_spec_sha256: str,
    artifact_root: Path,
    ranking_config_hash: str | None = None,
) -> dict[str, Any]:
    committed = validate_committed_file(lock_path, repo_root, expected_commit)
    lock = json.loads(committed)
    expected = {
        "schema": "geometry-experiment-lock-v1",
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "model_lock_sha256": model_lock_sha256,
        "implementation_sha256": implementation_tree_sha256(repo_root),
        "split": split,
        "backbone": backbone,
        "candidate_spec_sha256": candidate_spec_sha256,
        "artifact_root": str(artifact_root.resolve()),
    }
    mismatches = {
        key: (lock.get(key), value)
        for key, value in expected.items()
        if lock.get(key) != value
    }
    if mismatches:
        raise ValueError(f"experiment lock mismatch: {mismatches}")
    allowed = lock.get("authorized_ranking_config_hashes")
    if not isinstance(allowed, list) or not allowed:
        raise ValueError("experiment lock contains no authorized ranking configs")
    if ranking_config_hash is not None and ranking_config_hash not in allowed:
        raise ValueError(
            f"ranking config {ranking_config_hash} is not authorized for split {split}"
        )
    return lock


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


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_generation_sidecar(
    *,
    candidate: dict[str, Any],
    case_id: str,
    frozen_case: dict[str, Any],
    split: str,
    backbone: str,
    generation_profile: dict[str, Any],
    protocol_manifest_sha256: str,
    expected_git_commit: str | None,
    model_lock_sha256: str,
    locked_model_identity: dict[str, Any],
    experiment_lock_sha256: str,
    implementation_sha256: str,
    candidate_spec_sha256: str,
) -> None:
    metadata_value = candidate.get("generation_metadata")
    if not metadata_value:
        raise FileNotFoundError(f"formal candidate requires generation_metadata: {case_id}")
    metadata_path = Path(metadata_value).resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    video = Path(candidate["video"]).resolve()
    generation = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_generation = {
        "case_id": case_id,
        "manifest_sha256": protocol_manifest_sha256,
        "image_sha256": frozen_case["image_sha256"],
        "prompt": frozen_case["text_prompt"],
        "method": "baseline",
        "seed": candidate["seed"],
        "video_sha256": file_sha256(video),
        "backbone": "wan" if backbone.startswith("Wan") else "cosmos",
        "model_lock_sha256": model_lock_sha256,
        "model_content_verified": True,
        "locked_model_identity": locked_model_identity,
        "experiment_lock_sha256": experiment_lock_sha256,
        "implementation_sha256": implementation_sha256,
        "candidate_spec_sha256": candidate_spec_sha256,
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
            f"generation sidecar differs from candidate/protocol for {case_id}: {mismatches}"
        )
    expected_protocol = {
        "mode": "frozen",
        "split": split,
        "expected_split": split,
    }
    protocol_record = generation.get("protocol", {})
    protocol_mismatches = {
        key: (protocol_record.get(key), expected)
        for key, expected in expected_protocol.items()
        if protocol_record.get(key) != expected
    }
    if protocol_mismatches:
        raise ValueError(
            f"generation sidecar protocol mismatch for {case_id}: {protocol_mismatches}"
        )
    generation_record = generation.get("generation", {})
    profile_mismatches = {
        key: (generation_record.get(key), expected)
        for key, expected in generation_profile.items()
        if generation_record.get(key) != expected
    }
    if profile_mismatches:
        raise ValueError(
            f"generation profile sidecar mismatch for {case_id}: {profile_mismatches}"
        )
    if Path(generation.get("video", "")).resolve() != video:
        raise ValueError(f"generation sidecar video path mismatch: {case_id}")
    run_id = generation.get("run_id")
    run_config = generation.get("run_config")
    if not isinstance(run_id, str) or len(run_id) != 12 or not isinstance(run_config, dict):
        raise ValueError(f"generation sidecar lacks canonical run identity: {case_id}")
    run_config_sha256 = _canonical_sha256(run_config)
    if generation.get("run_config_sha256") != run_config_sha256:
        raise ValueError(f"generation run_config digest mismatch: {case_id}")
    if run_id != run_config_sha256[:12]:
        raise ValueError(f"generation run_id mismatch: {case_id}")
    run_expected = {
        "manifest_sha256": protocol_manifest_sha256,
        "case_id": case_id,
        "image_sha256": frozen_case["image_sha256"],
        "prompt": frozen_case["text_prompt"],
        "method": "baseline",
        "seed": candidate["seed"],
        "code_identity": expected_generation.get("code_identity"),
        "model_lock_sha256": model_lock_sha256,
        "locked_model_identity": locked_model_identity,
        "experiment_lock_sha256": experiment_lock_sha256,
        "implementation_sha256": implementation_sha256,
    }
    run_mismatches = {
        key: (run_config.get(key), expected)
        for key, expected in run_expected.items()
        if expected is not None and run_config.get(key) != expected
    }
    if run_mismatches:
        raise ValueError(f"generation run_config mismatch for {case_id}: {run_mismatches}")
    if generation.get("model") != run_config.get("model") or not generation.get("model"):
        raise ValueError(f"generation model identity mismatch: {case_id}")
    if generation.get("runner_sha256") != run_config.get("runner_sha256"):
        raise ValueError(f"generation runner digest mismatch: {case_id}")
    expected_metadata = video.parent / "metadata.json"
    complete = video.parent / "COMPLETE"
    canonical_parts = (
        video.name == "video.mp4"
        and metadata_path == expected_metadata
        and video.parent.name == f"run_{run_id}"
        and video.parent.parent.name == f"seed_{candidate['seed']}"
        and video.parent.parent.parent.name == case_id
        and video.parent.parent.parent.parent.name == "baseline"
        and video.parent.parent.parent.parent.parent.name == expected_generation["backbone"]
    )
    if not canonical_parts:
        raise ValueError(f"generation output is outside the canonical run layout: {case_id}")
    if not complete.is_file() or complete.read_text(encoding="utf-8").strip() != run_id:
        raise ValueError(f"generation COMPLETE marker mismatch: {case_id}")


def _validate_pool_cases_against_protocol(
    pool: dict[str, Any],
    protocol: dict[str, Any],
    dataset_root: Path,
    *,
    formal: bool,
    model_lock: dict[str, Any] | None,
    experiment_lock_sha256: str | None,
    expected_git_commit: str | None,
    expected_implementation_sha256: str | None,
) -> None:
    if formal and pool.get("artifact_mode") != "formal":
        raise ValueError("legacy-debug candidate pool cannot be promoted to formal")
    preparation = pool.get("preparation", {})
    if formal and preparation.get("dirty") is not False:
        raise ValueError("formal candidate pool was not prepared from clean code")
    if formal and model_lock is None:
        raise ValueError("formal candidate pool validation requires the frozen model lock")
    if formal and not experiment_lock_sha256:
        raise ValueError("formal candidate pool validation requires experiment lock digest")
    if formal and preparation.get("experiment_lock_sha256") != experiment_lock_sha256:
        raise ValueError("candidate pool preparation used a different experiment lock")
    if formal and preparation.get("commit") != expected_git_commit:
        raise ValueError("candidate pool preparation commit differs from frozen ranking commit")
    if formal and preparation.get("implementation_sha256") != expected_implementation_sha256:
        raise ValueError("candidate pool preparation implementation differs from experiment lock")
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
                validate_generation_sidecar(
                    candidate=candidate,
                    case_id=case_id,
                    frozen_case=frozen,
                    split=split,
                    backbone=case["backbone"],
                    generation_profile=case["generation"],
                    protocol_manifest_sha256=pool["protocol_manifest_sha256"],
                    expected_git_commit=preparation.get("commit"),
                    model_lock_sha256=protocol["_meta"]["model_lock_sha256"],
                    locked_model_identity=model_lock["generation_models"][case["backbone"]],
                    experiment_lock_sha256=experiment_lock_sha256,
                    implementation_sha256=expected_implementation_sha256,
                    candidate_spec_sha256=pool["candidate_spec_sha256"],
                )


def validate_candidate_pool_against_protocol(
    pool: dict[str, Any],
    protocol_path: Path,
    dataset_root: Path,
    *,
    formal: bool = True,
    model_lock: dict[str, Any] | None = None,
    experiment_lock_sha256: str | None = None,
    expected_git_commit: str | None = None,
    expected_implementation_sha256: str | None = None,
) -> None:
    protocol_path = protocol_path.resolve()
    if file_sha256(protocol_path) != pool["protocol_manifest_sha256"]:
        raise ValueError("candidate pool does not match the protocol digest")
    protocol = validate_formal_protocol(protocol_path) if formal else load_protocol(protocol_path)
    _validate_pool_cases_against_protocol(
        pool,
        protocol,
        dataset_root,
        formal=formal,
        model_lock=model_lock,
        experiment_lock_sha256=experiment_lock_sha256,
        expected_git_commit=expected_git_commit,
        expected_implementation_sha256=expected_implementation_sha256,
    )


def validate_candidate_spec_against_protocol(
    spec: dict[str, Any],
    protocol_path: Path,
    dataset_root: Path,
    *,
    formal: bool = True,
    expected_git_commit: str | None = None,
    model_lock: dict[str, Any] | None = None,
    experiment_lock_sha256: str | None = None,
    expected_implementation_sha256: str | None = None,
    candidate_spec_sha256: str | None = None,
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
    if formal and model_lock is None:
        raise ValueError("formal candidate spec validation requires the frozen model lock")
    if formal and not experiment_lock_sha256:
        raise ValueError("formal candidate spec validation requires experiment lock digest")
    if formal and not expected_implementation_sha256:
        raise ValueError("formal candidate spec validation requires implementation digest")
    if formal and not candidate_spec_sha256:
        raise ValueError("formal candidate spec validation requires candidate spec digest")
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
                validate_generation_sidecar(
                    candidate=candidate,
                    case_id=case_id,
                    frozen_case=frozen,
                    split=spec["split"],
                    backbone=spec["backbone"],
                    generation_profile=spec["generation"],
                    protocol_manifest_sha256=spec["protocol_manifest_sha256"],
                    expected_git_commit=expected_git_commit,
                    model_lock_sha256=protocol["_meta"]["model_lock_sha256"],
                    locked_model_identity=model_lock["generation_models"][spec["backbone"]],
                    experiment_lock_sha256=experiment_lock_sha256,
                    implementation_sha256=expected_implementation_sha256,
                    candidate_spec_sha256=candidate_spec_sha256,
                )
