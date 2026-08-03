from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import geometry_selection.protocol as protocol_module
from geometry_selection.protocol import (
    file_sha256,
    implementation_tree_sha256,
    implementation_tree_sha256_at_commit,
    validate_candidate_spec_against_protocol,
    validate_candidate_pool_against_protocol,
    validate_experiment_lock,
    validate_formal_protocol,
    validate_implementation_commit,
)
from geometry_selection.selection import CANDIDATE_SPEC_SCHEMA, materialize_candidate_pool


DEBUG_BUILDER = Path(__file__).resolve().parents[2] / "scripts/build_legacy_debug_smoke.py"


def test_candidate_spec_is_bound_to_protocol_split_prompt_and_image(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    scene = dataset / "scene-a"
    scene.mkdir(parents=True)
    image = scene / "frame.png"
    image.write_bytes(b"image")
    transforms = scene / "transforms.json"
    transforms.write_bytes(b"transforms")
    protocol = {
        "_meta": {"schema": "dl3dv-geometry-selection-v1"},
        "case-a": {
            "scene_uid": "dl3dv:scene-a",
            "split": "validation",
            "text_prompt": "camera moves forward",
            "dataset_relative_image": "scene-a/frame.png",
            "image_sha256": file_sha256(image),
            "dataset_relative_transforms": "scene-a/transforms.json",
            "transforms_sha256": file_sha256(transforms),
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    spec = {
        "schema": CANDIDATE_SPEC_SCHEMA,
        "protocol_manifest_sha256": file_sha256(protocol_path),
        "split": "validation",
        "candidate_count": 2,
        "backbone": "wan",
        "generation": {"steps": 50},
        "cases": [
            {
                "case_id": "case-a",
                "scene_uid": "dl3dv:scene-a",
                "conditioning_image": str(image),
                "prompt": "camera moves forward",
                "candidates": [
                    {
                        "candidate_id": f"candidate-{index}",
                        "seed": index,
                        "video": str(video),
                        "is_incumbent": index == 0,
                    }
                    for index in range(2)
                ],
            }
        ],
    }
    validate_candidate_spec_against_protocol(spec, protocol_path, dataset, formal=False)

    spec["split"] = "test"
    with pytest.raises(ValueError, match="differs from frozen protocol"):
        validate_candidate_spec_against_protocol(spec, protocol_path, dataset, formal=False)


def _small_formal_fixture(tmp_path, monkeypatch):
    counts = {"debug": 1, "validation": 1, "test": 1}
    monkeypatch.setattr(protocol_module, "FORMAL_SPLIT_COUNTS", counts)
    dataset = tmp_path / "dataset"
    profile = {
        "steps": 50,
        "frames": 121,
        "height": 704,
        "width": 1280,
        "fps": 24,
        "guidance_scale": 5.0,
        "wan_negative_prompt_mode": "none",
    }
    payload = {
        "_meta": {
            "schema": protocol_module.DL3DV_PROTOCOL_SCHEMA,
            "formal_protocol": True,
            "model_lock_sha256": "a" * 64,
            "split_counts": counts,
            "candidate_seed_policy": {
                "candidate_seeds": [0, 1, 2, 3],
                "incumbent_seed": 0,
            },
            "generation_profiles": {"Wan2.2-TI2V-5B": profile},
        }
    }
    paths = {}
    for split in counts:
        scene = dataset / split
        scene.mkdir(parents=True)
        image = scene / "frame.png"
        transforms = scene / "transforms.json"
        image.write_bytes(f"image-{split}".encode())
        transforms.write_bytes(f"transforms-{split}".encode())
        case_id = f"case-{split}"
        payload[case_id] = {
            "scene_uid": f"dl3dv:{split}",
            "split": split,
            "split_order": 0,
            "text_prompt": f"camera moves through {split}",
            "dataset_relative_image": f"{split}/frame.png",
            "image_sha256": file_sha256(image),
            "dataset_relative_transforms": f"{split}/transforms.json",
            "transforms_sha256": file_sha256(transforms),
        }
        paths[split] = (image, transforms)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(payload))
    return dataset, protocol_path, payload, profile, paths


def test_formal_protocol_binds_seeds_generation_sidecars_and_transforms(
    tmp_path, monkeypatch
) -> None:
    dataset, protocol_path, payload, profile, paths = _small_formal_fixture(
        tmp_path, monkeypatch
    )
    validate_formal_protocol(protocol_path)
    case_id = "case-validation"
    image, transforms = paths["validation"]
    locked_model_identity = {
        "snapshot_commit": "revision",
        "json_files": [],
        "weight_files": [{"path": "model.safetensors", "size": 1, "sha256": "b" * 64}],
        "support_files": [],
    }
    model_lock = {
        "generation_models": {"Wan2.2-TI2V-5B": locked_model_identity}
    }
    experiment_lock_sha256 = "c" * 64
    implementation_sha256 = "d" * 64
    candidate_spec_sha256 = "f" * 64
    frozen_commit = "a" * 40
    candidates = []
    for seed in range(4):
        run_config = {
            "manifest_sha256": file_sha256(protocol_path),
            "case_id": case_id,
            "image_sha256": file_sha256(image),
            "prompt": payload[case_id]["text_prompt"],
            "backbone": "wan",
            "method": "baseline",
            "code_identity": {"commit": frozen_commit, "dirty": False},
            "seed": seed,
            "model": {"requested": "model", "snapshot_commit": "revision"},
            "runner_sha256": "r" * 64,
            "model_lock_sha256": payload["_meta"]["model_lock_sha256"],
            "model_content_verified": True,
            "locked_model_identity": locked_model_identity,
            "experiment_lock_sha256": experiment_lock_sha256,
            "implementation_sha256": implementation_sha256,
        }
        run_config_sha256 = hashlib.sha256(
            json.dumps(run_config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        run_id = run_config_sha256[:12]
        run_dir = tmp_path / "wan" / "baseline" / case_id / f"seed_{seed}" / f"run_{run_id}"
        run_dir.mkdir(parents=True)
        video = run_dir / "video.mp4"
        video.write_bytes(f"video-{seed}".encode())
        metadata = run_dir / "metadata.json"
        metadata.write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "manifest_sha256": file_sha256(protocol_path),
                    "image_sha256": file_sha256(image),
                    "prompt": payload[case_id]["text_prompt"],
                    "backbone": "wan",
                    "method": "baseline",
                    "run_id": run_id,
                    "run_config": run_config,
                    "run_config_sha256": run_config_sha256,
                    "seed": seed,
                    "video": str(video),
                    "video_sha256": file_sha256(video),
                    "code_identity": {"commit": frozen_commit, "dirty": False},
                    "protocol": {
                        "mode": "frozen",
                        "split": "validation",
                        "expected_split": "validation",
                    },
                    "generation": profile,
                    "model": run_config["model"],
                    "runner_sha256": run_config["runner_sha256"],
                    "model_lock_sha256": payload["_meta"]["model_lock_sha256"],
                    "model_content_verified": True,
                    "locked_model_identity": locked_model_identity,
                    "experiment_lock_sha256": experiment_lock_sha256,
                    "implementation_sha256": implementation_sha256,
                    "candidate_spec_sha256": candidate_spec_sha256,
                }
            )
        )
        (run_dir / "COMPLETE").write_text(f"{run_id}\n")
        candidates.append(
            {
                "candidate_id": f"candidate-{seed}",
                "seed": seed,
                "video": str(video),
                "generation_metadata": str(metadata),
                "is_incumbent": seed == 0,
            }
        )
    spec = {
        "schema": CANDIDATE_SPEC_SCHEMA,
        "protocol_manifest_sha256": file_sha256(protocol_path),
        "split": "validation",
        "candidate_count": 4,
        "backbone": "Wan2.2-TI2V-5B",
        "generation": profile,
        "cases": [
            {
                "case_id": case_id,
                "scene_uid": payload[case_id]["scene_uid"],
                "conditioning_image": str(image),
                "prompt": payload[case_id]["text_prompt"],
                "candidates": candidates,
            }
        ],
    }
    validate_candidate_spec_against_protocol(
        spec,
        protocol_path,
        dataset,
        formal=True,
        expected_git_commit=frozen_commit,
        model_lock=model_lock,
        experiment_lock_sha256=experiment_lock_sha256,
        expected_implementation_sha256=implementation_sha256,
        candidate_spec_sha256=candidate_spec_sha256,
    )
    pool = materialize_candidate_pool(
        spec,
        {(case_id, f"candidate-{seed}"): str(seed) * 64 for seed in range(4)},
        artifact_mode="formal",
        producer_identity={
            "commit": frozen_commit,
            "dirty": False,
            "experiment_lock_sha256": experiment_lock_sha256,
            "implementation_sha256": implementation_sha256,
        },
        candidate_spec_sha256=candidate_spec_sha256,
    )
    validate_candidate_pool_against_protocol(
        pool,
        protocol_path,
        dataset,
        formal=True,
        model_lock=model_lock,
        experiment_lock_sha256=experiment_lock_sha256,
        expected_git_commit=frozen_commit,
        expected_implementation_sha256=implementation_sha256,
    )
    pool["artifact_mode"] = "legacy-debug"
    with pytest.raises(ValueError, match="cannot be promoted"):
        validate_candidate_pool_against_protocol(
            pool,
            protocol_path,
            dataset,
            formal=True,
            model_lock=model_lock,
            experiment_lock_sha256=experiment_lock_sha256,
            expected_git_commit=frozen_commit,
            expected_implementation_sha256=implementation_sha256,
        )
    pool["artifact_mode"] = "formal"

    sidecar = candidates[1]["generation_metadata"]
    metadata = json.loads(open(sidecar, encoding="utf-8").read())
    metadata["seed"] = 99
    open(sidecar, "w", encoding="utf-8").write(json.dumps(metadata))
    with pytest.raises(ValueError, match="generation sidecar differs"):
        validate_candidate_spec_against_protocol(
            spec,
            protocol_path,
            dataset,
            formal=True,
            expected_git_commit=frozen_commit,
            model_lock=model_lock,
            experiment_lock_sha256=experiment_lock_sha256,
            expected_implementation_sha256=implementation_sha256,
            candidate_spec_sha256=candidate_spec_sha256,
        )

    metadata["seed"] = 1
    open(sidecar, "w", encoding="utf-8").write(json.dumps(metadata))
    transforms.write_bytes(b"changed")
    with pytest.raises(ValueError, match="transforms hash mismatch"):
        validate_candidate_spec_against_protocol(
            spec,
            protocol_path,
            dataset,
            formal=True,
            expected_git_commit=frozen_commit,
            model_lock=model_lock,
            experiment_lock_sha256=experiment_lock_sha256,
            expected_implementation_sha256=implementation_sha256,
            candidate_spec_sha256=candidate_spec_sha256,
        )


def test_formal_protocol_rejects_conditioning_image_from_another_scene(
    tmp_path, monkeypatch
) -> None:
    _, protocol_path, payload, _, _ = _small_formal_fixture(tmp_path, monkeypatch)
    payload["case-validation"]["dataset_relative_image"] = "debug/frame.png"
    protocol_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="inside the scene directory"):
        validate_formal_protocol(protocol_path)


def test_resolved_protocol_paths_reject_cross_scene_symlinks(
    tmp_path, monkeypatch
) -> None:
    dataset, protocol_path, payload, _, paths = _small_formal_fixture(
        tmp_path, monkeypatch
    )
    source = paths["validation"][0]
    source.unlink()
    source.symlink_to(paths["debug"][0])
    protocol_path.write_text(json.dumps(payload))
    protocol = validate_formal_protocol(protocol_path)
    with pytest.raises(ValueError, match="resolved conditioning image escapes"):
        protocol_module.resolve_protocol_image(
            protocol["case-validation"], dataset
        )


@pytest.mark.parametrize(
    ("field", "unsafe_path", "message"),
    [
        ("dataset_relative_image", "../frame.png", "must not contain"),
        ("dataset_relative_transforms", "/scene/transforms.json", "must be relative"),
        ("dataset_relative_image", "validation/./frame.png", "normalized relative path"),
    ],
)
def test_formal_protocol_rejects_unsafe_or_unnormalized_dataset_paths(
    tmp_path, monkeypatch, field, unsafe_path, message
) -> None:
    _, protocol_path, payload, _, _ = _small_formal_fixture(tmp_path, monkeypatch)
    payload["case-validation"][field] = unsafe_path
    protocol_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        validate_formal_protocol(protocol_path)


def _commit_experiment_lock(
    tmp_path: Path, *, split: str, authorized_hashes: list[str]
) -> tuple[Path, Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    implementation = repo / "geometry_selection"
    implementation.mkdir()
    (implementation / "scorer.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"], check=True
    )
    lock_path = repo / "experiment_lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "schema": "geometry-experiment-lock-v1",
                "protocol_manifest_sha256": "a" * 64,
                "model_lock_sha256": "b" * 64,
                "implementation_sha256": implementation_tree_sha256(repo),
                "split": split,
                "backbone": "Wan2.2-TI2V-5B",
                "candidate_spec_sha256": "e" * 64,
                "artifact_root": str((tmp_path / "artifacts").resolve()),
                "authorized_ranking_config_hashes": authorized_hashes,
            }
        )
    )
    subprocess.run(["git", "-C", str(repo), "add", "experiment_lock.json"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "lock"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, lock_path, commit


def test_experiment_lock_authorizes_only_predeclared_config_hashes(tmp_path) -> None:
    repo, lock_path, commit = _commit_experiment_lock(
        tmp_path, split="validation", authorized_hashes=["c" * 64]
    )
    validate_experiment_lock(
        lock_path,
        repo,
        commit,
        protocol_manifest_sha256="a" * 64,
        model_lock_sha256="b" * 64,
        split="validation",
        backbone="Wan2.2-TI2V-5B",
        candidate_spec_sha256="e" * 64,
        artifact_root=tmp_path / "artifacts",
        ranking_config_hash="c" * 64,
    )
    with pytest.raises(ValueError, match="not authorized"):
        validate_experiment_lock(
            lock_path,
            repo,
            commit,
            protocol_manifest_sha256="a" * 64,
            model_lock_sha256="b" * 64,
            split="validation",
            backbone="Wan2.2-TI2V-5B",
            candidate_spec_sha256="e" * 64,
            artifact_root=tmp_path / "artifacts",
            ranking_config_hash="d" * 64,
        )


def test_implementation_tree_hash_can_be_recomputed_from_commit(tmp_path) -> None:
    repo = tmp_path / "repo"
    source = repo / "geometry_selection"
    source.mkdir(parents=True)
    (source / "module.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "geometry_selection/module.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "source"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert implementation_tree_sha256_at_commit(repo, commit) == implementation_tree_sha256(repo)
    (source / "module.py").write_text("VALUE = 2\n")
    assert implementation_tree_sha256_at_commit(repo, commit) != implementation_tree_sha256(repo)


def test_preparation_commit_must_exist_be_ancestor_and_match_implementation(
    tmp_path,
) -> None:
    repo = tmp_path / "repo"
    source = repo / "geometry_selection"
    source.mkdir(parents=True)
    (source / "module.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "geometry_selection/module.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "implementation"], check=True)
    preparation = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    implementation_hash = implementation_tree_sha256(repo)
    (repo / "artifact.json").write_text("{}\n")
    subprocess.run(["git", "-C", str(repo), "add", "artifact.json"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "artifact lock"], check=True)
    ranking = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    validate_implementation_commit(repo, preparation, ranking, implementation_hash)
    with pytest.raises(ValueError, match="does not exist"):
        validate_implementation_commit(repo, "f" * 40, ranking, implementation_hash)
    with pytest.raises(ValueError, match="differs from experiment lock"):
        validate_implementation_commit(repo, preparation, ranking, "0" * 64)


def test_test_experiment_lock_rejects_multiple_authorized_configs(tmp_path) -> None:
    repo, lock_path, commit = _commit_experiment_lock(
        tmp_path, split="test", authorized_hashes=["c" * 64, "d" * 64]
    )

    with pytest.raises(ValueError, match="exactly one ranking config"):
        validate_experiment_lock(
            lock_path,
            repo,
            commit,
            protocol_manifest_sha256="a" * 64,
            model_lock_sha256="b" * 64,
            split="test",
            backbone="Wan2.2-TI2V-5B",
            candidate_spec_sha256="e" * 64,
            artifact_root=tmp_path / "artifacts",
        )


@pytest.mark.parametrize(
    ("authorized_hashes", "message"),
    [
        (["C" * 64], "lowercase SHA-256"),
        (["c" * 64, "c" * 64], "must be unique"),
    ],
)
def test_experiment_lock_rejects_invalid_or_duplicate_config_hashes(
    tmp_path, authorized_hashes, message
) -> None:
    repo, lock_path, commit = _commit_experiment_lock(
        tmp_path, split="validation", authorized_hashes=authorized_hashes
    )

    with pytest.raises(ValueError, match=message):
        validate_experiment_lock(
            lock_path,
            repo,
            commit,
            protocol_manifest_sha256="a" * 64,
            model_lock_sha256="b" * 64,
            split="validation",
            backbone="Wan2.2-TI2V-5B",
            candidate_spec_sha256="e" * 64,
            artifact_root=tmp_path / "artifacts",
        )


def test_legacy_debug_builder_marks_artifacts_non_formal(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    videos = tmp_path / "videos"
    source = {}
    case_ids = []
    for index in range(3):
        case_id = f"case-{index}"
        case_ids.append(case_id)
        scene = dataset / f"scene-{index}"
        images = scene / "images_4"
        images.mkdir(parents=True)
        image = images / "frame.png"
        image.write_bytes(f"image-{index}".encode())
        (scene / "transforms.json").write_bytes(f"poses-{index}".encode())
        source[case_id] = {
            "image_prompt": str(image),
            "text_prompt": f"prompt-{index}",
        }
        case_video_root = videos / case_id
        case_video_root.mkdir(parents=True)
        for seed in (0, 1):
            (case_video_root / f"baseline_seed{seed}_steps50_frames121.mp4").write_bytes(
                f"video-{index}-{seed}".encode()
            )
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source))
    protocol_path = tmp_path / "protocol.json"
    spec_path = tmp_path / "spec.json"
    command = [
        sys.executable,
        str(DEBUG_BUILDER),
        "--source-manifest",
        str(source_path),
        "--dataset-root",
        str(dataset),
        "--video-root",
        str(videos),
        "--case-ids",
        *case_ids,
        "--seeds",
        "0",
        "1",
        "--protocol-output",
        str(protocol_path),
        "--spec-output",
        str(spec_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    protocol = json.loads(protocol_path.read_text())
    spec = json.loads(spec_path.read_text())
    assert protocol["_meta"]["formal_protocol"] is False
    assert spec["artifact_mode"] == "legacy-debug"
    assert spec["candidate_count"] == 2
    assert len(spec["cases"]) == 3
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "refusing to overwrite debug artifact" in completed.stderr
