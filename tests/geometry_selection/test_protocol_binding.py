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
    validate_candidate_spec_against_protocol,
    validate_candidate_pool_against_protocol,
    validate_committed_test_release,
    validate_formal_protocol,
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
    }
    model_lock = {
        "generation_models": {"Wan2.2-TI2V-5B": locked_model_identity}
    }
    candidates = []
    for seed in range(4):
        run_config = {
            "manifest_sha256": file_sha256(protocol_path),
            "case_id": case_id,
            "image_sha256": file_sha256(image),
            "prompt": payload[case_id]["text_prompt"],
            "backbone": "wan",
            "method": "baseline",
            "code_identity": {"commit": "abc", "dirty": False},
            "seed": seed,
            "model": {"requested": "model", "snapshot_commit": "revision"},
            "runner_sha256": "r" * 64,
            "model_lock_sha256": payload["_meta"]["model_lock_sha256"],
            "locked_model_identity": locked_model_identity,
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
                    "code_identity": {"commit": "abc", "dirty": False},
                    "protocol": {
                        "mode": "frozen",
                        "split": "validation",
                        "expected_split": "validation",
                    },
                    "generation": profile,
                    "model": run_config["model"],
                    "runner_sha256": run_config["runner_sha256"],
                    "model_lock_sha256": payload["_meta"]["model_lock_sha256"],
                    "locked_model_identity": locked_model_identity,
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
        expected_git_commit="abc",
        model_lock=model_lock,
    )
    pool = materialize_candidate_pool(
        spec,
        {(case_id, f"candidate-{seed}"): str(seed) * 64 for seed in range(4)},
        artifact_mode="formal",
        producer_identity={"commit": "abc", "dirty": False},
        candidate_spec_sha256="f" * 64,
    )
    validate_candidate_pool_against_protocol(
        pool, protocol_path, dataset, formal=True, model_lock=model_lock
    )
    pool["artifact_mode"] = "legacy-debug"
    with pytest.raises(ValueError, match="cannot be promoted"):
        validate_candidate_pool_against_protocol(
            pool, protocol_path, dataset, formal=True, model_lock=model_lock
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
            expected_git_commit="abc",
            model_lock=model_lock,
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
            expected_git_commit="abc",
            model_lock=model_lock,
        )


def test_test_release_must_be_committed_and_match_expected_fields(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    release = repo / "release.json"
    release.write_text(json.dumps({"schema": "geometry-test-release-v1", "phase": "ranking"}))
    subprocess.run(["git", "-C", str(repo), "add", "release.json"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "release"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    validate_committed_test_release(
        release,
        repo,
        commit,
        {"schema": "geometry-test-release-v1", "phase": "ranking"},
    )
    release.write_text(json.dumps({"schema": "geometry-test-release-v1", "phase": "changed"}))
    with pytest.raises(ValueError, match="differs from the file committed"):
        validate_committed_test_release(release, repo, commit, {"phase": "ranking"})


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
