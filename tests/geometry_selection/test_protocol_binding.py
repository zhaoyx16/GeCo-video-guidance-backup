from __future__ import annotations

import json
import subprocess

import pytest

import geometry_selection.protocol as protocol_module
from geometry_selection.protocol import (
    file_sha256,
    validate_candidate_spec_against_protocol,
    validate_committed_test_release,
    validate_formal_protocol,
)
from geometry_selection.selection import CANDIDATE_SPEC_SCHEMA


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
    candidates = []
    for seed in range(4):
        video = tmp_path / f"video-{seed}.mp4"
        video.write_bytes(f"video-{seed}".encode())
        metadata = tmp_path / f"metadata-{seed}.json"
        metadata.write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "manifest_sha256": file_sha256(protocol_path),
                    "image_sha256": file_sha256(image),
                    "prompt": payload[case_id]["text_prompt"],
                    "backbone": "wan",
                    "method": "baseline",
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
                }
            )
        )
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
    )

    sidecar = candidates[1]["generation_metadata"]
    metadata = json.loads(open(sidecar, encoding="utf-8").read())
    metadata["seed"] = 99
    open(sidecar, "w", encoding="utf-8").write(json.dumps(metadata))
    with pytest.raises(ValueError, match="generation sidecar differs"):
        validate_candidate_spec_against_protocol(
            spec, protocol_path, dataset, formal=True, expected_git_commit="abc"
        )

    metadata["seed"] = 1
    open(sidecar, "w", encoding="utf-8").write(json.dumps(metadata))
    transforms.write_bytes(b"changed")
    with pytest.raises(ValueError, match="transforms hash mismatch"):
        validate_candidate_spec_against_protocol(
            spec, protocol_path, dataset, formal=True, expected_git_commit="abc"
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
