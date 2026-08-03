from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/dl3dv_geco/freeze_protocol_split.py"


def build_source(tmp_path: Path, count: int = 8) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset"
    cases = {"_meta": {"source": "test"}}
    for index in range(count):
        scene = dataset / f"scene_{index:02d}"
        scene.mkdir(parents=True)
        image = scene / "frame.png"
        transforms = scene / "transforms.json"
        image.write_bytes(f"image-{index}".encode())
        transforms.write_text(json.dumps({"scene": index}))
        cases[f"case_{index:02d}"] = {
            "text_prompt": "camera moves forward",
            "image_prompt": str(image),
            "scene_id": scene.name,
            "motion_instruction": "forward" if index % 2 == 0 else "orbit_left",
            "pose_window": [0, 120],
            "pose_stats": {"selection_score": index / count},
            "transforms_path": str(transforms),
        }
    source = tmp_path / "source.json"
    source.write_text(json.dumps(cases))
    return source, dataset


def run_split(source: Path, dataset: Path, output: Path, preserve: bool = False) -> dict:
    command = [
            sys.executable,
            str(SCRIPT),
            "--source-manifest",
            str(source),
            "--dataset-root",
            str(dataset),
            "--output",
            str(output),
            "--split-seed",
            "7",
            "--test-count",
            "2",
            "--validation-count",
            "2",
            "--debug-count",
            "2",
            "--allow-nonstandard-counts",
        ]
    if preserve:
        command.append("--preserve-source-splits")
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output.read_text())


def test_split_is_deterministic_and_scene_disjoint(tmp_path: Path) -> None:
    source, dataset = build_source(tmp_path)
    first = run_split(source, dataset, tmp_path / "first.json")
    second = run_split(source, dataset, tmp_path / "second.json")
    assert first == second
    cases = [value for key, value in first.items() if not key.startswith("_")]
    assert {case["split"] for case in cases} == {"debug", "validation", "test"}
    assert len({case["scene_uid"] for case in cases}) == len(cases)
    assert len({case["image_sha256"] for case in cases}) == len(cases)
    assert first["_meta"]["split_counts"] == {"debug": 2, "test": 2, "validation": 2}
    assert first["_meta"]["candidate_seed_policy"]["candidate_seeds"] == [0, 1, 2, 3]
    assert first["_meta"]["formal_protocol"] is False
    assert all(not Path(case["image_prompt"]).is_absolute() for case in cases)


def test_split_fails_when_same_scene_has_multiple_cases(tmp_path: Path) -> None:
    source, dataset = build_source(tmp_path)
    payload = json.loads(source.read_text())
    duplicate = dict(payload["case_00"])
    duplicate["image_prompt"] = payload["case_01"]["image_prompt"]
    payload["duplicate_scene"] = duplicate
    source.write_text(json.dumps(payload))
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-manifest",
            str(source),
            "--dataset-root",
            str(dataset),
            "--output",
            str(tmp_path / "out.json"),
            "--test-count",
            "2",
            "--validation-count",
            "2",
            "--debug-count",
            "2",
            "--allow-nonstandard-counts",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "same scene" in completed.stderr


def test_nonstandard_counts_require_explicit_debug_flag(tmp_path: Path) -> None:
    source, dataset = build_source(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-manifest",
            str(source),
            "--dataset-root",
            str(dataset),
            "--output",
            str(tmp_path / "out.json"),
            "--test-count",
            "2",
            "--validation-count",
            "2",
            "--debug-count",
            "2",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "formal protocol requires split counts" in completed.stderr


def test_frozen_protocol_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    source, dataset = build_source(tmp_path)
    output = tmp_path / "protocol.json"
    run_split(source, dataset, output)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-manifest",
            str(source),
            "--dataset-root",
            str(dataset),
            "--output",
            str(output),
            "--test-count",
            "2",
            "--validation-count",
            "2",
            "--debug-count",
            "2",
            "--allow-nonstandard-counts",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "refusing to overwrite frozen protocol" in completed.stderr


def test_split_resolves_source_relative_paths_from_manifest_directory(tmp_path: Path) -> None:
    source, dataset = build_source(tmp_path)
    payload = json.loads(source.read_text())
    for key, case in payload.items():
        if key.startswith("_"):
            continue
        case["image_prompt"] = str(Path(case["image_prompt"]).relative_to(tmp_path))
        case["transforms_path"] = str(Path(case["transforms_path"]).relative_to(tmp_path))
    source.write_text(json.dumps(payload))
    result = run_split(source, dataset, tmp_path / "relative.json")
    assert result["_meta"]["split_counts"] == {"debug": 2, "test": 2, "validation": 2}


def test_scene_uid_is_stable_when_transforms_serialization_changes(tmp_path: Path) -> None:
    source, dataset = build_source(tmp_path)
    first = run_split(source, dataset, tmp_path / "first.json")
    transforms = dataset / "scene_00" / "transforms.json"
    transforms.write_text('{\n  "scene": 0\n}\n')
    second = run_split(source, dataset, tmp_path / "second.json")
    first_uid = {
        case["dataset_relative_transforms"]: case["scene_uid"]
        for key, case in first.items()
        if not key.startswith("_")
    }
    second_uid = {
        case["dataset_relative_transforms"]: case["scene_uid"]
        for key, case in second.items()
        if not key.startswith("_")
    }
    assert first_uid == second_uid


def test_preserve_source_splits_keeps_exact_assignments(tmp_path: Path) -> None:
    source, dataset = build_source(tmp_path, count=6)
    payload = json.loads(source.read_text())
    assignments = [
        ("validation", 0),
        ("validation", 1),
        ("debug", 0),
        ("debug", 1),
        ("test", 0),
        ("test", 1),
    ]
    for key, assignment in zip(
        [key for key in payload if not key.startswith("_")], assignments, strict=True
    ):
        payload[key]["split"], payload[key]["split_order"] = assignment
    source.write_text(json.dumps(payload))
    result = run_split(source, dataset, tmp_path / "preserved.json", preserve=True)
    actual = {
        key: (case["split"], case["split_order"])
        for key, case in result.items()
        if not key.startswith("_")
    }
    assert actual == {
        f"case_{index:02d}": assignment
        for index, assignment in enumerate(assignments)
    }
    assert result["_meta"]["split_algorithm"] == "preserved from frozen source assignments"


def test_preserve_source_splits_rejects_duplicate_order(tmp_path: Path) -> None:
    source, dataset = build_source(tmp_path, count=6)
    payload = json.loads(source.read_text())
    assignments = [
        ("validation", 0),
        ("validation", 0),
        ("debug", 0),
        ("debug", 1),
        ("test", 0),
        ("test", 1),
    ]
    for key, assignment in zip(
        [key for key in payload if not key.startswith("_")], assignments, strict=True
    ):
        payload[key]["split"], payload[key]["split_order"] = assignment
    source.write_text(json.dumps(payload))
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-manifest",
            str(source),
            "--dataset-root",
            str(dataset),
            "--output",
            str(tmp_path / "invalid.json"),
            "--test-count",
            "2",
            "--validation-count",
            "2",
            "--debug-count",
            "2",
            "--allow-nonstandard-counts",
            "--preserve-source-splits",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "split_order" in completed.stderr


def test_source_assignments_cannot_be_silently_reallocated(tmp_path: Path) -> None:
    source, dataset = build_source(tmp_path, count=6)
    payload = json.loads(source.read_text())
    for index, case in enumerate(
        [case for key, case in payload.items() if not key.startswith("_")]
    ):
        case["split"] = ("test", "validation", "debug")[index % 3]
        case["split_order"] = index // 3
    source.write_text(json.dumps(payload))
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-manifest",
            str(source),
            "--dataset-root",
            str(dataset),
            "--output",
            str(tmp_path / "reallocated.json"),
            "--test-count",
            "2",
            "--validation-count",
            "2",
            "--debug-count",
            "2",
            "--allow-nonstandard-counts",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "--preserve-source-splits is required" in completed.stderr
