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


def run_split(source: Path, dataset: Path, output: Path) -> dict:
    subprocess.run(
        [
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
        ],
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
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "same scene" in completed.stderr
