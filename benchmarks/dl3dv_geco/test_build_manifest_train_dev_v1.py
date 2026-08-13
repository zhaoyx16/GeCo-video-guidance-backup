from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("build_manifest.py")
SPEC = importlib.util.spec_from_file_location("train_dev_build_manifest", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_split(path: Path, split: str, count: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("split", "split_order", "source_order", "hash", "batch", "duration"),
        )
        writer.writeheader()
        for index in range(count):
            writer.writerow(
                {
                    "split": split,
                    "split_order": index,
                    "source_order": index + 1,
                    "hash": f"{index + 1:064x}",
                    "batch": "1K",
                    "duration": "1.0",
                }
            )


def test_load_frozen_train_dev_assignments(tmp_path: Path) -> None:
    split = tmp_path / "dev.csv"
    _write_split(split, "dev", 100)
    digest = hashlib.sha256(split.read_bytes()).hexdigest()
    assignments = MODULE.load_frozen_assignments(
        split, {"dev"}, MODULE.FORMAL_SPLIT_COUNTS, digest
    )
    assert len(assignments) == 100
    assert [item.split_order for item in assignments] == list(range(100))
    assert {item.split for item in assignments} == {"dev"}


def test_train_dev_count_is_strict(tmp_path: Path) -> None:
    split = tmp_path / "dev.csv"
    _write_split(split, "dev", 99)
    digest = hashlib.sha256(split.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        MODULE.load_frozen_assignments(
            split, {"dev"}, MODULE.FORMAL_SPLIT_COUNTS, digest
        )


def test_dev_loader_requires_sha_and_rejects_mixed_split_before_read(
    tmp_path: Path,
) -> None:
    nonexistent = tmp_path / "must-not-be-read.csv"
    with pytest.raises(ValueError, match="cannot be combined"):
        MODULE.load_frozen_assignments(
            nonexistent, {"dev", "test"}, MODULE.FORMAL_SPLIT_COUNTS, None
        )
    split = tmp_path / "dev.csv"
    _write_split(split, "dev", 100)
    with pytest.raises(ValueError, match="requires an expected"):
        MODULE.load_frozen_assignments(split, {"dev"}, MODULE.FORMAL_SPLIT_COUNTS)


def test_cli_rejects_mixed_dev_before_root_or_csv_access(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--roots",
            str(tmp_path / "missing-root"),
            "--output",
            str(tmp_path / "must-not-exist.json"),
            "--splits",
            "dev",
            "test",
            "--frozen-split-csv",
            str(tmp_path / "missing.csv"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "cannot be combined" in result.stderr
    assert not (tmp_path / "must-not-exist.json").exists()


def test_cli_missing_scene_error_is_identifier_free(tmp_path: Path) -> None:
    split = tmp_path / "dev.csv"
    _write_split(split, "dev", 100)
    digest = hashlib.sha256(split.read_bytes()).hexdigest()
    empty_root = tmp_path / "empty-root"
    empty_root.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--roots",
            str(empty_root),
            "--output",
            str(tmp_path / "must-not-exist.json"),
            "--splits",
            "dev",
            "--frozen-split-csv",
            str(split),
            "--expected-frozen-split-sha256",
            digest,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "missing downloaded data for 100 frozen scene" in result.stderr
    assert f"{1:064x}" not in result.stderr
    assert not (tmp_path / "must-not-exist.json").exists()


def test_cli_dev_forbids_descriptions_before_any_file_read(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--roots",
            str(tmp_path / "missing-root"),
            "--output",
            str(tmp_path / "must-not-exist.json"),
            "--splits",
            "dev",
            "--frozen-split-csv",
            str(tmp_path / "missing.csv"),
            "--expected-frozen-split-sha256",
            "0" * 64,
            "--scene-descriptions-json",
            str(tmp_path / "private-mixed-descriptions.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "forbids --scene-descriptions-json" in result.stderr
    assert not (tmp_path / "must-not-exist.json").exists()


def test_cli_malformed_dev_transforms_error_is_identifier_free(tmp_path: Path) -> None:
    split = tmp_path / "dev.csv"
    _write_split(split, "dev", 100)
    digest = hashlib.sha256(split.read_bytes()).hexdigest()
    root = tmp_path / "data"
    root.mkdir()
    for index in range(100):
        scene = root / f"{index + 1:064x}"
        scene.mkdir()
        (scene / "transforms.json").write_text("not-json", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--roots",
            str(root),
            "--output",
            str(tmp_path / "must-not-exist.json"),
            "--splits",
            "dev",
            "--frozen-split-csv",
            str(split),
            "--expected-frozen-split-sha256",
            digest,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "frozen source transforms validation failed" in result.stderr
    assert f"{1:064x}" not in result.stderr
    assert not (tmp_path / "must-not-exist.json").exists()


def test_dev_loader_rejects_mixed_split_before_identifier_validation(tmp_path: Path) -> None:
    split = tmp_path / "dev.csv"
    _write_split(split, "dev", 100)
    rows = split.read_text(encoding="utf-8").splitlines()
    parts = rows[-1].split(",")
    parts[0] = "test"
    parts[3] = "secret-not-a-digest"
    rows[-1] = ",".join(parts)
    split.write_text("\n".join(rows) + "\n", encoding="utf-8")
    digest = hashlib.sha256(split.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="non-dev") as error:
        MODULE.load_frozen_assignments(split, {"dev"}, MODULE.FORMAL_SPLIT_COUNTS, digest)
    assert "secret" not in str(error.value)


def test_dev_loader_rejects_overwide_rows_and_wrong_sha(tmp_path: Path) -> None:
    split = tmp_path / "dev.csv"
    _write_split(split, "dev", 100)
    digest = hashlib.sha256(split.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="SHA mismatch"):
        MODULE.load_frozen_assignments(split, {"dev"}, MODULE.FORMAL_SPLIT_COUNTS, "0" * 64)
    lines = split.read_text(encoding="utf-8").splitlines()
    lines[-1] += ",extra"
    split.write_text("\n".join(lines) + "\n", encoding="utf-8")
    digest = hashlib.sha256(split.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="malformed"):
        MODULE.load_frozen_assignments(split, {"dev"}, MODULE.FORMAL_SPLIT_COUNTS, digest)


def test_formal_scene_failures_are_identifier_free(tmp_path: Path) -> None:
    secret = "a" * 64
    assignment = MODULE.SplitAssignment("dev", 0, 1, secret)
    with pytest.raises(RuntimeError, match="1 frozen scene") as error:
        MODULE.select_best_per_scene([], [assignment])
    assert secret not in str(error.value)

    candidate = type(
        "CandidateStub",
        (),
        {
            "scene_id": secret,
            "transforms_path": tmp_path / secret / "transforms.json",
        },
    )()
    with pytest.raises(ValueError, match="missing frozen scene marker") as error:
        MODULE.load_source_scene_provenance(candidate, required=True)
    assert secret not in str(error.value)
    assert str(tmp_path) not in str(error.value)


def test_frozen_resolution_never_enumerates_non_allowlisted_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "mixed-private-root"
    root.mkdir()
    assignments = []
    for index in range(100):
        scene_id = f"{index + 1:064x}"
        scene = root / scene_id
        scene.mkdir()
        (scene / "transforms.json").write_text("{}", encoding="utf-8")
        assignments.append(MODULE.SplitAssignment("dev", index, index + 1, scene_id))

    reserved = root / ("f" * 64)
    reserved.mkdir()
    (reserved / "transforms.json").write_text("reserved", encoding="utf-8")

    def forbidden_enumeration(*args, **kwargs):
        raise AssertionError("formal resolution enumerated the mixed root")

    monkeypatch.setattr(Path, "iterdir", forbidden_enumeration)
    monkeypatch.setattr(Path, "rglob", forbidden_enumeration)
    resolved = MODULE.resolve_frozen_transforms([root], assignments)
    assert len(resolved) == 100
    assert reserved / "transforms.json" not in resolved


@pytest.mark.parametrize(
    "unsafe_path",
    ("/private/reserved/frame.png", "../reserved/frame.png", "images/../../frame.png"),
)
def test_resolve_image_rejects_absolute_and_traversal_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    scene = tmp_path / "allowlisted-scene"
    (scene / "images_8").mkdir(parents=True)
    (scene / "images_8" / "frame.png").write_bytes(b"image")
    with pytest.raises(ValueError, match="absolute or traverse"):
        MODULE.resolve_image(scene, {"file_path": unsafe_path}, "images_8")


def test_resolve_image_rejects_file_and_directory_symlinks(tmp_path: Path) -> None:
    scene = tmp_path / "allowlisted-scene"
    outside = tmp_path / "reserved"
    scene.mkdir()
    outside.mkdir()
    (outside / "frame.png").write_bytes(b"reserved")

    (scene / "images_8").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        MODULE.resolve_image(scene, {"file_path": "images/frame.png"}, "images_8")

    (scene / "images_8").unlink()
    (scene / "images_8").mkdir()
    (scene / "images_8" / "frame.png").symlink_to(outside / "frame.png")
    with pytest.raises(ValueError, match="symlink"):
        MODULE.resolve_image(scene, {"file_path": "images/frame.png"}, "images_8")


def test_resolve_image_rejects_symlinked_scene_root(tmp_path: Path) -> None:
    outside = tmp_path / "reserved-scene"
    (outside / "images_8").mkdir(parents=True)
    (outside / "images_8" / "frame.png").write_bytes(b"reserved")
    scene = tmp_path / "allowlisted-scene"
    scene.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        MODULE.resolve_image(scene, {"file_path": "images/frame.png"}, "images_8")


def test_frozen_resolution_rejects_scene_and_transforms_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    scene_id = "1" * 64
    assignment = MODULE.SplitAssignment("dev", 0, 1, scene_id)
    outside = tmp_path / "reserved"
    outside.mkdir()
    (outside / "transforms.json").write_text("{}", encoding="utf-8")

    (root / scene_id).symlink_to(outside, target_is_directory=True)
    with pytest.raises(MODULE.FrozenSourceError, match="symlink"):
        MODULE.resolve_frozen_transforms([root], [assignment])

    (root / scene_id).unlink()
    (root / scene_id).mkdir()
    (root / scene_id / "transforms.json").symlink_to(outside / "transforms.json")
    with pytest.raises(MODULE.FrozenSourceError, match="symlink"):
        MODULE.resolve_frozen_transforms([root], [assignment])


def test_provenance_rejects_symlinked_marker_before_json_read(tmp_path: Path) -> None:
    scene = tmp_path / "allowlisted-scene"
    scene.mkdir()
    outside = tmp_path / "reserved-marker.json"
    outside.write_text('{"schema":"dl3dv-scene-complete-v1"}', encoding="utf-8")
    (scene / ".dl3dv_complete.json").symlink_to(outside)
    candidate = type(
        "CandidateStub",
        (),
        {
            "scene_id": "1" * 64,
            "transforms_path": scene / "transforms.json",
            "image_path": scene / "images_8" / "frame.png",
        },
    )()
    with pytest.raises(ValueError, match="frozen source scene validation failed"):
        MODULE.load_source_scene_provenance(candidate, required=True)


def test_scene_candidates_rejects_symlinked_transforms(tmp_path: Path) -> None:
    scene = tmp_path / "allowlisted-scene"
    scene.mkdir()
    outside = tmp_path / "reserved-transforms.json"
    outside.write_text('{"frames":[]}', encoding="utf-8")
    transforms = scene / "transforms.json"
    transforms.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        MODULE.scene_candidates(transforms, 8, 1, "images_8")


@pytest.mark.parametrize("unsafe_path", ("../reserved/frame.png", "/reserved/frame.png"))
def test_provenance_validator_rejects_traversal_before_basename(
    tmp_path: Path, unsafe_path: str
) -> None:
    frames = [
        {
            "file_path": unsafe_path if index == 0 else f"images/frame_{index:05d}.png",
            "transform_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
        for index in range(8)
    ]
    payload = json.dumps({"frames": frames}).encode("utf-8")
    with pytest.raises(ValueError, match="relative to one allowlisted scene"):
        MODULE.validate_scene_files_no_follow(tmp_path, payload)
