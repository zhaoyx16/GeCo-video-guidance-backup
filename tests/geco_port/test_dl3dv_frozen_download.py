from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "dl3dv_geco"
    / "download_frozen_scenes.py"
)
SPEC = importlib.util.spec_from_file_location("download_frozen_scenes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _make_scene(root: Path, scene_hash: str, count: int = 8) -> Path:
    scene = root / scene_hash
    images = scene / "images_8"
    images.mkdir(parents=True)
    frames = []
    for index in range(count):
        name = f"frame_{index + 1:05d}.png"
        Image.new("RGB", (4, 3), color=(index, 2, 3)).save(images / name)
        frames.append(
            {
                "file_path": f"images/{name}",
                "transform_matrix": np.eye(4).tolist(),
            }
        )
    (scene / "transforms.json").write_text(json.dumps({"frames": frames}))
    return scene


def _write_marker(scene: Path, revision: str = "1" * 40) -> None:
    marker = {
        "schema": "dl3dv-scene-complete-v1",
        "repo_id": MODULE.REPO_ID,
        "dataset_revision": revision,
        "scene_hash": scene.name,
        "archive_filename": f"1K/{scene.name}.zip",
        "archive_sha256": "2" * 64,
        "archive_bytes": 123,
        **MODULE.validate_scene_files(scene),
    }
    (scene / MODULE.COMPLETION_MARKER).write_text(json.dumps(marker))


def _split_csv(path: Path, rows: list[str]) -> None:
    path.write_text(
        "split,split_order,source_order,hash,batch,duration\n" + "\n".join(rows) + "\n"
    )


def test_load_frozen_split_filters_and_orders_debug_first(tmp_path: Path) -> None:
    path = tmp_path / "split.csv"
    _split_csv(
        path,
        [
            f"validation,0,101,{'a' * 64},1K,60.0",
            f"debug,0,201,{'b' * 64},1K,61.0",
            f"test,0,1,{'c' * 64},1K,62.0",
        ],
    )
    records = MODULE.load_frozen_split(
        path,
        {"debug", "validation"},
        {"debug": 1, "validation": 1, "test": 1},
    )
    assert [record.split for record in records] == ["debug", "validation"]
    assert [record.scene_hash for record in records] == ["b" * 64, "a" * 64]


def test_cross_split_duplicate_is_rejected_before_filtering(tmp_path: Path) -> None:
    path = tmp_path / "split.csv"
    duplicate = "a" * 64
    _split_csv(
        path,
        [
            f"validation,0,101,{duplicate},1K,60.0",
            f"debug,0,201,{'b' * 64},1K,61.0",
            f"test,0,1,{duplicate},1K,62.0",
        ],
    )
    with pytest.raises(ValueError, match="scene hash"):
        MODULE.load_frozen_split(path, {"debug"})


def test_expected_counts_reject_truncated_split(tmp_path: Path) -> None:
    path = tmp_path / "split.csv"
    _split_csv(path, [f"debug,0,1,{'a' * 64},1K,60.0"])
    with pytest.raises(ValueError, match="counts"):
        MODULE.load_frozen_split(
            path,
            {"debug"},
            {"debug": 1, "validation": 100, "test": 100},
        )


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="unsafe zip member"):
        MODULE.safe_extract_zip(archive, tmp_path / "output", 1024)
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "output").exists()


def test_safe_extract_and_provenance_validation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    scene_hash = "d" * 64
    scene = _make_scene(source, scene_hash)
    archive = tmp_path / "scene.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in scene.rglob("*"):
            if path.is_file():
                bundle.write(path, path.relative_to(source))
    destination = tmp_path / "output"
    MODULE.safe_extract_zip(archive, destination, 1024 * 1024)
    extracted = MODULE.locate_extracted_scene(destination, scene_hash)
    assert extracted == destination / scene_hash
    assert not MODULE.scene_is_complete(extracted, "1" * 40, MODULE.REPO_ID, scene_hash)
    _write_marker(extracted)
    assert MODULE.scene_is_complete(extracted, "1" * 40, MODULE.REPO_ID, scene_hash)
    assert not MODULE.scene_is_complete(extracted, "3" * 40, MODULE.REPO_ID, scene_hash)


def test_corrupt_middle_image_invalidates_completion(tmp_path: Path) -> None:
    scene = _make_scene(tmp_path, "e" * 64)
    _write_marker(scene)
    assert MODULE.scene_is_complete(scene, "1" * 40, MODULE.REPO_ID, scene.name)
    (scene / "images_8" / "frame_00004.png").write_bytes(b"not a png")
    assert not MODULE.scene_is_complete(scene, "1" * 40, MODULE.REPO_ID, scene.name)


def test_transform_reference_must_exist(tmp_path: Path) -> None:
    scene = _make_scene(tmp_path, "f" * 64)
    (scene / "images_8" / "frame_00004.png").unlink()
    with pytest.raises(ValueError, match="missing"):
        MODULE.validate_scene_files(scene)


def test_unreferenced_extra_images_are_allowed(tmp_path: Path) -> None:
    scene = _make_scene(tmp_path, "0" * 64)
    Image.new("RGB", (4, 3), color=(9, 9, 9)).save(
        scene / "images_8" / "frame_99999.png"
    )

    validation = MODULE.validate_scene_files(scene)
    assert validation["frame_count"] == 8


def test_scene_directory_cannot_be_relabelled_as_another_scene(tmp_path: Path) -> None:
    source = _make_scene(tmp_path, "1" * 64)
    _write_marker(source)
    relabelled = tmp_path / ("2" * 64)
    shutil.copytree(source, relabelled)
    assert not MODULE.scene_is_complete(
        relabelled, "1" * 40, MODULE.REPO_ID, relabelled.name
    )


def test_resume_revalidates_cached_archive_digest(tmp_path: Path, monkeypatch) -> None:
    scene_hash = "3" * 64
    output_root = tmp_path / "output"
    target = _make_scene(output_root / "1K", scene_hash)
    _write_marker(target)
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"different archive bytes")
    monkeypatch.setattr(MODULE, "hf_hub_download", lambda **_: str(archive))
    record = MODULE.SceneRecord("debug", 0, 1, scene_hash, "1K", 60.0)
    with pytest.raises(RuntimeError, match="archive provenance mismatch"):
        MODULE.download_one(
            record,
            output_root=output_root,
            cache_root=tmp_path / "cache",
            repo_id=MODULE.REPO_ID,
            revision="1" * 40,
            max_uncompressed_bytes=1024,
            max_archive_members=100,
        )


def test_safe_extract_rejects_member_count_bomb(tmp_path: Path) -> None:
    archive = tmp_path / "many.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for index in range(4):
            bundle.writestr(f"empty_{index}", b"")
    with pytest.raises(ValueError, match="members"):
        MODULE.safe_extract_zip(archive, tmp_path / "output", 1024, max_members=3)
