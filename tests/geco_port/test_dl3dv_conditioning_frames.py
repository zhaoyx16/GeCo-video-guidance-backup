from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "dl3dv_geco"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "extract_conditioning_frames.py"
SPEC = importlib.util.spec_from_file_location("extract_conditioning_frames", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_find_exact_960p_frame_member(tmp_path: Path) -> None:
    scene_id = "a" * 64
    archive = tmp_path / "scene.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(f"{scene_id}/images_4/frame_00017.png", b"image")
        bundle.writestr(f"{scene_id}/images_8/frame_00017.png", b"wrong resolution")
    with zipfile.ZipFile(archive) as bundle:
        info = MODULE.find_frame_member(bundle, scene_id, "frame_00017.png", "images_4")
    assert info.filename == f"{scene_id}/images_4/frame_00017.png"


def test_find_frame_rejects_unsafe_archive_member(tmp_path: Path) -> None:
    archive = tmp_path / "scene.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.png", b"bad")
    with zipfile.ZipFile(archive) as bundle:
        with pytest.raises(ValueError, match="unsafe"):
            MODULE.find_frame_member(bundle, "a" * 64, "frame.png", "images_4")


def test_installed_frame_requires_matching_revision_and_content(tmp_path: Path) -> None:
    target = tmp_path / "scene"
    target.mkdir()
    image_path = target / "frame_00001.png"
    Image.new("RGB", (960, 540), color=(1, 2, 3)).save(image_path)
    transforms = target / "transforms.json"
    transforms.write_text("{}")
    marker = {
        "schema": "dl3dv-conditioning-v1",
        "repo_id": MODULE.DEFAULT_REPO_ID,
        "dataset_revision": "1" * 40,
        "scene_id": target.name,
        "archive_filename": f"1K/{target.name}.zip",
        "frame_name": image_path.name,
        "width": 960,
        "height": 540,
        "image_sha256": MODULE.sha256_file(image_path),
        "transforms_sha256": MODULE.sha256_file(transforms),
    }
    (target / MODULE.MARKER_NAME).write_text(json.dumps(marker))
    loaded = MODULE.validate_installed_frame(
        target,
        frame_name=image_path.name,
        revision="1" * 40,
        repo_id=MODULE.DEFAULT_REPO_ID,
        scene_id=target.name,
        min_long_side=900,
    )
    assert loaded["image_sha256"] == marker["image_sha256"]
    with pytest.raises(ValueError, match="does not match"):
        MODULE.validate_installed_frame(
            target,
            frame_name=image_path.name,
            revision="2" * 40,
            repo_id=MODULE.DEFAULT_REPO_ID,
            scene_id=target.name,
            min_long_side=900,
        )


def test_manifest_rejects_multiple_cases_from_one_scene(tmp_path: Path) -> None:
    scene_id = "b" * 64
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "_meta": {},
                "case_a": {
                    "scene_id": scene_id,
                    "image_prompt": "/x/a.png",
                    "transforms_path": "/x/transforms.json",
                    "source_scene_provenance": {"transforms_sha256": "1" * 64},
                },
                "case_b": {
                    "scene_id": scene_id,
                    "image_prompt": "/x/b.png",
                    "transforms_path": "/x/transforms.json",
                    "source_scene_provenance": {"transforms_sha256": "1" * 64},
                },
            }
        )
    )
    with pytest.raises(ValueError, match="multiple cases"):
        MODULE.load_cases(manifest)
