from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "validate_development_generation_inputs.py"
)
SPEC = importlib.util.spec_from_file_location("validate_development_inputs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    image = tmp_path / "frame.png"
    Image.new("RGB", (12, 8), color=(20, 30, 40)).save(image)
    transforms = tmp_path / "transforms.json"
    transforms.write_text("{}")
    model = tmp_path / "model"
    model.mkdir()
    model_lock = tmp_path / "model_lock.json"
    model_lock.write_text("locked")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "_meta": {},
                "case-0": {
                    "split": "validation",
                    "split_order": 0,
                    "scene_id": "scene-0",
                    "image_prompt": str(image),
                    "transforms_path": str(transforms),
                    "conditioning_image_provenance": {
                        "image_sha256": _sha(image),
                        "transforms_sha256": _sha(transforms),
                        "width": 12,
                        "height": 8,
                    },
                },
            }
        )
    )
    return manifest, image, model, model_lock


def _run(monkeypatch, manifest: Path, model: Path, model_lock: Path) -> None:
    monkeypatch.setattr(
        MODULE,
        "load_model_lock",
        lambda _: {"generation_models": {"Wan2.2-TI2V-5B": {"identity": "locked"}}},
    )
    monkeypatch.setattr(MODULE, "validate_frozen_model_snapshot", lambda _: None)
    monkeypatch.setattr(MODULE, "model_directory_identity", lambda *args, **kwargs: {"identity": "runtime"})
    monkeypatch.setattr(MODULE, "runtime_identity_matches_lock", lambda *args: True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--manifest-sha256",
            _sha(manifest),
            "--expected-split",
            "validation",
            "--expected-cases",
            "1",
            "--case-index",
            "0",
            "--model",
            str(model),
            "--model-lock",
            str(model_lock),
            "--model-lock-sha256",
            _sha(model_lock),
            "--model-profile",
            "Wan2.2-TI2V-5B",
        ],
    )
    MODULE.main()


def test_development_preflight_accepts_locked_inputs(tmp_path, monkeypatch, capsys):
    manifest, _, model, model_lock = _fixture(tmp_path)
    _run(monkeypatch, manifest, model, model_lock)
    assert json.loads(capsys.readouterr().out)["case_id"] == "case-0"


def test_development_preflight_rejects_tampered_conditioning_image(tmp_path, monkeypatch):
    manifest, image, model, model_lock = _fixture(tmp_path)
    Image.new("RGB", (12, 8), color=(90, 80, 70)).save(image)
    with pytest.raises(ValueError, match="conditioning-image SHA-256 mismatch"):
        _run(monkeypatch, manifest, model, model_lock)
