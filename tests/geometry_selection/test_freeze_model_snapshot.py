from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from geometry_selection.model_lock import FROZEN_MODEL_MARKER, model_directory_identity


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "freeze_model_snapshot.py"
SPEC = importlib.util.spec_from_file_location("freeze_model_snapshot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_freeze_snapshot_dereferences_links_and_atomically_publishes(tmp_path: Path) -> None:
    external = tmp_path / "external.safetensors"
    external.write_bytes(b"weights")
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").symlink_to(external)
    target = tmp_path / "published" / "model"
    source_identity = model_directory_identity(source, hash_weights=True)

    assert MODULE.freeze_snapshot(source, target) == target
    assert target.is_dir() and not target.is_symlink()
    assert (target / "model.safetensors").read_bytes() == b"weights"
    assert not (target / "model.safetensors").is_symlink()
    marker = json.loads((target / FROZEN_MODEL_MARKER).read_text())
    assert marker["publication"] == "closed-staging-read-only-atomic-rename"
    for path in [target, *target.rglob("*")]:
        assert not path.is_symlink()
        assert path.stat().st_mode & 0o222 == 0
    assert model_directory_identity(target, hash_weights=True) == source_identity

    with pytest.raises(FileExistsError, match="refusing to replace"):
        MODULE.freeze_snapshot(source, target)


def test_freeze_snapshot_rejects_target_inside_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(ValueError, match="must not be inside"):
        MODULE.freeze_snapshot(source, source / "published" / "model")
    assert not (source / "published").exists()


def test_freeze_snapshot_rejects_directory_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "loop").symlink_to(source, target_is_directory=True)

    with pytest.raises(ValueError, match="directory symlink"):
        MODULE.freeze_snapshot(source, tmp_path / "published")
