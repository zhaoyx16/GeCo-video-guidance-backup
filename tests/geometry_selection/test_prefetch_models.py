from __future__ import annotations

import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prefetch_models.py"
SPEC = importlib.util.spec_from_file_location("prefetch_models", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_model_keys_is_strict() -> None:
    assert MODULE.parse_model_keys("wan,cosmos") == ["wan", "cosmos"]
    with pytest.raises(ValueError, match="unique comma-separated subset"):
        MODULE.parse_model_keys("wan,wan")
    with pytest.raises(ValueError, match="unique comma-separated subset"):
        MODULE.parse_model_keys("unknown")


def test_prepare_models_pins_revision_and_freezes_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"weights")
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(source)

    model_root = tmp_path / "models"
    receipts = MODULE.prepare_models(
        ["wan"],
        cache_dir=tmp_path / "cache",
        model_root=model_root,
        download=fake_download,
    )

    assert calls[0]["revision"] == MODULE.MODEL_SPECS["wan"]["revision"]
    frozen = Path(receipts[0]["frozen_path"])
    assert frozen.is_dir() and frozen.stat().st_mode & 0o222 == 0
    assert (frozen / "model.safetensors").read_bytes() == b"weights"
    assert json.loads((model_root / "refs" / "wan.json").read_text()) == receipts[0]


def test_prepare_models_rejects_gated_model_without_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "get_token", lambda: None)

    with pytest.raises(RuntimeError, match="is gated"):
        MODULE.prepare_models(
            ["cosmos"],
            cache_dir=tmp_path / "cache",
            model_root=tmp_path / "models",
            download=lambda **_: pytest.fail("download must not start without a token"),
        )


def test_prepare_models_is_concurrently_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"weights")
    model_root = tmp_path / "models"

    def run_once(_):
        return MODULE.prepare_models(
            ["wan"],
            cache_dir=tmp_path / "cache",
            model_root=model_root,
            download=lambda **_: str(source),
        )[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(run_once, range(2)))

    assert first == second
    assert Path(first["frozen_path"]).is_dir()
    assert not list((model_root / "frozen").glob(".*.staging-*"))
