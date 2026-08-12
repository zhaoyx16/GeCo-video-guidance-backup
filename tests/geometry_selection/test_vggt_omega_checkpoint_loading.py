import inspect
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from geometry_selection.backbones.vggt_omega import _load_checkpoint_state


def test_safetensors_checkpoint_uses_tensor_only_loader(tmp_path: Path):
    path = tmp_path / "model.safetensors"
    expected = {"weight": torch.arange(6).reshape(2, 3)}
    save_file(expected, str(path))
    actual = _load_checkpoint_state(path)
    assert actual.keys() == expected.keys()
    assert torch.equal(actual["weight"], expected["weight"])


def test_legacy_torch_checkpoint_keeps_weights_only_true(tmp_path: Path, monkeypatch):
    path = tmp_path / "model.pt"
    path.write_bytes(b"fixture")
    calls = []

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        return {"model": {"weight": torch.ones(1)}}

    monkeypatch.setattr(torch, "load", fake_load)
    state = _load_checkpoint_state(path)
    assert torch.equal(state["weight"], torch.ones(1))
    assert calls[0][1] == {"map_location": "cpu", "weights_only": True}


def test_unsupported_and_non_tensor_payloads_are_rejected(tmp_path: Path, monkeypatch):
    unknown = tmp_path / "model.ckpt"
    unknown.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="unsupported"):
        _load_checkpoint_state(unknown)

    legacy = tmp_path / "model.pth"
    legacy.write_bytes(b"fixture")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"bad": "not-a-tensor"})
    with pytest.raises(TypeError, match="string-to-tensor"):
        _load_checkpoint_state(legacy)


def test_loader_source_never_disables_weights_only():
    source = inspect.getsource(_load_checkpoint_state)
    assert "weights_only=False" not in source
