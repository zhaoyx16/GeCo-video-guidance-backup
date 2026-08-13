from __future__ import annotations

import hashlib
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest


RUNNER = Path(__file__).with_name("run_generation_case.py")
SPEC = importlib.util.spec_from_file_location("fullgraph_dpo_train_dev_runner", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_checkpoint(root: Path, step: int = 64) -> tuple[Path, str]:
    checkpoint = root / "adapter-step-000064"
    checkpoint.mkdir()
    weight = checkpoint / "pytorch_lora_weights.safetensors"
    weight.write_bytes(b"model-level-lora")
    receipt = {
        "schema": "fullgraph-dpo-adapter-checkpoint-v1",
        "step": step,
        "files": [
            {
                "name": weight.name,
                "sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
                "size": weight.stat().st_size,
            }
        ],
    }
    receipt_path = checkpoint / "CHECKPOINT_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return checkpoint, hashlib.sha256(receipt_path.read_bytes()).hexdigest()


def test_validate_lora_checkpoint_binds_all_artifacts(tmp_path: Path) -> None:
    checkpoint, receipt_sha = _write_checkpoint(tmp_path)
    identity = MODULE.validate_lora_checkpoint(checkpoint, receipt_sha, 64)
    assert identity["step"] == 64
    assert identity["weight_name"] == "pytorch_lora_weights.safetensors"
    assert identity["checkpoint_receipt_sha256"] == receipt_sha

    (checkpoint / "unreceipted.txt").write_text("bad", encoding="utf-8")
    with pytest.raises(ValueError, match="unreceipted"):
        MODULE.validate_lora_checkpoint(checkpoint, receipt_sha, 64)


def test_validate_lora_checkpoint_rejects_content_and_step_mismatch(tmp_path: Path) -> None:
    checkpoint, receipt_sha = _write_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="contract mismatch"):
        MODULE.validate_lora_checkpoint(checkpoint, receipt_sha, 48)
    (checkpoint / "pytorch_lora_weights.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="content mismatch"):
        MODULE.validate_lora_checkpoint(checkpoint, receipt_sha, 64)


@pytest.mark.parametrize(
    ("mode", "expected_action"),
    (("base", "disabled"), ("adapted", "fullgraph_dpo")),
)
def test_build_pipeline_loads_model_level_lora_with_explicit_mode(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_action: str,
) -> None:
    import diffusers

    events: dict[str, object] = {}

    class FakeTransformer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            events["transformer_from_pretrained"] = (args, kwargs)
            return cls()

        def load_lora_adapter(self, *args, **kwargs):
            events["load"] = (args, kwargs)

        def disable_adapters(self):
            events["action"] = "disabled"

        def set_adapters(self, name):
            events["action"] = name

    class FakeVae:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            events["pipeline_from_pretrained"] = (args, kwargs)
            return cls()

        def to(self, device):
            events["device"] = device
            return self

    monkeypatch.setattr(diffusers, "WanTransformer3DModel", FakeTransformer)
    monkeypatch.setattr(diffusers, "AutoencoderKLWan", FakeVae)
    monkeypatch.setattr(diffusers, "WanImageToVideoPipeline", FakePipeline)
    monkeypatch.setattr(MODULE, "place_vae", lambda *args, **kwargs: "cuda:0")
    args = Namespace(
        repo=Path("/tmp/repo"),
        backbone="wan",
        method="lora_dpo",
        model="/model",
        pipe_device="cuda:0",
        vae_device="cuda:0",
        allow_split_vae=False,
        lora_checkpoint=Path("/checkpoint"),
        lora_identity={"weight_name": "pytorch_lora_weights.safetensors"},
        lora_mode=mode,
    )
    MODULE.build_pipeline(args)
    load_args, load_kwargs = events["load"]
    assert load_args == (Path("/checkpoint"),)
    assert load_kwargs["prefix"] is None
    assert load_kwargs["local_files_only"] is True
    assert load_kwargs["adapter_name"] == "fullgraph_dpo"
    assert events["action"] == expected_action
    _, pipeline_kwargs = events["pipeline_from_pretrained"]
    assert isinstance(pipeline_kwargs["transformer"], FakeTransformer)
    assert pipeline_kwargs["local_files_only"] is True
