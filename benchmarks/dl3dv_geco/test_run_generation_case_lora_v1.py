from __future__ import annotations

import hashlib
import importlib.util
import json
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


@pytest.mark.parametrize("invalid_step", (True, 64.0))
def test_validate_lora_checkpoint_requires_exact_integer_receipt_step(
    tmp_path: Path, invalid_step: object
) -> None:
    checkpoint, _ = _write_checkpoint(tmp_path)
    receipt_path = checkpoint / "CHECKPOINT_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["step"] = invalid_step
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="contract mismatch"):
        MODULE.validate_lora_checkpoint(checkpoint, receipt_sha, 64)


def test_private_snapshot_is_bound_to_validated_bytes(tmp_path: Path) -> None:
    checkpoint, receipt_sha = _write_checkpoint(tmp_path)
    identity = MODULE.validate_lora_checkpoint(checkpoint, receipt_sha, 64)
    temporary, snapshot = MODULE.snapshot_lora_checkpoint(checkpoint, identity)
    try:
        source = checkpoint / "pytorch_lora_weights.safetensors"
        source.write_bytes(b"same-path-mutated-after-snapshot")
        snap = snapshot / "pytorch_lora_weights.safetensors"
        assert snap.read_bytes() == b"model-level-lora"
        assert hashlib.sha256(snap.read_bytes()).hexdigest() == identity["files"][0]["sha256"]
    finally:
        temporary.cleanup()


def test_snapshot_never_reopens_authenticated_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, receipt_sha = _write_checkpoint(tmp_path)
    identity = MODULE.validate_lora_checkpoint(checkpoint, receipt_sha, 64)
    original_open = MODULE.os.open

    def reject_receipt_open(path, flags, *args, **kwargs):
        if Path(path).name == "CHECKPOINT_RECEIPT.json":
            raise AssertionError("authenticated receipt was reopened")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(MODULE.os, "open", reject_receipt_open)
    temporary, snapshot = MODULE.snapshot_lora_checkpoint(checkpoint, identity)
    try:
        assert (snapshot / "CHECKPOINT_RECEIPT.json").read_bytes() == identity[
            "_checkpoint_receipt_payload"
        ]
    finally:
        temporary.cleanup()


def test_snapshot_rejects_post_validation_mutation(tmp_path: Path) -> None:
    checkpoint, receipt_sha = _write_checkpoint(tmp_path)
    identity = MODULE.validate_lora_checkpoint(checkpoint, receipt_sha, 64)
    (checkpoint / "pytorch_lora_weights.safetensors").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="changed after validation"):
        MODULE.snapshot_lora_checkpoint(checkpoint, identity)


def test_receipt_is_hashed_and_parsed_from_one_no_follow_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, receipt_sha = _write_checkpoint(tmp_path)
    original_reader = MODULE.read_regular_file_bytes_no_follow
    calls = []

    def one_read(path: Path) -> bytes:
        calls.append(path)
        return original_reader(path)

    monkeypatch.setattr(MODULE, "read_regular_file_bytes_no_follow", one_read)
    # Any receipt Path.read_text call would re-open a mutable producer path.
    original_read_text = Path.read_text

    def reject_receipt_reopen(path: Path, *args, **kwargs):
        if path.name == "CHECKPOINT_RECEIPT.json":
            raise AssertionError("receipt path was reopened")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject_receipt_reopen)
    identity = MODULE.validate_lora_checkpoint(checkpoint, receipt_sha, 64)
    assert identity["checkpoint_receipt_sha256"] == receipt_sha
    assert calls == [checkpoint / "CHECKPOINT_RECEIPT.json"]


def test_snapshot_ignores_receipt_swap_after_validation(tmp_path: Path) -> None:
    checkpoint, receipt_sha = _write_checkpoint(tmp_path)
    identity = MODULE.validate_lora_checkpoint(checkpoint, receipt_sha, 64)
    receipt_path = checkpoint / "CHECKPOINT_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": "fullgraph-dpo-adapter-checkpoint-v1",
                "step": 64,
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    temporary, snapshot = MODULE.snapshot_lora_checkpoint(checkpoint, identity)
    try:
        assert hashlib.sha256(
            (snapshot / "CHECKPOINT_RECEIPT.json").read_bytes()
        ).hexdigest() == receipt_sha
    finally:
        temporary.cleanup()


@pytest.mark.parametrize(("mode", "expected_action"), (("base", "disabled"), ("adapted", "fullgraph_dpo")))
def test_load_lora_transformer_has_explicit_model_level_contract(
    mode: str, expected_action: str
) -> None:
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

    result = MODULE.load_lora_transformer(
        "/model",
        Path("/checkpoint"),
        {"weight_name": "pytorch_lora_weights.safetensors"},
        mode,
        transformer_class=FakeTransformer,
    )
    assert isinstance(result, FakeTransformer)
    from_args, from_kwargs = events["transformer_from_pretrained"]
    assert from_args == ("/model",)
    assert from_kwargs["subfolder"] == "transformer"
    assert from_kwargs["local_files_only"] is True
    load_args, load_kwargs = events["load"]
    assert load_args == (Path("/checkpoint"),)
    assert load_kwargs["prefix"] is None
    assert load_kwargs["local_files_only"] is True
    assert load_kwargs["adapter_name"] == "fullgraph_dpo"
    assert events["action"] == expected_action
