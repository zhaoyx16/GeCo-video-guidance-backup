from __future__ import annotations

import json
import os

import pytest

from geometry_selection.model_lock import (
    FROZEN_MODEL_MARKER,
    MODEL_LOCK_SCHEMA,
    load_model_lock,
    model_directory_identity,
    runtime_identity_matches_lock,
    verify_generation_model,
)


def _set_tree_read_only(root) -> None:
    (root / FROZEN_MODEL_MARKER).write_text(
        json.dumps(
            {
                "schema": "geometry-frozen-model-snapshot-v1",
                "publication": "closed-staging-read-only-atomic-rename",
            }
        )
    )
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _rewrite_frozen_weight(root, weight, content: bytes, *, offset: int | None = None) -> None:
    root.chmod(0o755)
    weight.chmod(0o644)
    if offset is None:
        weight.write_bytes(content)
    else:
        with weight.open("r+b") as handle:
            handle.seek(offset)
            handle.write(content)
    weight.chmod(0o444)
    root.chmod(0o555)


def test_model_lock_pins_snapshot_config_and_weight_content(tmp_path) -> None:
    revision = "1" * 40
    model = tmp_path / "models--example" / "snapshots" / revision
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({"layers": 1}))
    (model / "tokenizer.model").write_bytes(b"tokens")
    (model / "model.safetensors").write_bytes(b"weights")
    _set_tree_read_only(model)
    locked_identity = model_directory_identity(model, hash_weights=True)
    runtime_identity = model_directory_identity(model, hash_weights=False)
    assert runtime_identity_matches_lock(runtime_identity, locked_identity)

    payload = {
        "schema": MODEL_LOCK_SCHEMA,
        "generation_models": {"Wan2.2-TI2V-5B": locked_identity},
        "geometry_backbone": {
            "name": "VGGT-Omega-1B-512",
            "source_commit": "2" * 40,
            "source_tree_sha256": "3" * 64,
            "checkpoint_size": 7,
            "checkpoint_sha256": "4" * 64,
        },
    }
    path = tmp_path / "model_lock.json"
    path.write_text(json.dumps(payload))
    lock = load_model_lock(path)
    assert verify_generation_model(lock, "Wan2.2-TI2V-5B", model) == locked_identity

    _rewrite_frozen_weight(model, model / "model.safetensors", b"changed")
    with pytest.raises(ValueError, match="weight hash mismatch"):
        verify_generation_model(lock, "Wan2.2-TI2V-5B", model)

    _rewrite_frozen_weight(model, model / "model.safetensors", b"different-size")
    changed_runtime = model_directory_identity(model, hash_weights=False)
    assert not runtime_identity_matches_lock(changed_runtime, locked_identity)


def test_cached_weight_verification_invalidates_on_same_size_tamper(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    weight = model / "model.safetensors"
    weight.write_bytes(b"original")
    _set_tree_read_only(model)
    locked_identity = model_directory_identity(model, hash_weights=True)
    lock = {
        "schema": MODEL_LOCK_SCHEMA,
        "generation_models": {"model": locked_identity},
        "geometry_backbone": {
            "name": "VGGT-Omega-1B-512",
            "source_commit": "2" * 40,
            "source_tree_sha256": "3" * 64,
            "checkpoint_size": 7,
            "checkpoint_sha256": "4" * 64,
        },
    }
    cache = tmp_path / "verification"
    verify_generation_model(lock, "model", model, verification_cache=cache)
    _rewrite_frozen_weight(model, weight, b"tampered")
    with pytest.raises(ValueError, match="weight hash mismatch"):
        verify_generation_model(lock, "model", model, verification_cache=cache)


def test_cached_weight_verification_rehashes_unsampled_large_file_content(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    weight = model / "model.safetensors"
    weight.write_bytes(b"a" * (256 * 1024))
    _set_tree_read_only(model)
    locked_identity = model_directory_identity(model, hash_weights=True)
    lock = {
        "schema": MODEL_LOCK_SCHEMA,
        "generation_models": {"model": locked_identity},
        "geometry_backbone": {
            "name": "VGGT-Omega-1B-512",
            "source_commit": "2" * 40,
            "source_tree_sha256": "3" * 64,
            "checkpoint_size": 7,
            "checkpoint_sha256": "4" * 64,
        },
    }
    cache = tmp_path / "verification"
    verify_generation_model(lock, "model", model, verification_cache=cache)

    original_stat = weight.stat()
    _rewrite_frozen_weight(model, weight, b"tampered", offset=70 * 1024)
    os.utime(weight, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    with pytest.raises(ValueError, match="weight hash mismatch"):
        verify_generation_model(lock, "model", model, verification_cache=cache)


def test_model_verification_rejects_writable_snapshot(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"weights")
    (model / FROZEN_MODEL_MARKER).write_text(
        json.dumps(
            {
                "schema": "geometry-frozen-model-snapshot-v1",
                "publication": "closed-staging-read-only-atomic-rename",
            }
        )
    )
    locked_identity = model_directory_identity(model, hash_weights=True)
    lock = {
        "schema": MODEL_LOCK_SCHEMA,
        "generation_models": {"model": locked_identity},
        "geometry_backbone": {
            "name": "VGGT-Omega-1B-512",
            "source_commit": "2" * 40,
            "source_tree_sha256": "3" * 64,
            "checkpoint_size": 7,
            "checkpoint_sha256": "4" * 64,
        },
    }

    with pytest.raises(ValueError, match="frozen read-only snapshot"):
        verify_generation_model(lock, "model", model)


def test_model_verification_rejects_symlink_inside_snapshot(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    weight = model / "model.safetensors"
    weight.write_bytes(b"weights")
    _set_tree_read_only(model)
    model.chmod(0o755)
    (model / "weight-link.bin").symlink_to(weight)
    model.chmod(0o555)
    locked_identity = model_directory_identity(model, hash_weights=True)
    lock = {
        "schema": MODEL_LOCK_SCHEMA,
        "generation_models": {"model": locked_identity},
        "geometry_backbone": {
            "name": "VGGT-Omega-1B-512",
            "source_commit": "2" * 40,
            "source_tree_sha256": "3" * 64,
            "checkpoint_size": 7,
            "checkpoint_sha256": "4" * 64,
        },
    }

    with pytest.raises(ValueError, match="contains symlink"):
        verify_generation_model(lock, "model", model)
