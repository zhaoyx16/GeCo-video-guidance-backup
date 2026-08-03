from __future__ import annotations

import json

from geometry_selection.model_lock import (
    MODEL_LOCK_SCHEMA,
    load_model_lock,
    model_directory_identity,
    runtime_identity_matches_lock,
    verify_generation_model,
)


def test_model_lock_pins_snapshot_config_and_weight_content(tmp_path) -> None:
    revision = "1" * 40
    model = tmp_path / "models--example" / "snapshots" / revision
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({"layers": 1}))
    (model / "model.safetensors").write_bytes(b"weights")
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

    (model / "model.safetensors").write_bytes(b"different-size")
    changed_runtime = model_directory_identity(model, hash_weights=False)
    assert not runtime_identity_matches_lock(changed_runtime, locked_identity)
