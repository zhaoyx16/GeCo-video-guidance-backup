"""Portable, content-based model identities for frozen experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_LOCK_SCHEMA = "geometry-model-lock-v1"


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_commit(path: Path) -> str | None:
    parts = path.resolve().parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    return parts[index + 1] if index + 1 < len(parts) else None


def model_directory_identity(path: Path, *, hash_weights: bool) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"formal model must be a local pinned directory: {path}")
    json_files = [
        {"path": str(item.relative_to(path)), "sha256": file_sha256(item)}
        for item in sorted(path.rglob("*.json"))
    ]
    weights = []
    seen = set()
    for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth"):
        for item in sorted(path.rglob(pattern)):
            relative = str(item.relative_to(path))
            if relative in seen:
                continue
            seen.add(relative)
            record = {"path": relative, "size": item.stat().st_size}
            if hash_weights:
                record["sha256"] = file_sha256(item)
            weights.append(record)
    if not weights:
        raise ValueError(f"model directory contains no recognized weight files: {path}")
    return {
        "snapshot_commit": snapshot_commit(path),
        "json_files": json_files,
        "weight_files": weights,
    }


def runtime_identity_matches_lock(runtime: dict[str, Any], locked: dict[str, Any]) -> bool:
    locked_without_weight_hashes = {
        **locked,
        "weight_files": [
            {key: value for key, value in record.items() if key != "sha256"}
            for record in locked["weight_files"]
        ],
    }
    return runtime == locked_without_weight_hashes


def load_model_lock(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MODEL_LOCK_SCHEMA:
        raise ValueError(f"model lock schema must be {MODEL_LOCK_SCHEMA}")
    if not isinstance(payload.get("generation_models"), dict):
        raise ValueError("model lock must contain generation_models")
    geometry = payload.get("geometry_backbone")
    required_geometry = {
        "name",
        "source_commit",
        "source_tree_sha256",
        "checkpoint_size",
        "checkpoint_sha256",
    }
    if not isinstance(geometry, dict) or not required_geometry <= set(geometry):
        raise ValueError("model lock geometry_backbone is incomplete")
    return payload


def verify_generation_model(
    lock: dict[str, Any],
    profile_name: str,
    model_path: Path,
) -> dict[str, Any]:
    if profile_name not in lock["generation_models"]:
        raise ValueError(f"generation model is absent from lock: {profile_name}")
    runtime = model_directory_identity(model_path, hash_weights=False)
    locked = lock["generation_models"][profile_name]
    if not runtime_identity_matches_lock(runtime, locked):
        raise ValueError(f"generation model identity differs from lock: {profile_name}")
    return locked
