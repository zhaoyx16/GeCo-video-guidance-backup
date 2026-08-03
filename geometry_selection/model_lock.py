"""Portable, content-based model identities for frozen experiments."""

from __future__ import annotations

import hashlib
import json
import os
import fcntl
from pathlib import Path
from typing import Any


MODEL_LOCK_SCHEMA = "geometry-model-lock-v2"


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
    regular_files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    json_files = [
        {
            "path": item.relative_to(path).as_posix(),
            "size": item.stat().st_size,
            "sha256": file_sha256(item),
        }
        for item in regular_files
        if item.suffix == ".json"
    ]
    weights = []
    seen = set()
    weight_suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    for item in regular_files:
        if item.suffix in weight_suffixes:
            relative = str(item.relative_to(path))
            if relative in seen:
                continue
            seen.add(relative)
            record = {"path": relative, "size": item.stat().st_size}
            if hash_weights:
                record["sha256"] = file_sha256(item)
            weights.append(record)
    support_files = [
        {
            "path": item.relative_to(path).as_posix(),
            "size": item.stat().st_size,
            "sha256": file_sha256(item),
        }
        for item in regular_files
        if item.suffix != ".json" and item.suffix not in weight_suffixes
    ]
    if not weights:
        raise ValueError(f"model directory contains no recognized weight files: {path}")
    return {
        "snapshot_commit": snapshot_commit(path),
        "json_files": json_files,
        "weight_files": weights,
        "support_files": support_files,
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


def _identity_sha256(identity: dict[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _weight_stat_fingerprints(model_path: Path, locked: dict[str, Any]) -> list[dict[str, Any]]:
    root = model_path.resolve()
    fingerprints = []
    for record in locked["weight_files"]:
        path = (root / record["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"locked weight path escapes model directory: {path}") from error
        stat = path.stat()
        fingerprints.append(
            {
                "path": record["path"],
                "size": stat.st_size,
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }
        )
    return fingerprints


def _verify_weight_hashes_cached(
    model_path: Path,
    locked: dict[str, Any],
    verification_cache: Path,
) -> None:
    cache_root = verification_cache.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    identity_sha256 = _identity_sha256(locked)
    cache_path = cache_root / f"{identity_sha256}.json"
    lock_path = cache_root / f"{identity_sha256}.lock"
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        fingerprints = _weight_stat_fingerprints(model_path, locked)
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("schema") == "geometry-model-verification-v1"
                and cached.get("model_path") == str(model_path.resolve())
                and cached.get("locked_identity_sha256") == identity_sha256
                and cached.get("weight_stat_fingerprints") == fingerprints
            ):
                return
        for record in locked["weight_files"]:
            actual = file_sha256(model_path.resolve() / record["path"])
            if actual != record.get("sha256"):
                raise ValueError(f"generation model weight hash mismatch: {record['path']}")
        payload = {
            "schema": "geometry-model-verification-v1",
            "model_path": str(model_path.resolve()),
            "locked_identity_sha256": identity_sha256,
            "weight_stat_fingerprints": fingerprints,
        }
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(cache_path)


def load_model_lock(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MODEL_LOCK_SCHEMA:
        raise ValueError(f"model lock schema must be {MODEL_LOCK_SCHEMA}")
    if not isinstance(payload.get("generation_models"), dict):
        raise ValueError("model lock must contain generation_models")
    for name, identity in payload["generation_models"].items():
        if not isinstance(identity.get("support_files"), list):
            raise ValueError(f"model lock support_files are missing for {name}")
        if any("sha256" not in record for record in identity.get("weight_files", [])):
            raise ValueError(f"model lock weight hash is missing for {name}")
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
    *,
    verification_cache: Path | None = None,
) -> dict[str, Any]:
    if profile_name not in lock["generation_models"]:
        raise ValueError(f"generation model is absent from lock: {profile_name}")
    runtime = model_directory_identity(model_path, hash_weights=False)
    locked = lock["generation_models"][profile_name]
    if not runtime_identity_matches_lock(runtime, locked):
        raise ValueError(f"generation model identity differs from lock: {profile_name}")
    if verification_cache is None:
        for record in locked["weight_files"]:
            actual = file_sha256(model_path.resolve() / record["path"])
            if actual != record["sha256"]:
                raise ValueError(f"generation model weight hash mismatch: {record['path']}")
    else:
        _verify_weight_hashes_cached(model_path, locked, verification_cache)
    return locked
