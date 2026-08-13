"""Portable, content-based model identities for frozen experiments."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
import stat
from typing import Any


MODEL_LOCK_SCHEMA = "geometry-model-lock-v2"
FROZEN_MODEL_MARKER = ".geometry_frozen_snapshot.json"


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
    # The fixed root marker records how the snapshot was published; it is not
    # model content. Keeping it outside the content identity lets one lock bind
    # the source content and its frozen publication. Its exact schema and the
    # tree's immutability are enforced separately by
    # validate_frozen_model_snapshot().
    regular_files = sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file()
            and item.relative_to(path).as_posix() != FROZEN_MODEL_MARKER
        ),
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


def validate_frozen_model_snapshot(model_path: Path) -> Path:
    """Require an atomically published snapshot immutable to normal writers."""

    unresolved = model_path.expanduser().absolute()
    if unresolved.is_symlink():
        raise ValueError("formal generation model root must not be a symlink")
    root = unresolved.resolve()
    marker = root / FROZEN_MODEL_MARKER
    if not marker.is_file() or marker.is_symlink():
        raise ValueError(f"formal generation model lacks {FROZEN_MODEL_MARKER}")
    try:
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {FROZEN_MODEL_MARKER}") from error
    if marker_payload.get("schema") != "geometry-frozen-model-snapshot-v1":
        raise ValueError(f"invalid {FROZEN_MODEL_MARKER} schema")
    if marker_payload.get("publication") != "closed-staging-read-only-atomic-rename":
        raise ValueError(f"invalid {FROZEN_MODEL_MARKER} publication mode")

    entries = [root, *root.rglob("*")]
    writable = []
    for entry in entries:
        if entry.is_symlink():
            raise ValueError(f"formal generation model contains symlink: {entry}")
        mode = entry.stat().st_mode
        if stat.S_ISREG(mode) or stat.S_ISDIR(mode):
            if mode & 0o222:
                writable.append(str(entry.relative_to(root)) if entry != root else ".")
    if writable:
        preview = ", ".join(writable[:5])
        suffix = "" if len(writable) <= 5 else f" (+{len(writable) - 5} more)"
        raise ValueError(
            "formal generation model must be a frozen read-only snapshot; "
            f"writable entries: {preview}{suffix}"
        )
    return root


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
    """Fully verify weights and atomically publish an audit receipt.

    The receipt is deliberately never trusted to skip hashing: timestamps and
    sparse fingerprints are not authoritative on distributed filesystems.
    """

    cache_root = verification_cache.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    identity_sha256 = _identity_sha256(locked)
    cache_path = cache_root / f"{identity_sha256}.json"
    fingerprints = _weight_stat_fingerprints(model_path, locked)
    for record in locked["weight_files"]:
        actual = file_sha256(model_path.resolve() / record["path"])
        if actual != record.get("sha256"):
            raise ValueError(f"generation model weight hash mismatch: {record['path']}")
    verified_fingerprints = _weight_stat_fingerprints(model_path, locked)
    if verified_fingerprints != fingerprints:
        raise RuntimeError("generation model weights changed during verification")
    payload = {
        "schema": "geometry-model-verification-v2",
        "verification_mode": "full_sha256_each_call",
        "model_path": str(model_path.resolve()),
        "locked_identity_sha256": identity_sha256,
        "weight_stat_fingerprints": verified_fingerprints,
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=cache_root,
        prefix=f".{cache_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
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
    validate_frozen_model_snapshot(model_path)
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
    validate_frozen_model_snapshot(model_path)
    return locked
