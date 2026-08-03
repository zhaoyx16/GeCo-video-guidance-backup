"""Atomic, provenance-checked geometry cache."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .schema import GeometryPrediction


CACHE_SCHEMA_VERSION = 1


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def cache_paths(root: Path, cache_key: str) -> tuple[Path, Path]:
    if len(cache_key) != 64 or any(character not in "0123456789abcdef" for character in cache_key):
        raise ValueError("cache_key must be a lowercase SHA-256 hex digest")
    directory = root / cache_key[:2] / cache_key
    return directory / "geometry.npz", directory / "metadata.json"


def save_geometry_cache(
    root: Path,
    cache_key: str,
    prediction: GeometryPrediction,
    provenance: dict[str, Any],
) -> tuple[Path, Path]:
    prediction.validate()
    arrays_path, metadata_path = cache_paths(root, cache_key)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_arrays = arrays_path.with_name(f".{arrays_path.name}.{uuid.uuid4().hex}.tmp")
    temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.{uuid.uuid4().hex}.tmp")

    with temporary_arrays.open("wb") as handle:
        np.savez(
            handle,
            world_to_camera=prediction.world_to_camera.astype(np.float32),
            intrinsics=prediction.intrinsics.astype(np.float32),
            depth=prediction.depth.astype(np.float32),
            confidence=prediction.confidence.astype(np.float32),
            keyframe_indices=prediction.keyframe_indices.astype(np.int64),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_arrays, arrays_path)

    metadata = {
        "status": "COMPLETE",
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "arrays_file": arrays_path.name,
        "arrays_size": arrays_path.stat().st_size,
        "prediction_metadata": prediction.metadata,
        "provenance": provenance,
    }
    with temporary_metadata.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_metadata, metadata_path)
    return arrays_path, metadata_path


def load_geometry_cache(
    root: Path,
    cache_key: str,
    *,
    expected_provenance: dict[str, Any] | None = None,
) -> GeometryPrediction:
    arrays_path, metadata_path = cache_paths(root, cache_key)
    if not arrays_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"incomplete geometry cache: {arrays_path.parent}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "COMPLETE":
        raise ValueError(f"cache is not COMPLETE: {metadata_path}")
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"cache schema mismatch: {metadata.get('schema_version')} != {CACHE_SCHEMA_VERSION}"
        )
    if metadata.get("cache_key") != cache_key:
        raise ValueError("cache key does not match metadata")
    if metadata.get("arrays_size") != arrays_path.stat().st_size:
        raise ValueError("cached array size does not match metadata; output may be partial or stale")
    if expected_provenance is not None and metadata.get("provenance") != expected_provenance:
        raise ValueError("cache provenance does not match the requested run")

    with np.load(arrays_path, allow_pickle=False) as arrays:
        prediction = GeometryPrediction(
            world_to_camera=arrays["world_to_camera"],
            intrinsics=arrays["intrinsics"],
            depth=arrays["depth"],
            confidence=arrays["confidence"],
            keyframe_indices=arrays["keyframe_indices"],
            metadata=metadata.get("prediction_metadata", {}),
        )
    prediction.validate()
    return prediction
