"""Atomic, provenance-checked geometry cache."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .schema import GeometryPrediction


CACHE_SCHEMA_VERSION = 2


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if canonical_hash(provenance) != cache_key:
        raise ValueError("cache_key must equal canonical_hash(provenance)")
    final_directory = arrays_path.parent
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    if final_directory.exists():
        load_geometry_cache(root, cache_key, expected_provenance=provenance)
        return arrays_path, metadata_path

    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{cache_key}.tmp-",
            dir=final_directory.parent,
        )
    )
    temporary_arrays = temporary_directory / arrays_path.name
    temporary_metadata = temporary_directory / metadata_path.name
    try:
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

        metadata = {
            "status": "COMPLETE",
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_key": cache_key,
            "provenance_sha256": canonical_hash(provenance),
            "arrays_file": arrays_path.name,
            "arrays_size": temporary_arrays.stat().st_size,
            "arrays_sha256": file_sha256(temporary_arrays),
            "prediction_metadata": prediction.metadata,
            "provenance": provenance,
        }
        with temporary_metadata.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.rename(temporary_directory, final_directory)
        except FileExistsError:
            load_geometry_cache(root, cache_key, expected_provenance=provenance)
    finally:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)
    return arrays_path, metadata_path


def load_geometry_cache_metadata(root: Path, cache_key: str) -> dict[str, Any]:
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
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("cache provenance must be a mapping")
    provenance_hash = canonical_hash(provenance)
    if provenance_hash != cache_key or metadata.get("provenance_sha256") != provenance_hash:
        raise ValueError("cache key does not match canonical provenance")
    if metadata.get("arrays_size") != arrays_path.stat().st_size:
        raise ValueError("cached array size does not match metadata; output may be partial or stale")
    if metadata.get("arrays_sha256") != file_sha256(arrays_path):
        raise ValueError("cached array digest does not match metadata")
    return metadata


def load_geometry_cache(
    root: Path,
    cache_key: str,
    *,
    expected_provenance: dict[str, Any] | None = None,
    expected_video_sha256: str | None = None,
) -> GeometryPrediction:
    arrays_path, metadata_path = cache_paths(root, cache_key)
    metadata = load_geometry_cache_metadata(root, cache_key)
    if expected_provenance is not None and metadata.get("provenance") != expected_provenance:
        raise ValueError("cache provenance does not match the requested run")
    if (
        expected_video_sha256 is not None
        and metadata["provenance"].get("video_sha256") != expected_video_sha256
    ):
        raise ValueError("cache provenance video does not match the candidate video")

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
