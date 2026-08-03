"""Post-extraction lock binding a candidate pool and every geometry cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cache import cache_paths, canonical_hash, load_geometry_cache_metadata
from .protocol import file_sha256, validate_committed_file


CANDIDATE_POOL_LOCK_SCHEMA = "geometry-candidate-pool-lock-v1"


def _cache_keys(pool: dict[str, Any]) -> tuple[str, ...]:
    keys: list[str] = []
    for case in pool.get("cases", []):
        for candidate in case.get("candidates", []):
            keys.append(candidate["geometry_cache_key"])
            bundle = candidate.get("geometry_window_bundle")
            if bundle is not None:
                keys.extend(record["geometry_cache_key"] for record in bundle["windows"])
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("candidate pool cache keys must be non-empty and globally unique")
    return tuple(sorted(keys))


def cache_snapshot(pool: dict[str, Any], cache_root: Path) -> dict[str, Any]:
    records = []
    for cache_key in _cache_keys(pool):
        metadata = load_geometry_cache_metadata(cache_root, cache_key)
        _, metadata_path = cache_paths(cache_root, cache_key)
        records.append(
            {
                "cache_key": cache_key,
                "arrays_sha256": metadata["arrays_sha256"],
                "provenance_sha256": metadata["provenance_sha256"],
                "metadata_sha256": file_sha256(metadata_path),
            }
        )
    return {
        "count": len(records),
        "records_sha256": canonical_hash({"records": records}),
    }


def make_candidate_pool_lock(pool_path: Path, cache_root: Path) -> dict[str, Any]:
    pool_file = pool_path.resolve()
    pool = json.loads(pool_file.read_text(encoding="utf-8"))
    cases = pool.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("candidate pool contains no cases")
    splits = {case.get("split") for case in cases}
    backbones = {case.get("backbone") for case in cases}
    if len(splits) != 1 or len(backbones) != 1:
        raise ValueError("candidate pool must contain one split and one backbone")
    preparation = pool.get("preparation")
    if not isinstance(preparation, dict):
        raise ValueError("candidate pool lacks preparation identity")
    snapshot = cache_snapshot(pool, cache_root)
    return {
        "schema": CANDIDATE_POOL_LOCK_SCHEMA,
        "candidate_pool_sha256": file_sha256(pool_file),
        "candidate_spec_sha256": pool.get("candidate_spec_sha256"),
        "protocol_manifest_sha256": pool.get("protocol_manifest_sha256"),
        "artifact_mode": pool.get("artifact_mode"),
        "split": next(iter(splits)),
        "backbone": next(iter(backbones)),
        "case_count": len(cases),
        "candidate_count": pool.get("candidate_count"),
        "preparation_commit": preparation.get("commit"),
        "implementation_sha256": preparation.get("implementation_sha256"),
        "experiment_lock_sha256": preparation.get("experiment_lock_sha256"),
        "cache_record_count": snapshot["count"],
        "cache_records_sha256": snapshot["records_sha256"],
    }


def validate_candidate_pool_lock(
    lock_path: Path,
    repo_root: Path,
    expected_commit: str,
    *,
    pool_path: Path,
    cache_root: Path,
) -> dict[str, Any]:
    committed = validate_committed_file(lock_path, repo_root, expected_commit)
    lock = json.loads(committed)
    expected = make_candidate_pool_lock(pool_path, cache_root)
    if set(lock) != set(expected):
        raise ValueError("candidate pool lock fields are incomplete or unknown")
    mismatches = {
        key: (lock.get(key), value)
        for key, value in expected.items()
        if lock.get(key) != value
    }
    if mismatches:
        raise ValueError(f"candidate pool lock mismatch: {mismatches}")
    if lock["artifact_mode"] != "formal":
        raise ValueError("formal ranking requires a formal candidate pool lock")
    return lock
