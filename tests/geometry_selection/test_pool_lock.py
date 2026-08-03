from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from geometry_selection.cache import cache_paths, canonical_hash, save_geometry_cache
from geometry_selection.pool_lock import (
    make_candidate_pool_lock,
    validate_candidate_pool_lock,
)
from geometry_selection.schema import GeometryPrediction


def _cache(cache_root):
    prediction = GeometryPrediction(
        world_to_camera=np.broadcast_to(np.eye(4), (2, 4, 4)).copy(),
        intrinsics=np.broadcast_to(np.eye(3), (2, 3, 3)).copy(),
        depth=np.ones((2, 2, 2)),
        confidence=np.ones((2, 2, 2)),
        keyframe_indices=np.array([0, 1]),
    )
    provenance = {"video_sha256": "a" * 64, "kind": "test"}
    key = canonical_hash(provenance)
    save_geometry_cache(cache_root, key, prediction, provenance)
    return key


def test_candidate_pool_lock_binds_pool_and_cache_arrays(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    cache_key = _cache(cache_root)
    pool_path = tmp_path / "pool.json"
    pool = {
        "artifact_mode": "formal",
        "candidate_spec_sha256": "b" * 64,
        "protocol_manifest_sha256": "c" * 64,
        "candidate_count": 1,
        "preparation": {
            "commit": "d" * 40,
            "implementation_sha256": "e" * 64,
            "experiment_lock_sha256": "f" * 64,
        },
        "cases": [
            {
                "split": "validation",
                "backbone": "Wan2.2-TI2V-5B",
                "candidates": [
                    {
                        "geometry_cache_key": cache_key,
                        "geometry_window_bundle": None,
                    }
                ],
            }
        ],
    }
    pool_path.write_text(json.dumps(pool))
    lock = make_candidate_pool_lock(pool_path, cache_root)

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    lock_path = repo / "pool_lock.json"
    lock_path.write_text(json.dumps(lock))
    subprocess.run(["git", "-C", str(repo), "add", "pool_lock.json"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "lock"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    validate_candidate_pool_lock(
        lock_path,
        repo,
        commit,
        pool_path=pool_path,
        cache_root=cache_root,
    )

    _, metadata_path = cache_paths(cache_root, cache_key)
    original_metadata = json.loads(metadata_path.read_text())
    metadata = dict(original_metadata)
    metadata["prediction_metadata"] = {"inputs": [{"sha256": "0" * 64}]}
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="candidate pool lock mismatch"):
        validate_candidate_pool_lock(
            lock_path,
            repo,
            commit,
            pool_path=pool_path,
            cache_root=cache_root,
        )
    metadata_path.write_text(json.dumps(original_metadata, indent=2, sort_keys=True) + "\n")

    pool["candidate_spec_sha256"] = "0" * 64
    pool_path.write_text(json.dumps(pool))
    with pytest.raises(ValueError, match="candidate pool lock mismatch"):
        validate_candidate_pool_lock(
            lock_path,
            repo,
            commit,
            pool_path=pool_path,
            cache_root=cache_root,
        )
