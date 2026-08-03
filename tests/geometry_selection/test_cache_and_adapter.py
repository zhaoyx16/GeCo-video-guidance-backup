from __future__ import annotations

import json

import numpy as np
import pytest

from geometry_selection.backbones.vggt_omega import geometry_from_vggt_omega_outputs
from geometry_selection.cache import canonical_hash, load_geometry_cache, save_geometry_cache


def test_vggt_omega_output_normalization(plane_prediction) -> None:
    normalized = geometry_from_vggt_omega_outputs(
        extrinsics=plane_prediction.world_to_camera[:, :3, :4],
        intrinsics=plane_prediction.intrinsics,
        depth=plane_prediction.depth[..., None],
        confidence=plane_prediction.confidence,
        keyframe_indices=plane_prediction.keyframe_indices,
        metadata={"source": "test"},
    )
    assert normalized.world_to_camera.shape == plane_prediction.world_to_camera.shape
    assert normalized.depth.shape == plane_prediction.depth.shape
    assert normalized.depth.dtype == np.float32
    assert normalized.metadata["source"] == "test"


def test_cache_round_trip_and_provenance_rejection(tmp_path, plane_prediction) -> None:
    provenance = {"video_sha256": "a" * 64, "checkpoint": "b" * 64}
    key = canonical_hash(provenance)
    arrays_path, metadata_path = save_geometry_cache(
        tmp_path, key, plane_prediction, provenance
    )
    assert arrays_path.is_file()
    assert metadata_path.is_file()
    loaded = load_geometry_cache(tmp_path, key, expected_provenance=provenance)
    assert np.array_equal(loaded.keyframe_indices, plane_prediction.keyframe_indices)
    assert np.allclose(loaded.depth, plane_prediction.depth)

    with pytest.raises(ValueError, match="provenance"):
        load_geometry_cache(tmp_path, key, expected_provenance={"video_sha256": "c" * 64})


def test_cache_rejects_partial_or_stale_array(tmp_path, plane_prediction) -> None:
    provenance = {"case": "partial"}
    key = canonical_hash(provenance)
    arrays_path, metadata_path = save_geometry_cache(
        tmp_path, key, plane_prediction, provenance
    )
    metadata = json.loads(metadata_path.read_text())
    metadata["arrays_size"] += 1
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="partial or stale"):
        load_geometry_cache(tmp_path, key, expected_provenance=provenance)
    assert arrays_path.is_file()
