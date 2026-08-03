from __future__ import annotations

from copy import deepcopy

import pytest

from geometry_selection.cache import canonical_hash
from geometry_selection.window_bundle import (
    WINDOW_EXTRACTION_MODE,
    independent_run_id,
    make_window_bundle,
    validate_window_bundle,
    validate_window_cache_record,
)


def _config():
    return {
        "mode": WINDOW_EXTRACTION_MODE,
        "num_keyframes": 6,
        "local_window_size": 4,
        "local_stride": 2,
        "loop_context": 2,
        "min_loop_node_gap": 3,
        "max_loop_windows": 1,
    }


def _record(identifier, kind, frames, cache_digit):
    geometry = {"checkpoint_sha256": "a" * 64}
    producer = {"commit": "b" * 40}
    pixels = [str(frame % 10) * 64 for frame in frames]
    run = independent_run_id(
        video_sha256="c" * 64,
        window_id=identifier,
        kind=kind,
        frame_indices=frames,
        frame_pixels_sha256=pixels,
        geometry_backbone=geometry,
        producer=producer,
    )
    return {
        "window_id": identifier,
        "kind": kind,
        "frame_indices": list(frames),
        "frame_pixels_sha256": pixels,
        "geometry_cache_key": cache_digit * 64,
        "independent_run_id": run,
    }, geometry, producer


def test_window_bundle_binds_schedule_and_independent_runs() -> None:
    first, _, _ = _record("local-a", "local", (0, 1, 2, 3), "1")
    second, _, _ = _record("local-b", "local", (2, 3, 4, 5), "2")
    loop, _, _ = _record("loop-a", "loop", (0, 1, 4, 5), "3")
    bundle = make_window_bundle(
        video_sha256="c" * 64,
        global_geometry_cache_key="d" * 64,
        global_keyframe_indices=range(6),
        extraction_config=_config(),
        window_records=(first, second, loop),
    )
    validate_window_bundle(bundle, expected_video_sha256="c" * 64)
    tampered = deepcopy(bundle)
    tampered["windows"][0]["frame_indices"][0] = 1
    with pytest.raises(ValueError):
        validate_window_bundle(tampered)


def test_window_cache_must_match_content_derived_run_id() -> None:
    record, geometry, producer = _record("local-a", "local", (0, 1, 2, 3), "1")
    provenance = {
        "video_sha256": "c" * 64,
        "window_id": record["window_id"],
        "window_kind": record["kind"],
        "keyframe_indices": record["frame_indices"],
        "frame_pixels_sha256": record["frame_pixels_sha256"],
        "independent_run_id": record["independent_run_id"],
        "geometry_backbone": geometry,
        "producer": producer,
    }
    metadata = {
        "cache_key": record["geometry_cache_key"],
        "provenance": provenance,
    }
    validate_window_cache_record(record, metadata, video_sha256="c" * 64)
    metadata["provenance"]["independent_run_id"] = canonical_hash({"fake": True})
    with pytest.raises(ValueError, match="provenance mismatch|content-derived"):
        validate_window_cache_record(record, metadata, video_sha256="c" * 64)
