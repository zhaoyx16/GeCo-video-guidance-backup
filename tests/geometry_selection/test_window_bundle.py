from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from geometry_selection.cache import canonical_hash
from geometry_selection.schema import GeometryPrediction
from geometry_selection.window_bundle import (
    WINDOW_EXTRACTION_MODE,
    independent_run_id,
    make_window_bundle,
    validate_geometry_extraction_config,
    validate_global_prediction_bundle,
    validate_global_prediction_keyframes,
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
    files = [str((frame + 4) % 10) * 64 for frame in frames]
    run = independent_run_id(
        video_sha256="c" * 64,
        window_id=identifier,
        kind=kind,
        frame_indices=frames,
        frame_pixels_sha256=pixels,
        frame_file_sha256=files,
        geometry_backbone=geometry,
        producer=producer,
    )
    appearance = None
    if kind == "loop":
        appearance = {
            "source_frame": frames[0],
            "target_frame": frames[-1],
            "source_file_sha256": files[0],
            "target_file_sha256": files[-1],
            "source_keypoints": 30,
            "target_keypoints": 32,
            "ratio_matches": 20,
            "inliers": 15,
            "inlier_ratio": 0.75,
            "spatial_coverage": 0.25,
            "mean_descriptor_distance": 0.20,
            "status": "ok",
            "algorithm": "orb-mutual-ratio-fundamental-ransac-v1",
        }
    return {
        "window_id": identifier,
        "kind": kind,
        "frame_indices": list(frames),
        "frame_pixels_sha256": pixels,
        "frame_file_sha256": files,
        "appearance_evidence": appearance,
        "geometry_cache_key": cache_digit * 64,
        "independent_run_id": run,
    }, geometry, producer


def _prediction(record, *, keyframes=None, file_hashes=None):
    frames = record["frame_indices"] if keyframes is None else list(keyframes)
    hashes = record["frame_file_sha256"] if file_hashes is None else list(file_hashes)
    count = len(frames)
    world_to_camera = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    intrinsics = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    return GeometryPrediction(
        world_to_camera=world_to_camera,
        intrinsics=intrinsics,
        depth=np.ones((count, 2, 2)),
        confidence=np.ones((count, 2, 2)),
        keyframe_indices=np.asarray(frames, dtype=np.int64),
        metadata={
            "inputs": [
                {"path": f"/frames/{frame:04d}.png", "sha256": digest}
                for frame, digest in zip(frames, hashes)
            ]
        },
    )


def test_window_bundle_binds_schedule_and_independent_runs() -> None:
    first, _, _ = _record("local-00", "local", (0, 1, 2, 3), "1")
    second, _, _ = _record("local-01", "local", (2, 3, 4, 5), "2")
    loop, _, _ = _record("loop-00", "loop", (0, 1, 4, 5), "3")
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
    with pytest.raises(ValueError, match="sorted and unique|configured ordered schedule"):
        validate_window_bundle(tampered)


def test_local_only_extraction_allows_zero_loop_windows() -> None:
    config = {**_config(), "max_loop_windows": 0}
    validate_geometry_extraction_config(config)
    first = _record("local-00", "local", (0, 1, 2, 3), "1")[0]
    second = _record("local-01", "local", (2, 3, 4, 5), "2")[0]
    bundle = make_window_bundle(
        video_sha256="c" * 64,
        global_geometry_cache_key="d" * 64,
        global_keyframe_indices=range(6),
        extraction_config=config,
        window_records=(first, second),
    )
    validate_window_bundle(bundle, expected_video_sha256="c" * 64)
    assert [window["kind"] for window in bundle["windows"]] == ["local", "local"]


@pytest.mark.parametrize("tamper", ["order", "identifier", "kind", "frames"])
def test_window_bundle_rejects_schedule_tampering(tamper: str) -> None:
    records = [
        _record("local-00", "local", (0, 1, 2, 3), "1")[0],
        _record("local-01", "local", (2, 3, 4, 5), "2")[0],
        _record("loop-00", "loop", (0, 1, 4, 5), "3")[0],
    ]
    bundle = make_window_bundle(
        video_sha256="c" * 64,
        global_geometry_cache_key="d" * 64,
        global_keyframe_indices=range(6),
        extraction_config=_config(),
        window_records=records,
    )
    tampered = deepcopy(bundle)
    if tamper == "order":
        tampered["windows"][0], tampered["windows"][1] = (
            tampered["windows"][1],
            tampered["windows"][0],
        )
    elif tamper == "identifier":
        tampered["windows"][0]["window_id"] = "local-99"
    elif tamper == "kind":
        tampered["windows"][0]["kind"] = "loop"
    else:
        tampered["windows"][0]["frame_indices"] = [0, 1, 2, 4]
    with pytest.raises(
        ValueError,
        match=(
            "configured ordered schedule|appearance evidence|"
            "inconsistent file hashes"
        ),
    ):
        validate_window_bundle(tampered)


def test_window_bundle_rejects_duplicate_input_frame_sets() -> None:
    records = [
        _record("local-00", "local", (0, 1, 2, 3), "1")[0],
        _record("local-01", "local", (2, 3, 4, 5), "2")[0],
        _record("loop-00", "loop", (0, 1, 4, 5), "3")[0],
    ]
    bundle = make_window_bundle(
        video_sha256="c" * 64,
        global_geometry_cache_key="d" * 64,
        global_keyframe_indices=range(6),
        extraction_config=_config(),
        window_records=records,
    )
    tampered = deepcopy(bundle)
    tampered["windows"][1]["frame_indices"] = list(
        tampered["windows"][0]["frame_indices"]
    )
    with pytest.raises(ValueError, match="input frame sets must be unique"):
        validate_window_bundle(tampered)


def test_window_bundle_rejects_appearance_bound_to_wrong_frame_file() -> None:
    records = [
        _record("local-00", "local", (0, 1, 2, 3), "1")[0],
        _record("local-01", "local", (2, 3, 4, 5), "2")[0],
        _record("loop-00", "loop", (0, 1, 4, 5), "3")[0],
    ]
    bundle = make_window_bundle(
        video_sha256="c" * 64,
        global_geometry_cache_key="d" * 64,
        global_keyframe_indices=range(6),
        extraction_config=_config(),
        window_records=records,
    )
    tampered = deepcopy(bundle)
    tampered["windows"][2]["appearance_evidence"]["target_file_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="wrong frame files"):
        validate_window_bundle(tampered)


def test_window_cache_must_match_content_derived_run_id() -> None:
    record, geometry, producer = _record("local-00", "local", (0, 1, 2, 3), "1")
    prediction = _prediction(record)
    provenance = {
        "video_sha256": "c" * 64,
        "window_id": record["window_id"],
        "window_kind": record["kind"],
        "keyframe_indices": record["frame_indices"],
        "frame_pixels_sha256": record["frame_pixels_sha256"],
        "frame_file_sha256": record["frame_file_sha256"],
        "independent_run_id": record["independent_run_id"],
        "geometry_backbone": geometry,
        "producer": producer,
    }
    metadata = {
        "cache_key": record["geometry_cache_key"],
        "provenance": provenance,
    }
    validate_window_cache_record(
        record, metadata, prediction, video_sha256="c" * 64
    )
    metadata["provenance"]["independent_run_id"] = canonical_hash({"fake": True})
    with pytest.raises(ValueError, match="provenance mismatch|content-derived"):
        validate_window_cache_record(
            record, metadata, prediction, video_sha256="c" * 64
        )


def test_window_cache_rejects_prediction_keyframe_tampering() -> None:
    record, geometry, producer = _record("local-00", "local", (0, 1, 2, 3), "1")
    metadata = _cache_metadata(record, geometry, producer)
    prediction = _prediction(record, keyframes=(0, 1, 2, 4))
    with pytest.raises(ValueError, match="prediction keyframes differ"):
        validate_window_cache_record(
            record, metadata, prediction, video_sha256="c" * 64
        )


def test_window_cache_rejects_prediction_input_file_tampering() -> None:
    record, geometry, producer = _record("local-00", "local", (0, 1, 2, 3), "1")
    metadata = _cache_metadata(record, geometry, producer)
    tampered_hashes = list(record["frame_file_sha256"])
    tampered_hashes[1] = "f" * 64
    prediction = _prediction(record, file_hashes=tampered_hashes)
    with pytest.raises(ValueError, match="input file hashes differ"):
        validate_window_cache_record(
            record, metadata, prediction, video_sha256="c" * 64
        )


def test_independent_run_id_binds_ordered_input_file_hashes() -> None:
    record, geometry, producer = _record("local-00", "local", (0, 1, 2, 3), "1")
    reordered = list(record["frame_file_sha256"])
    reordered[0], reordered[1] = reordered[1], reordered[0]
    recomputed = independent_run_id(
        video_sha256="c" * 64,
        window_id=record["window_id"],
        kind=record["kind"],
        frame_indices=record["frame_indices"],
        frame_pixels_sha256=record["frame_pixels_sha256"],
        frame_file_sha256=reordered,
        geometry_backbone=geometry,
        producer=producer,
    )
    assert recomputed != record["independent_run_id"]


def test_global_prediction_keyframes_must_match_bundle() -> None:
    record, _, _ = _record("local-00", "local", (0, 1, 2, 3), "1")
    prediction = _prediction(record)
    validate_global_prediction_keyframes(prediction, (0, 1, 2, 3))
    with pytest.raises(ValueError, match="global prediction keyframes differ"):
        validate_global_prediction_keyframes(prediction, (0, 1, 2, 4))


def test_global_prediction_inputs_must_match_window_bundle() -> None:
    records = [
        _record("local-00", "local", (0, 1, 2, 3), "1")[0],
        _record("local-01", "local", (2, 3, 4, 5), "2")[0],
        _record("loop-00", "loop", (0, 1, 4, 5), "3")[0],
    ]
    # Shared decoded frames must have one content identity across all windows.
    canonical_files = {frame: str((frame + 4) % 10) * 64 for frame in range(6)}
    for record in records:
        record["frame_file_sha256"] = [
            canonical_files[frame] for frame in record["frame_indices"]
        ]
        if record["appearance_evidence"] is not None:
            record["appearance_evidence"]["source_file_sha256"] = canonical_files[0]
            record["appearance_evidence"]["target_file_sha256"] = canonical_files[5]
    bundle = make_window_bundle(
        video_sha256="c" * 64,
        global_geometry_cache_key="d" * 64,
        global_keyframe_indices=range(6),
        extraction_config=_config(),
        window_records=records,
    )
    template = records[0]
    global_prediction = _prediction(
        template,
        keyframes=range(6),
        file_hashes=[canonical_files[frame] for frame in range(6)],
    )
    validate_global_prediction_bundle(global_prediction, bundle)
    global_prediction.metadata["inputs"][3]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="global prediction input file hashes"):
        validate_global_prediction_bundle(global_prediction, bundle)


def _cache_metadata(record, geometry, producer):
    provenance = {
        "video_sha256": "c" * 64,
        "window_id": record["window_id"],
        "window_kind": record["kind"],
        "keyframe_indices": record["frame_indices"],
        "frame_pixels_sha256": record["frame_pixels_sha256"],
        "frame_file_sha256": record["frame_file_sha256"],
        "independent_run_id": record["independent_run_id"],
        "geometry_backbone": geometry,
        "producer": producer,
    }
    return {"cache_key": record["geometry_cache_key"], "provenance": provenance}
