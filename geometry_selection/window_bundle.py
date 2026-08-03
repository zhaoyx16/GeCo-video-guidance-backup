"""Immutable manifests for separately estimated geometry windows."""

from __future__ import annotations

from typing import Any, Sequence

from .appearance import AppearanceEvidence
from .cache import canonical_hash
from .schema import GeometryPrediction
from .window_graph import make_window_schedule


WINDOW_BUNDLE_SCHEMA = "geometry-window-bundle-v2"
WINDOW_EXTRACTION_MODE = "independent-window-pose-graph-v1"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expected_window_schedule(
    keyframe_indices: Sequence[int],
    extraction_config: dict[str, Any],
) -> list[dict[str, Any]]:
    schedule = make_window_schedule(
        keyframe_indices,
        local_window_size=extraction_config["local_window_size"],
        local_stride=extraction_config["local_stride"],
        loop_context=extraction_config["loop_context"],
        min_loop_node_gap=extraction_config["min_loop_node_gap"],
        max_loop_windows=extraction_config["max_loop_windows"],
    )
    return [
        {
            "window_id": window_id,
            "kind": kind,
            "frame_indices": list(frame_indices),
        }
        for window_id, kind, frame_indices in schedule
    ]


def validate_geometry_extraction_config(config: dict[str, Any]) -> None:
    expected = {
        "mode",
        "num_keyframes",
        "local_window_size",
        "local_stride",
        "loop_context",
        "min_loop_node_gap",
        "max_loop_windows",
    }
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError(
            f"geometry_extraction keys must be exactly {sorted(expected)}"
        )
    if config["mode"] != WINDOW_EXTRACTION_MODE:
        raise ValueError(f"unsupported geometry extraction mode: {config['mode']}")
    integer_fields = expected - {"mode"}
    for name in integer_fields:
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"geometry_extraction.{name} must be a non-negative integer")
    if config["num_keyframes"] < 4:
        raise ValueError("geometry extraction requires at least four keyframes")
    if config["local_window_size"] < 2:
        raise ValueError("local_window_size must be at least two")
    if not 1 <= config["local_stride"] < config["local_window_size"]:
        raise ValueError("local_stride must produce overlapping local windows")
    if config["loop_context"] < 2:
        raise ValueError("loop_context must be at least two")
    if config["min_loop_node_gap"] < 2:
        raise ValueError("min_loop_node_gap must be at least two")
    if config["max_loop_windows"] < 1:
        raise ValueError("at least one loop window is required")


def independent_run_id(
    *,
    video_sha256: str,
    window_id: str,
    kind: str,
    frame_indices: Sequence[int],
    frame_pixels_sha256: Sequence[str],
    frame_file_sha256: Sequence[str],
    geometry_backbone: dict[str, Any],
    producer: dict[str, Any],
) -> str:
    return canonical_hash(
        {
            "logical_inference": "separate-vggt-omega-window-forward-v2",
            "video_sha256": video_sha256,
            "window_id": window_id,
            "kind": kind,
            "frame_indices": list(frame_indices),
            "frame_pixels_sha256": list(frame_pixels_sha256),
            "frame_file_sha256": list(frame_file_sha256),
            "geometry_backbone": geometry_backbone,
            "producer": producer,
        }
    )


def make_window_bundle(
    *,
    video_sha256: str,
    global_geometry_cache_key: str,
    global_keyframe_indices: Sequence[int],
    extraction_config: dict[str, Any],
    window_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    validate_geometry_extraction_config(extraction_config)
    schedule = _expected_window_schedule(global_keyframe_indices, extraction_config)
    bundle = {
        "schema": WINDOW_BUNDLE_SCHEMA,
        "video_sha256": video_sha256,
        "global_geometry_cache_key": global_geometry_cache_key,
        "global_keyframe_indices": list(global_keyframe_indices),
        "geometry_extraction": dict(extraction_config),
        "schedule_sha256": canonical_hash({"windows": schedule}),
        "windows": [dict(record) for record in window_records],
    }
    validate_window_bundle(bundle)
    return bundle


def validate_window_bundle(
    bundle: dict[str, Any],
    *,
    expected_video_sha256: str | None = None,
    expected_global_cache_key: str | None = None,
) -> None:
    if not isinstance(bundle, dict) or bundle.get("schema") != WINDOW_BUNDLE_SCHEMA:
        raise ValueError(f"window bundle schema must be {WINDOW_BUNDLE_SCHEMA}")
    required = {
        "schema",
        "video_sha256",
        "global_geometry_cache_key",
        "global_keyframe_indices",
        "geometry_extraction",
        "schedule_sha256",
        "windows",
    }
    if set(bundle) != required:
        raise ValueError("window bundle fields are incomplete or unknown")
    for name in ("video_sha256", "global_geometry_cache_key", "schedule_sha256"):
        value = bundle[name]
        if not _is_sha256(value):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if expected_video_sha256 is not None and bundle["video_sha256"] != expected_video_sha256:
        raise ValueError("window bundle video digest differs from candidate")
    if (
        expected_global_cache_key is not None
        and bundle["global_geometry_cache_key"] != expected_global_cache_key
    ):
        raise ValueError("window bundle global cache key differs from candidate")
    keyframes = bundle["global_keyframe_indices"]
    if (
        not isinstance(keyframes, list)
        or len(keyframes) < 4
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in keyframes
        )
        or keyframes != sorted(set(keyframes))
    ):
        raise ValueError("global_keyframe_indices must be sorted unique integers")
    validate_geometry_extraction_config(bundle["geometry_extraction"])
    if bundle["geometry_extraction"]["num_keyframes"] != len(keyframes):
        raise ValueError("geometry extraction keyframe count differs from bundle")
    windows = bundle["windows"]
    if not isinstance(windows, list) or len(windows) < 2:
        raise ValueError("window bundle must contain multiple independent windows")
    ids: set[str] = set()
    runs: set[str] = set()
    cache_keys: set[str] = set()
    frame_sets: set[tuple[int, ...]] = set()
    frame_files_by_index: dict[int, str] = {}
    actual_schedule = []
    local_coverage: set[int] = set()
    loop_count = 0
    for record in windows:
        expected_record = {
            "window_id",
            "kind",
            "frame_indices",
            "frame_pixels_sha256",
            "frame_file_sha256",
            "appearance_evidence",
            "geometry_cache_key",
            "independent_run_id",
        }
        if not isinstance(record, dict) or set(record) != expected_record:
            raise ValueError("window record fields are incomplete or unknown")
        identifier = record["window_id"]
        kind = record["kind"]
        frames = record["frame_indices"]
        pixels = record["frame_pixels_sha256"]
        files = record["frame_file_sha256"]
        appearance_payload = record["appearance_evidence"]
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("window IDs must be unique non-empty strings")
        if kind not in {"local", "loop"}:
            raise ValueError("window kind must be local or loop")
        if not isinstance(frames, list) or len(frames) < 2 or frames != sorted(set(frames)):
            raise ValueError("window frame indices must be sorted and unique")
        if not set(frames).issubset(keyframes):
            raise ValueError("window references a frame outside global keyframes")
        frame_set = tuple(frames)
        if frame_set in frame_sets:
            raise ValueError("window input frame sets must be unique")
        if not isinstance(pixels, list) or len(pixels) != len(frames):
            raise ValueError("window pixel hashes must align with frame indices")
        if not isinstance(files, list) or len(files) != len(frames):
            raise ValueError("window file hashes must align with frame indices")
        for frame, file_digest in zip(frames, files, strict=True):
            previous = frame_files_by_index.setdefault(frame, file_digest)
            if previous != file_digest:
                raise ValueError(
                    "the same decoded frame has inconsistent file hashes across windows"
                )
        if kind == "local":
            if appearance_payload is not None:
                raise ValueError("local windows must not contain appearance evidence")
        else:
            if not isinstance(appearance_payload, dict):
                raise ValueError("loop windows must contain appearance evidence")
            appearance = AppearanceEvidence.from_dict(appearance_payload)
            if (
                appearance.source_frame != frames[0]
                or appearance.target_frame != frames[-1]
            ):
                raise ValueError("loop appearance evidence binds the wrong frame pair")
            if (
                appearance.source_file_sha256 != files[0]
                or appearance.target_file_sha256 != files[-1]
            ):
                raise ValueError("loop appearance evidence binds the wrong frame files")
        digests = [
            record["geometry_cache_key"],
            record["independent_run_id"],
            *pixels,
            *files,
        ]
        if any(not _is_sha256(value) for value in digests):
            raise ValueError(
                "window cache/run/pixel/file identities must be SHA-256 digests"
            )
        if record["independent_run_id"] in runs or record["geometry_cache_key"] in cache_keys:
            raise ValueError("window run IDs and cache keys must be unique")
        ids.add(identifier)
        runs.add(record["independent_run_id"])
        cache_keys.add(record["geometry_cache_key"])
        frame_sets.add(frame_set)
        if kind == "local":
            local_coverage.update(frames)
        else:
            loop_count += 1
        actual_schedule.append(
            {"window_id": identifier, "kind": kind, "frame_indices": frames}
        )
    expected_schedule = _expected_window_schedule(
        keyframes, bundle["geometry_extraction"]
    )
    if actual_schedule != expected_schedule:
        raise ValueError(
            "window records do not exactly match the configured ordered schedule"
        )
    expected_schedule_sha256 = canonical_hash({"windows": expected_schedule})
    if expected_schedule_sha256 != bundle["schedule_sha256"]:
        raise ValueError("window schedule digest mismatch")
    if local_coverage != set(keyframes):
        raise ValueError("local windows must cover every global keyframe")
    if loop_count < 1:
        raise ValueError("window bundle must contain at least one loop window")


def validate_global_prediction_keyframes(
    prediction: GeometryPrediction,
    global_keyframe_indices: Sequence[int],
) -> None:
    prediction.validate()
    expected = [int(value) for value in global_keyframe_indices]
    actual = prediction.keyframe_indices.tolist()
    if actual != expected:
        raise ValueError(
            f"global prediction keyframes differ from bundle: {actual} != {expected}"
        )


def validate_global_prediction_bundle(
    prediction: GeometryPrediction,
    bundle: dict[str, Any],
) -> None:
    """Bind a global cache prediction to the same decoded files as its windows."""

    validate_global_prediction_keyframes(
        prediction, bundle["global_keyframe_indices"]
    )
    files_by_frame: dict[int, str] = {}
    for record in bundle["windows"]:
        for frame, digest in zip(
            record["frame_indices"], record["frame_file_sha256"], strict=True
        ):
            previous = files_by_frame.setdefault(frame, digest)
            if previous != digest:
                raise ValueError("window records disagree on a decoded frame file hash")
    expected = [files_by_frame[frame] for frame in bundle["global_keyframe_indices"]]
    actual = _prediction_input_sha256(prediction)
    if actual != expected:
        raise ValueError("global prediction input file hashes differ from window bundle")


def _prediction_input_sha256(prediction: GeometryPrediction) -> list[str]:
    inputs = prediction.metadata.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("window prediction metadata.inputs must be an ordered list")
    digests: list[str] = []
    for input_record in inputs:
        if not isinstance(input_record, dict) or not _is_sha256(input_record.get("sha256")):
            raise ValueError(
                "window prediction metadata.inputs must contain valid sha256 fields"
            )
        digests.append(input_record["sha256"])
    return digests


def validate_window_cache_record(
    record: dict[str, Any],
    cache_metadata: dict[str, Any],
    prediction: GeometryPrediction,
    *,
    video_sha256: str,
) -> None:
    prediction.validate()
    expected_keyframes = list(record["frame_indices"])
    actual_keyframes = prediction.keyframe_indices.tolist()
    if actual_keyframes != expected_keyframes:
        raise ValueError(
            "window prediction keyframes differ from bundle record: "
            f"{actual_keyframes} != {expected_keyframes}"
        )
    input_sha256 = _prediction_input_sha256(prediction)
    if input_sha256 != record["frame_file_sha256"]:
        raise ValueError(
            "window prediction input file hashes differ from bundle record"
        )
    provenance = cache_metadata.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("window cache has no provenance mapping")
    expected = {
        "video_sha256": video_sha256,
        "window_id": record["window_id"],
        "window_kind": record["kind"],
        "keyframe_indices": record["frame_indices"],
        "frame_pixels_sha256": record["frame_pixels_sha256"],
        "frame_file_sha256": record["frame_file_sha256"],
        "independent_run_id": record["independent_run_id"],
    }
    mismatches = {
        key: (provenance.get(key), value)
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise ValueError(f"window cache provenance mismatch: {mismatches}")
    if cache_metadata.get("cache_key") != record["geometry_cache_key"]:
        raise ValueError("window cache key differs from bundle record")
    recomputed_run_id = independent_run_id(
        video_sha256=video_sha256,
        window_id=record["window_id"],
        kind=record["kind"],
        frame_indices=record["frame_indices"],
        frame_pixels_sha256=record["frame_pixels_sha256"],
        frame_file_sha256=record["frame_file_sha256"],
        geometry_backbone=provenance.get("geometry_backbone", {}),
        producer=provenance.get("producer", {}),
    )
    if recomputed_run_id != record["independent_run_id"]:
        raise ValueError("window independent_run_id is not content-derived")
