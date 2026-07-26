"""Versioned manifests for controlled navigation generation experiments.

The protocol deliberately keeps all condition-defining inputs inside the
condition mapping. Methods in a paired comparison may differ only in method,
output and execution. This makes a condition hash sufficient to catch
accidental changes to anchors, prompts, seeds, models or sampling settings.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "navigation-benchmark/v1"
GENERATION_RUN_RECORD_TYPE = "generation_run"
METRIC_RESULT_RECORD_TYPE = "metric_result"

_SPLITS = {"train", "dev", "test"}
_RUN_STATUSES = {"planned", "completed", "failed"}
_METRIC_DIRECTIONS = {"lower_is_better", "higher_is_better"}
_METRIC_ROLES = {
    "guidance_aligned",
    "independent_geometry",
    "trajectory_adherence",
    "motion_preservation",
    "visual_quality",
}
_ANCHOR_ROLES = {"first", "middle", "last"}
_GUIDANCE_SCHEDULE_STATES = {
    "baseline_all_zero_schedule",
    "positive_schedule_zero_lr",
    "active_guidance",
}


@dataclass(frozen=True)
class ValidationIssue:
    """One actionable manifest validation finding."""

    record_id: str | None
    field: str
    message: str

    def as_dict(self) -> dict[str, str | None]:
        return {
            "record_id": self.record_id,
            "field": self.field,
            "message": self.message,
        }

    def __str__(self) -> str:
        prefix = self.record_id or "<unknown record>"
        return f"{prefix}: {self.field}: {self.message}"


class ManifestValidationError(ValueError):
    """Raised when a manifest or metric result cannot support a fair comparison."""

    def __init__(self, issues: Sequence[ValidationIssue]):
        self.issues = list(issues)
        super().__init__("\n".join(str(issue) for issue in self.issues))


def canonical_json(value: Any) -> str:
    """Serialize data deterministically for reproducible condition fingerprints."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def condition_fingerprint(record: Mapping[str, Any]) -> str:
    """Return a SHA-256 hash of the immutable paired-condition fields."""

    condition = record.get("condition")
    if not isinstance(condition, Mapping):
        raise ValueError("generation record has no mapping-valued condition")
    return hashlib.sha256(canonical_json(condition).encode("utf-8")).hexdigest()


def with_condition_hash(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a generation record and attach its canonical condition hash."""

    copied = copy.deepcopy(dict(record))
    copied["condition_hash"] = condition_fingerprint(copied)
    return copied


def record_fingerprint(record: Mapping[str, Any]) -> str:
    """Return a SHA-256 hash of a completed run record excluding record_hash."""

    copied = copy.deepcopy(dict(record))
    copied.pop("record_hash", None)
    return hashlib.sha256(canonical_json(copied).encode("utf-8")).hexdigest()


def with_record_hash(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a record and attach a tamper-evident fingerprint for final storage."""

    copied = copy.deepcopy(dict(record))
    copied["record_hash"] = record_fingerprint(copied)
    return copied


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL records or a JSON array/object containing a records list."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        loaded = json.loads(text)
        if isinstance(loaded, Mapping):
            loaded = loaded.get("records")
        if not isinstance(loaded, list):
            raise ValueError(f"{source}: JSON manifest must be a list or contain a 'records' list")
        return [_require_mapping(item, None, f"{source}") for item in loaded]

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: invalid JSONL: {exc.msg}") from exc
        records.append(_require_mapping(loaded, None, f"{source}:{line_number}"))
    return records


def write_records(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write records as JSONL, or as a JSON array when the suffix is .json."""

    destination = Path(path)
    serializable = [dict(record) for record in records]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".json":
        destination.write_text(
            json.dumps(serializable, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    destination.write_text(
        "".join(canonical_json(record) + "\n" for record in serializable),
        encoding="utf-8",
    )


def validate_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_methods: Sequence[str] = (),
    require_static_scene: bool = False,
) -> list[ValidationIssue]:
    """Validate run records and paired-condition invariants.

    The optional expected_methods lets a manifest be built incrementally, while
    a final benchmark check can require all planned arms.
    """

    issues: list[ValidationIssue] = []
    run_ids: set[str] = set()
    generation_records: list[Mapping[str, Any]] = []

    for index, record in enumerate(records):
        record_id = _record_id(record, fallback=f"record[{index}]")
        record_type = record.get("record_type")
        if record_type == GENERATION_RUN_RECORD_TYPE:
            generation_records.append(record)
            issues.extend(validate_generation_run(record, require_static_scene=require_static_scene))
            run_id = record.get("run_id")
            if isinstance(run_id, str) and run_id:
                if run_id in run_ids:
                    issues.append(ValidationIssue(record_id, "run_id", "duplicate run_id"))
                run_ids.add(run_id)
        elif record_type == METRIC_RESULT_RECORD_TYPE:
            issues.extend(validate_metric_result(record))
        else:
            issues.append(
                ValidationIssue(
                    record_id,
                    "record_type",
                    f"must be '{GENERATION_RUN_RECORD_TYPE}' or '{METRIC_RESULT_RECORD_TYPE}'",
                )
            )

    issues.extend(validate_paired_conditions(generation_records, expected_methods=expected_methods))
    return issues


def validate_generation_run(
    record: Mapping[str, Any],
    *,
    require_static_scene: bool = False,
) -> list[ValidationIssue]:
    """Validate one generation record without comparing it to another method."""

    issues: list[ValidationIssue] = []
    record_id = _record_id(record)
    _require_equal(record, "schema_version", SCHEMA_VERSION, record_id, issues)
    _require_equal(record, "record_type", GENERATION_RUN_RECORD_TYPE, record_id, issues)
    _require_nonempty_string(record, "run_id", record_id, issues)
    _require_nonempty_string(record, "pair_id", record_id, issues)

    status = record.get("status")
    if status not in _RUN_STATUSES:
        issues.append(
            ValidationIssue(record_id, "status", f"must be one of {sorted(_RUN_STATUSES)}")
        )

    method = _require_mapping_field(record, "method", record_id, issues)
    if method is not None:
        _require_nonempty_string(method, "name", record_id, issues, prefix="method.")
        _require_nonempty_string(method, "version", record_id, issues, prefix="method.")
        _require_mapping_field(method, "parameters", record_id, issues, prefix="method.")
        mechanism = _require_mapping_field(method, "mechanism", record_id, issues, prefix="method.")
        if mechanism is not None:
            _require_nonempty_string(
                mechanism, "id", record_id, issues, prefix="method.mechanism."
            )
            _require_nonempty_string(
                mechanism, "version", record_id, issues, prefix="method.mechanism."
            )
            _require_nonempty_string(
                mechanism, "time_travel", record_id, issues, prefix="method.mechanism."
            )
            _require_nonempty_string(
                mechanism,
                "temporal_vae_context",
                record_id,
                issues,
                prefix="method.mechanism.",
            )
            schedule_state = mechanism.get("guidance_schedule_state")
            if schedule_state not in _GUIDANCE_SCHEDULE_STATES:
                issues.append(
                    ValidationIssue(
                        record_id,
                        "method.mechanism.guidance_schedule_state",
                        f"must be one of {sorted(_GUIDANCE_SCHEDULE_STATES)}",
                    )
                )

    condition = _require_mapping_field(record, "condition", record_id, issues)
    if condition is not None:
        _validate_condition(condition, record_id, issues, require_static_scene=require_static_scene)
        declared_hash = record.get("condition_hash")
        if not isinstance(declared_hash, str) or len(declared_hash) != 64:
            issues.append(
                ValidationIssue(
                    record_id,
                    "condition_hash",
                    "must be a 64-character SHA-256 hash of the normalized condition",
                )
            )
        elif declared_hash != condition_fingerprint(record):
            issues.append(
                ValidationIssue(
                    record_id,
                    "condition_hash",
                    "does not match the canonical condition fingerprint",
                )
            )

    output = _require_mapping_field(record, "output", record_id, issues)
    if output is not None:
        _require_nonempty_string(output, "video_uri", record_id, issues, prefix="output.")
        _validate_optional_sha256(output.get("sha256"), record_id, "output.sha256", issues)

    execution = record.get("execution")
    if status == "completed":
        execution_mapping = _require_mapping(execution, record_id, "execution", issues)
        if execution_mapping is not None:
            _require_nonempty_string(execution_mapping, "git_commit", record_id, issues, prefix="execution.")
            _require_nonnegative_number(
                execution_mapping, "runtime_sec", record_id, issues, prefix="execution."
            )
            _validate_device_assignments(execution_mapping.get("devices"), record_id, issues)
            _validate_peak_vram(execution_mapping.get("peak_vram_mib"), record_id, issues)
        declared_record_hash = record.get("record_hash")
        if not isinstance(declared_record_hash, str) or not _is_sha256(declared_record_hash):
            issues.append(
                ValidationIssue(
                    record_id,
                    "record_hash",
                    "completed records must carry a 64-character immutable record fingerprint",
                )
            )
        elif declared_record_hash != record_fingerprint(record):
            issues.append(
                ValidationIssue(
                    record_id,
                    "record_hash",
                    "does not match the full run record fingerprint",
                )
            )
    elif execution is not None and not isinstance(execution, Mapping):
        issues.append(ValidationIssue(record_id, "execution", "must be an object or null"))

    return issues


def validate_metric_result(record: Mapping[str, Any]) -> list[ValidationIssue]:
    """Validate a scalar metric record independently of GPU metric execution."""

    issues: list[ValidationIssue] = []
    record_id = _record_id(record)
    _require_equal(record, "schema_version", SCHEMA_VERSION, record_id, issues)
    _require_equal(record, "record_type", METRIC_RESULT_RECORD_TYPE, record_id, issues)
    _require_nonempty_string(record, "run_id", record_id, issues)
    _require_nonempty_string(record, "metric_name", record_id, issues)

    role = record.get("metric_role")
    if role not in _METRIC_ROLES:
        issues.append(
            ValidationIssue(record_id, "metric_role", f"must be one of {sorted(_METRIC_ROLES)}")
        )
    direction = record.get("direction")
    if direction not in _METRIC_DIRECTIONS:
        issues.append(
            ValidationIssue(record_id, "direction", f"must be one of {sorted(_METRIC_DIRECTIONS)}")
        )

    value = record.get("value")
    if not _is_finite_number(value):
        issues.append(ValidationIssue(record_id, "value", "must be a finite number"))

    evaluator = _require_mapping_field(record, "evaluator", record_id, issues)
    if evaluator is not None:
        _require_nonempty_string(evaluator, "name", record_id, issues, prefix="evaluator.")
        _require_nonempty_string(evaluator, "version", record_id, issues, prefix="evaluator.")
        _require_nonempty_string(evaluator, "model_id", record_id, issues, prefix="evaluator.")
        _require_nonempty_string(
            evaluator, "checkpoint_revision", record_id, issues, prefix="evaluator."
        )
        _require_mapping_field(evaluator, "config", record_id, issues, prefix="evaluator.")

    return issues


def validate_paired_conditions(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_methods: Sequence[str] = (),
) -> list[ValidationIssue]:
    """Ensure methods assigned to a pair share exactly the same condition."""

    issues: list[ValidationIssue] = []
    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        pair_id = record.get("pair_id")
        if isinstance(pair_id, str) and pair_id:
            by_pair[pair_id].append(record)

    expected = set(expected_methods)
    for pair_id, pair_records in sorted(by_pair.items()):
        methods: dict[str, Mapping[str, Any]] = {}
        hashes: set[str] = set()
        for record in pair_records:
            record_id = _record_id(record)
            method = record.get("method")
            method_name = method.get("name") if isinstance(method, Mapping) else None
            if not isinstance(method_name, str) or not method_name:
                continue
            if method_name in methods:
                issues.append(
                    ValidationIssue(
                        record_id,
                        "method.name",
                        f"duplicate method '{method_name}' in pair '{pair_id}'",
                    )
                )
            methods[method_name] = record
            try:
                hashes.add(condition_fingerprint(record))
            except ValueError:
                pass

        if len(hashes) > 1:
            details = ", ".join(sorted(hashes))
            issues.append(
                ValidationIssue(
                    pair_id,
                    "condition",
                    "paired methods do not share identical non-method settings "
                    f"(condition hashes: {details})",
                )
            )

        if expected and set(methods) != expected:
            missing = sorted(expected - set(methods))
            unexpected = sorted(set(methods) - expected)
            parts: list[str] = []
            if missing:
                parts.append(f"missing {missing}")
            if unexpected:
                parts.append(f"unexpected {unexpected}")
            issues.append(
                ValidationIssue(pair_id, "methods", "; ".join(parts) or "method set mismatch")
            )
    return issues


def _validate_condition(
    condition: Mapping[str, Any],
    record_id: str | None,
    issues: list[ValidationIssue],
    *,
    require_static_scene: bool,
) -> None:
    protocol = _require_mapping_field(condition, "protocol", record_id, issues, prefix="condition.")
    if protocol is not None:
        _require_nonempty_string(protocol, "id", record_id, issues, prefix="condition.protocol.")
        _require_nonempty_string(protocol, "version", record_id, issues, prefix="condition.protocol.")

    scene = _require_mapping_field(condition, "scene", record_id, issues, prefix="condition.")
    if scene is not None:
        _require_nonempty_string(scene, "scene_id", record_id, issues, prefix="condition.scene.")
        _require_nonempty_string(scene, "dataset_id", record_id, issues, prefix="condition.scene.")
        split = scene.get("split")
        if split not in _SPLITS:
            issues.append(
                ValidationIssue(
                    record_id,
                    "condition.scene.split",
                    f"must be one of {sorted(_SPLITS)}",
                )
            )
        eligibility = _require_mapping_field(
            scene,
            "static_scene_eligibility",
            record_id,
            issues,
            prefix="condition.scene.",
        )
        if eligibility is not None:
            if not isinstance(eligibility.get("eligible"), bool):
                issues.append(
                    ValidationIssue(
                        record_id,
                        "condition.scene.static_scene_eligibility.eligible",
                        "must be a boolean",
                    )
                )
            elif require_static_scene and not eligibility["eligible"]:
                issues.append(
                    ValidationIssue(
                        record_id,
                        "condition.scene.static_scene_eligibility.eligible",
                        "must be true when --require-static-scene is used",
                    )
                )
            _require_nonempty_string(
                eligibility,
                "criteria_version",
                record_id,
                issues,
                prefix="condition.scene.static_scene_eligibility.",
            )
            _require_nonempty_string(
                eligibility,
                "rationale",
                record_id,
                issues,
                prefix="condition.scene.static_scene_eligibility.",
            )

    source = _require_mapping_field(condition, "source_clip", record_id, issues, prefix="condition.")
    if source is not None:
        for field in ("source_uri", "sequence_id", "clip_id"):
            _require_nonempty_string(source, field, record_id, issues, prefix="condition.source_clip.")
        _require_sha256(source.get("source_sha256"), record_id, "condition.source_clip.source_sha256", issues)
        start = _require_integer(source, "start_frame", record_id, issues, prefix="condition.source_clip.")
        end = _require_integer(source, "end_frame", record_id, issues, prefix="condition.source_clip.")
        fps = _require_positive_number(source, "source_fps", record_id, issues, prefix="condition.source_clip.")
        time_origin = source.get("time_origin")
        if time_origin not in {"clip_relative", "source_absolute"}:
            issues.append(
                ValidationIssue(
                    record_id,
                    "condition.source_clip.time_origin",
                    "must be 'clip_relative' or 'source_absolute'",
                )
            )
        if start is not None and end is not None and start > end:
            issues.append(
                ValidationIssue(
                    record_id,
                    "condition.source_clip",
                    "start_frame must be less than or equal to end_frame",
                )
            )
        _validate_reference(source.get("intrinsics_ref"), record_id, "condition.source_clip.intrinsics_ref", issues)
        _validate_reference(source.get("poses_ref"), record_id, "condition.source_clip.poses_ref", issues)
        _validate_anchors(source.get("anchors"), record_id, issues, start, end, fps, time_origin)

    _require_nonempty_string(condition, "prompt", record_id, issues, prefix="condition.")
    _require_integer(condition, "seed", record_id, issues, prefix="condition.")

    model = _require_mapping_field(condition, "model", record_id, issues, prefix="condition.")
    if model is not None:
        _require_nonempty_string(model, "model_id", record_id, issues, prefix="condition.model.")
        _require_nonempty_string(model, "checkpoint_revision", record_id, issues, prefix="condition.model.")

    sampling = _require_mapping_field(condition, "sampling", record_id, issues, prefix="condition.")
    if sampling is not None:
        for field in ("height", "width", "num_frames", "num_inference_steps"):
            value = _require_integer(sampling, field, record_id, issues, prefix="condition.sampling.")
            if value is not None and value <= 0:
                issues.append(
                    ValidationIssue(record_id, f"condition.sampling.{field}", "must be greater than zero")
                )
        fps = _require_positive_number(sampling, "fps", record_id, issues, prefix="condition.sampling.")
        scheduler = _require_mapping_field(
            sampling, "scheduler", record_id, issues, prefix="condition.sampling."
        )
        if scheduler is not None:
            _require_nonempty_string(
                scheduler, "name", record_id, issues, prefix="condition.sampling.scheduler."
            )
        if fps is not None and fps <= 0:
            issues.append(ValidationIssue(record_id, "condition.sampling.fps", "must be greater than zero"))

    frame_guidance = _require_mapping_field(condition, "frame_guidance", record_id, issues, prefix="condition.")
    if frame_guidance is not None:
        if not isinstance(frame_guidance.get("enabled"), bool):
            issues.append(
                ValidationIssue(record_id, "condition.frame_guidance.enabled", "must be a boolean")
            )
        anchor_roles = frame_guidance.get("anchor_roles")
        if not isinstance(anchor_roles, list) or set(anchor_roles) != _ANCHOR_ROLES:
            issues.append(
                ValidationIssue(
                    record_id,
                    "condition.frame_guidance.anchor_roles",
                    "must contain exactly ['first', 'middle', 'last']",
                )
            )


def _validate_anchors(
    anchors: Any,
    record_id: str | None,
    issues: list[ValidationIssue],
    start: int | None,
    end: int | None,
    source_fps: float | None,
    time_origin: Any,
) -> None:
    if not isinstance(anchors, list) or len(anchors) != 3:
        issues.append(
            ValidationIssue(record_id, "condition.source_clip.anchors", "must be a list of three anchors")
        )
        return
    seen_roles: set[str] = set()
    last_frame: int | None = None
    last_timestamp: float | None = None
    for index, anchor in enumerate(anchors):
        path = f"condition.source_clip.anchors[{index}]"
        if not isinstance(anchor, Mapping):
            issues.append(ValidationIssue(record_id, path, "must be an object"))
            continue
        role = anchor.get("role")
        if role not in _ANCHOR_ROLES:
            issues.append(ValidationIssue(record_id, f"{path}.role", "must be first, middle or last"))
        elif role in seen_roles:
            issues.append(ValidationIssue(record_id, f"{path}.role", f"duplicate role '{role}'"))
        else:
            seen_roles.add(role)
        frame_index = _require_integer(anchor, "frame_index", record_id, issues, prefix=f"{path}.")
        timestamp = _require_nonnegative_number(anchor, "timestamp_sec", record_id, issues, prefix=f"{path}.")
        _require_nonempty_string(anchor, "frame_uri", record_id, issues, prefix=f"{path}.")
        _require_sha256(anchor.get("sha256"), record_id, f"{path}.sha256", issues)
        if frame_index is not None:
            if start is not None and frame_index < start:
                issues.append(ValidationIssue(record_id, f"{path}.frame_index", "precedes start_frame"))
            if end is not None and frame_index > end:
                issues.append(ValidationIssue(record_id, f"{path}.frame_index", "exceeds end_frame"))
            if last_frame is not None and frame_index <= last_frame:
                issues.append(
                    ValidationIssue(record_id, f"{path}.frame_index", "anchors must be strictly increasing")
                )
            last_frame = frame_index
        if timestamp is not None:
            if last_timestamp is not None and timestamp <= last_timestamp:
                issues.append(
                    ValidationIssue(
                        record_id, f"{path}.timestamp_sec", "anchors must be strictly increasing"
                    )
                )
            if frame_index is not None and source_fps is not None:
                origin_frame = start if time_origin == "clip_relative" and start is not None else 0
                expected = (frame_index - origin_frame) / source_fps
                if abs(timestamp - expected) > max(1e-6, 0.51 / source_fps):
                    issues.append(
                        ValidationIssue(
                            record_id,
                            f"{path}.timestamp_sec",
                            "is inconsistent with frame_index/source_fps; document a nonzero clip time offset in source_uri if intended",
                        )
                    )
            last_timestamp = timestamp
    if seen_roles != _ANCHOR_ROLES:
        issues.append(
            ValidationIssue(
                record_id,
                "condition.source_clip.anchors",
                "roles must cover first, middle and last exactly once",
            )
        )


def _validate_reference(value: Any, record_id: str | None, field: str, issues: list[ValidationIssue]) -> None:
    reference = _require_mapping(value, record_id, field, issues)
    if reference is None:
        return
    _require_nonempty_string(reference, "uri", record_id, issues, prefix=f"{field}.")
    _require_nonempty_string(reference, "format", record_id, issues, prefix=f"{field}.")
    _require_sha256(reference.get("sha256"), record_id, f"{field}.sha256", issues)


def _validate_peak_vram(value: Any, record_id: str | None, issues: list[ValidationIssue]) -> None:
    if not isinstance(value, Mapping) or not value:
        issues.append(
            ValidationIssue(
                record_id,
                "execution.peak_vram_mib",
                "must be a non-empty device mapping with allocated and reserved values",
            )
        )
        return
    for device, stats in value.items():
        if not isinstance(device, str) or not device:
            issues.append(
                ValidationIssue(record_id, "execution.peak_vram_mib", "device keys must be non-empty strings")
            )
            continue
        if not isinstance(stats, Mapping):
            issues.append(
                ValidationIssue(
                    record_id,
                    f"execution.peak_vram_mib.{device}",
                    "must contain allocated and reserved values",
                )
            )
            continue
        for key in ("allocated", "reserved"):
            if not _is_nonnegative_number(stats.get(key)):
                issues.append(
                    ValidationIssue(
                        record_id,
                        f"execution.peak_vram_mib.{device}.{key}",
                        "must be a finite non-negative number",
                    )
                )


def _validate_device_assignments(value: Any, record_id: str | None, issues: list[ValidationIssue]) -> None:
    if not isinstance(value, Mapping) or not value:
        issues.append(
            ValidationIssue(
                record_id,
                "execution.devices",
                "must be a non-empty role-to-device mapping",
            )
        )
        return
    for role, device in value.items():
        if not isinstance(role, str) or not role:
            issues.append(ValidationIssue(record_id, "execution.devices", "roles must be non-empty strings"))
        if not isinstance(device, str) or not device:
            issues.append(
                ValidationIssue(
                    record_id,
                    f"execution.devices.{role}",
                    "device must be a non-empty string",
                )
            )


def _require_sha256(
    value: Any,
    record_id: str | None,
    field: str,
    issues: list[ValidationIssue],
) -> None:
    if not _is_sha256(value):
        issues.append(ValidationIssue(record_id, field, "must be a 64-character SHA-256 hex string"))


def _validate_optional_sha256(
    value: Any,
    record_id: str | None,
    field: str,
    issues: list[ValidationIssue],
) -> None:
    if value is not None:
        _require_sha256(value, record_id, field, issues)


def _require_mapping_field(
    container: Mapping[str, Any],
    key: str,
    record_id: str | None,
    issues: list[ValidationIssue],
    *,
    prefix: str = "",
) -> Mapping[str, Any] | None:
    return _require_mapping(container.get(key), record_id, f"{prefix}{key}", issues)


def _require_mapping(
    value: Any,
    record_id: str | None,
    field: str,
    issues: list[ValidationIssue] | None = None,
) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if issues is not None:
        issues.append(ValidationIssue(record_id, field, "must be an object"))
        return None
    raise ValueError(f"{field}: expected an object")


def _require_nonempty_string(
    container: Mapping[str, Any],
    key: str,
    record_id: str | None,
    issues: list[ValidationIssue],
    *,
    prefix: str = "",
) -> None:
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        issues.append(ValidationIssue(record_id, f"{prefix}{key}", "must be a non-empty string"))


def _require_equal(
    container: Mapping[str, Any],
    key: str,
    expected: Any,
    record_id: str | None,
    issues: list[ValidationIssue],
) -> None:
    if container.get(key) != expected:
        issues.append(ValidationIssue(record_id, key, f"must equal {expected!r}"))


def _require_integer(
    container: Mapping[str, Any],
    key: str,
    record_id: str | None,
    issues: list[ValidationIssue],
    *,
    prefix: str = "",
) -> int | None:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        issues.append(ValidationIssue(record_id, f"{prefix}{key}", "must be an integer"))
        return None
    return value


def _require_positive_number(
    container: Mapping[str, Any],
    key: str,
    record_id: str | None,
    issues: list[ValidationIssue],
    *,
    prefix: str = "",
) -> float | None:
    value = container.get(key)
    if not _is_finite_number(value) or float(value) <= 0:
        issues.append(
            ValidationIssue(record_id, f"{prefix}{key}", "must be a finite number greater than zero")
        )
        return None
    return float(value)


def _require_nonnegative_number(
    container: Mapping[str, Any],
    key: str,
    record_id: str | None,
    issues: list[ValidationIssue],
    *,
    prefix: str = "",
) -> float | None:
    value = container.get(key)
    if not _is_nonnegative_number(value):
        issues.append(
            ValidationIssue(record_id, f"{prefix}{key}", "must be a finite non-negative number")
        )
        return None
    return float(value)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_nonnegative_number(value: Any) -> bool:
    return _is_finite_number(value) and float(value) >= 0


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _record_id(record: Mapping[str, Any], fallback: str | None = None) -> str | None:
    run_id = record.get("run_id")
    if isinstance(run_id, str) and run_id:
        return run_id
    return fallback
