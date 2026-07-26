"""Strict, versioned manifests for controlled navigation experiments.

The module is intentionally protocol-only.  It does not run a generator,
estimate poses, or choose a guidance schedule.  Instead it makes the inputs
and outputs of such runs immutable enough that a paired comparison can be
audited later.
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
SPLIT_MANIFEST_RECORD_TYPE = "split_manifest"
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
_EVALUATOR_POLICIES = {"independent", "guidance_aligned", "human_annotation"}
_POSE_CONVENTIONS = {"W2C", "C2W"}
_STATISTICAL_UNIT_LEVELS = {"scene", "sequence"}
_ANCHOR_ROLES = ("first", "middle", "last")
_GUIDANCE_SCHEDULE_STATES = {
    "baseline_all_zero_schedule",
    "positive_schedule_zero_lr",
    "active_guidance",
}
_ANCHOR_POLICY_ID = "first_middle_last_floor_v1"
_ANCHOR_MAPPING_POLICY_ID = "first_middle_last_index_v1"


@dataclass(frozen=True)
class ValidationIssue:
    """One actionable manifest-validation finding."""

    record_id: str | None
    field: str
    message: str

    def as_dict(self) -> dict[str, str | None]:
        return {"record_id": self.record_id, "field": self.field, "message": self.message}

    def __str__(self) -> str:
        return f"{self.record_id or '<unknown record>'}: {self.field}: {self.message}"


class ManifestValidationError(ValueError):
    """Raised when records cannot support the requested fair comparison."""

    def __init__(self, issues: Sequence[ValidationIssue]):
        self.issues = list(issues)
        super().__init__("\n".join(str(issue) for issue in self.issues))


def canonical_json(value: Any) -> str:
    """Serialize data deterministically for fingerprints and config equality."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return the SHA-256 of a canonical JSON value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def condition_fingerprint(record: Mapping[str, Any]) -> str:
    """Return the immutable paired-condition fingerprint for a run record."""

    condition = record.get("condition")
    if not isinstance(condition, Mapping):
        raise ValueError("generation record has no mapping-valued condition")
    return sha256_json(condition)


def with_condition_hash(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a generation record and attach its canonical condition hash."""

    copied = copy.deepcopy(dict(record))
    copied["condition_hash"] = condition_fingerprint(copied)
    return copied


def split_manifest_fingerprint(record: Mapping[str, Any]) -> str:
    """Return the immutable fingerprint of a frozen split-manifest record."""

    copied = copy.deepcopy(dict(record))
    copied.pop("split_manifest_hash", None)
    return sha256_json(copied)


def with_split_manifest_hash(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a split manifest and attach its frozen-manifest fingerprint."""

    copied = copy.deepcopy(dict(record))
    copied["split_manifest_hash"] = split_manifest_fingerprint(copied)
    return copied


def record_fingerprint(record: Mapping[str, Any]) -> str:
    """Return a SHA-256 hash of a completed run record excluding record_hash."""

    copied = copy.deepcopy(dict(record))
    copied.pop("record_hash", None)
    return sha256_json(copied)


def with_record_hash(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a completed run record and attach its immutable record fingerprint."""

    copied = copy.deepcopy(dict(record))
    copied["record_hash"] = record_fingerprint(copied)
    return copied


def evaluator_fingerprint(evaluator: Mapping[str, Any]) -> str:
    """Fingerprint an evaluator identity, policy, checkpoint, and config."""

    copied = copy.deepcopy(dict(evaluator))
    copied.pop("fingerprint", None)
    return sha256_json(copied)


def with_evaluator_fingerprint(evaluator: Mapping[str, Any]) -> dict[str, Any]:
    """Copy an evaluator descriptor and attach its canonical fingerprint."""

    copied = copy.deepcopy(dict(evaluator))
    copied["fingerprint"] = evaluator_fingerprint(copied)
    return copied


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL, a JSON array, or a JSON object containing a records list."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        loaded = json.loads(text)
        if isinstance(loaded, Mapping):
            loaded = loaded.get("records")
        if not isinstance(loaded, list):
            raise ValueError(f"{source}: JSON manifest must be a list or contain a records list")
        records: list[dict[str, Any]] = []
        for index, item in enumerate(loaded):
            issues: list[ValidationIssue] = []
            mapping = _require_mapping(item, None, f"{source}[{index}]", issues)
            if mapping is None:
                raise ValueError(str(issues[0]))
            records.append(mapping)
        return records

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: invalid JSONL: {exc.msg}") from exc
        issues: list[ValidationIssue] = []
        mapping = _require_mapping(loaded, None, f"{source}:{line_number}", issues)
        if mapping is None:
            raise ValueError(str(issues[0]))
        records.append(mapping)
    return records


def write_records(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write records as JSONL, or as a JSON array when the suffix is .json."""

    destination = Path(path)
    serializable = [dict(record) for record in records]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".json":
        destination.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    destination.write_text("".join(canonical_json(record) + "\n" for record in serializable), encoding="utf-8")


def validate_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_methods: Sequence[str] = (),
    require_static_scene: bool = False,
    require_completed: bool = False,
) -> list[ValidationIssue]:
    """Validate a whole benchmark manifest, including cross-record contracts.

    A generation record is accepted only when it references a frozen split
    manifest in the same manifest.  Final aggregation should set
    require_completed and expected_methods to make absent or failed arms hard
    errors rather than silently omitted observations.
    """

    issues: list[ValidationIssue] = []
    split_records: list[Mapping[str, Any]] = []
    generation_records: list[Mapping[str, Any]] = []
    metric_records: list[Mapping[str, Any]] = []
    run_ids: set[str] = set()

    for index, record in enumerate(records):
        record_id = _record_id(record, fallback=f"record[{index}]")
        record_type = record.get("record_type")
        if record_type == SPLIT_MANIFEST_RECORD_TYPE:
            split_records.append(record)
            issues.extend(validate_split_manifest(record))
        elif record_type == GENERATION_RUN_RECORD_TYPE:
            generation_records.append(record)
            run_id = record.get("run_id")
            if isinstance(run_id, str) and run_id:
                if run_id in run_ids:
                    issues.append(ValidationIssue(record_id, "run_id", "duplicate run_id"))
                run_ids.add(run_id)
        elif record_type == METRIC_RESULT_RECORD_TYPE:
            metric_records.append(record)
            issues.extend(validate_metric_result(record))
        else:
            issues.append(
                ValidationIssue(
                    record_id,
                    "record_type",
                    "must be split_manifest, generation_run, or metric_result",
                )
            )

    split_index = _build_split_index(split_records, issues)
    for record in generation_records:
        issues.extend(
            validate_generation_run(
                record,
                require_static_scene=require_static_scene,
                split_index=split_index,
            )
        )
    issues.extend(
        validate_paired_conditions(
            generation_records,
            expected_methods=expected_methods,
            require_completed=require_completed,
        )
    )
    issues.extend(_validate_metric_bindings(metric_records, generation_records))
    return issues


def validate_split_manifest(record: Mapping[str, Any]) -> list[ValidationIssue]:
    """Validate one immutable scene/sequence-disjoint frozen split manifest."""

    issues: list[ValidationIssue] = []
    record_id = _record_id(record)
    _require_equal(record, "schema_version", SCHEMA_VERSION, record_id, issues)
    _require_equal(record, "record_type", SPLIT_MANIFEST_RECORD_TYPE, record_id, issues)
    _require_nonempty_string(record, "split_manifest_id", record_id, issues)
    _require_equal(record, "status", "frozen", record_id, issues)
    declared = record.get("split_manifest_hash")
    if not _is_sha256(declared):
        issues.append(ValidationIssue(record_id, "split_manifest_hash", "must be a 64-character SHA-256 hash"))
    elif declared != split_manifest_fingerprint(record):
        issues.append(ValidationIssue(record_id, "split_manifest_hash", "does not match frozen split-manifest content"))

    assignments = record.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        issues.append(ValidationIssue(record_id, "assignments", "must be a non-empty list"))
        return issues

    seen_scenes: set[tuple[str, str]] = set()
    seen_sequences: set[tuple[str, str]] = set()
    for index, assignment in enumerate(assignments):
        field = f"assignments[{index}]"
        mapping = _require_mapping(assignment, record_id, field, issues)
        if mapping is None:
            continue
        dataset_id = _required_string(mapping, "dataset_id", record_id, issues, prefix=f"{field}.")
        scene_id = _required_string(mapping, "scene_id", record_id, issues, prefix=f"{field}.")
        sequence_id = _required_string(mapping, "sequence_id", record_id, issues, prefix=f"{field}.")
        split = mapping.get("split")
        if split not in _SPLITS:
            issues.append(ValidationIssue(record_id, f"{field}.split", f"must be one of {sorted(_SPLITS)}"))
        if dataset_id and scene_id:
            scene_key = (dataset_id, scene_id)
            if scene_key in seen_scenes:
                issues.append(ValidationIssue(record_id, field, "duplicate scene assignment"))
            seen_scenes.add(scene_key)
        if dataset_id and sequence_id:
            sequence_key = (dataset_id, sequence_id)
            if sequence_key in seen_sequences:
                issues.append(ValidationIssue(record_id, field, "duplicate sequence assignment"))
            seen_sequences.add(sequence_key)
    _validate_extensions(record, record_id, issues)
    return issues


def validate_generation_run(
    record: Mapping[str, Any],
    *,
    require_static_scene: bool = False,
    split_index: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[ValidationIssue]:
    """Validate one generation record without comparing it to a paired arm."""

    issues: list[ValidationIssue] = []
    record_id = _record_id(record)
    _require_equal(record, "schema_version", SCHEMA_VERSION, record_id, issues)
    _require_equal(record, "record_type", GENERATION_RUN_RECORD_TYPE, record_id, issues)
    _require_nonempty_string(record, "run_id", record_id, issues)
    _require_nonempty_string(record, "pair_id", record_id, issues)

    status = record.get("status")
    if status not in _RUN_STATUSES:
        issues.append(ValidationIssue(record_id, "status", f"must be one of {sorted(_RUN_STATUSES)}"))

    method = _require_mapping_field(record, "method", record_id, issues)
    if method is not None:
        _required_string(method, "name", record_id, issues, prefix="method.")
        _required_string(method, "version", record_id, issues, prefix="method.")
        parameters = _require_mapping_field(method, "parameters", record_id, issues, prefix="method.")
        if parameters is not None:
            _validate_extensions(parameters, record_id, issues, field="method.parameters.extensions")
        mechanism = _require_mapping_field(method, "mechanism", record_id, issues, prefix="method.")
        if mechanism is not None:
            _required_string(mechanism, "id", record_id, issues, prefix="method.mechanism.")
            _required_string(mechanism, "version", record_id, issues, prefix="method.mechanism.")
            _required_string(mechanism, "time_travel", record_id, issues, prefix="method.mechanism.")
            _required_string(mechanism, "temporal_vae_context", record_id, issues, prefix="method.mechanism.")
            if mechanism.get("guidance_schedule_state") not in _GUIDANCE_SCHEDULE_STATES:
                issues.append(
                    ValidationIssue(
                        record_id,
                        "method.mechanism.guidance_schedule_state",
                        f"must be one of {sorted(_GUIDANCE_SCHEDULE_STATES)}",
                    )
                )

    condition = _require_mapping_field(record, "condition", record_id, issues)
    if condition is not None:
        _validate_condition(
            condition,
            record_id,
            issues,
            require_static_scene=require_static_scene,
            split_index=split_index,
        )
        declared_hash = record.get("condition_hash")
        if not _is_sha256(declared_hash):
            issues.append(ValidationIssue(record_id, "condition_hash", "must be a 64-character SHA-256 hash"))
        elif declared_hash != condition_fingerprint(record):
            issues.append(ValidationIssue(record_id, "condition_hash", "does not match normalized condition"))

    output = _require_mapping_field(record, "output", record_id, issues)
    if output is not None:
        _required_string(output, "video_uri", record_id, issues, prefix="output.")
        if status == "completed":
            _require_sha256(output.get("sha256"), record_id, "output.sha256", issues)
        elif output.get("sha256") is not None:
            _require_sha256(output.get("sha256"), record_id, "output.sha256", issues)

    execution = record.get("execution")
    if status == "completed":
        execution_mapping = _require_mapping(execution, record_id, "execution", issues)
        if execution_mapping is not None:
            _required_string(execution_mapping, "git_commit", record_id, issues, prefix="execution.")
            _require_nonnegative_number(execution_mapping, "runtime_sec", record_id, issues, prefix="execution.")
            _validate_device_assignments(execution_mapping.get("devices"), record_id, issues)
            _validate_peak_vram(execution_mapping.get("peak_vram_mib"), record_id, issues)
        declared_record_hash = record.get("record_hash")
        if not _is_sha256(declared_record_hash):
            issues.append(ValidationIssue(record_id, "record_hash", "completed runs require an immutable 64-character record hash"))
        elif declared_record_hash != record_fingerprint(record):
            issues.append(ValidationIssue(record_id, "record_hash", "does not match full completed run content"))
    elif execution is not None and not isinstance(execution, Mapping):
        issues.append(ValidationIssue(record_id, "execution", "must be an object or null"))
    _validate_extensions(record, record_id, issues)
    return issues


def validate_metric_result(record: Mapping[str, Any]) -> list[ValidationIssue]:
    """Validate one scalar metric record before binding it to a completed run."""

    issues: list[ValidationIssue] = []
    record_id = _record_id(record)
    _require_equal(record, "schema_version", SCHEMA_VERSION, record_id, issues)
    _require_equal(record, "record_type", METRIC_RESULT_RECORD_TYPE, record_id, issues)
    _required_string(record, "run_id", record_id, issues)
    _required_string(record, "metric_name", record_id, issues)
    if record.get("metric_role") not in _METRIC_ROLES:
        issues.append(ValidationIssue(record_id, "metric_role", f"must be one of {sorted(_METRIC_ROLES)}"))
    if record.get("direction") not in _METRIC_DIRECTIONS:
        issues.append(ValidationIssue(record_id, "direction", f"must be one of {sorted(_METRIC_DIRECTIONS)}"))
    if not _is_finite_number(record.get("value")):
        issues.append(ValidationIssue(record_id, "value", "must be a finite number"))
    _require_sha256(record.get("run_record_hash"), record_id, "run_record_hash", issues)
    _require_sha256(record.get("evaluated_output_sha256"), record_id, "evaluated_output_sha256", issues)
    _validate_artifact(record.get("metric_artifact"), record_id, "metric_artifact", issues)

    evaluator = _require_mapping_field(record, "evaluator", record_id, issues)
    if evaluator is not None:
        _required_string(evaluator, "name", record_id, issues, prefix="evaluator.")
        _required_string(evaluator, "version", record_id, issues, prefix="evaluator.")
        _required_string(evaluator, "model_id", record_id, issues, prefix="evaluator.")
        _required_string(evaluator, "checkpoint_revision", record_id, issues, prefix="evaluator.")
        _require_mapping_field(evaluator, "config", record_id, issues, prefix="evaluator.")
        if evaluator.get("independence_policy") not in _EVALUATOR_POLICIES:
            issues.append(ValidationIssue(record_id, "evaluator.independence_policy", f"must be one of {sorted(_EVALUATOR_POLICIES)}"))
        fingerprint = evaluator.get("fingerprint")
        if not _is_sha256(fingerprint):
            issues.append(ValidationIssue(record_id, "evaluator.fingerprint", "must be a 64-character SHA-256 hash"))
        elif fingerprint != evaluator_fingerprint(evaluator):
            issues.append(ValidationIssue(record_id, "evaluator.fingerprint", "does not match evaluator identity, policy, and config"))

    if record.get("metric_role") == "trajectory_adherence":
        _validate_trajectory_binding(record.get("trajectory_binding"), record_id, issues)
    _validate_extensions(record, record_id, issues)
    return issues


def validate_paired_conditions(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_methods: Sequence[str] = (),
    require_completed: bool = False,
) -> list[ValidationIssue]:
    """Require every arm of a pair to share one immutable condition mapping."""

    issues: list[ValidationIssue] = []
    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        pair_id = record.get("pair_id")
        if isinstance(pair_id, str) and pair_id:
            by_pair[pair_id].append(record)

    expected = set(expected_methods)
    if len(expected) != len(tuple(expected_methods)):
        issues.append(ValidationIssue(None, "expected_methods", "must not contain duplicates"))
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
                issues.append(ValidationIssue(record_id, "method.name", f"duplicate method '{method_name}' in pair '{pair_id}'"))
            methods[method_name] = record
            try:
                hashes.add(condition_fingerprint(record))
            except ValueError:
                pass

        if len(hashes) != 1:
            issues.append(ValidationIssue(pair_id, "condition", "all paired methods must share one identical normalized condition"))
        if expected and set(methods) != expected:
            missing = sorted(expected - set(methods))
            unexpected = sorted(set(methods) - expected)
            details = []
            if missing:
                details.append(f"missing {missing}")
            if unexpected:
                details.append(f"unexpected {unexpected}")
            issues.append(ValidationIssue(pair_id, "methods", "; ".join(details)))
        if require_completed:
            required_names = expected or set(methods)
            for method_name in required_names:
                record = methods.get(method_name)
                if record is None:
                    continue
                if record.get("status") != "completed":
                    issues.append(ValidationIssue(_record_id(record), "status", "final paired aggregation requires every expected arm to be completed"))
    return issues


def _build_split_index(
    records: Sequence[Mapping[str, Any]],
    issues: list[ValidationIssue],
) -> dict[str, Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    global_scene_owner: dict[tuple[str, str], tuple[str, str]] = {}
    global_sequence_owner: dict[tuple[str, str], tuple[str, str]] = {}
    for record in records:
        record_id = _record_id(record)
        manifest_id = record.get("split_manifest_id")
        if not isinstance(manifest_id, str) or not manifest_id:
            continue
        if manifest_id in by_id:
            issues.append(ValidationIssue(record_id, "split_manifest_id", "duplicate frozen split-manifest id"))
            continue
        by_id[manifest_id] = record
        assignments = record.get("assignments")
        if not isinstance(assignments, list):
            continue
        for index, assignment in enumerate(assignments):
            if not isinstance(assignment, Mapping):
                continue
            dataset_id = assignment.get("dataset_id")
            scene_id = assignment.get("scene_id")
            sequence_id = assignment.get("sequence_id")
            split = assignment.get("split")
            if not all(isinstance(value, str) and value for value in (dataset_id, scene_id, sequence_id, split)):
                continue
            owner = (str(split), manifest_id)
            scene_key = (str(dataset_id), str(scene_id))
            sequence_key = (str(dataset_id), str(sequence_id))
            _check_global_split_owner(global_scene_owner, scene_key, owner, record_id, f"assignments[{index}].scene_id", issues)
            _check_global_split_owner(global_sequence_owner, sequence_key, owner, record_id, f"assignments[{index}].sequence_id", issues)
    return by_id


def _check_global_split_owner(
    owners: dict[tuple[str, str], tuple[str, str]],
    key: tuple[str, str],
    owner: tuple[str, str],
    record_id: str | None,
    field: str,
    issues: list[ValidationIssue],
) -> None:
    previous = owners.get(key)
    if previous is not None:
        if previous[0] != owner[0]:
            issues.append(ValidationIssue(record_id, field, f"globally assigned to both '{previous[0]}' and '{owner[0]}' splits"))
        else:
            issues.append(ValidationIssue(record_id, field, "appears in more than one frozen split manifest"))
        return
    owners[key] = owner


def _validate_condition(
    condition: Mapping[str, Any],
    record_id: str | None,
    issues: list[ValidationIssue],
    *,
    require_static_scene: bool,
    split_index: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    protocol = _require_mapping_field(condition, "protocol", record_id, issues, prefix="condition.")
    if protocol is not None:
        _required_string(protocol, "id", record_id, issues, prefix="condition.protocol.")
        _required_string(protocol, "version", record_id, issues, prefix="condition.protocol.")

    scene = _require_mapping_field(condition, "scene", record_id, issues, prefix="condition.")
    dataset_id = scene_id = sequence_id = None
    if scene is not None:
        dataset_id = _required_string(scene, "dataset_id", record_id, issues, prefix="condition.scene.")
        scene_id = _required_string(scene, "scene_id", record_id, issues, prefix="condition.scene.")
        split = scene.get("split")
        if split not in _SPLITS:
            issues.append(ValidationIssue(record_id, "condition.scene.split", f"must be one of {sorted(_SPLITS)}"))
        statistical_unit = _require_mapping_field(scene, "statistical_unit", record_id, issues, prefix="condition.scene.")
        if statistical_unit is not None:
            _required_string(statistical_unit, "cluster_id", record_id, issues, prefix="condition.scene.statistical_unit.")
            if statistical_unit.get("level") not in _STATISTICAL_UNIT_LEVELS:
                issues.append(ValidationIssue(record_id, "condition.scene.statistical_unit.level", f"must be one of {sorted(_STATISTICAL_UNIT_LEVELS)}"))
        eligibility = _require_mapping_field(scene, "static_scene_eligibility", record_id, issues, prefix="condition.scene.")
        if eligibility is not None:
            if not isinstance(eligibility.get("eligible"), bool):
                issues.append(ValidationIssue(record_id, "condition.scene.static_scene_eligibility.eligible", "must be a boolean"))
            elif require_static_scene and not eligibility["eligible"]:
                issues.append(ValidationIssue(record_id, "condition.scene.static_scene_eligibility.eligible", "must be true when static-scene validation is requested"))
            _required_string(eligibility, "criteria_version", record_id, issues, prefix="condition.scene.static_scene_eligibility.")
            _required_string(eligibility, "rationale", record_id, issues, prefix="condition.scene.static_scene_eligibility.")

    source = _require_mapping_field(condition, "source_clip", record_id, issues, prefix="condition.")
    start = end = None
    source_fps = None
    anchors_by_role: dict[str, Mapping[str, Any]] = {}
    if source is not None:
        for field in ("source_uri", "clip_id"):
            _required_string(source, field, record_id, issues, prefix="condition.source_clip.")
        sequence_id = _required_string(source, "sequence_id", record_id, issues, prefix="condition.source_clip.")
        _require_sha256(source.get("source_sha256"), record_id, "condition.source_clip.source_sha256", issues)
        start = _require_integer(source, "start_frame", record_id, issues, prefix="condition.source_clip.")
        end = _require_integer(source, "end_frame", record_id, issues, prefix="condition.source_clip.")
        source_fps = _require_positive_number(source, "source_fps", record_id, issues, prefix="condition.source_clip.")
        time_origin = source.get("time_origin")
        if time_origin not in {"clip_relative", "source_absolute"}:
            issues.append(ValidationIssue(record_id, "condition.source_clip.time_origin", "must be clip_relative or source_absolute"))
        if start is not None and end is not None:
            if end <= start:
                issues.append(ValidationIssue(record_id, "condition.source_clip", "end_frame must be greater than start_frame"))
            elif end - start < 2:
                issues.append(ValidationIssue(record_id, "condition.source_clip", "clip must contain three distinct first/middle/last frames"))
        anchor_policy = _require_mapping_field(source, "anchor_policy", record_id, issues, prefix="condition.source_clip.")
        if anchor_policy is not None:
            _require_equal(anchor_policy, "id", _ANCHOR_POLICY_ID, record_id, issues, prefix="condition.source_clip.anchor_policy.")
            _require_equal(anchor_policy, "middle_rule", "floor_midpoint", record_id, issues, prefix="condition.source_clip.anchor_policy.")
        _validate_reference(source.get("intrinsics_ref"), record_id, "condition.source_clip.intrinsics_ref", issues)
        _validate_reference(source.get("poses_ref"), record_id, "condition.source_clip.poses_ref", issues, pose_reference=True)
        anchors_by_role = _validate_anchors(source.get("anchors"), record_id, issues, start, end, source_fps, time_origin)

    _required_string(condition, "prompt", record_id, issues, prefix="condition.")
    _require_integer(condition, "seed", record_id, issues, prefix="condition.")

    model = _require_mapping_field(condition, "model", record_id, issues, prefix="condition.")
    if model is not None:
        _required_string(model, "model_id", record_id, issues, prefix="condition.model.")
        _required_string(model, "checkpoint_revision", record_id, issues, prefix="condition.model.")
        _require_mapping_field(model, "config", record_id, issues, prefix="condition.model.")

    sampling = _require_mapping_field(condition, "sampling", record_id, issues, prefix="condition.")
    num_frames = generated_fps = None
    if sampling is not None:
        for field in ("height", "width", "num_frames", "num_inference_steps"):
            value = _require_integer(sampling, field, record_id, issues, prefix="condition.sampling.")
            if value is not None and value <= 0:
                issues.append(ValidationIssue(record_id, f"condition.sampling.{field}", "must be greater than zero"))
            if field == "num_frames":
                num_frames = value
        generated_fps = _require_positive_number(sampling, "fps", record_id, issues, prefix="condition.sampling.")
        scheduler = _require_mapping_field(sampling, "scheduler", record_id, issues, prefix="condition.sampling.")
        if scheduler is not None:
            _required_string(scheduler, "name", record_id, issues, prefix="condition.sampling.scheduler.")
            _require_mapping_field(scheduler, "config", record_id, issues, prefix="condition.sampling.scheduler.")

    frame_guidance = _require_mapping_field(condition, "frame_guidance", record_id, issues, prefix="condition.")
    if frame_guidance is not None:
        _validate_frame_guidance(
            frame_guidance,
            anchors_by_role,
            num_frames,
            generated_fps,
            record_id,
            issues,
        )

    _validate_split_reference(condition, scene, source, dataset_id, scene_id, sequence_id, record_id, issues, split_index)
    _validate_extensions(condition, record_id, issues, field="condition.extensions")


def _validate_split_reference(
    condition: Mapping[str, Any],
    scene: Mapping[str, Any] | None,
    source: Mapping[str, Any] | None,
    dataset_id: str | None,
    scene_id: str | None,
    sequence_id: str | None,
    record_id: str | None,
    issues: list[ValidationIssue],
    split_index: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    reference = _require_mapping_field(condition, "split_manifest", record_id, issues, prefix="condition.")
    if reference is None:
        return
    manifest_id = _required_string(reference, "id", record_id, issues, prefix="condition.split_manifest.")
    manifest_hash = reference.get("sha256")
    _require_sha256(manifest_hash, record_id, "condition.split_manifest.sha256", issues)
    if split_index is None or not manifest_id:
        return
    split_record = split_index.get(manifest_id)
    if split_record is None:
        issues.append(ValidationIssue(record_id, "condition.split_manifest.id", "does not resolve to a frozen split manifest in this manifest"))
        return
    if manifest_hash != split_record.get("split_manifest_hash"):
        issues.append(ValidationIssue(record_id, "condition.split_manifest.sha256", "does not match the referenced frozen split manifest"))
    if not all(isinstance(value, str) and value for value in (dataset_id, scene_id, sequence_id)):
        return
    assignment = _find_split_assignment(split_record, dataset_id, scene_id, sequence_id)
    if assignment is None:
        issues.append(ValidationIssue(record_id, "condition.split_manifest", "does not assign this dataset/scene/sequence tuple"))
        return
    if isinstance(scene, Mapping) and scene.get("split") != assignment.get("split"):
        issues.append(ValidationIssue(record_id, "condition.scene.split", "does not match the frozen split-manifest assignment"))


def _find_split_assignment(
    split_record: Mapping[str, Any], dataset_id: str, scene_id: str, sequence_id: str
) -> Mapping[str, Any] | None:
    assignments = split_record.get("assignments")
    if not isinstance(assignments, list):
        return None
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            continue
        if (
            assignment.get("dataset_id") == dataset_id
            and assignment.get("scene_id") == scene_id
            and assignment.get("sequence_id") == sequence_id
        ):
            return assignment
    return None


def _validate_anchors(
    anchors: Any,
    record_id: str | None,
    issues: list[ValidationIssue],
    start: int | None,
    end: int | None,
    source_fps: float | None,
    time_origin: Any,
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(anchors, list) or len(anchors) != 3:
        issues.append(ValidationIssue(record_id, "condition.source_clip.anchors", "must be exactly three ordered anchors"))
        return {}
    roles = [anchor.get("role") if isinstance(anchor, Mapping) else None for anchor in anchors]
    if roles != list(_ANCHOR_ROLES):
        issues.append(ValidationIssue(record_id, "condition.source_clip.anchors", "must be ordered exactly as first, middle, last"))
    by_role: dict[str, Mapping[str, Any]] = {}
    frame_indices: dict[str, int] = {}
    for index, anchor in enumerate(anchors):
        path = f"condition.source_clip.anchors[{index}]"
        if not isinstance(anchor, Mapping):
            issues.append(ValidationIssue(record_id, path, "must be an object"))
            continue
        role = anchor.get("role")
        if role not in _ANCHOR_ROLES:
            issues.append(ValidationIssue(record_id, f"{path}.role", "must be first, middle, or last"))
        elif role in by_role:
            issues.append(ValidationIssue(record_id, f"{path}.role", f"duplicate role '{role}'"))
        else:
            by_role[str(role)] = anchor
        frame_index = _require_integer(anchor, "frame_index", record_id, issues, prefix=f"{path}.")
        timestamp = _require_nonnegative_number(anchor, "timestamp_sec", record_id, issues, prefix=f"{path}.")
        _required_string(anchor, "frame_uri", record_id, issues, prefix=f"{path}.")
        _require_sha256(anchor.get("sha256"), record_id, f"{path}.sha256", issues)
        if isinstance(role, str) and frame_index is not None:
            frame_indices[role] = frame_index
        if frame_index is not None and timestamp is not None and source_fps is not None:
            origin = start if time_origin == "clip_relative" and start is not None else 0
            expected_timestamp = (frame_index - origin) / source_fps
            if abs(timestamp - expected_timestamp) > 1e-6:
                issues.append(ValidationIssue(record_id, f"{path}.timestamp_sec", "must equal the declared source-frame time exactly"))
    if set(by_role) != set(_ANCHOR_ROLES):
        return by_role
    if start is not None and frame_indices.get("first") != start:
        issues.append(ValidationIssue(record_id, "condition.source_clip.anchors[0].frame_index", "first anchor must equal start_frame"))
    if end is not None and frame_indices.get("last") != end:
        issues.append(ValidationIssue(record_id, "condition.source_clip.anchors[2].frame_index", "last anchor must equal end_frame"))
    if start is not None and end is not None:
        expected_middle = start + (end - start) // 2
        if frame_indices.get("middle") != expected_middle:
            issues.append(ValidationIssue(record_id, "condition.source_clip.anchors[1].frame_index", "middle anchor must use deterministic floor midpoint"))
    if all(role in frame_indices for role in _ANCHOR_ROLES):
        if not frame_indices["first"] < frame_indices["middle"] < frame_indices["last"]:
            issues.append(ValidationIssue(record_id, "condition.source_clip.anchors", "anchor frame indices must be strictly increasing"))
    return by_role


def _validate_frame_guidance(
    frame_guidance: Mapping[str, Any],
    source_anchors: Mapping[str, Mapping[str, Any]],
    num_frames: int | None,
    generated_fps: float | None,
    record_id: str | None,
    issues: list[ValidationIssue],
) -> None:
    if frame_guidance.get("enabled") is not True:
        issues.append(ValidationIssue(record_id, "condition.frame_guidance.enabled", "must be true for this frame-guidance protocol"))
    if frame_guidance.get("anchor_roles") != list(_ANCHOR_ROLES):
        issues.append(ValidationIssue(record_id, "condition.frame_guidance.anchor_roles", "must be ordered exactly as first, middle, last"))
    timing = _require_mapping_field(frame_guidance, "generated_timing", record_id, issues, prefix="condition.frame_guidance.")
    if timing is None:
        return
    _require_equal(timing, "mapping_policy", _ANCHOR_MAPPING_POLICY_ID, record_id, issues, prefix="condition.frame_guidance.generated_timing.")
    reported_count = _require_integer(timing, "generated_frame_count", record_id, issues, prefix="condition.frame_guidance.generated_timing.")
    reported_fps = _require_positive_number(timing, "generated_fps", record_id, issues, prefix="condition.frame_guidance.generated_timing.")
    if num_frames is not None and reported_count is not None and reported_count != num_frames:
        issues.append(ValidationIssue(record_id, "condition.frame_guidance.generated_timing.generated_frame_count", "must equal sampling.num_frames"))
    if generated_fps is not None and reported_fps is not None and abs(reported_fps - generated_fps) > 1e-9:
        issues.append(ValidationIssue(record_id, "condition.frame_guidance.generated_timing.generated_fps", "must equal sampling.fps"))
    anchor_map = timing.get("anchor_map")
    if not isinstance(anchor_map, list) or len(anchor_map) != 3:
        issues.append(ValidationIssue(record_id, "condition.frame_guidance.generated_timing.anchor_map", "must contain exactly three contractual mappings"))
        return
    roles = [entry.get("role") if isinstance(entry, Mapping) else None for entry in anchor_map]
    if roles != list(_ANCHOR_ROLES):
        issues.append(ValidationIssue(record_id, "condition.frame_guidance.generated_timing.anchor_map", "must be ordered exactly as first, middle, last"))
    expected_generated_indices = None
    if num_frames is not None and num_frames > 0:
        expected_generated_indices = {"first": 0, "middle": (num_frames - 1) // 2, "last": num_frames - 1}
    for index, entry in enumerate(anchor_map):
        path = f"condition.frame_guidance.generated_timing.anchor_map[{index}]"
        if not isinstance(entry, Mapping):
            issues.append(ValidationIssue(record_id, path, "must be an object"))
            continue
        role = entry.get("role")
        source_frame = _require_integer(entry, "source_frame_index", record_id, issues, prefix=f"{path}.")
        source_time = _require_nonnegative_number(entry, "source_timestamp_sec", record_id, issues, prefix=f"{path}.")
        generated_frame = _require_integer(entry, "generated_frame_index", record_id, issues, prefix=f"{path}.")
        generated_time = _require_nonnegative_number(entry, "generated_timestamp_sec", record_id, issues, prefix=f"{path}.")
        if role in source_anchors:
            source_anchor = source_anchors[str(role)]
            if source_frame != source_anchor.get("frame_index"):
                issues.append(ValidationIssue(record_id, f"{path}.source_frame_index", "must match the corresponding source anchor"))
            if source_time is not None and abs(source_time - float(source_anchor.get("timestamp_sec", -1))) > 1e-6:
                issues.append(ValidationIssue(record_id, f"{path}.source_timestamp_sec", "must match the corresponding source anchor"))
        if expected_generated_indices is not None and role in expected_generated_indices and generated_frame != expected_generated_indices[str(role)]:
            issues.append(ValidationIssue(record_id, f"{path}.generated_frame_index", "does not match the contractual first/middle/last generated frame"))
        if generated_frame is not None and generated_time is not None and reported_fps is not None:
            if abs(generated_time - generated_frame / reported_fps) > 1e-6:
                issues.append(ValidationIssue(record_id, f"{path}.generated_timestamp_sec", "must equal generated_frame_index/generated_fps"))


def _validate_reference(
    value: Any,
    record_id: str | None,
    field: str,
    issues: list[ValidationIssue],
    *,
    pose_reference: bool = False,
) -> None:
    reference = _require_mapping(value, record_id, field, issues)
    if reference is None:
        return
    _required_string(reference, "uri", record_id, issues, prefix=f"{field}.")
    _required_string(reference, "format", record_id, issues, prefix=f"{field}.")
    _require_sha256(reference.get("sha256"), record_id, f"{field}.sha256", issues)
    if pose_reference:
        if reference.get("pose_convention") not in _POSE_CONVENTIONS:
            issues.append(ValidationIssue(record_id, f"{field}.pose_convention", "must be W2C or C2W"))
        _required_string(reference, "translation_unit", record_id, issues, prefix=f"{field}.")


def _validate_artifact(value: Any, record_id: str | None, field: str, issues: list[ValidationIssue]) -> None:
    artifact = _require_mapping(value, record_id, field, issues)
    if artifact is None:
        return
    _required_string(artifact, "uri", record_id, issues, prefix=f"{field}.")
    _require_sha256(artifact.get("sha256"), record_id, f"{field}.sha256", issues)


def _validate_trajectory_binding(value: Any, record_id: str | None, issues: list[ValidationIssue]) -> None:
    binding = _require_mapping(value, record_id, "trajectory_binding", issues)
    if binding is None:
        return
    for field in ("reference_pose_artifact_sha256", "predicted_pose_artifact_sha256", "anchor_mapping_hash"):
        _require_sha256(binding.get(field), record_id, f"trajectory_binding.{field}", issues)
    if binding.get("pose_convention") not in _POSE_CONVENTIONS:
        issues.append(ValidationIssue(record_id, "trajectory_binding.pose_convention", "must be W2C or C2W"))
    _required_string(binding, "translation_unit", record_id, issues, prefix="trajectory_binding.")
    if binding.get("scale_alignment") not in {"none", "least_squares"}:
        issues.append(ValidationIssue(record_id, "trajectory_binding.scale_alignment", "must be none or least_squares"))


def _validate_metric_bindings(
    metric_records: Sequence[Mapping[str, Any]], generation_records: Sequence[Mapping[str, Any]]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    runs_by_id = {record.get("run_id"): record for record in generation_records if isinstance(record.get("run_id"), str)}
    for metric in metric_records:
        record_id = _record_id(metric)
        run_id = metric.get("run_id")
        run = runs_by_id.get(run_id)
        if run is None:
            issues.append(ValidationIssue(record_id, "run_id", "does not resolve to a generation run in this manifest"))
            continue
        if run.get("status") != "completed":
            issues.append(ValidationIssue(record_id, "run_id", "metric input must bind to a completed run"))
            continue
        if metric.get("run_record_hash") != run.get("record_hash"):
            issues.append(ValidationIssue(record_id, "run_record_hash", "does not match the completed run manifest"))
        output = run.get("output")
        expected_output_hash = output.get("sha256") if isinstance(output, Mapping) else None
        if metric.get("evaluated_output_sha256") != expected_output_hash:
            issues.append(ValidationIssue(record_id, "evaluated_output_sha256", "does not match the completed output artifact"))
    return issues


def _validate_device_assignments(value: Any, record_id: str | None, issues: list[ValidationIssue]) -> None:
    if not isinstance(value, Mapping) or not value:
        issues.append(ValidationIssue(record_id, "execution.devices", "must be a non-empty role-to-device mapping"))
        return
    for role, device in value.items():
        if not isinstance(role, str) or not role or not isinstance(device, str) or not device:
            issues.append(ValidationIssue(record_id, "execution.devices", "every role and device must be non-empty strings"))


def _validate_peak_vram(value: Any, record_id: str | None, issues: list[ValidationIssue]) -> None:
    if not isinstance(value, Mapping) or not value:
        issues.append(ValidationIssue(record_id, "execution.peak_vram_mib", "must be a non-empty per-device mapping"))
        return
    for device, measurements in value.items():
        mapping = _require_mapping(measurements, record_id, f"execution.peak_vram_mib.{device}", issues)
        if mapping is None:
            continue
        _require_nonnegative_number(mapping, "allocated", record_id, issues, prefix=f"execution.peak_vram_mib.{device}.")
        _require_nonnegative_number(mapping, "reserved", record_id, issues, prefix=f"execution.peak_vram_mib.{device}.")


def _validate_extensions(
    value: Mapping[str, Any], record_id: str | None, issues: list[ValidationIssue], *, field: str = "extensions"
) -> None:
    if "extensions" in value and not isinstance(value.get("extensions"), Mapping):
        issues.append(ValidationIssue(record_id, field, "must be an object when present"))


def _record_id(record: Mapping[str, Any], fallback: str | None = None) -> str | None:
    for key in ("run_id", "split_manifest_id", "metric_name"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return fallback


def _require_mapping(value: Any, record_id: str | None, field: str, issues: list[ValidationIssue]) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        issues.append(ValidationIssue(record_id, field, "must be an object"))
        return None
    return dict(value)


def _require_mapping_field(
    mapping: Mapping[str, Any], key: str, record_id: str | None, issues: list[ValidationIssue], *, prefix: str = ""
) -> dict[str, Any] | None:
    return _require_mapping(mapping.get(key), record_id, f"{prefix}{key}", issues)


def _required_string(
    mapping: Mapping[str, Any], key: str, record_id: str | None, issues: list[ValidationIssue], *, prefix: str = ""
) -> str | None:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        issues.append(ValidationIssue(record_id, f"{prefix}{key}", "must be a non-empty string"))
        return None
    return value


def _require_nonempty_string(mapping: Mapping[str, Any], key: str, record_id: str | None, issues: list[ValidationIssue], *, prefix: str = "") -> None:
    _required_string(mapping, key, record_id, issues, prefix=prefix)


def _require_equal(
    mapping: Mapping[str, Any], key: str, expected: Any, record_id: str | None, issues: list[ValidationIssue], *, prefix: str = ""
) -> None:
    if mapping.get(key) != expected:
        issues.append(ValidationIssue(record_id, f"{prefix}{key}", f"must equal {expected!r}"))


def _require_integer(
    mapping: Mapping[str, Any], key: str, record_id: str | None, issues: list[ValidationIssue], *, prefix: str = ""
) -> int | None:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        issues.append(ValidationIssue(record_id, f"{prefix}{key}", "must be an integer"))
        return None
    return value


def _require_positive_number(
    mapping: Mapping[str, Any], key: str, record_id: str | None, issues: list[ValidationIssue], *, prefix: str = ""
) -> float | None:
    value = mapping.get(key)
    if not _is_finite_number(value) or float(value) <= 0:
        issues.append(ValidationIssue(record_id, f"{prefix}{key}", "must be a positive finite number"))
        return None
    return float(value)


def _require_nonnegative_number(
    mapping: Mapping[str, Any], key: str, record_id: str | None, issues: list[ValidationIssue], *, prefix: str = ""
) -> float | None:
    value = mapping.get(key)
    if not _is_finite_number(value) or float(value) < 0:
        issues.append(ValidationIssue(record_id, f"{prefix}{key}", "must be a non-negative finite number"))
        return None
    return float(value)


def _require_sha256(value: Any, record_id: str | None, field: str, issues: list[ValidationIssue]) -> None:
    if not _is_sha256(value):
        issues.append(ValidationIssue(record_id, field, "must be a 64-character SHA-256 hash"))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
