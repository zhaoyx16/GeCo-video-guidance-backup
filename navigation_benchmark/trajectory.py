"""Bound pose-based trajectory-adherence evaluation.

Trajectory matrices are deliberately not accepted as free-floating arrays.
They must be wrapped in pose artifacts whose hashes, anchor indices, pose
convention, and units are bound to a completed generation-run manifest.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import (
    METRIC_RESULT_RECORD_TYPE,
    SCHEMA_VERSION,
    canonical_json,
    evaluator_fingerprint,
    validate_generation_run,
    with_evaluator_fingerprint,
)


Pose = list[list[float]]
_ROLES = ("first", "middle", "last")
_CONVENTIONS = {"W2C", "C2W"}


def load_pose_artifact(path: str | Path) -> dict[str, Any]:
    """Load a bound pose-artifact JSON object, never a raw pose list."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("pose artifact JSON must be an object with provenance and anchor poses")
    return dict(payload)


def evaluate_bound_anchor_trajectory(
    run_record: Mapping[str, Any],
    reference_pose_artifact: Mapping[str, Any],
    predicted_pose_artifact: Mapping[str, Any],
    *,
    evaluator: Mapping[str, Any],
    scale_alignment: str = "none",
) -> dict[str, Any]:
    """Evaluate an independently estimated trajectory bound to one completed run.

    The reference artifact must hash-match source_clip.poses_ref.  The
    predicted artifact must state the SHA-256 of the completed output video.
    Both expose exactly the first/middle/last anchors encoded by the run
    manifest.  This prevents a metric result from being produced for arbitrary
    matrices that cannot be traced to the generated video and source clip.
    """

    _assert_completed_run(run_record)
    if scale_alignment not in {"none", "least_squares"}:
        raise ValueError("scale_alignment must be none or least_squares")
    _assert_independent_evaluator(evaluator)

    condition = _mapping(run_record.get("condition"), "condition")
    source = _mapping(condition.get("source_clip"), "source_clip")
    frame_guidance = _mapping(condition.get("frame_guidance"), "frame_guidance")
    timing = _mapping(frame_guidance.get("generated_timing"), "generated_timing")
    output = _mapping(run_record.get("output"), "output")
    poses_ref = _mapping(source.get("poses_ref"), "source_clip.poses_ref")

    reference_poses = _bound_reference_poses(reference_pose_artifact, source, poses_ref)
    predicted_poses = _bound_predicted_poses(predicted_pose_artifact, timing, output)
    convention = str(poses_ref["pose_convention"])
    reference_unit = str(poses_ref["translation_unit"])
    predicted_unit = str(predicted_pose_artifact["translation_unit"])
    if reference_pose_artifact.get("pose_convention") != convention:
        raise ValueError("reference pose convention does not match source pose reference")
    if predicted_pose_artifact.get("pose_convention") != convention:
        raise ValueError("predicted pose convention does not match source pose reference")
    if scale_alignment == "none" and predicted_unit != reference_unit:
        raise ValueError("scale_alignment=none requires equal reference and predicted translation units")

    report = _evaluate_relative_trajectory(
        reference_poses,
        predicted_poses,
        pose_convention=convention,
        scale_alignment=scale_alignment,
    )
    report["binding"] = {
        "reference_pose_artifact_sha256": str(reference_pose_artifact["sha256"]),
        "predicted_pose_artifact_sha256": str(predicted_pose_artifact["sha256"]),
        "predicted_input_video_sha256": str(output["sha256"]),
        "anchor_mapping_hash": _anchor_mapping_fingerprint(timing),
        "run_condition_hash": str(run_record["condition_hash"]),
        "pose_convention": convention,
        "translation_unit": reference_unit,
        "scale_alignment": scale_alignment,
    }
    return report


def trajectory_metric_records(
    run_record: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    evaluator: Mapping[str, Any],
    metric_artifact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Turn a bound trajectory report into provenance-complete metric records."""

    _assert_completed_run(run_record)
    _assert_independent_evaluator(evaluator)
    output = _mapping(run_record.get("output"), "output")
    binding = report.get("binding")
    if not isinstance(binding, Mapping):
        raise ValueError("trajectory report is unbound; use evaluate_bound_anchor_trajectory")
    _assert_artifact(metric_artifact, "metric_artifact")
    normalized_evaluator = with_evaluator_fingerprint(evaluator)
    details = {key: copy.deepcopy(value) for key, value in report.items() if key != "binding"}
    common = {
        "schema_version": SCHEMA_VERSION,
        "record_type": METRIC_RESULT_RECORD_TYPE,
        "run_id": str(run_record["run_id"]),
        "run_record_hash": str(run_record["record_hash"]),
        "evaluated_output_sha256": str(output["sha256"]),
        "metric_role": "trajectory_adherence",
        "direction": "lower_is_better",
        "evaluator": normalized_evaluator,
        "metric_artifact": copy.deepcopy(dict(metric_artifact)),
        "trajectory_binding": copy.deepcopy(dict(binding)),
        "details": details,
    }
    return [
        {
            **common,
            "metric_name": "trajectory_translation_rmse",
            "value": float(report["translation_rmse"]),
        },
        {
            **common,
            "metric_name": "trajectory_rotation_deg_mean",
            "value": float(report["rotation_deg_mean"]),
        },
    ]


def _assert_completed_run(run_record: Mapping[str, Any]) -> None:
    issues = validate_generation_run(run_record)
    if issues:
        raise ValueError("invalid generation run: " + "; ".join(str(issue) for issue in issues))
    if run_record.get("status") != "completed":
        raise ValueError("trajectory evaluation requires a completed generation run")
    output = _mapping(run_record.get("output"), "output")
    _assert_sha256(output.get("sha256"), "completed output sha256")
    _assert_sha256(run_record.get("record_hash"), "completed run record_hash")


def _assert_independent_evaluator(evaluator: Mapping[str, Any]) -> None:
    required = ("name", "version", "model_id", "checkpoint_revision", "config", "independence_policy")
    for field in required:
        if field not in evaluator:
            raise ValueError(f"evaluator is missing {field}")
    if not isinstance(evaluator.get("config"), Mapping):
        raise ValueError("evaluator config must be an object")
    if evaluator.get("independence_policy") != "independent":
        raise ValueError("trajectory adherence requires evaluator.independence_policy=independent")
    expected = evaluator_fingerprint(evaluator)
    supplied = evaluator.get("fingerprint")
    if supplied is not None and supplied != expected:
        raise ValueError("evaluator fingerprint does not match evaluator config")


def _bound_reference_poses(
    artifact: Mapping[str, Any], source: Mapping[str, Any], poses_ref: Mapping[str, Any]
) -> list[Pose]:
    _assert_artifact(artifact, "reference pose artifact")
    if artifact.get("sha256") != poses_ref.get("sha256"):
        raise ValueError("reference pose artifact hash does not match source_clip.poses_ref")
    if artifact.get("pose_convention") not in _CONVENTIONS:
        raise ValueError("reference pose artifact must declare W2C or C2W")
    if not isinstance(artifact.get("translation_unit"), str) or not artifact["translation_unit"]:
        raise ValueError("reference pose artifact must declare a translation unit")
    expected = _mapping_by_role(source.get("anchors"), "source anchor")
    entries = _mapping_by_role(artifact.get("anchor_poses"), "reference anchor poses")
    poses: list[Pose] = []
    for role in _ROLES:
        entry = entries[role]
        anchor = expected[role]
        if entry.get("source_frame_index") != anchor.get("frame_index"):
            raise ValueError(f"reference {role} frame does not match manifest anchor")
        if not _close(entry.get("source_timestamp_sec"), anchor.get("timestamp_sec")):
            raise ValueError(f"reference {role} timestamp does not match manifest anchor")
        poses.append(_coerce_pose(entry.get("matrix")))
    return poses


def _bound_predicted_poses(
    artifact: Mapping[str, Any], timing: Mapping[str, Any], output: Mapping[str, Any]
) -> list[Pose]:
    _assert_artifact(artifact, "predicted pose artifact")
    if artifact.get("input_video_sha256") != output.get("sha256"):
        raise ValueError("predicted pose artifact is not bound to the completed output video")
    if artifact.get("pose_convention") not in _CONVENTIONS:
        raise ValueError("predicted pose artifact must declare W2C or C2W")
    if not isinstance(artifact.get("translation_unit"), str) or not artifact["translation_unit"]:
        raise ValueError("predicted pose artifact must declare a translation unit")
    expected = _mapping_by_role(timing.get("anchor_map"), "generated anchor map")
    entries = _mapping_by_role(artifact.get("anchor_poses"), "predicted anchor poses")
    poses: list[Pose] = []
    for role in _ROLES:
        entry = entries[role]
        mapping = expected[role]
        if entry.get("generated_frame_index") != mapping.get("generated_frame_index"):
            raise ValueError(f"predicted {role} frame does not match manifest mapping")
        if not _close(entry.get("generated_timestamp_sec"), mapping.get("generated_timestamp_sec")):
            raise ValueError(f"predicted {role} timestamp does not match manifest mapping")
        poses.append(_coerce_pose(entry.get("matrix")))
    return poses


def _evaluate_relative_trajectory(
    gt_poses: Sequence[Pose],
    predicted_poses: Sequence[Pose],
    *,
    pose_convention: str,
    scale_alignment: str,
) -> dict[str, float | int | str]:
    if len(gt_poses) != len(predicted_poses) or len(gt_poses) < 2:
        raise ValueError("bound reference and predicted artifacts need the same number of at least two poses")
    gt_relative = _relative_to_first(gt_poses, pose_convention)
    predicted_relative = _relative_to_first(predicted_poses, pose_convention)
    scale = 1.0
    if scale_alignment == "least_squares":
        scale = _least_squares_scale(gt_relative, predicted_relative)
        for pose in predicted_relative:
            for row in range(3):
                pose[row][3] *= scale
    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    for gt_pose, predicted_pose in zip(gt_relative[1:], predicted_relative[1:]):
        translation_errors.append(_norm(_subtract(_translation(gt_pose), _translation(predicted_pose))))
        rotation_errors.append(_rotation_error_deg(gt_pose, predicted_pose))
    gt_path = _path_length(gt_relative)
    predicted_path = _path_length(predicted_relative)
    return {
        "anchor_count": len(gt_relative),
        "scale_alignment": scale_alignment,
        "translation_scale_factor": scale,
        "translation_mean": _mean(translation_errors),
        "translation_rmse": math.sqrt(_mean([value * value for value in translation_errors])),
        "translation_max": max(translation_errors),
        "rotation_deg_mean": _mean(rotation_errors),
        "rotation_deg_max": max(rotation_errors),
        "gt_path_length": gt_path,
        "predicted_path_length": predicted_path,
        "motion_ratio": predicted_path / gt_path if gt_path > 1e-12 else "undefined_zero_gt_path",
    }


def _anchor_mapping_fingerprint(timing: Mapping[str, Any]) -> str:
    anchor_map = timing.get("anchor_map")
    if not isinstance(anchor_map, list):
        raise ValueError("run frame-guidance timing has no anchor map")
    import hashlib

    return hashlib.sha256(canonical_json(anchor_map).encode("utf-8")).hexdigest()


def _mapping_by_role(value: Any, label: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly three first/middle/last entries")
    result: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{label}[{index}] must be an object")
        role = entry.get("role")
        if role not in _ROLES or role in result:
            raise ValueError(f"{label} must use each first/middle/last role exactly once")
        result[str(role)] = entry
    if tuple(result) != _ROLES:
        raise ValueError(f"{label} must be ordered first, middle, last")
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _assert_artifact(value: Mapping[str, Any], label: str) -> None:
    if not isinstance(value.get("uri"), str) or not value["uri"]:
        raise ValueError(f"{label} must declare a non-empty uri")
    _assert_sha256(value.get("sha256"), f"{label} sha256")


def _assert_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError(f"{label} must be a 64-character SHA-256 hash")


def _coerce_pose(value: Any) -> Pose:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError("each pose matrix must be a 4x4 matrix")
    pose: Pose = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 4:
            raise ValueError("each pose matrix must be 4x4")
        cast = [float(entry) for entry in row]
        if not all(math.isfinite(entry) for entry in cast):
            raise ValueError("pose matrix entries must be finite")
        pose.append(cast)
    if any(abs(pose[3][index] - expected) > 1e-6 for index, expected in enumerate((0.0, 0.0, 0.0, 1.0))):
        raise ValueError("pose matrix must be homogeneous")
    return pose


def _relative_to_first(poses: Sequence[Pose], pose_convention: str) -> list[Pose]:
    inverse_first = _invert_rigid(poses[0])
    if pose_convention == "C2W":
        return [_matmul(inverse_first, pose) for pose in poses]
    if pose_convention == "W2C":
        return [_matmul(pose, inverse_first) for pose in poses]
    raise ValueError("pose convention must be W2C or C2W")


def _least_squares_scale(gt_relative: Sequence[Pose], predicted_relative: Sequence[Pose]) -> float:
    numerator = denominator = 0.0
    for gt_pose, predicted_pose in zip(gt_relative[1:], predicted_relative[1:]):
        gt_translation = _translation(gt_pose)
        predicted_translation = _translation(predicted_pose)
        numerator += sum(left * right for left, right in zip(gt_translation, predicted_translation))
        denominator += sum(value * value for value in predicted_translation)
    if denominator <= 1e-12:
        raise ValueError("cannot align scale for a zero-length predicted trajectory")
    scale = numerator / denominator
    if scale <= 0:
        raise ValueError("least-squares scale must be positive")
    return scale


def _invert_rigid(pose: Pose) -> Pose:
    rotation = [[pose[row][column] for column in range(3)] for row in range(3)]
    translation = _translation(pose)
    rotation_t = [[rotation[column][row] for column in range(3)] for row in range(3)]
    inverse_translation = [-sum(rotation_t[row][column] * translation[column] for column in range(3)) for row in range(3)]
    return [
        rotation_t[0] + [inverse_translation[0]],
        rotation_t[1] + [inverse_translation[1]],
        rotation_t[2] + [inverse_translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matmul(left: Pose, right: Pose) -> Pose:
    return [[sum(left[row][axis] * right[axis][column] for axis in range(4)) for column in range(4)] for row in range(4)]


def _translation(pose: Pose) -> list[float]:
    return [pose[row][3] for row in range(3)]


def _rotation_error_deg(left: Pose, right: Pose) -> float:
    left_rotation = [[left[row][column] for column in range(3)] for row in range(3)]
    right_rotation = [[right[row][column] for column in range(3)] for row in range(3)]
    relative = [[sum(left_rotation[axis][row] * right_rotation[axis][column] for axis in range(3)) for column in range(3)] for row in range(3)]
    cosine = max(-1.0, min(1.0, (sum(relative[index][index] for index in range(3)) - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def _path_length(poses: Sequence[Pose]) -> float:
    return sum(_norm(_subtract(_translation(current), _translation(previous))) for previous, current in zip(poses, poses[1:]))


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _subtract(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _close(left: Any, right: Any) -> bool:
    return isinstance(left, (int, float)) and isinstance(right, (int, float)) and abs(float(left) - float(right)) <= 1e-6
