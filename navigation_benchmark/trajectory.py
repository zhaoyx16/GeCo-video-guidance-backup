"""Pose-based trajectory-adherence interface.

This module does not estimate poses. It compares pose series supplied by an
external estimator or reconstructor against source-clip GT poses. Keeping
estimation outside this protocol avoids treating a guidance-aligned metric as
an independent trajectory metric.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import METRIC_RESULT_RECORD_TYPE, SCHEMA_VERSION


Pose = list[list[float]]


def load_pose_series(path: str | Path) -> list[Pose]:
    """Load a JSON list of 4x4 matrices or an object containing a poses list."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("poses")
    if not isinstance(payload, list):
        raise ValueError("pose JSON must be a list or contain a 'poses' list")
    return [_coerce_pose(item) for item in payload]


def evaluate_anchor_trajectory(
    gt_poses: Sequence[Sequence[Sequence[float]]],
    predicted_poses: Sequence[Sequence[Sequence[float]]],
    *,
    scale_alignment: str = "none",
) -> dict[str, float | int | str]:
    """Compare trajectories after anchoring both pose series at their first frame.

    Scale alignment is appropriate only for an estimator without metric scale.
    When an estimator returns metric poses, leave scale alignment disabled so
    translation error remains physically meaningful.
    """

    if len(gt_poses) != len(predicted_poses):
        raise ValueError("GT and predicted pose series must have the same length")
    if len(gt_poses) < 2:
        raise ValueError("at least two anchor poses are required")
    if scale_alignment not in {"none", "least_squares"}:
        raise ValueError("scale_alignment must be 'none' or 'least_squares'")

    gt_relative = _relative_to_first([_coerce_pose(pose) for pose in gt_poses])
    pred_relative = _relative_to_first([_coerce_pose(pose) for pose in predicted_poses])

    scale = 1.0
    if scale_alignment == "least_squares":
        scale = _least_squares_scale(gt_relative, pred_relative)
        for pose in pred_relative:
            for row in range(3):
                pose[row][3] *= scale

    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    for gt_pose, pred_pose in zip(gt_relative[1:], pred_relative[1:]):
        translation_errors.append(_norm(_subtract(_translation(gt_pose), _translation(pred_pose))))
        rotation_errors.append(_rotation_error_deg(gt_pose, pred_pose))

    gt_path = _path_length(gt_relative)
    pred_path = _path_length(pred_relative)
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
        "predicted_path_length": pred_path,
        "motion_ratio": pred_path / gt_path if gt_path > 1e-12 else "undefined_zero_gt_path",
    }


def trajectory_metric_records(
    run_id: str,
    report: Mapping[str, float | int | str],
    *,
    evaluator_name: str,
    evaluator_version: str,
    evaluator_model_id: str,
    evaluator_checkpoint_revision: str,
    evaluator_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Turn a trajectory report into scalar metric-result JSONL records."""

    details = dict(report)
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": METRIC_RESULT_RECORD_TYPE,
            "run_id": run_id,
            "metric_name": "trajectory_translation_rmse",
            "metric_role": "trajectory_adherence",
            "direction": "lower_is_better",
            "value": float(report["translation_rmse"]),
            "evaluator": {
                "name": evaluator_name,
                "version": evaluator_version,
                "model_id": evaluator_model_id,
                "checkpoint_revision": evaluator_checkpoint_revision,
                "config": dict(evaluator_config),
            },
            "details": details,
        },
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": METRIC_RESULT_RECORD_TYPE,
            "run_id": run_id,
            "metric_name": "trajectory_rotation_deg_mean",
            "metric_role": "trajectory_adherence",
            "direction": "lower_is_better",
            "value": float(report["rotation_deg_mean"]),
            "evaluator": {
                "name": evaluator_name,
                "version": evaluator_version,
                "model_id": evaluator_model_id,
                "checkpoint_revision": evaluator_checkpoint_revision,
                "config": dict(evaluator_config),
            },
            "details": details,
        },
    ]


def _coerce_pose(value: Any) -> Pose:
    if isinstance(value, Mapping):
        value = value.get("matrix", value.get("pose"))
    if not isinstance(value, Sequence) or len(value) != 4:
        raise ValueError("each pose must be a 4x4 matrix or an object with matrix")
    pose: Pose = []
    for row in value:
        if not isinstance(row, Sequence) or len(row) != 4:
            raise ValueError("each pose must be a 4x4 matrix")
        cast_row = [float(entry) for entry in row]
        if not all(math.isfinite(entry) for entry in cast_row):
            raise ValueError("pose entries must be finite")
        pose.append(cast_row)
    expected_last_row = (0.0, 0.0, 0.0, 1.0)
    if any(abs(pose[3][column] - expected) > 1e-6 for column, expected in enumerate(expected_last_row)):
        raise ValueError("pose must be a homogeneous 4x4 transform")
    return pose


def _relative_to_first(poses: Sequence[Pose]) -> list[Pose]:
    inverse_first = _invert_rigid(poses[0])
    return [_matmul(inverse_first, pose) for pose in poses]


def _least_squares_scale(gt_relative: Sequence[Pose], pred_relative: Sequence[Pose]) -> float:
    numerator = 0.0
    denominator = 0.0
    for gt_pose, pred_pose in zip(gt_relative[1:], pred_relative[1:]):
        gt_t = _translation(gt_pose)
        pred_t = _translation(pred_pose)
        numerator += sum(a * b for a, b in zip(gt_t, pred_t))
        denominator += sum(value * value for value in pred_t)
    if denominator <= 1e-12:
        raise ValueError("cannot align scale: predicted trajectory has zero translation")
    scale = numerator / denominator
    if scale <= 0:
        raise ValueError("cannot align scale: least-squares scale is non-positive")
    return scale


def _invert_rigid(pose: Pose) -> Pose:
    rotation = [[pose[row][column] for column in range(3)] for row in range(3)]
    translation = _translation(pose)
    rotation_t = [[rotation[column][row] for column in range(3)] for row in range(3)]
    inverse_translation = [
        -sum(rotation_t[row][column] * translation[column] for column in range(3))
        for row in range(3)
    ]
    return [
        rotation_t[0] + [inverse_translation[0]],
        rotation_t[1] + [inverse_translation[1]],
        rotation_t[2] + [inverse_translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matmul(left: Pose, right: Pose) -> Pose:
    return [
        [sum(left[row][axis] * right[axis][column] for axis in range(4)) for column in range(4)]
        for row in range(4)
    ]


def _translation(pose: Pose) -> list[float]:
    return [pose[row][3] for row in range(3)]


def _rotation_error_deg(left: Pose, right: Pose) -> float:
    left_rotation = [[left[row][column] for column in range(3)] for row in range(3)]
    right_rotation = [[right[row][column] for column in range(3)] for row in range(3)]
    relative = [
        [
            sum(left_rotation[axis][row] * right_rotation[axis][column] for axis in range(3))
            for column in range(3)
        ]
        for row in range(3)
    ]
    cosine = max(-1.0, min(1.0, (sum(relative[index][index] for index in range(3)) - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def _path_length(poses: Sequence[Pose]) -> float:
    return sum(
        _norm(_subtract(_translation(current), _translation(previous)))
        for previous, current in zip(poses, poses[1:])
    )


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _subtract(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0
