from __future__ import annotations

from dataclasses import replace

import numpy as np

from geometry_selection.schema import GeometryPrediction
from geometry_selection.scorer import ScorerConfig, score_geometry


def config(**overrides) -> ScorerConfig:
    base = ScorerConfig(
        local_offsets=(1,),
        min_long_range_gap=3,
        confidence_quantile=0.0,
        depth_edge_relative_threshold=0.5,
        occlusion_relative_tolerance=0.02,
        min_pair_overlap=0.20,
        min_long_range_overlap=0.20,
        min_valid_pixels=20,
        pixel_stride=2,
        huber_delta=0.05,
        cycle_weight=0.25,
        require_long_range=True,
    )
    return replace(base, **overrides)


def clone_prediction(prediction: GeometryPrediction, **changes) -> GeometryPrediction:
    values = {
        "world_to_camera": prediction.world_to_camera.copy(),
        "intrinsics": prediction.intrinsics.copy(),
        "depth": prediction.depth.copy(),
        "confidence": prediction.confidence.copy(),
        "keyframe_indices": prediction.keyframe_indices.copy(),
        "metadata": dict(prediction.metadata),
    }
    values.update(changes)
    result = GeometryPrediction(**values)
    result.validate()
    return result


def test_perfect_geometry_scores_near_zero(plane_prediction: GeometryPrediction) -> None:
    report = score_geometry(plane_prediction, config())
    assert report.status == "ok"
    assert report.valid_local_edges > 0
    assert report.valid_long_range_edges > 0
    assert report.total_score < 1e-6


def test_wrong_pose_and_wrong_depth_rank_after_correct_candidate(
    plane_prediction: GeometryPrediction,
) -> None:
    correct = score_geometry(plane_prediction, config()).total_score

    wrong_poses = plane_prediction.world_to_camera.copy()
    wrong_poses[2, 0, 3] += 0.35
    wrong_pose = score_geometry(
        clone_prediction(plane_prediction, world_to_camera=wrong_poses),
        config(),
    ).total_score

    wrong_depths = plane_prediction.depth.copy()
    wrong_depths[2] *= 1.25
    wrong_depth = score_geometry(
        clone_prediction(plane_prediction, depth=wrong_depths),
        config(),
    ).total_score

    assert wrong_pose > correct + 1e-5
    assert wrong_depth > correct + 1e-5


def test_shared_scale_is_invariant_but_depth_only_scale_is_not(
    plane_prediction: GeometryPrediction,
) -> None:
    base = score_geometry(plane_prediction, config()).total_score
    for scale in (0.1, 10.0):
        poses = plane_prediction.world_to_camera.copy()
        poses[:, :3, 3] *= scale
        scaled = clone_prediction(
            plane_prediction,
            world_to_camera=poses,
            depth=plane_prediction.depth * scale,
        )
        assert np.isclose(score_geometry(scaled, config()).total_score, base, atol=1e-7)

    depth_only = clone_prediction(plane_prediction, depth=plane_prediction.depth * 1.5)
    assert score_geometry(depth_only, config()).total_score > base + 1e-5


def test_c2w_translation_mistake_is_detected(plane_prediction: GeometryPrediction) -> None:
    wrong = plane_prediction.world_to_camera.copy()
    wrong[:, :3, 3] *= -1.0
    correct_score = score_geometry(plane_prediction, config()).total_score
    wrong_score = score_geometry(
        clone_prediction(plane_prediction, world_to_camera=wrong), config()
    ).total_score
    assert wrong_score > correct_score + 1e-5


def test_no_overlap_abstains_instead_of_returning_a_good_score(
    plane_prediction: GeometryPrediction,
) -> None:
    poses = plane_prediction.world_to_camera.copy()
    poses[1:, 0, 3] -= 100.0 * np.arange(1, plane_prediction.num_frames)  # camera centres move right
    report = score_geometry(
        clone_prediction(plane_prediction, world_to_camera=poses),
        config(),
    )
    assert np.isinf(report.total_score)
    assert report.status == "no_valid_local_edges"


def test_raw_confidence_above_one_is_handled_as_relative_weight(
    plane_prediction: GeometryPrediction,
) -> None:
    confidence = plane_prediction.confidence * 500.0
    report = score_geometry(
        clone_prediction(plane_prediction, confidence=confidence),
        config(confidence_quantile=0.2),
    )
    assert report.status == "ok"
    assert np.isfinite(report.total_score)
