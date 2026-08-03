from __future__ import annotations

import numpy as np
import pytest

from geometry_selection.appearance import AppearanceEvidence
from geometry_selection.graph import optimize_pose_graph, se3_exp
from geometry_selection.schema import GeometryPrediction
from geometry_selection.window_graph import (
    IndependentWindow,
    WindowGraphConfig,
    build_window_graph_measurements,
    make_window_schedule,
)


def _prediction(frames, positions, *, scale: float, noise: float = 0.0) -> GeometryPrediction:
    extrinsics = []
    first_position = float(positions[0])
    for frame, position in zip(frames, positions, strict=True):
        world_from_camera = se3_exp(
            np.array(
                [
                    scale * position,
                    0.0,
                    0.0,
                    0.0,
                    0.01 * frame + noise * (float(position) - first_position),
                    0.0,
                ]
            )
        )
        extrinsics.append(np.linalg.inv(world_from_camera))
    count = len(frames)
    depth = np.full((count, 16, 16), 3.0 * scale, dtype=np.float64)
    intrinsics = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    intrinsics[:, 0, 0] = 10.0
    intrinsics[:, 1, 1] = 10.0
    intrinsics[:, 0, 2] = 8.0
    intrinsics[:, 1, 2] = 8.0
    return GeometryPrediction(
        world_to_camera=np.stack(extrinsics),
        intrinsics=intrinsics,
        depth=depth,
        confidence=np.full_like(depth, 3.0),
        keyframe_indices=np.asarray(frames, dtype=np.int64),
    )


def _window(identifier, kind, frames, *, scale, noise=0.0):
    positions = [float(frame) for frame in frames]
    return IndependentWindow(
        window_id=identifier,
        kind=kind,
        prediction=_prediction(frames, positions, scale=scale, noise=noise),
        independent_run_id=f"run-{identifier}",
    )


def test_overlapping_window_scales_are_recovered_before_se3_graph() -> None:
    windows = (
        _window("local-a", "local", (0, 1, 2, 3), scale=1.0),
        _window("local-b", "local", (2, 3, 4, 5), scale=2.0),
        _window("loop-a", "loop", (0, 1, 4, 5), scale=0.5),
    )
    measurements = build_window_graph_measurements(
        windows,
        WindowGraphConfig(
            min_depth_scale_pixels=2,
            depth_sample_stride=2,
            require_reobservation_support=False,
            require_appearance_support=False,
        ),
    )

    np.testing.assert_allclose(
        measurements.window_scales,
        (1.0 / 3.0, 1.0 / 6.0, 2.0 / 3.0),
        atol=1e-8,
    )
    assert np.isclose(measurements.candidate_depth_normalizer, 1.0 / 3.0)
    assert measurements.scale_rank == 2
    assert measurements.scale_residual_rms < 1e-10
    assert measurements.node_frame_indices == (0, 1, 2, 3, 4, 5)
    assert len(measurements.loop_measurements) == 1

    report = optimize_pose_graph(
        measurements.initial_world_from_camera,
        measurements.local_measurements,
        measurements.loop_measurements,
    )
    assert report.status == "converged"
    assert report.local_residual_rms < 1e-7
    assert report.loop_residual_rms is not None
    assert report.loop_residual_rms < 1e-7


def test_inconsistent_independent_loop_leaves_graph_residual() -> None:
    windows = (
        _window("local-a", "local", (0, 1, 2, 3), scale=1.0),
        _window("local-b", "local", (2, 3, 4, 5), scale=1.0),
        _window("loop-bad", "loop", (0, 1, 4, 5), scale=1.0, noise=0.25),
    )
    measurements = build_window_graph_measurements(
        windows,
        WindowGraphConfig(
            min_depth_scale_pixels=2,
            depth_sample_stride=2,
            require_reobservation_support=False,
            require_appearance_support=False,
        ),
    )
    report = optimize_pose_graph(
        measurements.initial_world_from_camera,
        measurements.local_measurements,
        measurements.loop_measurements,
    )
    assert report.loop_residual_rms is not None
    assert report.loop_residual_rms > 1e-3


def test_local_only_ablation_ignores_loop_windows_completely() -> None:
    local = (
        _window("local-a", "local", (0, 1, 2, 3), scale=1.0),
        _window("local-b", "local", (2, 3, 4, 5), scale=2.0),
    )
    good_loop = _window("loop-a", "loop", (0, 1, 4, 5), scale=0.5)
    bad_loop = _window("loop-a", "loop", (0, 1, 4, 5), scale=7.0, noise=0.8)
    config = WindowGraphConfig(
        use_loop_edges=False,
        require_loop_edges=False,
        min_depth_scale_pixels=2,
        depth_sample_stride=2,
    )

    first = build_window_graph_measurements(local + (good_loop,), config)
    second = build_window_graph_measurements(local + (bad_loop,), config)

    assert not first.loop_measurements
    assert first.potential_loop_edges == 0
    assert first.rejected_loop_edges == ()
    assert first.window_scale_ids == ("local-a", "local-b")
    np.testing.assert_allclose(first.window_scales, second.window_scales)
    assert first.candidate_depth_normalizer == second.candidate_depth_normalizer
    assert first.potential_scale_constraint_ids == second.potential_scale_constraint_ids
    assert first.accepted_scale_constraint_ids == second.accepted_scale_constraint_ids
    for first_edge, second_edge in zip(
        first.local_measurements, second.local_measurements, strict=True
    ):
        np.testing.assert_allclose(
            first_edge.target_from_source,
            second_edge.target_from_source,
        )
        assert first_edge.provenance == second_edge.provenance
        assert first_edge.confidence == second_edge.confidence

    malformed_loop = IndependentWindow(
        window_id="loop-malformed",
        kind="loop",
        prediction=good_loop.prediction,
        independent_run_id="run-local-a",
    )
    ignored = build_window_graph_measurements(local + (malformed_loop,), config)
    assert ignored.window_scale_ids == first.window_scale_ids
    np.testing.assert_allclose(ignored.window_scales, first.window_scales)


def test_loop_requirement_cannot_be_enabled_for_local_only_ablation() -> None:
    with pytest.raises(ValueError, match="require_loop_edges"):
        WindowGraphConfig(use_loop_edges=False, require_loop_edges=True).validate()


def test_window_scale_graph_must_be_connected() -> None:
    windows = (
        _window("local-a", "local", (0, 1, 2, 3), scale=1.0),
        _window("local-b", "local", (2, 3, 4, 5), scale=1.0),
        _window("loop", "loop", (0, 1, 4, 5), scale=1.0),
    )
    # The loop overlaps both local windows and therefore connects all scales.
    build_window_graph_measurements(
        windows,
        WindowGraphConfig(
            min_depth_scale_pixels=2,
            depth_sample_stride=2,
            require_reobservation_support=False,
            require_appearance_support=False,
        ),
    )
    disconnected = (
        windows[0],
        _window("local-isolated", "local", (4, 5, 6, 7), scale=1.0),
    )
    with pytest.raises(ValueError, match="no scale constraints|disconnected"):
        build_window_graph_measurements(
            disconnected,
            WindowGraphConfig(
                min_depth_scale_pixels=2,
                depth_sample_stride=2,
                require_loop_edges=False,
                require_reobservation_support=False,
                require_appearance_support=False,
            ),
        )


def test_reused_forward_pass_is_rejected() -> None:
    first = _window("local-a", "local", (0, 1, 2, 3), scale=1.0)
    second = IndependentWindow(
        window_id="loop-a",
        kind="loop",
        prediction=_prediction((0, 1, 2, 3), (0.0, 1.0, 2.0, 3.0), scale=1.0),
        independent_run_id=first.independent_run_id,
    )
    with pytest.raises(ValueError, match="distinct model forward pass"):
        build_window_graph_measurements((first, second))


def test_schedule_contains_overlapping_local_and_distant_loop_windows() -> None:
    schedule = make_window_schedule(tuple(range(8)))
    local = [frames for _, kind, frames in schedule if kind == "local"]
    loops = [frames for _, kind, frames in schedule if kind == "loop"]
    assert local == [(0, 1, 2, 3), (2, 3, 4, 5), (4, 5, 6, 7)]
    assert loops
    assert all(len(frames) == 4 for frames in loops)
    assert all(frames[-1] - frames[0] >= 3 for frames in loops)


def test_candidate_global_scale_and_window_order_do_not_change_measurements() -> None:
    base = (
        _window("local-a", "local", (0, 1, 2, 3), scale=1.0),
        _window("local-b", "local", (2, 3, 4, 5), scale=2.0),
        _window("loop-a", "loop", (0, 1, 4, 5), scale=0.5),
    )
    scaled = (
        _window("local-a", "local", (0, 1, 2, 3), scale=0.1),
        _window("local-b", "local", (2, 3, 4, 5), scale=0.2),
        _window("loop-a", "loop", (0, 1, 4, 5), scale=0.05),
    )
    config = WindowGraphConfig(
        min_depth_scale_pixels=2,
        depth_sample_stride=2,
        require_reobservation_support=False,
        require_appearance_support=False,
    )
    first = build_window_graph_measurements(base, config)
    second = build_window_graph_measurements(tuple(reversed(base)), config)
    third = build_window_graph_measurements(scaled, config)

    assert first.node_frame_indices == second.node_frame_indices == third.node_frame_indices
    np.testing.assert_allclose(first.window_scales, second.window_scales, atol=1e-10)
    for first_edge, second_edge, third_edge in zip(
        first.local_measurements + first.loop_measurements,
        second.local_measurements + second.loop_measurements,
        third.local_measurements + third.loop_measurements,
        strict=True,
    ):
        assert first_edge.provenance == second_edge.provenance
        np.testing.assert_allclose(
            first_edge.target_from_source,
            second_edge.target_from_source,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            first_edge.target_from_source,
            third_edge.target_from_source,
            atol=1e-9,
        )


def test_accepted_edge_weights_do_not_reward_low_model_confidence() -> None:
    high = (
        _window("local-a", "local", (0, 1, 2, 3), scale=1.0),
        _window("local-b", "local", (2, 3, 4, 5), scale=1.0),
        _window("loop-a", "loop", (0, 1, 4, 5), scale=1.0),
    )
    low = (
        _window("local-a", "local", (0, 1, 2, 3), scale=1.0),
        _window("local-b", "local", (2, 3, 4, 5), scale=1.0),
        _window("loop-a", "loop", (0, 1, 4, 5), scale=1.0),
    )
    for window in low:
        window.prediction.confidence.fill(1.1)
    config = WindowGraphConfig(
        min_depth_scale_pixels=2,
        depth_sample_stride=2,
        require_reobservation_support=False,
        require_appearance_support=False,
    )
    first = build_window_graph_measurements(high, config)
    second = build_window_graph_measurements(low, config)
    assert [edge.confidence for edge in first.local_measurements] == [
        edge.confidence for edge in second.local_measurements
    ]
    assert [edge.confidence for edge in first.loop_measurements] == [
        edge.confidence for edge in second.loop_measurements
    ]
    assert [item.confidence for item in first.scale_constraints] == [
        item.confidence for item in second.scale_constraints
    ]


def test_loop_edge_requires_independent_appearance_evidence() -> None:
    local_windows = (
        _window("local-a", "local", (0, 1, 2, 3), scale=1.0),
        _window("local-b", "local", (2, 3, 4, 5), scale=1.0),
    )
    raw_loop = _window("loop-a", "loop", (0, 1, 4, 5), scale=1.0)
    accepted_loop = IndependentWindow(
        window_id=raw_loop.window_id,
        kind=raw_loop.kind,
        prediction=raw_loop.prediction,
        independent_run_id=raw_loop.independent_run_id,
        appearance_evidence=AppearanceEvidence(
            source_frame=0,
            target_frame=5,
            source_file_sha256="a" * 64,
            target_file_sha256="b" * 64,
            source_keypoints=100,
            target_keypoints=90,
            ratio_matches=40,
            inliers=30,
            inlier_ratio=0.75,
            spatial_coverage=0.25,
            mean_descriptor_distance=0.20,
            status="ok",
        ),
    )
    config = WindowGraphConfig(
        min_depth_scale_pixels=2,
        depth_sample_stride=2,
        require_reobservation_support=False,
        require_appearance_support=True,
        require_loop_edges=False,
    )
    missing = build_window_graph_measurements(local_windows + (raw_loop,), config)
    assert not missing.loop_measurements
    assert missing.rejected_loop_edges[0].reason == "missing_appearance_evidence"
    accepted = build_window_graph_measurements(local_windows + (accepted_loop,), config)
    assert len(accepted.loop_measurements) == 1
