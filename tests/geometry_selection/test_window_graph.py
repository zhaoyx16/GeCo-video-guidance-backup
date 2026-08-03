from __future__ import annotations

import numpy as np
import pytest

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
        ),
    )

    np.testing.assert_allclose(measurements.window_scales, (1.0, 0.5, 2.0), atol=1e-8)
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
        ),
    )
    report = optimize_pose_graph(
        measurements.initial_world_from_camera,
        measurements.local_measurements,
        measurements.loop_measurements,
    )
    assert report.loop_residual_rms is not None
    assert report.loop_residual_rms > 1e-3


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
