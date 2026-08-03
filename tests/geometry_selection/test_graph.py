from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from geometry_selection.graph import (
    PoseGraphConfig,
    RelativePoseMeasurement,
    _objective,
    invert_se3,
    optimize_pose_graph,
    pose_edge_residual,
    relative_target_from_source,
    se3_exp,
)


def _pose(*, x: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    return se3_exp(np.array([x, 0.0, 0.0, 0.0, 0.0, yaw], dtype=np.float64))


def _chain(num_nodes: int) -> np.ndarray:
    return np.stack([_pose(x=float(index), yaw=0.03 * index) for index in range(num_nodes)])


def _edge(states: np.ndarray, source: int, target: int, confidence: float = 1.0) -> RelativePoseMeasurement:
    return RelativePoseMeasurement(
        source=source,
        target=target,
        target_from_source=relative_target_from_source(states[source], states[target]),
        confidence=confidence,
        independently_estimated=True,
        provenance=f"independent-window-{source}-{target}",
    )


def _local_chain_edges(states: np.ndarray, confidence: float = 1.0) -> tuple[RelativePoseMeasurement, ...]:
    return tuple(_edge(states, index, index + 1, confidence) for index in range(len(states) - 1))


def _perturbed(states: np.ndarray) -> np.ndarray:
    result = states.copy()
    for index in range(1, len(result)):
        delta = np.array(
            [0.08 * (-1) ** index, 0.02 * index, 0.0, 0.0, 0.0, 0.02 * (-1) ** index]
        )
        result[index] = se3_exp(delta) @ result[index]
    return result


def test_exact_chain_converges_with_fixed_gauge() -> None:
    truth = _chain(4)
    initial = _perturbed(truth)
    report = optimize_pose_graph(initial, _local_chain_edges(truth))

    assert report.converged
    assert report.status == "converged"
    assert not report.degenerate
    assert report.normal_matrix_rank == report.variable_dimension
    assert report.local_residual_rms < 1e-7
    assert report.loop_residual_rms is None
    # Node zero is the fixed gauge, not an implicit identity reset.
    np.testing.assert_allclose(report.optimized_world_from_camera[0], initial[0], atol=1e-12)
    for edge in _local_chain_edges(truth):
        np.testing.assert_allclose(
            relative_target_from_source(
                report.optimized_world_from_camera[edge.source],
                report.optimized_world_from_camera[edge.target],
            ),
            edge.target_from_source,
            atol=1e-7,
        )


def test_inconsistent_loop_leaves_explicit_local_and_loop_residuals() -> None:
    truth = _chain(4)
    inconsistent_loop = RelativePoseMeasurement(
        source=0,
        target=3,
        target_from_source=_pose(x=-2.4, yaw=-0.09),
        confidence=1.0,
        independently_estimated=True,
        provenance="independent-long-window-0-3",
    )
    config = replace(PoseGraphConfig(), use_switchable_loops=False, huber_delta=100.0)
    report = optimize_pose_graph(
        truth,
        _local_chain_edges(truth),
        (inconsistent_loop,),
        config,
    )

    assert report.converged
    assert report.local_residual_rms > 1e-3
    assert report.loop_residual_rms is not None
    assert report.loop_residual_rms > 1e-3
    assert report.loop_switches == (1.0,)


def test_outlier_loop_is_softly_downweighted() -> None:
    truth = _chain(4)
    outlier = RelativePoseMeasurement(
        source=0,
        target=3,
        target_from_source=_pose(x=6.0, yaw=1.0),
        confidence=1.0,
        independently_estimated=True,
        provenance="independent-but-false-loop-0-3",
    )
    base = PoseGraphConfig(huber_delta=100.0, loop_switch_prior=0.1)
    switched = optimize_pose_graph(
        truth,
        _local_chain_edges(truth, confidence=5.0),
        (outlier,),
        base,
    )
    forced = optimize_pose_graph(
        truth,
        _local_chain_edges(truth, confidence=5.0),
        (outlier,),
        replace(base, use_switchable_loops=False),
    )

    switched_endpoint_error = np.linalg.norm(
        switched.optimized_world_from_camera[-1][:3, 3] - truth[-1][:3, 3]
    )
    forced_endpoint_error = np.linalg.norm(
        forced.optimized_world_from_camera[-1][:3, 3] - truth[-1][:3, 3]
    )
    assert switched.loop_switches[0] < 0.05
    assert switched.loop_effective_weights[0] < 0.01
    assert switched_endpoint_error < forced_endpoint_error * 0.1


def test_global_world_gauge_does_not_change_optimized_relations() -> None:
    truth = _chain(4)
    initial = _perturbed(truth)
    global_world_change = _pose(x=4.0, yaw=0.4)
    transformed_initial = np.stack([global_world_change @ pose for pose in initial])
    local = _local_chain_edges(truth)
    loop = (_edge(truth, 0, 3, confidence=0.8),)

    first = optimize_pose_graph(initial, local, loop)
    second = optimize_pose_graph(transformed_initial, local, loop)

    assert np.isclose(first.total_cost, second.total_cost, atol=1e-10)
    assert np.isclose(first.local_residual_rms, second.local_residual_rms, atol=1e-9)
    for source, target in ((0, 1), (1, 3), (0, 3)):
        first_relative = relative_target_from_source(
            first.optimized_world_from_camera[source],
            first.optimized_world_from_camera[target],
        )
        second_relative = relative_target_from_source(
            second.optimized_world_from_camera[source],
            second.optimized_world_from_camera[target],
        )
        np.testing.assert_allclose(first_relative, second_relative, atol=1e-7)


def test_wrong_transform_direction_has_nonzero_residual() -> None:
    states = np.stack([_pose(x=0.0), _pose(x=1.0)])
    target_from_source = relative_target_from_source(states[0], states[1])
    correct = pose_edge_residual(states[0], states[1], target_from_source)
    wrong_direction = pose_edge_residual(states[0], states[1], invert_se3(target_from_source))

    np.testing.assert_allclose(correct, np.zeros(6), atol=1e-12)
    assert np.linalg.norm(wrong_direction) > 1.9


def test_shared_global_prediction_is_not_accepted_as_independent_evidence() -> None:
    states = _chain(2)
    derived = RelativePoseMeasurement(
        source=0,
        target=1,
        target_from_source=relative_target_from_source(states[0], states[1]),
        independently_estimated=False,
        provenance="derived-from-one-global-vggt-output",
    )
    with pytest.raises(ValueError, match="shared/global prediction"):
        optimize_pose_graph(states, (derived,))


def test_independence_is_fail_closed_by_default() -> None:
    states = _chain(2)
    omitted = RelativePoseMeasurement(
        source=0,
        target=1,
        target_from_source=relative_target_from_source(states[0], states[1]),
    )
    with pytest.raises(ValueError, match="shared/global prediction"):
        optimize_pose_graph(states, (omitted,))


def test_total_cost_uses_exact_huber_loss() -> None:
    states = np.stack([_pose(x=0.0), _pose(x=1.0)])
    wrong = RelativePoseMeasurement(
        source=0,
        target=1,
        target_from_source=_pose(x=-3.0),
        confidence=2.0,
        independently_estimated=True,
        provenance="independent-window",
    )
    delta = 0.25
    config = PoseGraphConfig(max_iterations=1, huber_delta=delta)
    cost, _, _, _ = _objective(states, (wrong,), (), config)
    residual_norm = 2.0
    expected = wrong.confidence * delta * (residual_norm - 0.5 * delta)
    assert np.isclose(cost, expected)


def test_effectively_closed_loop_cannot_hide_disconnected_local_graph() -> None:
    truth = _chain(4)
    local = (_edge(truth, 0, 1), _edge(truth, 2, 3))
    false_bridge = RelativePoseMeasurement(
        source=1,
        target=2,
        target_from_source=_pose(x=100.0, yaw=2.0),
        confidence=1.0,
        independently_estimated=True,
        provenance="independent-false-bridge",
    )
    report = optimize_pose_graph(
        truth,
        local,
        (false_bridge,),
        PoseGraphConfig(
            huber_delta=0.1,
            loop_switch_prior=1e-4,
            min_effective_edge_weight=1e-3,
        ),
    )
    assert report.degenerate
    assert report.status == "degenerate"


def test_max_iterations_must_be_an_integer() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        PoseGraphConfig(max_iterations=1.5).validate()


def test_se3_exp_log_round_trip_with_general_motion() -> None:
    from geometry_selection.graph import se3_log

    tangent = np.array([0.3, -0.2, 0.5, 0.4, -0.25, 0.15])
    np.testing.assert_allclose(se3_log(se3_exp(tangent)), tangent, atol=1e-10)
