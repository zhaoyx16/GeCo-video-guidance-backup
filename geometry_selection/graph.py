"""Strict SE(3) pose-graph scoring for independently estimated measurements.

Conventions
-----------
Each node stores ``T_world_from_camera[i]`` (also written ``T_wc``), which
maps a homogeneous point from camera ``i`` coordinates into one common world
frame.  A directed edge from source ``i`` to target ``j`` stores
``T_target_from_source`` (``T_cj_ci``), which maps points from camera ``i``
coordinates into camera ``j`` coordinates.  The relation predicted by two
node states is therefore::

    T_cj_ci_pred = inv(T_world_from_camera[j]) @ T_world_from_camera[i]

and the unscaled six-dimensional edge residual is exactly::

    Log(inv(T_cj_ci_measured) @ T_cj_ci_pred)

with tangent-vector ordering ``[translation, rotation]``.

This module consumes relative-pose measurements that were estimated directly
and independently (for example, from separate local VGGT windows).  Relative
transforms derived from one global camera prediction are explicitly rejected:
such transforms are algebraically self-consistent and cannot provide an
independent loop-closure test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


_EPS = 1e-12


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _validate_transform(transform: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation determinant is not +1")
    return matrix.copy()


def invert_se3(transform: np.ndarray) -> np.ndarray:
    """Invert one rigid transform without a general matrix inverse."""

    matrix = _validate_transform(transform, "transform")
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.T
    result[:3, 3] = -(rotation.T @ translation)
    return result


def so3_exp(rotation_vector: np.ndarray) -> np.ndarray:
    """Exponential map from an axis-angle vector to SO(3)."""

    phi = np.asarray(rotation_vector, dtype=np.float64)
    if phi.shape != (3,) or not np.isfinite(phi).all():
        raise ValueError("rotation_vector must be a finite length-3 vector")
    theta = float(np.linalg.norm(phi))
    phi_hat = _skew(phi)
    if theta < 1e-8:
        return np.eye(3) + phi_hat + 0.5 * (phi_hat @ phi_hat)
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * phi_hat + b * (phi_hat @ phi_hat)


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """Logarithm map from SO(3) to an axis-angle vector."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cosine))
    vee = np.array(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]],
        dtype=np.float64,
    )
    if theta < 1e-8:
        return 0.5 * vee
    if np.pi - theta < 1e-5:
        # Recover a stable axis near pi from the symmetric part of R.
        diagonal = np.maximum((np.diag(matrix) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        largest = int(np.argmax(axis))
        if axis[largest] < _EPS:
            raise ValueError("could not recover rotation axis near pi")
        if largest == 0:
            axis[1] = np.copysign(axis[1], matrix[0, 1] + matrix[1, 0])
            axis[2] = np.copysign(axis[2], matrix[0, 2] + matrix[2, 0])
        elif largest == 1:
            axis[0] = np.copysign(axis[0], matrix[0, 1] + matrix[1, 0])
            axis[2] = np.copysign(axis[2], matrix[1, 2] + matrix[2, 1])
        else:
            axis[0] = np.copysign(axis[0], matrix[0, 2] + matrix[2, 0])
            axis[1] = np.copysign(axis[1], matrix[1, 2] + matrix[2, 1])
        axis /= np.linalg.norm(axis)
        return theta * axis
    return theta * vee / (2.0 * np.sin(theta))


def se3_exp(tangent: np.ndarray) -> np.ndarray:
    """SE(3) exponential for ``[rho_x, rho_y, rho_z, phi_x, phi_y, phi_z]``."""

    xi = np.asarray(tangent, dtype=np.float64)
    if xi.shape != (6,) or not np.isfinite(xi).all():
        raise ValueError("tangent must be a finite length-6 vector")
    rho = xi[:3]
    phi = xi[3:]
    theta = float(np.linalg.norm(phi))
    phi_hat = _skew(phi)
    if theta < 1e-8:
        left_jacobian = np.eye(3) + 0.5 * phi_hat + (phi_hat @ phi_hat) / 6.0
    else:
        theta2 = theta * theta
        left_jacobian = (
            np.eye(3)
            + (1.0 - np.cos(theta)) / theta2 * phi_hat
            + (theta - np.sin(theta)) / (theta2 * theta) * (phi_hat @ phi_hat)
        )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = so3_exp(phi)
    result[:3, 3] = left_jacobian @ rho
    return result


def se3_log(transform: np.ndarray) -> np.ndarray:
    """SE(3) logarithm with tangent ordering ``[translation, rotation]``."""

    matrix = _validate_transform(transform, "transform")
    phi = so3_log(matrix[:3, :3])
    theta = float(np.linalg.norm(phi))
    phi_hat = _skew(phi)
    if theta < 1e-8:
        inverse_left_jacobian = np.eye(3) - 0.5 * phi_hat + (phi_hat @ phi_hat) / 12.0
    else:
        theta2 = theta * theta
        coefficient = 1.0 / theta2 - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
        inverse_left_jacobian = np.eye(3) - 0.5 * phi_hat + coefficient * (phi_hat @ phi_hat)
    rho = inverse_left_jacobian @ matrix[:3, 3]
    return np.concatenate([rho, phi])


@dataclass(frozen=True)
class RelativePoseMeasurement:
    """One direct relative-pose observation.

    ``independently_estimated`` must only be true when this transform was
    estimated from its own image/window evidence, rather than computed from
    the same global node poses used to initialize the graph. ``provenance`` is
    a human-readable estimator/window identifier retained for auditing.
    """

    source: int
    target: int
    target_from_source: np.ndarray
    confidence: float = 1.0
    independently_estimated: bool = False
    provenance: str = ""

    def validated(self, num_nodes: int, label: str) -> "RelativePoseMeasurement":
        if isinstance(self.source, bool) or not isinstance(self.source, (int, np.integer)):
            raise TypeError(f"{label} source must be an integer")
        if isinstance(self.target, bool) or not isinstance(self.target, (int, np.integer)):
            raise TypeError(f"{label} target must be an integer")
        if self.source == self.target:
            raise ValueError(f"{label} cannot connect a node to itself")
        if not (0 <= int(self.source) < num_nodes and 0 <= int(self.target) < num_nodes):
            raise IndexError(f"{label} endpoints ({self.source}, {self.target}) outside {num_nodes} nodes")
        if not np.isfinite(self.confidence) or self.confidence <= 0:
            raise ValueError(f"{label} confidence must be finite and positive")
        if not self.independently_estimated:
            raise ValueError(
                f"{label} is marked as derived from a shared/global prediction; "
                "it is not an independent pose-graph measurement"
            )
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError(f"{label} provenance must be a non-empty string")
        transform = _validate_transform(self.target_from_source, f"{label}.target_from_source")
        return RelativePoseMeasurement(
            source=int(self.source),
            target=int(self.target),
            target_from_source=transform,
            confidence=float(self.confidence),
            independently_estimated=True,
            provenance=self.provenance.strip(),
        )


@dataclass(frozen=True)
class PoseGraphConfig:
    translation_scale: float = 1.0
    rotation_scale: float = 1.0
    huber_delta: float = 1.0
    loop_switch_prior: float = 1.0
    use_switchable_loops: bool = True
    max_iterations: int = 40
    finite_difference_epsilon: float = 1e-6
    step_tolerance: float = 1e-8
    cost_tolerance: float = 1e-10
    initial_damping: float = 1e-5
    condition_threshold: float = 1e12
    rank_tolerance: float = 1e-9
    min_effective_edge_weight: float = 1e-6

    def validate(self) -> None:
        positive = {
            "translation_scale": self.translation_scale,
            "rotation_scale": self.rotation_scale,
            "huber_delta": self.huber_delta,
            "loop_switch_prior": self.loop_switch_prior,
            "finite_difference_epsilon": self.finite_difference_epsilon,
            "step_tolerance": self.step_tolerance,
            "cost_tolerance": self.cost_tolerance,
            "initial_damping": self.initial_damping,
            "condition_threshold": self.condition_threshold,
            "rank_tolerance": self.rank_tolerance,
            "min_effective_edge_weight": self.min_effective_edge_weight,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, (int, np.integer))
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")
        if not isinstance(self.use_switchable_loops, bool):
            raise TypeError("use_switchable_loops must be boolean")


@dataclass(frozen=True)
class PoseGraphReport:
    """Optimization result and diagnostics in the fixed first-node gauge."""

    optimized_world_from_camera: np.ndarray
    local_residual_rms: float
    loop_residual_rms: float | None
    total_cost: float
    normalized_cost: float
    evidence_weight: float
    loop_switches: tuple[float, ...]
    local_effective_weights: tuple[float, ...]
    loop_effective_weights: tuple[float, ...]
    converged: bool
    iterations: int
    status: Literal["converged", "max_iterations", "degenerate"]
    normal_matrix_condition: float
    normal_matrix_rank: int
    variable_dimension: int
    degenerate: bool


def relative_target_from_source(
    world_from_source_camera: np.ndarray,
    world_from_target_camera: np.ndarray,
) -> np.ndarray:
    """Predict ``T_target_from_source`` from two camera-to-world states."""

    source = _validate_transform(world_from_source_camera, "world_from_source_camera")
    target = _validate_transform(world_from_target_camera, "world_from_target_camera")
    return invert_se3(target) @ source


def pose_edge_residual(
    world_from_source_camera: np.ndarray,
    world_from_target_camera: np.ndarray,
    measured_target_from_source: np.ndarray,
) -> np.ndarray:
    """Return ``Log(inv(T_measured) @ T_predicted)`` without unit scaling."""

    measured = _validate_transform(measured_target_from_source, "measured_target_from_source")
    predicted = relative_target_from_source(world_from_source_camera, world_from_target_camera)
    return se3_log(invert_se3(measured) @ predicted)


def _scaled_edge_residual(
    states: np.ndarray,
    edge: RelativePoseMeasurement,
    config: PoseGraphConfig,
) -> np.ndarray:
    residual = pose_edge_residual(
        states[edge.source], states[edge.target], edge.target_from_source
    )
    result = residual.copy()
    result[:3] /= config.translation_scale
    result[3:] /= config.rotation_scale
    return result


def _huber_weight(norm: float, delta: float) -> float:
    if norm <= delta or norm < _EPS:
        return 1.0
    return delta / norm


def _huber_loss(norm: float, delta: float) -> float:
    """Exact scalar Huber loss for a vector residual norm."""

    if norm <= delta:
        return 0.5 * norm * norm
    return delta * (norm - 0.5 * delta)


def _weights_and_switches(
    states: np.ndarray,
    local_edges: Sequence[RelativePoseMeasurement],
    loop_edges: Sequence[RelativePoseMeasurement],
    config: PoseGraphConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_weights = []
    for edge in local_edges:
        norm = float(np.linalg.norm(_scaled_edge_residual(states, edge, config)))
        local_weights.append(edge.confidence * _huber_weight(norm, config.huber_delta))

    loop_weights = []
    switches = []
    for edge in loop_edges:
        residual = _scaled_edge_residual(states, edge, config)
        norm = float(np.linalg.norm(residual))
        robust_information = edge.confidence * _huber_weight(norm, config.huber_delta)
        if config.use_switchable_loops:
            # Minimise s^2 c rho(||r||) + 0.5 lambda (1-s)^2 exactly
            # for fixed graph states.  This keeps the switch update consistent
            # with the exact Huber objective used for acceptance and reporting.
            robust_energy = edge.confidence * _huber_loss(norm, config.huber_delta)
            switch = config.loop_switch_prior / (
                config.loop_switch_prior + 2.0 * robust_energy
            )
        else:
            switch = 1.0
        switches.append(float(np.clip(switch, 0.0, 1.0)))
        loop_weights.append(robust_information)
    return (
        np.asarray(local_weights, dtype=np.float64),
        np.asarray(loop_weights, dtype=np.float64),
        np.asarray(switches, dtype=np.float64),
    )


def _weighted_residual_vector(
    states: np.ndarray,
    local_edges: Sequence[RelativePoseMeasurement],
    loop_edges: Sequence[RelativePoseMeasurement],
    config: PoseGraphConfig,
    local_weights: np.ndarray,
    loop_weights: np.ndarray,
    loop_switches: np.ndarray,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for edge, weight in zip(local_edges, local_weights, strict=True):
        blocks.append(np.sqrt(weight) * _scaled_edge_residual(states, edge, config))
    for edge, weight, switch in zip(loop_edges, loop_weights, loop_switches, strict=True):
        blocks.append(np.sqrt(weight) * switch * _scaled_edge_residual(states, edge, config))
        if config.use_switchable_loops:
            blocks.append(np.array([np.sqrt(config.loop_switch_prior) * (1.0 - switch)]))
    return np.concatenate(blocks) if blocks else np.empty(0, dtype=np.float64)


def _objective(
    states: np.ndarray,
    local_edges: Sequence[RelativePoseMeasurement],
    loop_edges: Sequence[RelativePoseMeasurement],
    config: PoseGraphConfig,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    local_weights, loop_weights, switches = _weights_and_switches(
        states, local_edges, loop_edges, config
    )
    cost = 0.0
    for edge in local_edges:
        norm = float(np.linalg.norm(_scaled_edge_residual(states, edge, config)))
        cost += edge.confidence * _huber_loss(norm, config.huber_delta)
    for edge, switch in zip(loop_edges, switches, strict=True):
        norm = float(np.linalg.norm(_scaled_edge_residual(states, edge, config)))
        cost += (
            switch * switch * edge.confidence * _huber_loss(norm, config.huber_delta)
            + 0.5 * config.loop_switch_prior * (1.0 - switch) ** 2
        )
    return float(cost), local_weights, loop_weights, switches


def _apply_step(states: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Apply left perturbations to all nodes except gauge-fixed node zero."""

    result = states.copy()
    for node in range(1, len(states)):
        offset = 6 * (node - 1)
        result[node] = se3_exp(step[offset : offset + 6]) @ result[node]
    return result


def _numerical_jacobian(
    states: np.ndarray,
    local_edges: Sequence[RelativePoseMeasurement],
    loop_edges: Sequence[RelativePoseMeasurement],
    config: PoseGraphConfig,
    local_weights: np.ndarray,
    loop_weights: np.ndarray,
    switches: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base = _weighted_residual_vector(
        states, local_edges, loop_edges, config, local_weights, loop_weights, switches
    )
    dimension = 6 * (len(states) - 1)
    jacobian = np.empty((base.size, dimension), dtype=np.float64)
    epsilon = config.finite_difference_epsilon
    for column in range(dimension):
        delta = np.zeros(dimension, dtype=np.float64)
        delta[column] = epsilon
        plus = _apply_step(states, delta)
        minus = _apply_step(states, -delta)
        plus_residual = _weighted_residual_vector(
            plus, local_edges, loop_edges, config, local_weights, loop_weights, switches
        )
        minus_residual = _weighted_residual_vector(
            minus, local_edges, loop_edges, config, local_weights, loop_weights, switches
        )
        jacobian[:, column] = (plus_residual - minus_residual) / (2.0 * epsilon)
    return base, jacobian


def _conditioning(jacobian: np.ndarray, config: PoseGraphConfig) -> tuple[float, int, bool]:
    dimension = jacobian.shape[1]
    if dimension == 0:
        return 1.0, 0, False
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] <= _EPS:
        return float("inf"), 0, True
    threshold = config.rank_tolerance * singular_values[0]
    rank = int(np.count_nonzero(singular_values > threshold))
    if rank < dimension or singular_values[-1] <= _EPS:
        condition = float("inf")
    else:
        # cond(J^T J) = cond(J)^2.
        condition = float((singular_values[0] / singular_values[-1]) ** 2)
    return condition, rank, rank < dimension or condition > config.condition_threshold


def _rms_residual(
    states: np.ndarray,
    edges: Sequence[RelativePoseMeasurement],
    config: PoseGraphConfig,
) -> float | None:
    if not edges:
        return None
    values = np.concatenate([_scaled_edge_residual(states, edge, config) for edge in edges])
    return float(np.sqrt(np.mean(values * values)))


def _effective_graph_connected(
    num_nodes: int,
    local_edges: Sequence[RelativePoseMeasurement],
    loop_edges: Sequence[RelativePoseMeasurement],
    local_weights: np.ndarray,
    loop_weights: np.ndarray,
    loop_switches: np.ndarray,
    threshold: float,
) -> bool:
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for edge, weight in zip(local_edges, local_weights, strict=True):
        if float(weight) >= threshold:
            adjacency[edge.source].append(edge.target)
            adjacency[edge.target].append(edge.source)
    for edge, weight, switch in zip(
        loop_edges, loop_weights, loop_switches, strict=True
    ):
        if float(weight * switch * switch) >= threshold:
            adjacency[edge.source].append(edge.target)
            adjacency[edge.target].append(edge.source)
    visited = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbour in adjacency[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                frontier.append(neighbour)
    return len(visited) == num_nodes


def _effective_local_graph_connected(
    num_nodes: int,
    local_edges: Sequence[RelativePoseMeasurement],
    local_weights: np.ndarray,
    threshold: float,
) -> bool:
    return _effective_graph_connected(
        num_nodes,
        local_edges,
        (),
        local_weights,
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        threshold,
    )


def optimize_pose_graph(
    initial_world_from_camera: np.ndarray,
    local_measurements: Sequence[RelativePoseMeasurement],
    loop_measurements: Sequence[RelativePoseMeasurement] = (),
    config: PoseGraphConfig | None = None,
) -> PoseGraphReport:
    """Optimize a camera pose graph with node zero fixed as the gauge.

    The caller is responsible for producing each relative measurement from
    independent evidence.  In particular, do not derive these edges from one
    global VGGT camera prediction and then use them to score that same global
    prediction.  Local measurements normally come from overlapping short
    windows; long-range measurements come from separately matched windows and
    are treated as confidence-weighted switchable constraints.
    """

    resolved = config or PoseGraphConfig()
    resolved.validate()
    states = np.asarray(initial_world_from_camera, dtype=np.float64)
    if states.ndim != 3 or states.shape[1:] != (4, 4) or len(states) < 2:
        raise ValueError("initial_world_from_camera must have shape (N,4,4) with N >= 2")
    states = np.stack(
        [_validate_transform(state, f"initial_world_from_camera[{index}]") for index, state in enumerate(states)]
    )
    local_edges = tuple(
        edge.validated(len(states), f"local_measurements[{index}]")
        for index, edge in enumerate(local_measurements)
    )
    loop_edges = tuple(
        edge.validated(len(states), f"loop_measurements[{index}]")
        for index, edge in enumerate(loop_measurements)
    )
    if not local_edges:
        raise ValueError("at least one independent local measurement is required")

    dimension = 6 * (len(states) - 1)
    damping = resolved.initial_damping
    converged = False
    iterations = 0
    current_cost, local_weights, loop_weights, switches = _objective(
        states, local_edges, loop_edges, resolved
    )

    for iteration in range(1, resolved.max_iterations + 1):
        iterations = iteration
        residual, jacobian = _numerical_jacobian(
            states,
            local_edges,
            loop_edges,
            resolved,
            local_weights,
            loop_weights,
            switches,
        )
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ residual
        if np.linalg.norm(gradient, ord=np.inf) < resolved.step_tolerance:
            converged = True
            break
        try:
            step = np.linalg.solve(normal + damping * np.eye(dimension), -gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(normal + damping * np.eye(dimension), -gradient, rcond=None)[0]
        if not np.isfinite(step).all():
            break
        if np.linalg.norm(step) < resolved.step_tolerance:
            # A tiny damped step is not convergence when the gradient remains
            # large; it may mean that LM rejected proposals until damping
            # effectively froze the state.
            break

        proposal = _apply_step(states, step)
        proposal_cost, proposal_local, proposal_loop, proposal_switches = _objective(
            proposal, local_edges, loop_edges, resolved
        )
        if proposal_cost < current_cost:
            improvement = current_cost - proposal_cost
            states = proposal
            current_cost = proposal_cost
            local_weights = proposal_local
            loop_weights = proposal_loop
            switches = proposal_switches
            damping = max(damping * 0.3, 1e-12)
            if improvement < resolved.cost_tolerance:
                converged = True
                break
        else:
            damping = min(damping * 10.0, 1e12)

    # Recompute final linearization and diagnostics at the returned state.
    current_cost, local_weights, loop_weights, switches = _objective(
        states, local_edges, loop_edges, resolved
    )
    _, final_jacobian = _numerical_jacobian(
        states,
        local_edges,
        loop_edges,
        resolved,
        local_weights,
        loop_weights,
        switches,
    )
    condition, rank, numerical_degenerate = _conditioning(final_jacobian, resolved)
    connected = _effective_graph_connected(
        len(states),
        local_edges,
        loop_edges,
        local_weights,
        loop_weights,
        switches,
        resolved.min_effective_edge_weight,
    )
    local_connected = _effective_local_graph_connected(
        len(states),
        local_edges,
        local_weights,
        resolved.min_effective_edge_weight,
    )
    degenerate = numerical_degenerate or not connected or not local_connected
    status: Literal["converged", "max_iterations", "degenerate"]
    if degenerate:
        status = "degenerate"
    elif converged:
        status = "converged"
    else:
        status = "max_iterations"

    evidence_weight = float(
        sum(edge.confidence for edge in local_edges)
        + sum(
            edge.confidence * switch * switch
            for edge, switch in zip(loop_edges, switches, strict=True)
        )
    )
    return PoseGraphReport(
        optimized_world_from_camera=states,
        local_residual_rms=float(_rms_residual(states, local_edges, resolved)),
        loop_residual_rms=_rms_residual(states, loop_edges, resolved),
        total_cost=current_cost,
        normalized_cost=current_cost / max(evidence_weight, _EPS),
        evidence_weight=evidence_weight,
        loop_switches=tuple(float(value) for value in switches),
        local_effective_weights=tuple(float(value) for value in local_weights),
        loop_effective_weights=tuple(
            float(weight * switch * switch)
            for weight, switch in zip(loop_weights, switches, strict=True)
        ),
        converged=converged,
        iterations=iterations,
        status=status,
        normal_matrix_condition=condition,
        normal_matrix_rank=rank,
        variable_dimension=dimension,
        degenerate=degenerate,
    )
