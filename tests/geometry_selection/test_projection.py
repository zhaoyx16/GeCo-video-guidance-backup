from __future__ import annotations

import numpy as np

from geometry_selection.projection import (
    bilinear_sample,
    camera_to_world,
    project_camera,
    unproject_z_depth,
    world_to_camera,
)


def test_identity_projection_round_trip() -> None:
    height, width = 24, 32
    intrinsics = np.array(
        [[50.0, 0.0, 15.5], [0.0, 55.0, 11.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    depth = np.linspace(2.0, 7.0, height * width).reshape(height, width)
    points, x, y = unproject_z_depth(depth, intrinsics)
    projected_x, projected_y, projected_z = project_camera(points, intrinsics)
    assert np.allclose(projected_x, x, atol=1e-10)
    assert np.allclose(projected_y, y, atol=1e-10)
    assert np.allclose(projected_z, depth, atol=1e-10)


def test_known_world_camera_transform_round_trip() -> None:
    angle = np.deg2rad(30.0)
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    world_to_camera_matrix = np.eye(4)
    world_to_camera_matrix[:3, :3] = rotation
    world_to_camera_matrix[:3, 3] = np.array([0.3, -0.4, 1.2])
    points_world = np.array([[1.0, 2.0, 5.0], [-2.0, 0.5, 8.0], [0.1, -1.0, 3.0]])
    points_camera = world_to_camera(points_world, world_to_camera_matrix)
    recovered = camera_to_world(points_camera, world_to_camera_matrix)
    assert np.allclose(recovered, points_world, atol=1e-10)


def test_bilinear_sample_does_not_clip_invalid_points() -> None:
    image = np.arange(16, dtype=np.float64).reshape(4, 4)
    result = bilinear_sample(
        image,
        np.array([1.5, -0.1, 3.1]),
        np.array([1.5, 1.0, 2.0]),
    )
    assert np.isclose(result.values[0], 7.5)
    assert np.isnan(result.values[1])
    assert np.isnan(result.values[2])
    assert result.in_bounds.tolist() == [True, False, False]


def test_projection_preserves_small_positive_gauge() -> None:
    intrinsics = np.array(
        [[50.0, 0.0, 15.5], [0.0, 50.0, 11.5], [0.0, 0.0, 1.0]]
    )
    point = np.array([[2e-15, -1e-15, 1e-14]])
    x, y, z = project_camera(point, intrinsics)
    assert np.allclose(x, [25.5])
    assert np.allclose(y, [6.5])
    assert np.allclose(z, [1e-14])
