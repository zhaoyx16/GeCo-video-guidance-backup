from __future__ import annotations

import numpy as np
import pytest

from geometry_selection.schema import GeometryPrediction


def slanted_plane_prediction(
    *,
    num_frames: int = 5,
    height: int = 48,
    width: int = 64,
    camera_step: float = 0.08,
) -> GeometryPrediction:
    fx = fy = 80.0
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    intrinsics = np.broadcast_to(
        np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64),
        (num_frames, 3, 3),
    ).copy()
    centers = np.stack(
        [
            np.arange(num_frames, dtype=np.float64) * camera_step,
            np.zeros(num_frames),
            np.zeros(num_frames),
        ],
        axis=-1,
    )
    world_to_camera = np.broadcast_to(np.eye(4), (num_frames, 4, 4)).copy()
    world_to_camera[:, :3, 3] = -centers

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    ray_x = (x - cx) / fx
    ray_y = (y - cy) / fy
    plane_z = 5.0
    plane_x_slope = 0.25
    plane_y_slope = -0.10
    depth = []
    for center in centers:
        numerator = plane_z + plane_x_slope * center[0] + plane_y_slope * center[1] - center[2]
        denominator = 1.0 - plane_x_slope * ray_x - plane_y_slope * ray_y
        depth.append(numerator / denominator)
    depth = np.asarray(depth, dtype=np.float64)
    confidence = np.broadcast_to(
        2.0 + 0.2 * np.cos(x / width * np.pi)[None],
        depth.shape,
    ).copy()
    prediction = GeometryPrediction(
        world_to_camera=world_to_camera,
        intrinsics=intrinsics,
        depth=depth,
        confidence=confidence,
        keyframe_indices=np.arange(num_frames, dtype=np.int64) * 10,
        metadata={"kind": "synthetic_slanted_plane"},
    )
    prediction.validate()
    return prediction


@pytest.fixture
def plane_prediction() -> GeometryPrediction:
    return slanted_plane_prediction()
