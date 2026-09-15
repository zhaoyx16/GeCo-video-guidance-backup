"""Projection utilities for OpenCV cameras and camera-space z depth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SampledImage:
    values: np.ndarray
    in_bounds: np.ndarray


def pixel_grid(height: int, width: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    if height < 1 or width < 1:
        raise ValueError("height and width must be positive")
    if stride < 1:
        raise ValueError("stride must be positive")
    y, x = np.meshgrid(
        np.arange(0, height, stride, dtype=np.float64),
        np.arange(0, width, stride, dtype=np.float64),
        indexing="ij",
    )
    return x, y


def unproject_z_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    *,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unproject z depth to OpenCV camera points.

    Returns points ``[...,3]`` and the corresponding pixel x/y grids.
    """

    if depth.ndim != 2:
        raise ValueError(f"depth must have shape [H,W], got {depth.shape}")
    x, y = pixel_grid(*depth.shape, stride=stride)
    sampled_depth = depth[::stride, ::stride].astype(np.float64, copy=False)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if fx <= 0 or fy <= 0:
        raise ValueError("focal lengths must be positive")
    points = np.stack(
        [
            (x - cx) / fx * sampled_depth,
            (y - cy) / fy * sampled_depth,
            sampled_depth,
        ],
        axis=-1,
    )
    return points, x, y


def camera_to_world(points_camera: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    rotation = world_to_camera[:3, :3].astype(np.float64, copy=False)
    translation = world_to_camera[:3, 3].astype(np.float64, copy=False)
    return (points_camera - translation) @ rotation


def world_to_camera(points_world: np.ndarray, world_to_camera_matrix: np.ndarray) -> np.ndarray:
    rotation = world_to_camera_matrix[:3, :3].astype(np.float64, copy=False)
    translation = world_to_camera_matrix[:3, 3].astype(np.float64, copy=False)
    return points_world @ rotation.T + translation


def project_camera(points_camera: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = points_camera[..., 2]
    x_normalized = np.full_like(z, np.nan, dtype=np.float64)
    y_normalized = np.full_like(z, np.nan, dtype=np.float64)
    np.divide(points_camera[..., 0], z, out=x_normalized, where=z != 0)
    np.divide(points_camera[..., 1], z, out=y_normalized, where=z != 0)
    x = intrinsics[0, 0] * x_normalized + intrinsics[0, 2]
    y = intrinsics[1, 1] * y_normalized + intrinsics[1, 2]
    return x, y, z


def bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> SampledImage:
    """Sample a scalar image without silently clipping out-of-bounds points."""

    if image.ndim != 2:
        raise ValueError(f"image must have shape [H,W], got {image.shape}")
    if x.shape != y.shape:
        raise ValueError(f"x/y shapes differ: {x.shape} != {y.shape}")
    height, width = image.shape
    finite = np.isfinite(x) & np.isfinite(y)
    in_bounds = finite & (x >= 0.0) & (x <= width - 1) & (y >= 0.0) & (y <= height - 1)

    x_safe = np.clip(np.where(finite, x, 0.0), 0.0, width - 1)
    y_safe = np.clip(np.where(finite, y, 0.0), 0.0, height - 1)
    x0 = np.floor(x_safe).astype(np.int64)
    y0 = np.floor(y_safe).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x_safe - x0
    wy = y_safe - y0

    values = (
        (1.0 - wx) * (1.0 - wy) * image[y0, x0]
        + wx * (1.0 - wy) * image[y0, x1]
        + (1.0 - wx) * wy * image[y1, x0]
        + wx * wy * image[y1, x1]
    )
    values = np.where(in_bounds, values, np.nan)
    return SampledImage(values=values, in_bounds=in_bounds)


def relative_depth_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Symmetric scale-normalized depth disagreement in [0, 2]."""

    denominator = np.abs(a) + np.abs(b)
    result = np.full_like(denominator, np.nan, dtype=np.float64)
    np.divide(2.0 * np.abs(a - b), denominator, out=result, where=denominator > 0)
    return result


def camera_centers(world_to_camera_matrices: np.ndarray) -> np.ndarray:
    rotation = world_to_camera_matrices[:, :3, :3].astype(np.float64, copy=False)
    translation = world_to_camera_matrices[:, :3, 3].astype(np.float64, copy=False)
    return -np.einsum("tji,tj->ti", rotation, translation)
