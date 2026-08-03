"""Canonical geometry data exchanged by backbones, scorers, and caches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GeometryPrediction:
    """Camera and depth predictions in one explicit convention.

    ``world_to_camera`` contains homogeneous OpenCV world-to-camera matrices:
    x points right, y down, z forward. ``depth`` is camera-space z depth at the
    resolution described by ``intrinsics``. This matches the public
    VGGT-Omega inference code and is intentionally not called a generic
    "extrinsic" to avoid direction ambiguity.
    """

    world_to_camera: np.ndarray
    intrinsics: np.ndarray
    depth: np.ndarray
    confidence: np.ndarray
    keyframe_indices: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.depth.shape[0])

    @property
    def image_size_hw(self) -> tuple[int, int]:
        return int(self.depth.shape[1]), int(self.depth.shape[2])

    def validate(self) -> None:
        arrays = {
            "world_to_camera": self.world_to_camera,
            "intrinsics": self.intrinsics,
            "depth": self.depth,
            "confidence": self.confidence,
            "keyframe_indices": self.keyframe_indices,
        }
        for name, value in arrays.items():
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{name} must be a numpy array, got {type(value)!r}")

        if self.depth.ndim != 3:
            raise ValueError(f"depth must have shape [T,H,W], got {self.depth.shape}")
        frames, height, width = self.depth.shape
        if frames < 2 or height < 2 or width < 2:
            raise ValueError(f"geometry requires at least 2 frames and 2x2 maps, got {self.depth.shape}")
        if self.confidence.shape != self.depth.shape:
            raise ValueError(
                f"confidence must match depth: {self.confidence.shape} != {self.depth.shape}"
            )
        if self.world_to_camera.shape != (frames, 4, 4):
            raise ValueError(
                "world_to_camera must have shape "
                f"[{frames},4,4], got {self.world_to_camera.shape}"
            )
        if self.intrinsics.shape != (frames, 3, 3):
            raise ValueError(
                f"intrinsics must have shape [{frames},3,3], got {self.intrinsics.shape}"
            )
        if self.keyframe_indices.shape != (frames,):
            raise ValueError(
                f"keyframe_indices must have shape [{frames}], got {self.keyframe_indices.shape}"
            )
        if np.any(np.diff(self.keyframe_indices.astype(np.int64)) <= 0):
            raise ValueError("keyframe_indices must be strictly increasing")

        for name in ("world_to_camera", "intrinsics", "depth", "confidence"):
            if not np.isfinite(arrays[name]).all():
                raise ValueError(f"{name} contains NaN or infinity")
        if np.any(self.depth <= 0):
            raise ValueError("depth must be strictly positive")
        if np.any(self.confidence <= 0):
            raise ValueError("confidence must be strictly positive")

        expected_bottom = np.broadcast_to(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            (frames, 4),
        )
        if not np.allclose(self.world_to_camera[:, 3], expected_bottom, atol=1e-5):
            raise ValueError("world_to_camera matrices must have homogeneous bottom row [0,0,0,1]")

        rotation = self.world_to_camera[:, :3, :3].astype(np.float64)
        identity = np.broadcast_to(np.eye(3), rotation.shape)
        if not np.allclose(rotation @ np.swapaxes(rotation, -1, -2), identity, atol=2e-3):
            raise ValueError("camera rotations are not orthonormal")
        determinants = np.linalg.det(rotation)
        if not np.allclose(determinants, 1.0, atol=2e-3):
            raise ValueError(f"camera rotation determinants must be +1, got {determinants}")

        fx = self.intrinsics[:, 0, 0]
        fy = self.intrinsics[:, 1, 1]
        if np.any(fx <= 0) or np.any(fy <= 0):
            raise ValueError("focal lengths must be positive")
        if not np.allclose(self.intrinsics[:, 2, 2], 1.0, atol=1e-6):
            raise ValueError("intrinsics[:,2,2] must equal 1")
        if not np.allclose(self.intrinsics[:, 2, :2], 0.0, atol=1e-6):
            raise ValueError("intrinsics bottom-left entries must equal 0")

    def as_float32(self) -> "GeometryPrediction":
        return GeometryPrediction(
            world_to_camera=self.world_to_camera.astype(np.float32, copy=False),
            intrinsics=self.intrinsics.astype(np.float32, copy=False),
            depth=self.depth.astype(np.float32, copy=False),
            confidence=self.confidence.astype(np.float32, copy=False),
            keyframe_indices=self.keyframe_indices.astype(np.int64, copy=False),
            metadata=dict(self.metadata),
        )
