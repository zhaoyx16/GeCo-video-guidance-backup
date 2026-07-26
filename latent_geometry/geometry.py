"""Pose targets and metrics for the latent-geometry probe.

The probe predicts a relative camera transform for an ordered pair of views.
It predicts rotation and *translation direction* only.  Translation magnitude
is deliberately omitted because monocular videos do not determine metric scale.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


def validate_world_to_camera_se3(
    world_to_camera: torch.Tensor,
    *,
    atol: float = 1e-4,
    rotation_atol: float = 1e-3,
) -> None:
    """Reject malformed transforms before treating them as W2C SE(3) poses."""
    if world_to_camera.ndim != 3 or world_to_camera.shape[-2:] != (4, 4):
        raise ValueError(
            "world_to_camera must have shape [num_frames, 4, 4], "
            f"got {tuple(world_to_camera.shape)}"
        )
    if not torch.is_floating_point(world_to_camera) or not torch.isfinite(world_to_camera).all():
        raise ValueError("world_to_camera must contain finite floating-point values")
    expected_bottom_row = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], device=world_to_camera.device, dtype=world_to_camera.dtype
    )
    if not torch.allclose(world_to_camera[:, 3, :], expected_bottom_row.expand_as(world_to_camera[:, 3, :]), atol=atol, rtol=0.0):
        raise ValueError("world_to_camera transforms must use homogeneous SE(3) bottom row [0, 0, 0, 1]")

    rotation = world_to_camera[:, :3, :3]
    identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype).expand_as(rotation)
    orthogonality_error = (rotation.transpose(-1, -2) @ rotation - identity).abs().amax()
    if float(orthogonality_error.detach().cpu()) > rotation_atol:
        raise ValueError(
            "world_to_camera rotation is not orthonormal; "
            f"max |R^T R - I|={float(orthogonality_error.detach().cpu()):.3e}"
        )
    determinant = torch.linalg.det(rotation)
    determinant_error = (determinant - 1.0).abs().amax()
    if bool((determinant <= 0.0).any()) or float(determinant_error.detach().cpu()) > rotation_atol:
        raise ValueError(
            "world_to_camera rotation must be right-handed with determinant approximately +1; "
            f"max |det(R)-1|={float(determinant_error.detach().cpu()):.3e}"
        )


def rotation_matrix_to_6d(rotation: torch.Tensor) -> torch.Tensor:
    """Encode a rotation matrix by its first two columns, column-major."""
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [..., 3, 3] rotation matrix, got {tuple(rotation.shape)}")
    return rotation[..., :, :2].transpose(-1, -2).reshape(*rotation.shape[:-2], 6)


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Decode Zhou et al.'s continuous 6D rotation representation."""
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Expected [..., 6] rotation representation, got {tuple(rotation_6d.shape)}")

    first = rotation_6d[..., :3]
    second = rotation_6d[..., 3:]
    basis_1 = F.normalize(first, dim=-1, eps=1e-6)
    second = second - (basis_1 * second).sum(dim=-1, keepdim=True) * basis_1
    basis_2 = F.normalize(second, dim=-1, eps=1e-6)
    basis_3 = torch.cross(basis_1, basis_2, dim=-1)
    return torch.stack((basis_1, basis_2, basis_3), dim=-1)


def relative_transform_from_world_to_camera(
    world_to_camera: torch.Tensor,
    source_index: int,
    target_index: int,
) -> torch.Tensor:
    """Return ``T_target_from_source`` for world-to-camera poses.

    ``world_to_camera[i]`` maps homogeneous world points to camera ``i``.
    The returned transform maps source-camera coordinates to target-camera
    coordinates, so its translation direction has an unambiguous sign once the
    ordered source/target pair is fixed.
    """
    validate_world_to_camera_se3(world_to_camera)
    num_frames = world_to_camera.shape[0]
    if not 0 <= source_index < num_frames or not 0 <= target_index < num_frames:
        raise IndexError(
            f"Pose pair ({source_index}, {target_index}) is outside [0, {num_frames - 1}]"
        )
    return world_to_camera[target_index] @ torch.linalg.inv(world_to_camera[source_index])


def make_relative_pose_target(
    world_to_camera: torch.Tensor,
    source_index: int,
    target_index: int,
    translation_epsilon: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Construct scale-free relative-pose supervision for one ordered pair.

    Targets:
      - ``rotation_6d``: first two columns of ``R_target_from_source``.
      - ``translation_direction``: normalized ``t_target_from_source`` in the
        target camera coordinate system.
      - ``translation_valid``: false for negligible baseline, where direction
        is mathematically undefined and must not contribute to the loss.
    """
    transform = relative_transform_from_world_to_camera(world_to_camera, source_index, target_index)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    magnitude = torch.linalg.vector_norm(translation)
    translation_valid = magnitude > translation_epsilon
    direction = translation / magnitude.clamp_min(translation_epsilon)
    return {
        "rotation_6d": rotation_matrix_to_6d(rotation),
        "translation_direction": direction,
        "translation_valid": translation_valid.to(dtype=torch.bool),
    }


def _as_batch(target: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Accept a single target or a batch target without changing its device."""
    rotation_6d = target["rotation_6d"]
    translation_direction = target["translation_direction"]
    translation_valid = target["translation_valid"]
    if rotation_6d.ndim == 1:
        rotation_6d = rotation_6d.unsqueeze(0)
        translation_direction = translation_direction.unsqueeze(0)
        translation_valid = translation_valid.reshape(1)
    return {
        "rotation_6d": rotation_6d,
        "translation_direction": translation_direction,
        "translation_valid": translation_valid.to(dtype=torch.bool),
    }


def pose_losses(
    prediction: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    rotation_weight: float = 1.0,
    translation_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Stable training losses for scale-free relative pose.

    The rotation term is a chordal loss on orthonormalized matrices.  The
    translation term is cosine distance, masked when no translation direction
    is defined.  Geodesic angle is reported in metrics rather than optimized
    directly to avoid unstable gradients near identical rotations.
    """
    target_batch = _as_batch(target)
    predicted_rotation = rotation_6d_to_matrix(prediction["rotation_6d"])
    target_rotation = rotation_6d_to_matrix(target_batch["rotation_6d"])
    rotation_loss = (predicted_rotation - target_rotation).square().mean(dim=(-2, -1)).mean()

    predicted_direction = F.normalize(prediction["translation_direction"], dim=-1, eps=1e-6)
    target_direction = F.normalize(target_batch["translation_direction"], dim=-1, eps=1e-6)
    direction_distance = 1.0 - (predicted_direction * target_direction).sum(dim=-1).clamp(-1.0, 1.0)
    valid = target_batch["translation_valid"]
    if bool(valid.any()):
        translation_loss = direction_distance[valid].mean()
    else:
        translation_loss = prediction["translation_direction"].sum() * 0.0

    total = rotation_weight * rotation_loss + translation_weight * translation_loss
    return {
        "loss": total,
        "rotation_loss": rotation_loss,
        "translation_loss": translation_loss,
    }


def pose_metrics(
    prediction: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return per-example angular errors and valid-direction mask."""
    target_batch = _as_batch(target)
    predicted_rotation = rotation_6d_to_matrix(prediction["rotation_6d"])
    target_rotation = rotation_6d_to_matrix(target_batch["rotation_6d"])
    relative_rotation = predicted_rotation @ target_rotation.transpose(-1, -2)
    trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    rotation_deg = torch.rad2deg(torch.acos(cosine))

    predicted_direction = F.normalize(prediction["translation_direction"], dim=-1, eps=1e-6)
    target_direction = F.normalize(target_batch["translation_direction"], dim=-1, eps=1e-6)
    direction_deg = torch.rad2deg(
        torch.acos((predicted_direction * target_direction).sum(dim=-1).clamp(-1.0, 1.0))
    )
    return {
        "rotation_deg": rotation_deg,
        "translation_direction_deg": direction_deg,
        "translation_valid": target_batch["translation_valid"],
    }
