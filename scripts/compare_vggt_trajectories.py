#!/usr/bin/env python3
"""Compare two VGGT camera trajectories after similarity alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def camera_centers(extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.swapaxes(rotation, 1, 2), translation)


def align_similarity(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ target_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    scale = float(singular_values.sum() / max(np.square(source_centered).sum(), 1e-12))
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = scale * (source @ rotation.T) + translation
    return aligned, scale, rotation, translation


def path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def rotation_angle_degrees(rotation: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation, axis1=1, axis2=2)
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    if reference["frame_indices"] != candidate["frame_indices"]:
        raise ValueError("Reference and candidate must contain the same frame indices")

    reference_extrinsic = reference["extrinsic"].float().numpy()
    candidate_extrinsic = candidate["extrinsic"].float().numpy()
    reference_centers = camera_centers(reference_extrinsic)
    candidate_centers = camera_centers(candidate_extrinsic)
    aligned_centers, scale, alignment_rotation, _ = align_similarity(candidate_centers, reference_centers)

    center_error = np.linalg.norm(aligned_centers - reference_centers, axis=1)
    reference_path_length = path_length(reference_centers)
    candidate_path_length = path_length(candidate_centers)
    aligned_path_length = path_length(aligned_centers)
    reference_displacement = float(np.linalg.norm(reference_centers[-1] - reference_centers[0]))

    reference_c2w = np.swapaxes(reference_extrinsic[:, :3, :3], 1, 2)
    candidate_c2w = np.swapaxes(candidate_extrinsic[:, :3, :3], 1, 2)
    # VGGT reconstructions have an arbitrary global world orientation. Align
    # orientations with the shared first frame instead of the center-only
    # Kabsch rotation, which is underconstrained for a nearly straight path.
    orientation_alignment = reference_c2w[0] @ candidate_c2w[0].T
    candidate_c2w_aligned = np.einsum("ij,njk->nik", orientation_alignment, candidate_c2w)
    relative_rotation = np.einsum(
        "nij,njk->nik",
        np.swapaxes(reference_c2w, 1, 2),
        candidate_c2w_aligned,
    )
    angle_error = rotation_angle_degrees(relative_rotation)

    reference_intrinsic = reference["intrinsic"].float().numpy()
    candidate_intrinsic = candidate["intrinsic"].float().numpy()
    focal_relative_error = np.abs(
        candidate_intrinsic[:, (0, 1), (0, 1)] - reference_intrinsic[:, (0, 1), (0, 1)]
    ) / np.maximum(np.abs(reference_intrinsic[:, (0, 1), (0, 1)]), 1e-8)

    result = {
        "frame_indices": reference["frame_indices"],
        "similarity_scale_candidate_to_reference": scale,
        "reference_path_length": reference_path_length,
        "candidate_path_length_raw": candidate_path_length,
        "candidate_path_length_aligned": aligned_path_length,
        "aligned_path_length_ratio": aligned_path_length / max(reference_path_length, 1e-12),
        "reference_endpoint_displacement": reference_displacement,
        "center_rmse": float(np.sqrt(np.square(center_error).mean())),
        "center_rmse_over_path_length": float(
            np.sqrt(np.square(center_error).mean()) / max(reference_path_length, 1e-12)
        ),
        "center_error_mean": float(center_error.mean()),
        "center_error_max": float(center_error.max()),
        "rotation_error_deg_mean": float(angle_error.mean()),
        "rotation_error_deg_median": float(np.median(angle_error)),
        "rotation_error_deg_max": float(angle_error.max()),
        "focal_relative_error_mean": float(focal_relative_error.mean()),
        "focal_relative_error_max": float(focal_relative_error.max()),
        "per_frame_center_error": center_error.tolist(),
        "per_frame_rotation_error_deg": angle_error.tolist(),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")


if __name__ == "__main__":
    main()
