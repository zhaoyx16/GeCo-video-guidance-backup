"""Content-bound RGB evidence for proposed loop-closure edges."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cache import file_sha256


APPEARANCE_ALGORITHM = "orb-mutual-ratio-fundamental-ransac-v1"


@dataclass(frozen=True)
class AppearanceEvidence:
    source_frame: int
    target_frame: int
    source_file_sha256: str
    target_file_sha256: str
    source_keypoints: int
    target_keypoints: int
    ratio_matches: int
    inliers: int
    inlier_ratio: float
    spatial_coverage: float
    mean_descriptor_distance: float | None
    status: str
    algorithm: str = APPEARANCE_ALGORITHM

    def validate(self) -> None:
        if self.algorithm != APPEARANCE_ALGORITHM:
            raise ValueError(f"unsupported appearance algorithm: {self.algorithm}")
        if self.source_frame < 0 or self.target_frame <= self.source_frame:
            raise ValueError("appearance frames must be ordered non-negative indices")
        for name in ("source_file_sha256", "target_file_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        for name in (
            "source_keypoints",
            "target_keypoints",
            "ratio_matches",
            "inliers",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.inliers > self.ratio_matches:
            raise ValueError("appearance inliers cannot exceed ratio matches")
        for name in ("inlier_ratio", "spatial_coverage"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0,1]")
        if self.mean_descriptor_distance is not None and (
            not np.isfinite(self.mean_descriptor_distance)
            or not 0.0 <= self.mean_descriptor_distance <= 1.0
        ):
            raise ValueError("mean_descriptor_distance must be null or in [0,1]")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("appearance status must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AppearanceEvidence":
        if not isinstance(payload, dict):
            raise TypeError("appearance evidence must be a mapping")
        expected = {field.name for field in __import__("dataclasses").fields(cls)}
        if set(payload) != expected:
            raise ValueError("appearance evidence fields are incomplete or unknown")
        result = cls(**payload)
        result.validate()
        return result


def _ratio_matches(first_descriptors, second_descriptors, ratio: float):
    import cv2

    if first_descriptors is None or second_descriptors is None:
        return []
    if len(first_descriptors) < 2 or len(second_descriptors) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    forward = matcher.knnMatch(first_descriptors, second_descriptors, k=2)
    reverse = matcher.knnMatch(second_descriptors, first_descriptors, k=2)
    forward_good = {
        match.queryIdx: match
        for pair in forward
        if len(pair) == 2
        for match, alternative in [pair]
        if match.distance < ratio * alternative.distance
    }
    reverse_good = {
        match.queryIdx: match
        for pair in reverse
        if len(pair) == 2
        for match, alternative in [pair]
        if match.distance < ratio * alternative.distance
    }
    return [
        match
        for query, match in sorted(forward_good.items())
        if match.trainIdx in reverse_good
        and reverse_good[match.trainIdx].trainIdx == query
    ]


def _grid_coverage(points: np.ndarray, shape: tuple[int, int], grid_size: int) -> float:
    if points.size == 0:
        return 0.0
    height, width = shape
    x = np.clip((points[:, 0] / max(width, 1) * grid_size).astype(int), 0, grid_size - 1)
    y = np.clip((points[:, 1] / max(height, 1) * grid_size).astype(int), 0, grid_size - 1)
    return len(set(zip(x.tolist(), y.tolist()))) / float(grid_size * grid_size)


def score_appearance_pair(
    source_path: Path,
    target_path: Path,
    *,
    source_frame: int,
    target_frame: int,
    max_features: int = 2000,
    ratio_threshold: float = 0.75,
    ransac_threshold_px: float = 1.5,
    grid_size: int = 4,
) -> AppearanceEvidence:
    """Match decoded RGB frames and verify matches by epipolar RANSAC.

    This is intentionally independent of VGGT geometry.  It answers only
    whether the two decoded frames contain enough repeated visual content for
    a long-range edge to be treated as a loop-closure proposal.
    """

    import cv2

    source = Path(source_path).resolve()
    target = Path(target_path).resolve()
    if not source.is_file() or not target.is_file():
        raise FileNotFoundError(source if not source.is_file() else target)
    if max_features < 32 or not 0.0 < ratio_threshold < 1.0:
        raise ValueError("appearance matcher parameters are invalid")
    if ransac_threshold_px <= 0 or grid_size < 2:
        raise ValueError("RANSAC/grid parameters are invalid")
    first = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    second = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
    if first is None or second is None:
        raise RuntimeError("could not decode appearance-matching image")
    cv2.setRNGSeed(0)
    detector = cv2.ORB_create(nfeatures=max_features, fastThreshold=10)
    first_keypoints, first_descriptors = detector.detectAndCompute(first, None)
    second_keypoints, second_descriptors = detector.detectAndCompute(second, None)
    matches = _ratio_matches(first_descriptors, second_descriptors, ratio_threshold)
    source_hash = file_sha256(source)
    target_hash = file_sha256(target)
    if len(matches) < 8:
        return AppearanceEvidence(
            source_frame=source_frame,
            target_frame=target_frame,
            source_file_sha256=source_hash,
            target_file_sha256=target_hash,
            source_keypoints=len(first_keypoints),
            target_keypoints=len(second_keypoints),
            ratio_matches=len(matches),
            inliers=0,
            inlier_ratio=0.0,
            spatial_coverage=0.0,
            mean_descriptor_distance=None,
            status="insufficient_ratio_matches",
        )
    first_points = np.float32([first_keypoints[match.queryIdx].pt for match in matches])
    second_points = np.float32([second_keypoints[match.trainIdx].pt for match in matches])
    _, mask = cv2.findFundamentalMat(
        first_points,
        second_points,
        cv2.FM_RANSAC,
        ransac_threshold_px,
        0.999,
    )
    inlier_mask = (
        np.zeros(len(matches), dtype=bool)
        if mask is None
        else np.asarray(mask).reshape(-1).astype(bool)
    )
    if inlier_mask.size != len(matches):
        inlier_mask = np.zeros(len(matches), dtype=bool)
    inliers = int(inlier_mask.sum())
    source_coverage = _grid_coverage(
        first_points[inlier_mask], first.shape[:2], grid_size
    )
    target_coverage = _grid_coverage(
        second_points[inlier_mask], second.shape[:2], grid_size
    )
    distances = np.asarray(
        [match.distance / 256.0 for match, keep in zip(matches, inlier_mask) if keep],
        dtype=np.float64,
    )
    return AppearanceEvidence(
        source_frame=source_frame,
        target_frame=target_frame,
        source_file_sha256=source_hash,
        target_file_sha256=target_hash,
        source_keypoints=len(first_keypoints),
        target_keypoints=len(second_keypoints),
        ratio_matches=len(matches),
        inliers=inliers,
        inlier_ratio=inliers / float(len(matches)),
        spatial_coverage=min(source_coverage, target_coverage),
        mean_descriptor_distance=(float(distances.mean()) if distances.size else None),
        status="ok" if inliers >= 8 else "insufficient_ransac_inliers",
    )
