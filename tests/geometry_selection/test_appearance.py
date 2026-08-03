from __future__ import annotations

import numpy as np
import pytest

from geometry_selection.appearance import AppearanceEvidence, score_appearance_pair


cv2 = pytest.importorskip("cv2")


def _textured_image(size: int = 256) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.uint8)
    rng = np.random.default_rng(7)
    for _ in range(120):
        x, y = rng.integers(12, size - 12, size=2)
        radius = int(rng.integers(2, 6))
        colour = int(rng.integers(80, 255))
        cv2.circle(image, (int(x), int(y)), radius, colour, -1)
    cv2.putText(image, "LOOP", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)
    return image


def test_appearance_match_accepts_reobserved_content(tmp_path) -> None:
    source = _textured_image()
    transform = np.float32([[1.0, 0.0, 7.0], [0.0, 1.0, 4.0]])
    target = cv2.warpAffine(source, transform, source.shape[::-1])
    source_path = tmp_path / "source.png"
    target_path = tmp_path / "target.png"
    assert cv2.imwrite(str(source_path), source)
    assert cv2.imwrite(str(target_path), target)

    evidence = score_appearance_pair(
        source_path,
        target_path,
        source_frame=0,
        target_frame=7,
    )
    evidence.validate()
    assert evidence.status == "ok"
    assert evidence.inliers >= 8
    assert evidence.inlier_ratio > 0.5
    assert evidence.spatial_coverage > 0.1


def test_appearance_match_rejects_untextured_pair(tmp_path) -> None:
    source_path = tmp_path / "source.png"
    target_path = tmp_path / "target.png"
    assert cv2.imwrite(str(source_path), np.zeros((128, 128), dtype=np.uint8))
    assert cv2.imwrite(str(target_path), np.full((128, 128), 255, dtype=np.uint8))
    evidence = score_appearance_pair(
        source_path,
        target_path,
        source_frame=1,
        target_frame=9,
    )
    assert evidence.status == "insufficient_ratio_matches"
    assert evidence.inliers == 0


def test_appearance_evidence_rejects_tampered_digest() -> None:
    payload = {
        "source_frame": 0,
        "target_frame": 7,
        "source_file_sha256": "a" * 64,
        "target_file_sha256": "b" * 64,
        "source_keypoints": 20,
        "target_keypoints": 21,
        "ratio_matches": 10,
        "inliers": 8,
        "inlier_ratio": 0.8,
        "spatial_coverage": 0.25,
        "mean_descriptor_distance": 0.2,
        "status": "ok",
        "algorithm": "orb-mutual-ratio-fundamental-ransac-v1",
    }
    evidence = AppearanceEvidence.from_dict(payload)
    assert evidence.to_dict() == payload
    payload["source_file_sha256"] = "bad"
    with pytest.raises(ValueError, match="source_file_sha256"):
        AppearanceEvidence.from_dict(payload)
