from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.c2f_geometry_validation.run_stage_a_geometry import (
    project_rows,
    sample_record_columns,
    token_to_processed,
)
from geometry_selection.schema import GeometryPrediction


def make_geometry(target_depth: float = 5.0) -> GeometryPrediction:
    height, width = 220, 400
    intrinsics = np.array(
        [[200.0, 0.0, 199.5], [0.0, 200.0, 109.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    geometry = GeometryPrediction(
        world_to_camera=np.stack([np.eye(4), np.eye(4)]).astype(np.float32),
        intrinsics=np.stack([intrinsics, intrinsics]),
        depth=np.stack(
            [
                np.full((height, width), 5.0, dtype=np.float32),
                np.full((height, width), target_depth, dtype=np.float32),
            ]
        ),
        confidence=np.ones((2, height, width), dtype=np.float32),
        keyframe_indices=np.array([36, 40], dtype=np.int64),
    )
    geometry.validate()
    return geometry


def row(target_x: int = 12) -> dict:
    return {
        "target_time": 10,
        "source_time": 9,
        "source_x": 10,
        "source_y": 8,
        "target_x": target_x,
        "target_y": 8,
        "confidence": 0.8,
        "similarity": 0.9,
        "selected_lag": 1,
    }


def test_token_centres_cover_processed_grid() -> None:
    values = token_to_processed(np.array([0, 39]), 40, 400)
    assert np.allclose(values, [4.5, 394.5])


def test_identical_camera_same_token_is_accepted() -> None:
    result = project_rows(
        [row(target_x=10)],
        make_geometry(),
        {36: 0, 40: 1},
        (31, 22, 40),
        4,
        20.0,
        0.15,
        1.5,
    )[0]
    assert result["geometry_status"] == "accept"
    assert result["reprojection_error_tokens"] < 1e-6


def test_wrong_target_location_is_rejected() -> None:
    result = project_rows(
        [row(target_x=15)],
        make_geometry(),
        {36: 0, 40: 1},
        (31, 22, 40),
        4,
        20.0,
        0.15,
        1.5,
    )[0]
    assert result["geometry_status"] == "reject_reprojection"
    assert result["reprojection_error_tokens"] == 5.0


def test_source_behind_target_abstains_as_occluded() -> None:
    result = project_rows(
        [row(target_x=10)],
        make_geometry(target_depth=4.0),
        {36: 0, 40: 1},
        (31, 22, 40),
        4,
        20.0,
        0.15,
        1.5,
    )[0]
    assert result["geometry_status"] == "abstain_occluded"


def test_columnar_samples_are_filtered_by_target_time() -> None:
    record = {
        "samples": {
            "target_time": [9, 10, 25],
            "source_time": [8, 9, 24],
            "target_x": [0, 1, 2],
        }
    }
    rows = sample_record_columns(record, {10, 25})
    assert [item["target_time"] for item in rows] == [10, 25]
