from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_selection.c2f_gate import geometry_validation_gate, prepare_geometry_bundle, uniform_update_scale
from geometry_selection.schema import GeometryPrediction
from benchmarks.c2f_geometry_validation.run_stage_a_geometry import project_rows


def make_bundle(*, frames: int = 2, height: int = 4, width: int = 4):
    world_to_camera = torch.eye(4).repeat(frames, 1, 1)
    intrinsics = torch.tensor(
        [[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]]
    ).repeat(frames, 1, 1)
    return prepare_geometry_bundle(
        {
            "world_to_camera": world_to_camera,
            "intrinsics": intrinsics,
            "depth": torch.ones(frames, height, width),
            "confidence": torch.ones(frames, height, width),
            "confidence_thresholds": torch.zeros(frames),
        },
        device=torch.device("cpu"),
        expected_frames=frames,
    )


def test_identity_geometry_accepts_identity_correspondence():
    geometry = make_bundle()
    source_index = torch.arange(4).reshape(1, 1, 4)
    source_time = torch.zeros_like(source_index)
    gate, info = geometry_validation_gate(source_index, source_time, (2, 2, 2), geometry)
    assert gate.shape == (1, 1, 4, 1)
    assert gate.bool().all()
    assert info["accepted"].all()
    assert torch.allclose(info["reprojection_error_tokens"], torch.zeros(1, 1, 4))


def test_camera_translation_rejects_unshifted_c2f_target():
    geometry = make_bundle(height=8, width=8)
    geometry["world_to_camera"][1, 0, 3] = 1.0
    geometry["camera_to_world"] = torch.linalg.inv(geometry["world_to_camera"])
    source_index = torch.tensor([[[0, 1, 2, 3]]])
    source_time = torch.zeros_like(source_index)
    gate, info = geometry_validation_gate(
        source_index,
        source_time,
        (2, 2, 2),
        geometry,
        max_error_tokens=0.25,
    )
    assert not gate.bool().any()
    assert info["rejected_reprojection"].any()


def test_nearer_target_surface_abstains_as_occluded():
    geometry = make_bundle()
    geometry["depth"][1].fill_(0.5)
    source_index = torch.arange(4).reshape(1, 1, 4)
    source_time = torch.zeros_like(source_index)
    gate, info = geometry_validation_gate(source_index, source_time, (2, 2, 2), geometry)
    assert not gate.bool().any()
    assert info["occluded"].all()
    assert not info["front_conflict"].any()


def test_uniform_control_matches_gated_update_norm():
    residual = torch.tensor([[[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]]])
    base_weight = torch.full((1, 1, 4, 1), 0.25)
    gate = torch.tensor([[[[1.0], [0.0], [1.0], [0.0]]]])
    scale, norms = uniform_update_scale(residual, base_weight, gate)
    expected = torch.sqrt(torch.tensor((1.0**2 + 3.0**2) / (1.0**2 + 2.0**2 + 3.0**2 + 4.0**2)))
    assert torch.allclose(scale, expected)
    assert torch.allclose(norms["uniform_update_norm"], norms["gated_update_norm"])


def test_geometry_bundle_rejects_wrong_temporal_length():
    bundle = {
        "world_to_camera": torch.eye(4).repeat(2, 1, 1),
        "intrinsics": torch.eye(3).repeat(2, 1, 1),
        "depth": torch.ones(2, 4, 4),
        "confidence": torch.ones(2, 4, 4),
    }
    try:
        prepare_geometry_bundle(bundle, device=torch.device("cpu"), expected_frames=3)
    except ValueError as error:
        assert "world_to_camera" in str(error)
    else:
        raise AssertionError("expected temporal shape validation to fail")


def test_torch_gate_matches_stage_a_numpy_evidence_rule():
    intrinsics = np.array(
        [[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    prediction = GeometryPrediction(
        world_to_camera=np.stack([np.eye(4), np.eye(4)]).astype(np.float32),
        intrinsics=np.stack([intrinsics, intrinsics]),
        depth=np.ones((2, 4, 4), dtype=np.float32),
        confidence=np.ones((2, 4, 4), dtype=np.float32),
        keyframe_indices=np.array([0, 4]),
    )
    prediction.validate()
    source_indices = [0, 0, 2, 3]
    rows = []
    for target_index, source_index in enumerate(source_indices):
        rows.append(
            {
                "source_time": 0,
                "target_time": 1,
                "source_x": source_index % 2,
                "source_y": source_index // 2,
                "target_x": target_index % 2,
                "target_y": target_index // 2,
            }
        )
    numpy_rows = project_rows(
        rows,
        prediction,
        {0: 0, 4: 1},
        (2, 2, 2),
        4,
        20.0,
        0.15,
        0.25,
    )
    expected = torch.tensor([row["geometry_status"] == "accept" for row in numpy_rows])
    prepared = prepare_geometry_bundle(
        {
            "world_to_camera": prediction.world_to_camera,
            "intrinsics": prediction.intrinsics,
            "depth": prediction.depth,
            "confidence": prediction.confidence,
        },
        device=torch.device("cpu"),
        expected_frames=2,
    )
    gate, _ = geometry_validation_gate(
        torch.tensor(source_indices).reshape(1, 1, 4),
        torch.zeros((1, 1, 4), dtype=torch.long),
        (2, 2, 2),
        prepared,
        max_error_tokens=0.25,
    )
    assert torch.equal(gate.reshape(-1).bool(), expected)
