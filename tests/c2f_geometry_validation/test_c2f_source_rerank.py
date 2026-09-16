from __future__ import annotations

import torch

from geometry_selection.c2f_gate import prepare_geometry_bundle
from geometry_selection.c2f_source_rerank import (
    candidate_pairwise_geometry,
    candidate_policy_selections,
    candidate_source_support,
    select_candidate_source,
    uniform_scale_to_target_update,
)


TOKEN_GRID = (4, 2, 2)


def _geometry(depth: torch.Tensor | None = None) -> dict:
    frames = TOKEN_GRID[0]
    height = width = 8
    if depth is None:
        depth = torch.full((frames, height, width), 2.0)
    intrinsics = torch.tensor(
        [[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]]
    ).repeat(frames, 1, 1)
    return prepare_geometry_bundle(
        {
            "world_to_camera": torch.eye(4).repeat(frames, 1, 1),
            "intrinsics": intrinsics,
            "depth": depth,
            "confidence": torch.ones_like(depth),
            "confidence_thresholds": torch.zeros(frames),
        },
        device=torch.device("cpu"),
        expected_frames=frames,
    )


def _candidates() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = 1
    target_frames = TOKEN_GRID[0] - 1
    spatial = TOKEN_GRID[1] * TOKEN_GRID[2]
    candidates = 3
    source_index = torch.arange(spatial).view(1, 1, spatial, 1).expand(
        batch, target_frames, spatial, candidates
    ).clone()
    source_time = torch.full_like(source_index, -1)
    candidate_valid = torch.zeros_like(source_index, dtype=torch.bool)
    confidence = torch.zeros_like(source_index, dtype=torch.float32)
    for target_slot in range(target_frames):
        target_time = target_slot + 1
        for lag in range(1, candidates + 1):
            if target_time - lag >= 0:
                source_time[:, target_slot, :, lag - 1] = target_time - lag
                candidate_valid[:, target_slot, :, lag - 1] = True
                confidence[:, target_slot, :, lag - 1] = 0.9 - 0.1 * (lag - 1)
    return source_index, source_time, confidence, candidate_valid


def test_rigid_identity_geometry_accepts_candidates_and_supports_sources() -> None:
    source_index, source_time, confidence, valid = _candidates()
    geometry = _geometry()
    hard, info = candidate_pairwise_geometry(
        source_index,
        source_time,
        confidence,
        valid,
        TOKEN_GRID,
        geometry,
        weighting="hard",
    )
    soft, _ = candidate_pairwise_geometry(
        source_index,
        source_time,
        confidence,
        valid,
        TOKEN_GRID,
        geometry,
        weighting="soft",
    )
    score, source_info = candidate_source_support(
        source_index,
        source_time,
        confidence,
        valid,
        TOKEN_GRID,
        geometry,
    )

    assert torch.all(hard[valid] == 1)
    assert torch.allclose(soft[valid], torch.ones_like(soft[valid]), atol=1e-6)
    assert torch.all(info["accepted"][valid])
    assert torch.allclose(score[valid], torch.ones_like(score[valid]))
    assert int(source_info["conflict"].sum()) == 0


def test_source_front_conflict_lowers_score() -> None:
    source_index, source_time, confidence, valid = _candidates()
    depth = torch.full((TOKEN_GRID[0], 8, 8), 2.0)
    depth[0] = 4.0
    score, info = candidate_source_support(
        source_index,
        source_time,
        confidence,
        valid,
        TOKEN_GRID,
        _geometry(depth),
    )
    # Target time 2, lag 1 selects source time 1, whose only prior support is frame 0.
    assert torch.allclose(score[0, 1, :, 0], torch.full((4,), 0.5))
    assert torch.all(info["conflict"][0, 1, :, 0, 0])
    assert not torch.any(info["support"][0, 1, :, 0, 0])


def test_occlusion_is_unknown_not_conflict() -> None:
    source_index, source_time, confidence, valid = _candidates()
    depth = torch.full((TOKEN_GRID[0], 8, 8), 2.0)
    depth[0] = 1.0
    score, info = candidate_source_support(
        source_index,
        source_time,
        confidence,
        valid,
        TOKEN_GRID,
        _geometry(depth),
    )
    assert torch.allclose(score[0, 1, :, 0], torch.ones(4))
    assert not torch.any(info["conflict"][0, 1, :, 0, 0])
    assert torch.all(info["unknown"][0, 1, :, 0, 0])


def test_source_score_changes_selection_without_rescaling_update() -> None:
    source_index = torch.tensor([[[[11, 22]]]])
    source_time = torch.tensor([[[[2, 1]]]])
    confidence = torch.tensor([[[[0.9, 0.8]]]])
    valid = torch.ones_like(source_index, dtype=torch.bool)
    pair_weight = torch.ones_like(confidence)
    source_score = torch.tensor([[[[0.25, 1.0]]]])

    p = select_candidate_source(
        source_index,
        source_time,
        confidence,
        valid,
        pair_weight,
        source_score,
        policy="P",
    )
    s = select_candidate_source(
        source_index,
        source_time,
        confidence,
        valid,
        pair_weight,
        source_score,
        policy="S",
    )
    assert p["source_index"].item() == 11
    assert s["source_index"].item() == 22
    assert torch.allclose(s["update_weight"], torch.tensor([[[0.8]]]))


def test_unit_source_score_recovers_pairwise_policy_exactly() -> None:
    source_index, source_time, confidence, valid = _candidates()
    pair_weight = torch.rand_like(confidence)
    selections = candidate_policy_selections(
        source_index,
        source_time,
        confidence,
        valid,
        pair_weight,
        torch.ones_like(confidence),
    )
    for field in ("choice", "source_index", "source_time", "valid"):
        assert torch.equal(selections["P"][field], selections["S"][field])
    for field in ("confidence", "pair_weight", "update_weight"):
        assert torch.equal(selections["P"][field], selections["S"][field])


def test_all_invalid_candidates_are_noop() -> None:
    source_index = torch.zeros((1, 1, 1, 3), dtype=torch.long)
    source_time = torch.zeros_like(source_index)
    confidence = torch.zeros_like(source_index, dtype=torch.float32)
    valid = torch.zeros_like(source_index, dtype=torch.bool)
    selected = select_candidate_source(
        source_index,
        source_time,
        confidence,
        valid,
        torch.zeros_like(confidence),
        torch.ones_like(confidence),
        policy="S",
    )
    assert not selected["valid"].item()
    assert selected["update_weight"].item() == 0.0


def test_uniform_control_matches_target_norm_when_target_is_weaker() -> None:
    base_residual = torch.tensor([[[[3.0, 4.0], [0.0, 2.0]]]])
    target_residual = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    base_weight = torch.tensor([[[1.0, 1.0]]])
    target_weight = torch.tensor([[[1.0, 1.0]]])
    scale, info = uniform_scale_to_target_update(
        base_residual,
        base_weight,
        target_residual,
        target_weight,
    )
    assert 0.0 < scale.item() < 1.0
    assert torch.allclose(info["actual_update_norm"], info["target_update_norm"])
    assert info["norm_match_relative_error"].item() < 1e-6
