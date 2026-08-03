from __future__ import annotations

import json

import numpy as np
import pytest

from geometry_selection.config import (
    load_offline_ranking_config,
    write_resolved_config,
)
from geometry_selection.scorer import GeometryScoreReport, PairScore, ScorerConfig
from geometry_selection.selection import (
    CANDIDATE_POOL_SCHEMA,
    CANDIDATE_SPEC_SCHEMA,
    CandidateScore,
    SelectionConfig,
    candidate_pool_id,
    deterministic_random_candidate,
    file_sha256,
    load_and_validate_candidate_pool,
    materialize_candidate_pool,
    pairing_id,
    select_candidate,
)


def report(score: float, motion: float, status: str = "ok") -> GeometryScoreReport:
    pairs = ()
    if status.startswith("ok") and np.isfinite(score):
        pairs = (
            PairScore(0, 1, 1, 0.8, 0.8, 100, score, 0.0, score, "ok"),
            PairScore(1, 0, 1, 0.8, 0.8, 100, score, 0.0, score, "ok"),
            PairScore(1, 2, 1, 0.8, 0.8, 100, score, 0.0, score, "ok"),
            PairScore(2, 1, 1, 0.8, 0.8, 100, score, 0.0, score, "ok"),
            PairScore(0, 3, 3, 0.5, 0.5, 80, score, 0.0, score, "ok"),
            PairScore(3, 0, 3, 0.5, 0.5, 80, score, 0.0, score, "ok"),
        )
    return GeometryScoreReport(
        total_score=score,
        local_score=score if np.isfinite(score) else None,
        long_range_score=None,
        camera_path_length=motion * 5.0,
        normalized_translation_motion=motion,
        camera_angular_path_deg=0.0,
        normalized_camera_motion=motion,
        local_edge_fraction=1.0 if pairs else 0.0,
        long_range_edge_fraction=1.0 if pairs else 0.0,
        valid_local_edges=2 if np.isfinite(score) else 0,
        valid_long_range_edges=2 if pairs else 0,
        status=status,
        pairs=pairs,
        config=ScorerConfig(require_long_range=False),
    )


def candidate(identifier: str, score: float, motion: float, *, incumbent: bool = False) -> CandidateScore:
    return CandidateScore(
        candidate_id=identifier,
        seed=int(identifier[-1]),
        video_sha256=identifier[0] * 64,
        geometry_cache_key=identifier[-1] * 64,
        is_incumbent=incumbent,
        report=report(score, motion),
    )


def test_selection_uses_geometry_but_abstains_on_motion_suppression() -> None:
    incumbent = candidate("candidate0", 0.20, 0.10, incumbent=True)
    improved = candidate("candidate1", 0.10, 0.095)
    selected = select_candidate([incumbent, improved], SelectionConfig())
    assert selected.selected_candidate_id == "candidate1"
    assert selected.decision == "select_geometry_best"

    frozen = candidate("candidate1", 0.01, 0.02)
    abstained = select_candidate([incumbent, frozen], SelectionConfig())
    assert abstained.selected_candidate_id == "candidate0"
    assert abstained.decision == "abstain_margin_or_motion_guard"


def test_selection_uses_eligible_runner_up_when_lowest_score_suppresses_motion() -> None:
    incumbent = candidate("candidate0", 0.20, 0.10, incumbent=True)
    frozen_best = candidate("candidate1", 0.01, 0.02)
    eligible_runner_up = candidate("candidate2", 0.10, 0.095)
    selected = select_candidate(
        [incumbent, frozen_best, eligible_runner_up],
        SelectionConfig(),
    )
    assert selected.selected_candidate_id == "candidate2"
    assert selected.decision == "select_geometry_best"


def test_all_invalid_candidates_retain_incumbent() -> None:
    candidates = [
        CandidateScore("candidate0", 0, "a" * 64, "0" * 64, True, report(float("inf"), 0.0, "no_valid_local_edges")),
        CandidateScore("candidate1", 1, "b" * 64, "1" * 64, False, report(float("inf"), 0.0, "no_valid_local_edges")),
    ]
    result = select_candidate(candidates, SelectionConfig())
    assert result.selected_candidate_id == "candidate0"
    assert result.decision == "abstain_insufficient_common_evidence"
    serialized = result.to_dict()
    assert serialized["candidates"][0]["report"]["total_score"] is None
    json.dumps(serialized, allow_nan=False)


def test_candidate_pool_enforces_pairing_hashes_and_incumbent(tmp_path) -> None:
    videos = []
    for index in range(2):
        path = tmp_path / f"candidate_{index}.mp4"
        path.write_bytes(f"video-{index}".encode())
        videos.append(path)
    case = {
        "case_id": "case-1",
        "protocol_manifest_sha256": "d" * 64,
        "scene_uid": "dl3dv:abc",
        "split": "validation",
        "conditioning_image": str(videos[0]),
        "conditioning_image_sha256": "c" * 64,
        "prompt": "camera moves forward",
        "backbone": "wan2.2-ti2v-5b",
        "generation": {"steps": 50, "frames": 121, "height": 704, "width": 1280},
        "candidates": [
            {
                "candidate_id": f"candidate-{index}",
                "seed": index,
                "video": str(path),
                "video_sha256": file_sha256(path),
                "geometry_cache_key": str(index) * 64,
                "is_incumbent": index == 0,
            }
            for index, path in enumerate(videos)
        ],
    }
    case["conditioning_image_sha256"] = file_sha256(videos[0])
    case["pairing_id"] = pairing_id(case)
    case["candidate_pool_id"] = candidate_pool_id(case)
    manifest = {
        "schema": CANDIDATE_POOL_SCHEMA,
        "protocol_manifest_sha256": "d" * 64,
        "candidate_count": 2,
        "cases": [case],
    }
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(manifest))
    loaded = load_and_validate_candidate_pool(path, expected_split="validation")
    assert loaded["cases"][0]["candidate_pool_id"] == case["candidate_pool_id"]
    assert deterministic_random_candidate(case) in {"candidate-0", "candidate-1"}

    loaded["cases"][0]["candidates"][0]["geometry_cache_key"] = "f" * 64
    path.write_text(json.dumps(loaded))
    with pytest.raises(ValueError, match="candidate_pool_id mismatch"):
        load_and_validate_candidate_pool(path, expected_split="validation")

    loaded = json.loads(json.dumps(manifest))
    loaded["cases"][0]["split"] = "test"
    path.write_text(json.dumps(loaded))
    with pytest.raises(ValueError, match="belongs to split|pairing_id mismatch"):
        load_and_validate_candidate_pool(path, expected_split="validation")

    loaded = json.loads(json.dumps(manifest))
    loaded["cases"][0]["prompt"] = "changed"
    path.write_text(json.dumps(loaded))
    with pytest.raises(ValueError, match="pairing_id mismatch"):
        load_and_validate_candidate_pool(path, expected_split="validation")

    case["candidates"][0].pop("geometry_cache_key")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="geometry_cache_key"):
        load_and_validate_candidate_pool(path, expected_split="validation")


def test_materialize_candidate_pool_hashes_inputs(tmp_path) -> None:
    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    videos = []
    for index in range(2):
        video = tmp_path / f"video-{index}.mp4"
        video.write_bytes(f"video-{index}".encode())
        videos.append(video)
    spec = {
        "schema": CANDIDATE_SPEC_SCHEMA,
        "split": "debug",
        "candidate_count": 2,
        "backbone": "wan2.2-ti2v-5b",
        "protocol_manifest_sha256": "d" * 64,
        "generation": {"steps": 50, "frames": 121},
        "cases": [
            {
                "case_id": "case-1",
                "scene_uid": "dl3dv:scene-1",
                "conditioning_image": str(image),
                "prompt": "camera moves forward",
                "candidates": [
                    {
                        "candidate_id": f"candidate-{index}",
                        "seed": index,
                        "video": str(video),
                        "is_incumbent": index == 0,
                    }
                    for index, video in enumerate(videos)
                ],
            }
        ],
    }
    keys = {("case-1", f"candidate-{index}"): str(index) * 64 for index in range(2)}
    pool = materialize_candidate_pool(spec, keys)
    assert pool["candidate_count"] == 2
    assert pool["cases"][0]["conditioning_image_sha256"] == file_sha256(image)
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(pool))
    load_and_validate_candidate_pool(path, expected_split="debug")


def test_yaml_config_is_strict_and_hash_is_stable(tmp_path) -> None:
    text = """
schema_version: 1
method_version: offline_v1
experiment_name: test
candidate_manifest: /tmp/candidates.json
geometry_cache_root: /tmp/cache
output_root: /tmp/results
expected_split: validation
score:
  local_offsets: [1]
selection:
  abstain: true
"""
    path = tmp_path / "config.yaml"
    path.write_text(text)
    first = load_offline_ranking_config(path)
    second = load_offline_ranking_config(path)
    assert first.config_hash == second.config_hash
    assert first.scorer.local_offsets == (1,)
    resolved = tmp_path / "resolved.yaml"
    write_resolved_config(first, resolved)
    round_trip = load_offline_ranking_config(resolved)
    assert round_trip == first
    assert round_trip.config_hash == first.config_hash

    relocated = text.replace("output_root: /tmp/results", "output_root: /another/place")
    path.write_text(relocated)
    assert load_offline_ranking_config(path).config_hash == first.config_hash

    path.write_text(text + "unexpected: 1\n")
    with pytest.raises(ValueError, match="unknown"):
        load_offline_ranking_config(path)

    path.write_text(text.replace("abstain: true", 'abstain: "false"'))
    with pytest.raises(TypeError, match="boolean"):
        load_offline_ranking_config(path)
