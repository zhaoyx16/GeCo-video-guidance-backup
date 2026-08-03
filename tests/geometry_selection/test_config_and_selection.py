from __future__ import annotations

import json
from dataclasses import replace

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
from geometry_selection.window_bundle import WINDOW_EXTRACTION_MODE, make_window_bundle


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
        long_range_score=score if pairs else None,
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


def test_motion_guard_checks_net_translation_separately() -> None:
    incumbent = candidate("candidate0", 0.20, 0.10, incumbent=True)
    challenger = candidate("candidate1", 0.05, 0.10)
    incumbent = replace(
        incumbent,
        report=replace(
            incumbent.report,
            normalized_net_translation_motion=0.10,
            camera_angular_path_deg=10.0,
        ),
    )
    challenger = replace(
        challenger,
        report=replace(
            challenger.report,
            normalized_net_translation_motion=0.01,
            camera_angular_path_deg=10.0,
        ),
    )
    result = select_candidate([incumbent, challenger], SelectionConfig())
    assert result.selected_candidate_id == "candidate0"
    assert result.decision == "abstain_margin_or_motion_guard"


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


def test_sparse_common_evidence_reports_measured_edge_counts() -> None:
    incumbent = candidate("candidate0", 0.20, 0.10, incumbent=True)
    challenger = candidate("candidate1", 0.10, 0.10)
    result = select_candidate(
        [incumbent, challenger],
        SelectionConfig(min_common_local_edges=3, min_common_long_range_edges=2),
    )
    assert result.decision == "abstain_insufficient_common_evidence"
    assert result.common_local_edges == 2
    assert result.common_long_range_edges == 1


def test_common_support_requires_the_same_projection_direction() -> None:
    forward_pairs = (
        PairScore(0, 1, 1, 0.8, 0.8, 100, 0.2, 0.0, 0.2, "ok"),
        PairScore(0, 3, 3, 0.5, 0.5, 80, 0.2, 0.0, 0.2, "ok"),
    )
    reverse_pairs = (
        PairScore(1, 0, 1, 0.8, 0.8, 100, 0.1, 0.0, 0.1, "ok"),
        PairScore(3, 0, 3, 0.5, 0.5, 80, 0.1, 0.0, 0.1, "ok"),
    )

    def directional(identifier, pairs, incumbent):
        base = report(0.2 if incumbent else 0.1, 0.1)
        directional_report = GeometryScoreReport(
            **{**base.__dict__, "pairs": pairs}
        )
        return CandidateScore(
            identifier,
            int(identifier[-1]),
            identifier[-1] * 64,
            identifier[-1] * 64,
            incumbent,
            directional_report,
        )

    result = select_candidate(
        [
            directional("candidate0", forward_pairs, True),
            directional("candidate1", reverse_pairs, False),
        ],
        SelectionConfig(min_common_local_edges=1, min_common_long_range_edges=1),
    )
    assert result.decision == "abstain_insufficient_common_evidence"
    assert result.common_local_edges == 0


def test_common_support_rejects_large_coverage_imbalance() -> None:
    incumbent = candidate("candidate0", 0.20, 0.10, incumbent=True)
    challenger = candidate("candidate1", 0.10, 0.10)
    challenger_report = replace(
        challenger.report,
        pairs=tuple(
            replace(pair, comparable_fraction=0.10)
            for pair in challenger.report.pairs
        ),
    )
    challenger = replace(challenger, report=challenger_report)
    result = select_candidate(
        [incumbent, challenger],
        SelectionConfig(min_support_ratio=0.5),
    )
    assert result.decision == "abstain_insufficient_common_evidence"


def test_candidates_must_share_keyframe_timestamps() -> None:
    incumbent = candidate("candidate0", 0.20, 0.10, incumbent=True)
    challenger = candidate("candidate1", 0.10, 0.10)
    incumbent = replace(
        incumbent,
        report=replace(incumbent.report, keyframe_indices=(0, 10, 20, 30)),
    )
    challenger = replace(
        challenger,
        report=replace(challenger.report, keyframe_indices=(0, 11, 22, 33)),
    )
    with pytest.raises(ValueError, match="identical keyframe timestamps"):
        select_candidate([incumbent, challenger], SelectionConfig())


def test_selection_requires_each_candidate_to_pass_long_range_coverage() -> None:
    incumbent = candidate("candidate0", 0.20, 0.10, incumbent=True)
    challenger = candidate("candidate1", 0.10, 0.10)
    challenger = CandidateScore(
        **{
            **challenger.__dict__,
            "report": GeometryScoreReport(
                **{
                    **challenger.report.__dict__,
                    "status": "ok_local_only",
                    "long_range_score": None,
                }
            ),
        }
    )
    result = select_candidate([incumbent, challenger], SelectionConfig())
    assert result.decision == "abstain_insufficient_common_evidence"
    assert result.common_local_edges == 2
    assert result.common_long_range_edges == 1


def test_motion_guard_treats_two_zero_motion_candidates_as_equal_motion() -> None:
    result = select_candidate(
        [
            candidate("candidate0", 0.20, 0.0, incumbent=True),
            candidate("candidate1", 0.10, 0.0),
        ],
        SelectionConfig(),
    )
    assert result.selected_candidate_id == "candidate1"
    assert result.motion_ratio == 1.0


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
                "generation_metadata": None,
                "generation_metadata_sha256": None,
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
        "artifact_mode": "legacy-debug",
        "candidate_spec_sha256": "e" * 64,
        "preparation": {"commit": "d" * 40, "dirty": True},
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

    missing_provenance = json.loads(json.dumps(manifest))
    missing_provenance.pop("candidate_spec_sha256")
    path.write_text(json.dumps(missing_provenance))
    with pytest.raises(ValueError, match="candidate_spec_sha256"):
        load_and_validate_candidate_pool(path, expected_split="validation")

    missing_provenance = json.loads(json.dumps(manifest))
    missing_provenance["preparation"].pop("commit")
    path.write_text(json.dumps(missing_provenance))
    with pytest.raises(ValueError, match="preparation commit"):
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
    pool = materialize_candidate_pool(
        spec,
        keys,
        artifact_mode="legacy-debug",
        producer_identity={"commit": "d" * 40, "dirty": True},
        candidate_spec_sha256="e" * 64,
    )
    assert pool["candidate_count"] == 2
    assert pool["cases"][0]["conditioning_image_sha256"] == file_sha256(image)
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(pool))
    load_and_validate_candidate_pool(path, expected_split="debug")


def test_candidate_pool_id_binds_window_bundle(tmp_path) -> None:
    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    videos = []
    for index in range(2):
        path = tmp_path / f"candidate-{index}.mp4"
        path.write_bytes(f"video-{index}".encode())
        videos.append(path)
    extraction = {
        "mode": WINDOW_EXTRACTION_MODE,
        "num_keyframes": 6,
        "local_window_size": 4,
        "local_stride": 2,
        "loop_context": 2,
        "min_loop_node_gap": 2,
        "max_loop_windows": 1,
    }
    spec = {
        "schema": CANDIDATE_SPEC_SCHEMA,
        "split": "debug",
        "candidate_count": 2,
        "backbone": "wan2.2-ti2v-5b",
        "protocol_manifest_sha256": "d" * 64,
        "generation": {"steps": 50, "frames": 121},
        "geometry_extraction": extraction,
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
    bundles = {}
    for index, video in enumerate(videos):
        records = [
            {
                "window_id": "local-00",
                "kind": "local",
                "frame_indices": [0, 1, 2, 3],
                "frame_pixels_sha256": ["1" * 64] * 4,
                "frame_file_sha256": ["2" * 64] * 4,
                "appearance_evidence": None,
                "geometry_cache_key": f"{index + 2}" * 64,
                "independent_run_id": f"{index + 4}" * 64,
            },
            {
                "window_id": "local-01",
                "kind": "local",
                "frame_indices": [2, 3, 4, 5],
                "frame_pixels_sha256": ["1" * 64] * 4,
                "frame_file_sha256": ["2" * 64] * 4,
                "appearance_evidence": None,
                "geometry_cache_key": f"{index + 6}" * 64,
                "independent_run_id": f"{index + 8}" * 64,
            },
            {
                "window_id": "loop-00",
                "kind": "loop",
                "frame_indices": [0, 1, 4, 5],
                "frame_pixels_sha256": ["1" * 64] * 4,
                "frame_file_sha256": ["2" * 64] * 4,
                "appearance_evidence": {
                    "source_frame": 0,
                    "target_frame": 5,
                    "source_file_sha256": "2" * 64,
                    "target_file_sha256": "2" * 64,
                    "source_keypoints": 30,
                    "target_keypoints": 32,
                    "ratio_matches": 20,
                    "inliers": 15,
                    "inlier_ratio": 0.75,
                    "spatial_coverage": 0.25,
                    "mean_descriptor_distance": 0.20,
                    "status": "ok",
                    "algorithm": "orb-mutual-ratio-fundamental-ransac-v1",
                },
                "geometry_cache_key": f"{index + 10}" * 32,
                "independent_run_id": f"{index + 12}" * 32,
            },
        ]
        bundles[("case-1", f"candidate-{index}")] = make_window_bundle(
            video_sha256=file_sha256(video),
            global_geometry_cache_key=keys[("case-1", f"candidate-{index}")],
            global_keyframe_indices=range(6),
            extraction_config=extraction,
            window_records=records,
        )
    pool = materialize_candidate_pool(
        spec,
        keys,
        artifact_mode="legacy-debug",
        producer_identity={"commit": "a" * 40, "dirty": True},
        candidate_spec_sha256="e" * 64,
        geometry_window_bundles=bundles,
    )
    original_id = pool["cases"][0]["candidate_pool_id"]
    pool["cases"][0]["candidates"][0]["geometry_window_bundle"]["windows"][0][
        "independent_run_id"
    ] = "f" * 64
    assert candidate_pool_id(pool["cases"][0]) != original_id
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(pool))
    with pytest.raises(ValueError, match="candidate_pool_id mismatch"):
        load_and_validate_candidate_pool(path, expected_split="debug")


def test_yaml_config_is_strict_and_hash_is_stable(tmp_path) -> None:
    text = """
schema_version: 2
method_version: offline_v1
experiment_name: test
candidate_manifest: /tmp/candidates.json
protocol_manifest: /tmp/protocol.json
model_lock: /tmp/model_lock.json
experiment_lock: /tmp/experiment_lock.json
artifact_root: /tmp/formal_artifacts
dataset_root: /tmp/dataset
geometry_cache_root: /tmp/cache
geometry_checkpoint_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
geometry_source_tree_sha256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
geometry_source_commit: 39a0cb8af88554f15ddcb5354cd52bde588fa014
output_root: /tmp/results
expected_split: validation
score_mode: direct_reprojection
score:
  local_offsets: [1]
graph_score:
  window: {}
  optimizer: {}
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
