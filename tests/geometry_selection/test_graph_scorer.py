from __future__ import annotations

from dataclasses import replace

import numpy as np

from geometry_selection.graph import se3_exp
from geometry_selection.graph_scorer import GraphScoreConfig, score_window_pose_graph
from geometry_selection.schema import GeometryPrediction
from geometry_selection.selection import CandidateScore, SelectionConfig, select_candidate
from geometry_selection.window_graph import IndependentWindow, WindowGraphConfig


def _prediction(frames, *, scale=1.0, loop_error=0.0):
    poses = []
    for offset, frame in enumerate(frames):
        pose = se3_exp(
            np.array(
                [
                    scale * float(frame),
                    0.0,
                    0.0,
                    0.0,
                    0.01 * frame + loop_error * offset,
                    0.0,
                ]
            )
        )
        poses.append(np.linalg.inv(pose))
    count = len(frames)
    depth = np.full((count, 16, 16), 3.0 * scale)
    intrinsics = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    intrinsics[:, 0, 0] = intrinsics[:, 1, 1] = 10.0
    intrinsics[:, 0, 2] = intrinsics[:, 1, 2] = 8.0
    return GeometryPrediction(
        world_to_camera=np.stack(poses),
        intrinsics=intrinsics,
        depth=depth,
        confidence=np.full_like(depth, 3.0),
        keyframe_indices=np.asarray(frames),
    )


def _windows(loop_error=0.0):
    specs = (
        ("local-a", "local", (0, 1, 2, 3), 1.0, 0.0),
        ("local-b", "local", (2, 3, 4, 5), 2.0, 0.0),
        ("loop", "loop", (0, 1, 4, 5), 0.5, loop_error),
    )
    return tuple(
        IndependentWindow(
            window_id=identifier,
            kind=kind,
            prediction=_prediction(frames, scale=scale, loop_error=error),
            independent_run_id=f"run-{identifier}-{loop_error}",
        )
        for identifier, kind, frames, scale, error in specs
    )


def _config():
    return GraphScoreConfig(
        window=WindowGraphConfig(
            min_depth_scale_pixels=2,
            depth_sample_stride=2,
            require_reobservation_support=False,
            require_appearance_support=False,
        )
    )


def test_graph_scorer_prefers_consistent_independent_windows() -> None:
    global_prediction = _prediction((0, 1, 2, 3, 4, 5))
    consistent = score_window_pose_graph(
        global_prediction,
        _windows(0.0),
        graph_config=_config(),
    )
    inconsistent = score_window_pose_graph(
        global_prediction,
        _windows(0.20),
        graph_config=_config(),
    )
    assert consistent.status == "ok_pose_graph"
    assert inconsistent.status == "ok_pose_graph"
    assert consistent.total_score < inconsistent.total_score
    assert consistent.score_kind == "pose_graph"


def test_local_only_graph_score_is_independent_of_loop_prediction() -> None:
    global_prediction = _prediction((0, 1, 2, 3, 4, 5))
    config = GraphScoreConfig(
        window=WindowGraphConfig(
            use_loop_edges=False,
            require_loop_edges=False,
            min_depth_scale_pixels=2,
            depth_sample_stride=2,
        ),
        missing_loop_penalty_weight=0.0,
    )
    first = score_window_pose_graph(
        global_prediction,
        _windows(0.0),
        graph_config=config,
    )
    second = score_window_pose_graph(
        global_prediction,
        _windows(0.8),
        graph_config=config,
    )

    assert first.status == second.status == "ok_pose_graph"
    assert first.total_score == second.total_score
    assert first.long_range_score is None
    assert first.graph_diagnostics["window_scale_ids"] == ["local-a", "local-b"]
    assert first.graph_diagnostics["potential_loop_edge_ids"] == []


def test_candidate_selection_uses_normalized_graph_score() -> None:
    global_prediction = _prediction((0, 1, 2, 3, 4, 5))
    worse = score_window_pose_graph(global_prediction, _windows(0.20), graph_config=_config())
    better = score_window_pose_graph(global_prediction, _windows(0.0), graph_config=_config())
    result = select_candidate(
        [
            CandidateScore("candidate0", 0, "a" * 64, "0" * 64, True, worse),
            CandidateScore("candidate1", 1, "b" * 64, "1" * 64, False, better),
        ],
        SelectionConfig(
            min_relative_improvement=0.0,
            min_motion_ratio=0.1,
            max_motion_ratio=10.0,
            min_net_translation_ratio=0.1,
            max_net_translation_ratio=10.0,
            min_rotation_ratio=0.1,
            max_rotation_ratio=10.0,
            min_common_local_edges=1,
            min_common_long_range_edges=1,
        ),
    )
    assert result.selected_candidate_id == "candidate1"


def test_pose_graph_selection_abstains_on_different_accepted_evidence() -> None:
    global_prediction = _prediction((0, 1, 2, 3, 4, 5))
    incumbent = score_window_pose_graph(
        global_prediction, _windows(0.20), graph_config=_config()
    )
    challenger = score_window_pose_graph(
        global_prediction, _windows(0.0), graph_config=_config()
    )
    diagnostics = dict(challenger.graph_diagnostics)
    diagnostics["accepted_loop_edge_ids"] = []
    challenger = replace(challenger, graph_diagnostics=diagnostics)
    result = select_candidate(
        [
            CandidateScore("candidate0", 0, "a" * 64, "0" * 64, True, incumbent),
            CandidateScore("candidate1", 1, "b" * 64, "1" * 64, False, challenger),
        ],
        SelectionConfig(
            min_relative_improvement=0.0,
            min_motion_ratio=0.1,
            max_motion_ratio=10.0,
            min_net_translation_ratio=0.1,
            max_net_translation_ratio=10.0,
            min_rotation_ratio=0.1,
            max_rotation_ratio=10.0,
            min_common_local_edges=1,
            min_common_long_range_edges=1,
        ),
    )
    assert result.selected_candidate_id == "candidate0"
    assert result.decision == "abstain_insufficient_common_evidence"


def test_pose_graph_selection_abstains_on_different_graph_config_hash() -> None:
    global_prediction = _prediction((0, 1, 2, 3, 4, 5))
    incumbent = score_window_pose_graph(
        global_prediction, _windows(0.20), graph_config=_config()
    )
    challenger = score_window_pose_graph(
        global_prediction, _windows(0.0), graph_config=_config()
    )
    diagnostics = dict(challenger.graph_diagnostics)
    diagnostics["graph_score_config_sha256"] = "f" * 64
    challenger = replace(challenger, graph_diagnostics=diagnostics)
    result = select_candidate(
        [
            CandidateScore("candidate0", 0, "a" * 64, "0" * 64, True, incumbent),
            CandidateScore("candidate1", 1, "b" * 64, "1" * 64, False, challenger),
        ],
        SelectionConfig(
            min_relative_improvement=0.0,
            min_motion_ratio=0.1,
            max_motion_ratio=10.0,
            min_net_translation_ratio=0.1,
            max_net_translation_ratio=10.0,
            min_rotation_ratio=0.1,
            max_rotation_ratio=10.0,
            min_common_local_edges=1,
            min_common_long_range_edges=1,
        ),
    )
    assert result.selected_candidate_id == "candidate0"
    assert result.decision == "abstain_insufficient_common_evidence"


def test_insufficient_graph_evidence_returns_invalid_report() -> None:
    global_prediction = _prediction((0, 1, 2, 3, 4, 5))
    disconnected = (
        IndependentWindow(
            "local-a",
            "local",
            _prediction((0, 1, 2, 3)),
            "run-a",
        ),
        IndependentWindow(
            "local-b",
            "local",
            _prediction((4, 5)),
            "run-b",
        ),
    )
    report = score_window_pose_graph(
        global_prediction,
        disconnected,
        graph_config=GraphScoreConfig(
            window=WindowGraphConfig(require_loop_edges=False)
        ),
    )
    assert report.status == "invalid_insufficient_graph_evidence"
    assert np.isinf(report.total_score)
