from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from geometry_selection.appearance import AppearanceEvidence
from geometry_selection.graph import se3_exp
from geometry_selection.graph_scorer import GraphScoreConfig
from geometry_selection.online import OnlineCandidate, OnlineSelectionContext, tensor_sha256
from geometry_selection.online_vggt import (
    OnlinePoseGraphScorerConfig,
    OnlineVGGTPoseGraphScorer,
    uniformly_spaced_keyframes,
)
from geometry_selection.schema import GeometryPrediction
from geometry_selection.scorer import ScorerConfig
from geometry_selection.window_graph import WindowGraphConfig, make_window_schedule


def _extraction() -> dict:
    return {
        "mode": "independent-window-pose-graph-v1",
        "num_keyframes": 4,
        "local_window_size": 4,
        "local_stride": 2,
        "loop_context": 2,
        "min_loop_node_gap": 3,
        "max_loop_windows": 1,
    }


def _prediction(frames: tuple[int, ...]) -> GeometryPrediction:
    world_to_camera = []
    for frame in frames:
        world_from_camera = se3_exp(
            np.array([float(frame), 0.0, 0.0, 0.0, 0.01 * frame, 0.0])
        )
        world_to_camera.append(np.linalg.inv(world_from_camera))
    count = len(frames)
    depth = np.full((count, 16, 16), 3.0, dtype=np.float64)
    intrinsics = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    intrinsics[:, 0, 0] = intrinsics[:, 1, 1] = 10.0
    intrinsics[:, 0, 2] = intrinsics[:, 1, 2] = 8.0
    return GeometryPrediction(
        world_to_camera=np.stack(world_to_camera),
        intrinsics=intrinsics,
        depth=depth,
        confidence=np.full_like(depth, 3.0),
        keyframe_indices=np.asarray(frames, dtype=np.int64),
    )


def test_uniform_keyframes_are_deterministic_and_cover_endpoints() -> None:
    assert uniformly_spaced_keyframes(121, 8) == (0, 17, 34, 51, 69, 86, 103, 120)


def test_online_scorer_turns_geometry_failure_into_invalid_report(tmp_path: Path) -> None:
    latent = torch.ones((1, 1, 2, 2, 2))
    candidate = OnlineCandidate(
        candidate_id="incumbent",
        latents=latent,
        x0=latent,
        is_incumbent=True,
        perturbation_rms=0.0,
        branch_seed=None,
        noise_sha256=None,
        latent_sha256=tensor_sha256(latent),
        x0_sha256=tensor_sha256(latent),
    )
    context = OnlineSelectionContext(
        step_index=7,
        timestep=100,
        incumbent_latents=latent,
        incumbent_x0=latent,
        mutable_mask=None,
        predict_x0=lambda value: value,
    )

    def failing_decoder(_x0, _indices, _directory):
        raise RuntimeError("synthetic decode failure")

    scorer = OnlineVGGTPoseGraphScorer(
        adapter=object(),
        decode_keyframes=failing_decoder,
        work_root=tmp_path,
        config=OnlinePoseGraphScorerConfig(
            total_frames=8,
            extraction=_extraction(),
            scorer=ScorerConfig(require_long_range=False),
            graph_score=GraphScoreConfig(),
        ),
    )
    report = scorer([candidate], context)[0]

    assert report.status == "invalid_geometry_RuntimeError"
    assert report.total_score == float("inf")
    assert scorer.last_artifacts[0]["candidate_id"] == "incumbent"
    assert not list(tmp_path.iterdir())


def test_online_scorer_runs_full_pose_graph_and_cleans_temporary_frames(
    tmp_path: Path, monkeypatch
) -> None:
    latent = torch.ones((1, 1, 2, 2, 2))
    candidate = OnlineCandidate(
        candidate_id="incumbent",
        latents=latent,
        x0=latent,
        is_incumbent=True,
        perturbation_rms=0.0,
        branch_seed=None,
        noise_sha256=None,
        latent_sha256=tensor_sha256(latent),
        x0_sha256=tensor_sha256(latent),
    )
    context = OnlineSelectionContext(
        step_index=7,
        timestep=100,
        incumbent_latents=latent,
        incumbent_x0=latent,
        mutable_mask=None,
        predict_x0=lambda value: value,
    )
    extraction = {
        **_extraction(),
        "num_keyframes": 8,
        "local_window_size": 4,
        "local_stride": 2,
        "max_loop_windows": 2,
    }
    keyframes = uniformly_spaced_keyframes(12, 8)
    expected_schedule = make_window_schedule(
        keyframes,
        local_window_size=4,
        local_stride=2,
        loop_context=2,
        min_loop_node_gap=3,
        max_loop_windows=2,
    )

    class FakeAdapter:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[int, ...], tuple[str, ...]]] = []

        def predict_image_paths(self, paths, *, keyframe_indices):
            self.calls.append(
                (tuple(keyframe_indices), tuple(path.name for path in paths))
            )
            return _prediction(tuple(keyframe_indices))

    def decoder(_x0, indices, directory):
        result = {}
        for index in indices:
            path = directory / f"frame_{index:03d}.png"
            path.write_bytes(f"frame-{index}".encode())
            result[index] = path
        return result

    def appearance(source_path, target_path, *, source_frame, target_frame):
        return AppearanceEvidence(
            source_frame=source_frame,
            target_frame=target_frame,
            source_file_sha256="a" * 64,
            target_file_sha256="b" * 64,
            source_keypoints=50,
            target_keypoints=50,
            ratio_matches=30,
            inliers=25,
            inlier_ratio=25.0 / 30.0,
            spatial_coverage=0.50,
            mean_descriptor_distance=0.20,
            status="ok",
        )

    monkeypatch.setattr("geometry_selection.online_vggt.score_appearance_pair", appearance)
    adapter = FakeAdapter()
    scorer = OnlineVGGTPoseGraphScorer(
        adapter=adapter,
        decode_keyframes=decoder,
        work_root=tmp_path,
        config=OnlinePoseGraphScorerConfig(
            total_frames=12,
            extraction=extraction,
            scorer=ScorerConfig(require_long_range=False),
            graph_score=GraphScoreConfig(
                window=WindowGraphConfig(
                    min_depth_scale_pixels=2,
                    depth_sample_stride=2,
                    require_reobservation_support=False,
                    require_appearance_support=True,
                )
            ),
        ),
    )

    report = scorer([candidate], context)[0]

    assert report.status == "ok_pose_graph"
    assert adapter.calls[0][0] == keyframes
    assert [call[0] for call in adapter.calls[1:]] == [item[2] for item in expected_schedule]
    assert report.graph_diagnostics["window_scale_ids"] == [item[0] for item in expected_schedule]
    assert len(report.graph_diagnostics["accepted_loop_edge_ids"]) == 2
    artifact = scorer.last_artifacts[0]
    assert artifact["window_ids"] == [item[0] for item in expected_schedule]
    assert len(artifact["frame_file_sha256"]) == len(keyframes)
    assert not list(tmp_path.iterdir())
