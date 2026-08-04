from __future__ import annotations

from pathlib import Path

import torch

from geometry_selection.graph_scorer import GraphScoreConfig
from geometry_selection.online import OnlineCandidate, OnlineSelectionContext, tensor_sha256
from geometry_selection.online_vggt import (
    OnlinePoseGraphScorerConfig,
    OnlineVGGTPoseGraphScorer,
    uniformly_spaced_keyframes,
)
from geometry_selection.scorer import ScorerConfig


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
