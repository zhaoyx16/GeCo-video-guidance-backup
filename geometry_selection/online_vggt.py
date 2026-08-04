"""VGGT-Omega pose-graph reports for provisional denoising predictions."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from .appearance import score_appearance_pair
from .backbones.vggt_omega import VGGTOmegaAdapter
from .cache import file_sha256
from .graph_scorer import GraphScoreConfig, score_window_pose_graph
from .online import OnlineCandidate, OnlineSelectionContext
from .scorer import GeometryScoreReport, ScorerConfig
from .window_bundle import validate_geometry_extraction_config
from .window_graph import IndependentWindow, make_window_schedule


DecodeKeyframesFn = Callable[[torch.Tensor, Sequence[int], Path], dict[int, Path]]


def uniformly_spaced_keyframes(total_frames: int, count: int) -> tuple[int, ...]:
    if not isinstance(total_frames, int) or total_frames < 2:
        raise ValueError("total_frames must be an integer >= 2")
    if not isinstance(count, int) or not 2 <= count <= total_frames:
        raise ValueError("keyframe count must be in [2, total_frames]")
    return tuple(
        int(value)
        for value in np.linspace(0, total_frames - 1, count).round().astype(np.int64)
    )


@dataclass(frozen=True)
class OnlinePoseGraphScorerConfig:
    total_frames: int
    extraction: dict[str, int | str]
    scorer: ScorerConfig
    graph_score: GraphScoreConfig
    retain_frames: bool = False

    def validate(self) -> None:
        if self.total_frames < 2:
            raise ValueError("total_frames must be >= 2")
        validate_geometry_extraction_config(self.extraction)
        if self.extraction["mode"] != "independent-window-pose-graph-v1":
            raise ValueError("online scorer only supports independent-window pose graphs")
        num_keyframes = self.extraction["num_keyframes"]
        if not isinstance(num_keyframes, int) or not 4 <= num_keyframes <= self.total_frames:
            raise ValueError("online num_keyframes must be in [4, total_frames]")
        self.scorer.validate()
        self.graph_score.validate()
        if not isinstance(self.retain_frames, bool):
            raise TypeError("retain_frames must be boolean")


class OnlineVGGTPoseGraphScorer:
    """Sequentially decode and score each provisional ``x0`` candidate.

    Each local/loop window is a separate VGGT-Omega forward pass, exactly as in
    the offline pose-graph implementation.  The temporary PNGs are merely a
    bridge to the official adapter API and are removed unless explicitly kept
    for a debugging run.
    """

    def __init__(
        self,
        *,
        adapter: VGGTOmegaAdapter,
        decode_keyframes: DecodeKeyframesFn,
        work_root: Path,
        config: OnlinePoseGraphScorerConfig,
    ) -> None:
        config.validate()
        self.adapter = adapter
        self.decode_keyframes = decode_keyframes
        self.work_root = Path(work_root).resolve()
        self.config = config
        self.keyframe_indices = uniformly_spaced_keyframes(
            config.total_frames, int(config.extraction["num_keyframes"])
        )
        self.last_artifacts: list[dict] = []

    def _event_directory(
        self,
        context: OnlineSelectionContext,
        candidate: OnlineCandidate,
    ) -> tuple[Path, Callable[[], None]]:
        self.work_root.mkdir(parents=True, exist_ok=True)
        name = f"step_{context.step_index:03d}_{candidate.candidate_id}"
        if self.config.retain_frames:
            directory = self.work_root / name
            directory.mkdir(parents=True, exist_ok=False)
            return directory, lambda: None
        directory = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=self.work_root))
        return directory, lambda: shutil.rmtree(directory, ignore_errors=True)

    def _invalid_report(self, status: str) -> GeometryScoreReport:
        return GeometryScoreReport(
            total_score=float("inf"),
            local_score=None,
            long_range_score=None,
            camera_path_length=0.0,
            normalized_translation_motion=0.0,
            camera_angular_path_deg=0.0,
            normalized_camera_motion=0.0,
            normalized_net_translation_motion=0.0,
            local_edge_fraction=0.0,
            long_range_edge_fraction=0.0,
            valid_local_edges=0,
            valid_long_range_edges=0,
            status=status,
            pairs=(),
            config=self.config.scorer,
            keyframe_indices=self.keyframe_indices,
            score_kind="pose_graph",
        )

    def _score_one(
        self,
        candidate: OnlineCandidate,
        context: OnlineSelectionContext,
    ) -> GeometryScoreReport:
        directory, cleanup = self._event_directory(context, candidate)
        artifact = {
            "candidate_id": candidate.candidate_id,
            "step_index": context.step_index,
            "frame_indices": list(self.keyframe_indices),
            "retained": self.config.retain_frames,
        }
        try:
            paths_by_frame = self.decode_keyframes(
                candidate.x0, self.keyframe_indices, directory
            )
            if tuple(sorted(paths_by_frame)) != self.keyframe_indices:
                raise ValueError("decoder returned the wrong provisional keyframes")
            paths = [Path(paths_by_frame[index]).resolve() for index in self.keyframe_indices]
            if any(not path.is_file() for path in paths):
                raise FileNotFoundError("provisional frame decoder did not write all PNGs")
            artifact["frame_file_sha256"] = [file_sha256(path) for path in paths]
            artifact["frames_directory"] = str(directory) if self.config.retain_frames else None
            global_prediction = self.adapter.predict_image_paths(
                paths,
                keyframe_indices=self.keyframe_indices,
            )
            schedule = make_window_schedule(
                self.keyframe_indices,
                local_window_size=int(self.config.extraction["local_window_size"]),
                local_stride=int(self.config.extraction["local_stride"]),
                loop_context=int(self.config.extraction["loop_context"]),
                min_loop_node_gap=int(self.config.extraction["min_loop_node_gap"]),
                max_loop_windows=int(self.config.extraction["max_loop_windows"]),
            )
            windows: list[IndependentWindow] = []
            for window_id, kind, indices in schedule:
                window_paths = [Path(paths_by_frame[index]).resolve() for index in indices]
                window_prediction = self.adapter.predict_image_paths(
                    window_paths, keyframe_indices=indices
                )
                appearance = (
                    score_appearance_pair(
                        window_paths[0],
                        window_paths[-1],
                        source_frame=indices[0],
                        target_frame=indices[-1],
                    )
                    if kind == "loop"
                    else None
                )
                independent_id = hashlib.sha256(
                    (
                        f"online-window-v1:{context.step_index}:{candidate.candidate_id}:"
                        f"{window_id}:{candidate.x0_sha256}"
                    ).encode("utf-8")
                ).hexdigest()
                windows.append(
                    IndependentWindow(
                        window_id=window_id,
                        kind=kind,
                        prediction=window_prediction,
                        independent_run_id=independent_id,
                        appearance_evidence=appearance,
                    )
                )
            report = score_window_pose_graph(
                global_prediction,
                windows,
                direct_config=self.config.scorer,
                graph_config=self.config.graph_score,
            )
            artifact["status"] = report.status
            artifact["window_ids"] = [window.window_id for window in windows]
            return report
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            artifact["status"] = f"invalid_geometry_{type(error).__name__}"
            artifact["error"] = str(error)
            return self._invalid_report(artifact["status"])
        finally:
            self.last_artifacts.append(artifact)
            cleanup()

    def __call__(
        self,
        candidates: Sequence[OnlineCandidate],
        context: OnlineSelectionContext,
    ) -> Sequence[GeometryScoreReport]:
        self.last_artifacts = []
        return [self._score_one(candidate, context) for candidate in candidates]
