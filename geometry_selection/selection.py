"""Candidate-pool validation and confidence-aware selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cache import canonical_hash
from .scorer import GeometryScoreReport


CANDIDATE_POOL_SCHEMA = "geometry-candidate-pool-v1"


@dataclass(frozen=True)
class SelectionConfig:
    abstain: bool = True
    min_relative_improvement: float = 0.01
    min_motion_ratio: float = 0.80
    max_motion_ratio: float = 1.25
    retain_unguided_incumbent: bool = True

    def validate(self) -> None:
        if self.min_relative_improvement < 0:
            raise ValueError("min_relative_improvement must be non-negative")
        if self.min_motion_ratio <= 0:
            raise ValueError("min_motion_ratio must be positive")
        if self.max_motion_ratio < self.min_motion_ratio:
            raise ValueError("max_motion_ratio must be >= min_motion_ratio")
        if self.abstain and not self.retain_unguided_incumbent:
            raise ValueError("abstention requires retain_unguided_incumbent=true")


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    seed: int
    video_sha256: str
    is_incumbent: bool
    report: GeometryScoreReport

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["report"] = self.report.to_dict()
        return payload


@dataclass(frozen=True)
class SelectionResult:
    selected_candidate_id: str
    incumbent_candidate_id: str
    decision: str
    score_improvement: float | None
    motion_ratio: float | None
    candidates: tuple[CandidateScore, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_candidate_id": self.selected_candidate_id,
            "incumbent_candidate_id": self.incumbent_candidate_id,
            "decision": self.decision,
            "score_improvement": self.score_improvement,
            "motion_ratio": self.motion_ratio,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pairing_id(case: dict[str, Any]) -> str:
    fields = {
        "scene_uid": case["scene_uid"],
        "conditioning_image_sha256": case["conditioning_image_sha256"],
        "prompt": case["prompt"],
        "backbone": case["backbone"],
        "generation": case["generation"],
    }
    return canonical_hash(fields)


def candidate_pool_id(case: dict[str, Any]) -> str:
    fields = {
        "pairing_id": case["pairing_id"],
        "candidates": [
            {
                "candidate_id": candidate["candidate_id"],
                "seed": candidate["seed"],
                "video_sha256": candidate["video_sha256"],
            }
            for candidate in case["candidates"]
        ],
    }
    return canonical_hash(fields)


def load_and_validate_candidate_pool(
    path: Path,
    *,
    expected_split: str,
    verify_video_hashes: bool = True,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CANDIDATE_POOL_SCHEMA:
        raise ValueError(f"candidate pool schema must be {CANDIDATE_POOL_SCHEMA}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("candidate pool must contain a non-empty cases list")

    seen_cases: set[str] = set()
    seen_scenes: set[str] = set()
    expected_count = payload.get("candidate_count")
    if not isinstance(expected_count, int) or expected_count < 2:
        raise ValueError("candidate_count must be an integer >= 2")
    for case in cases:
        required = {
            "case_id",
            "scene_uid",
            "split",
            "conditioning_image_sha256",
            "prompt",
            "backbone",
            "generation",
            "pairing_id",
            "candidate_pool_id",
            "candidates",
        }
        missing = sorted(required - set(case))
        if missing:
            raise ValueError(f"case is missing fields {missing}: {case.get('case_id')}")
        if case["split"] != expected_split:
            raise ValueError(f"case {case['case_id']} belongs to split {case['split']}, not {expected_split}")
        if case["case_id"] in seen_cases or case["scene_uid"] in seen_scenes:
            raise ValueError(f"duplicate case or scene: {case['case_id']} / {case['scene_uid']}")
        seen_cases.add(case["case_id"])
        seen_scenes.add(case["scene_uid"])
        if case["pairing_id"] != pairing_id(case):
            raise ValueError(f"pairing_id mismatch for {case['case_id']}")
        candidates = case["candidates"]
        if not isinstance(candidates, list):
            raise TypeError(f"candidates must be a list for {case['case_id']}")
        if len(candidates) != expected_count:
            raise ValueError(
                f"case {case['case_id']} has {len(candidates)} candidates, expected {expected_count}"
            )
        candidate_required = {
            "candidate_id",
            "seed",
            "video",
            "video_sha256",
            "geometry_cache_key",
            "is_incumbent",
        }
        for candidate in candidates:
            candidate_missing = sorted(candidate_required - set(candidate))
            if candidate_missing:
                raise ValueError(
                    f"candidate is missing fields {candidate_missing} for {case['case_id']}"
                )
        ids = [candidate["candidate_id"] for candidate in candidates]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate candidate IDs for {case['case_id']}")
        if sum(bool(candidate.get("is_incumbent")) for candidate in candidates) != 1:
            raise ValueError(f"case {case['case_id']} must have exactly one incumbent")
        if case["candidate_pool_id"] != candidate_pool_id(case):
            raise ValueError(f"candidate_pool_id mismatch for {case['case_id']}")
        if verify_video_hashes:
            for candidate in candidates:
                video = Path(candidate["video"])
                if not video.is_file():
                    raise FileNotFoundError(video)
                if file_sha256(video) != candidate["video_sha256"]:
                    raise ValueError(f"video hash mismatch: {video}")
    return payload


def deterministic_random_candidate(case: dict[str, Any], control_seed: int = 0) -> str:
    token = canonical_hash(
        {
            "control": "random-of-k",
            "control_seed": control_seed,
            "candidate_pool_id": case["candidate_pool_id"],
        }
    )
    index = int(token[:16], 16) % len(case["candidates"])
    return case["candidates"][index]["candidate_id"]


def select_candidate(
    candidates: list[CandidateScore],
    config: SelectionConfig,
) -> SelectionResult:
    config.validate()
    if len(candidates) < 2:
        raise ValueError("selection requires at least two candidates")
    incumbents = [candidate for candidate in candidates if candidate.is_incumbent]
    if len(incumbents) != 1:
        raise ValueError("exactly one candidate must be the incumbent")
    incumbent = incumbents[0]
    valid = [
        candidate
        for candidate in candidates
        if candidate.report.status.startswith("ok") and np.isfinite(candidate.report.total_score)
    ]
    if not valid:
        return SelectionResult(
            incumbent.candidate_id,
            incumbent.candidate_id,
            "abstain_no_valid_candidate",
            None,
            None,
            tuple(candidates),
        )
    best = min(valid, key=lambda candidate: (candidate.report.total_score, candidate.candidate_id))
    if best.candidate_id == incumbent.candidate_id:
        return SelectionResult(
            best.candidate_id,
            incumbent.candidate_id,
            "incumbent_is_best",
            0.0,
            1.0,
            tuple(candidates),
        )

    incumbent_score = incumbent.report.total_score
    if not np.isfinite(incumbent_score):
        return SelectionResult(
            best.candidate_id,
            incumbent.candidate_id,
            "replace_invalid_incumbent",
            None,
            None,
            tuple(candidates),
        )
    improvement = (incumbent_score - best.report.total_score) / max(abs(incumbent_score), 1e-8)
    incumbent_motion = incumbent.report.normalized_camera_motion
    motion_ratio = best.report.normalized_camera_motion / max(incumbent_motion, 1e-8)
    passes = (
        improvement >= config.min_relative_improvement
        and config.min_motion_ratio <= motion_ratio <= config.max_motion_ratio
    )
    if config.abstain and not passes:
        return SelectionResult(
            incumbent.candidate_id,
            incumbent.candidate_id,
            "abstain_margin_or_motion_guard",
            float(improvement),
            float(motion_ratio),
            tuple(candidates),
        )
    return SelectionResult(
        best.candidate_id,
        incumbent.candidate_id,
        "select_geometry_best",
        float(improvement),
        float(motion_ratio),
        tuple(candidates),
    )
