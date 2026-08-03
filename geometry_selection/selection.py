"""Candidate-pool validation and confidence-aware selection."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cache import canonical_hash
from .scorer import GeometryScoreReport, PairScore


CANDIDATE_POOL_SCHEMA = "geometry-candidate-pool-v1"
CANDIDATE_SPEC_SCHEMA = "geometry-candidate-spec-v1"


@dataclass(frozen=True)
class SelectionConfig:
    abstain: bool = True
    min_relative_improvement: float = 0.01
    min_motion_ratio: float = 0.80
    max_motion_ratio: float = 1.25
    retain_unguided_incumbent: bool = True
    min_common_local_edges: int = 2
    min_common_long_range_edges: int = 1
    require_common_long_range: bool = True
    random_control_seed: int = 0

    def validate(self) -> None:
        if not isinstance(self.abstain, bool) or not isinstance(
            self.retain_unguided_incumbent, bool
        ):
            raise TypeError("abstain and retain_unguided_incumbent must be boolean")
        if not isinstance(self.require_common_long_range, bool):
            raise TypeError("require_common_long_range must be boolean")
        if not isinstance(self.random_control_seed, int) or isinstance(
            self.random_control_seed, bool
        ):
            raise TypeError("random_control_seed must be an integer")
        numeric = (
            self.min_relative_improvement,
            self.min_motion_ratio,
            self.max_motion_ratio,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("selection thresholds must be finite")
        if self.min_relative_improvement < 0:
            raise ValueError("min_relative_improvement must be non-negative")
        if self.min_motion_ratio <= 0:
            raise ValueError("min_motion_ratio must be positive")
        if self.max_motion_ratio < self.min_motion_ratio:
            raise ValueError("max_motion_ratio must be >= min_motion_ratio")
        if self.abstain and not self.retain_unguided_incumbent:
            raise ValueError("abstention requires retain_unguided_incumbent=true")
        if self.min_common_local_edges < 1 or self.min_common_long_range_edges < 0:
            raise ValueError("common-edge thresholds must be non-negative with local >= 1")


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    seed: int
    video_sha256: str
    geometry_cache_key: str
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
    common_local_edges: int
    common_long_range_edges: int
    comparable_scores: dict[str, float]
    candidates: tuple[CandidateScore, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_candidate_id": self.selected_candidate_id,
            "incumbent_candidate_id": self.incumbent_candidate_id,
            "decision": self.decision,
            "score_improvement": self.score_improvement,
            "motion_ratio": self.motion_ratio,
            "common_local_edges": self.common_local_edges,
            "common_long_range_edges": self.common_long_range_edges,
            "comparable_scores": self.comparable_scores,
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
                "geometry_cache_key": candidate["geometry_cache_key"],
                "is_incumbent": candidate["is_incumbent"],
            }
            for candidate in case["candidates"]
        ],
    }
    return canonical_hash(fields)


def materialize_candidate_pool(
    spec: dict[str, Any],
    geometry_cache_keys: dict[tuple[str, str], str],
) -> dict[str, Any]:
    """Hash immutable inputs and turn a human-written spec into a frozen pool."""

    validate_candidate_spec(spec)
    cases = spec["cases"]

    frozen_cases: list[dict[str, Any]] = []
    for source_case in cases:
        image = Path(source_case["conditioning_image"]).resolve()
        candidates = []
        for source_candidate in source_case["candidates"]:
            video = Path(source_candidate["video"]).resolve()
            key = (source_case["case_id"], source_candidate["candidate_id"])
            if key not in geometry_cache_keys:
                raise ValueError(f"missing geometry cache key for {key}")
            candidates.append(
                {
                    "candidate_id": source_candidate["candidate_id"],
                    "seed": int(source_candidate["seed"]),
                    "video": str(video),
                    "video_sha256": file_sha256(video),
                    "geometry_cache_key": geometry_cache_keys[key],
                    "is_incumbent": bool(source_candidate["is_incumbent"]),
                }
            )
        case = {
            "case_id": source_case["case_id"],
            "scene_uid": source_case["scene_uid"],
            "split": spec["split"],
            "conditioning_image": str(image),
            "conditioning_image_sha256": file_sha256(image),
            "prompt": source_case["prompt"],
            "backbone": spec["backbone"],
            "generation": spec["generation"],
            "candidates": candidates,
        }
        case["pairing_id"] = pairing_id(case)
        case["candidate_pool_id"] = candidate_pool_id(case)
        frozen_cases.append(case)
    return {
        "schema": CANDIDATE_POOL_SCHEMA,
        "candidate_count": spec["candidate_count"],
        "cases": frozen_cases,
    }


def validate_candidate_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema") != CANDIDATE_SPEC_SCHEMA:
        raise ValueError(f"candidate spec schema must be {CANDIDATE_SPEC_SCHEMA}")
    required = {"split", "candidate_count", "backbone", "generation", "cases"}
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"candidate spec is missing fields: {missing}")
    if spec["split"] not in {"debug", "validation", "test"}:
        raise ValueError("candidate spec split must be debug, validation, or test")
    if not isinstance(spec["backbone"], str) or not spec["backbone"]:
        raise ValueError("candidate spec backbone must be a non-empty string")
    if not isinstance(spec["generation"], dict) or not spec["generation"]:
        raise ValueError("candidate spec generation must be a non-empty mapping")
    candidate_count = spec["candidate_count"]
    if not isinstance(candidate_count, int) or candidate_count < 2:
        raise ValueError("candidate_count must be an integer >= 2")
    cases = spec["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("candidate spec must contain a non-empty cases list")

    seen_cases: set[str] = set()
    seen_scenes: set[str] = set()
    for source_case in cases:
        case_required = {"case_id", "scene_uid", "conditioning_image", "prompt", "candidates"}
        case_missing = sorted(case_required - set(source_case))
        if case_missing:
            raise ValueError(
                f"candidate spec case is missing fields {case_missing}: "
                f"{source_case.get('case_id')}"
            )
        image = Path(source_case["conditioning_image"]).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        if source_case["case_id"] in seen_cases or source_case["scene_uid"] in seen_scenes:
            raise ValueError(
                f"duplicate candidate spec case or scene: "
                f"{source_case['case_id']} / {source_case['scene_uid']}"
            )
        seen_cases.add(source_case["case_id"])
        seen_scenes.add(source_case["scene_uid"])
        if not isinstance(source_case["prompt"], str) or not source_case["prompt"]:
            raise ValueError(f"case {source_case['case_id']} has an empty prompt")
        if not isinstance(source_case["candidates"], list):
            raise TypeError(f"candidates must be a list for {source_case['case_id']}")
        if len(source_case["candidates"]) != candidate_count:
            raise ValueError(
                f"case {source_case['case_id']} has {len(source_case['candidates'])} "
                f"candidates, expected {candidate_count}"
            )
        candidate_ids: set[str] = set()
        candidate_seeds: set[int] = set()
        incumbent_count = 0
        for source_candidate in source_case["candidates"]:
            candidate_required = {"candidate_id", "seed", "video", "is_incumbent"}
            candidate_missing = sorted(candidate_required - set(source_candidate))
            if candidate_missing:
                raise ValueError(
                    f"candidate spec entry is missing fields {candidate_missing}: "
                    f"{source_case['case_id']}"
                )
            video = Path(source_candidate["video"]).resolve()
            if not video.is_file():
                raise FileNotFoundError(video)
            candidate_id = source_candidate["candidate_id"]
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError(f"candidate ID must be non-empty for {source_case['case_id']}")
            if not isinstance(source_candidate["seed"], int) or isinstance(
                source_candidate["seed"], bool
            ):
                raise TypeError(f"candidate seed must be an integer for {source_case['case_id']}")
            if not isinstance(source_candidate["is_incumbent"], bool):
                raise TypeError(f"is_incumbent must be boolean for {source_case['case_id']}")
            seed = int(source_candidate["seed"])
            if candidate_id in candidate_ids or seed in candidate_seeds:
                raise ValueError(f"duplicate candidate ID or seed for {source_case['case_id']}")
            candidate_ids.add(candidate_id)
            candidate_seeds.add(seed)
            incumbent_count += bool(source_candidate["is_incumbent"])
        if incumbent_count != 1:
            raise ValueError(f"case {source_case['case_id']} must have exactly one incumbent")


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
    seen_videos: set[str] = set()
    expected_count = payload.get("candidate_count")
    if not isinstance(expected_count, int) or expected_count < 2:
        raise ValueError("candidate_count must be an integer >= 2")
    for case in cases:
        required = {
            "case_id",
            "scene_uid",
            "split",
            "conditioning_image",
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
        if verify_video_hashes:
            image = Path(case["conditioning_image"])
            if not image.is_file():
                raise FileNotFoundError(image)
            if file_sha256(image) != case["conditioning_image_sha256"]:
                raise ValueError(f"conditioning image hash mismatch: {image}")
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
            if not isinstance(candidate["is_incumbent"], bool):
                raise TypeError(f"is_incumbent must be boolean for {case['case_id']}")
            for field in ("video_sha256", "geometry_cache_key"):
                value = candidate[field]
                if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                    raise ValueError(f"{field} must be a lowercase SHA-256 digest")
        ids = [candidate["candidate_id"] for candidate in candidates]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate candidate IDs for {case['case_id']}")
        seeds = [int(candidate["seed"]) for candidate in candidates]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate candidate seeds for {case['case_id']}")
        video_hashes = [candidate["video_sha256"] for candidate in candidates]
        if len(set(video_hashes)) != len(video_hashes):
            raise ValueError(f"duplicate candidate videos for {case['case_id']}")
        reused = sorted(set(video_hashes) & seen_videos)
        if reused:
            raise ValueError(f"candidate video reused across cases: {reused}")
        seen_videos.update(video_hashes)
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
    candidates = sorted(case["candidates"], key=lambda item: item["candidate_id"])
    index = int(token[:16], 16) % len(candidates)
    return candidates[index]["candidate_id"]


def _pair_map(report: GeometryScoreReport) -> dict[tuple[int, int], float]:
    grouped: dict[tuple[int, int], list[PairScore]] = {}
    for pair in report.pairs:
        key = (min(pair.source, pair.target), max(pair.source, pair.target))
        grouped.setdefault(key, []).append(pair)
    result = {}
    for key, directions in grouped.items():
        if len({(pair.source, pair.target) for pair in directions}) != 2:
            continue
        if all(pair.status == "ok" and math.isfinite(pair.score) for pair in directions):
            result[key] = float(np.mean([pair.score for pair in directions]))
    return result


def _common_comparable_scores(
    candidates: list[CandidateScore],
    config: SelectionConfig,
) -> tuple[dict[str, float], int, int] | None:
    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate.report.status.startswith("ok")
        and math.isfinite(candidate.report.total_score)
    ]
    incumbent = next(candidate for candidate in candidates if candidate.is_incumbent)
    if incumbent not in valid_candidates or len(valid_candidates) < 2:
        return None
    first_config = asdict(valid_candidates[0].report.config)
    if any(asdict(candidate.report.config) != first_config for candidate in valid_candidates[1:]):
        raise ValueError("all candidates must use the same scorer configuration")
    maps = [_pair_map(candidate.report) for candidate in valid_candidates]
    common = set(maps[0])
    for pair_map in maps[1:]:
        common &= set(pair_map)
    common = {
        key
        for key in common
        if all(math.isfinite(pair_map[key]) for pair_map in maps)
    }
    scorer_config = valid_candidates[0].report.config
    local_keys = sorted(
        key for key in common if abs(key[1] - key[0]) < scorer_config.min_long_range_gap
    )
    long_keys = sorted(common - set(local_keys))
    if len(local_keys) < config.min_common_local_edges:
        return None
    if config.require_common_long_range and len(long_keys) < config.min_common_long_range_edges:
        return None

    scores: dict[str, float] = {}
    for candidate, pair_map in zip(valid_candidates, maps, strict=True):
        local_score = float(np.mean([pair_map[key] for key in local_keys]))
        components = [(scorer_config.local_weight, local_score)]
        if long_keys and scorer_config.long_range_weight > 0:
            long_score = float(np.mean([pair_map[key] for key in long_keys]))
            components.append((scorer_config.long_range_weight, long_score))
        weight = sum(item[0] for item in components)
        scores[candidate.candidate_id] = sum(
            component_weight * score for component_weight, score in components
        ) / weight
    return scores, len(local_keys), len(long_keys)


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
    comparable = _common_comparable_scores(candidates, config)
    if comparable is None:
        return SelectionResult(
            incumbent.candidate_id,
            incumbent.candidate_id,
            "abstain_insufficient_common_evidence",
            None,
            None,
            0,
            0,
            {},
            tuple(candidates),
        )
    comparable_scores, common_local, common_long = comparable
    comparable_candidates = [
        candidate for candidate in candidates if candidate.candidate_id in comparable_scores
    ]
    best = min(
        comparable_candidates,
        key=lambda candidate: (comparable_scores[candidate.candidate_id], candidate.candidate_id),
    )
    if best.candidate_id == incumbent.candidate_id:
        return SelectionResult(
            best.candidate_id,
            incumbent.candidate_id,
            "incumbent_is_best",
            0.0,
            1.0,
            common_local,
            common_long,
            comparable_scores,
            tuple(candidates),
        )

    incumbent_score = comparable_scores[incumbent.candidate_id]
    incumbent_motion = incumbent.report.normalized_camera_motion
    eligible: list[tuple[CandidateScore, float, float]] = []
    for challenger in comparable_candidates:
        if challenger.is_incumbent:
            continue
        challenger_score = comparable_scores[challenger.candidate_id]
        improvement = (incumbent_score - challenger_score) / max(abs(incumbent_score), 1e-8)
        motion_ratio = challenger.report.normalized_camera_motion / max(incumbent_motion, 1e-8)
        if (
            improvement >= config.min_relative_improvement
            and config.min_motion_ratio <= motion_ratio <= config.max_motion_ratio
        ):
            eligible.append((challenger, float(improvement), float(motion_ratio)))
    if config.abstain and not eligible:
        return SelectionResult(
            incumbent.candidate_id,
            incumbent.candidate_id,
            "abstain_margin_or_motion_guard",
            None,
            None,
            common_local,
            common_long,
            comparable_scores,
            tuple(candidates),
        )
    if eligible:
        best, improvement, motion_ratio = min(
            eligible,
            key=lambda item: (comparable_scores[item[0].candidate_id], item[0].candidate_id),
        )
    else:
        improvement = (incumbent_score - comparable_scores[best.candidate_id]) / max(
            abs(incumbent_score), 1e-8
        )
        motion_ratio = best.report.normalized_camera_motion / max(incumbent_motion, 1e-8)
    return SelectionResult(
        best.candidate_id,
        incumbent.candidate_id,
        "select_geometry_best",
        float(improvement),
        float(motion_ratio),
        common_local,
        common_long,
        comparable_scores,
        tuple(candidates),
    )
