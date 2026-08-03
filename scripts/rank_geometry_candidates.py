#!/usr/bin/env python3
"""Rank a frozen candidate pool using cached geometry and one resolved config."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_selection.cache import load_geometry_cache, load_geometry_cache_metadata
from geometry_selection.config import load_offline_ranking_config, write_resolved_config
from geometry_selection.scorer import score_geometry
from geometry_selection.selection import (
    CandidateScore,
    deterministic_random_candidate,
    file_sha256,
    load_and_validate_candidate_pool,
    select_candidate,
)


def git_identity(repo: Path) -> dict:
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(dirty)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--debug-skip-video-hash-verification", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_offline_ranking_config(config_path)

    def resolve_config_path(value: str) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else config_path.parent / path).resolve()

    candidate_manifest = resolve_config_path(config.candidate_manifest)
    geometry_cache_root = resolve_config_path(config.geometry_cache_root)
    output_root = resolve_config_path(config.output_root)
    code = git_identity(REPO_ROOT)
    if code["dirty"]:
        raise RuntimeError("formal ranking refuses to run from a dirty Git worktree")
    if code["commit"] != args.expected_git_commit:
        raise RuntimeError(
            f"code commit mismatch: expected {args.expected_git_commit}, got {code['commit']}"
        )
    formal = not args.debug_skip_video_hash_verification
    pool = load_and_validate_candidate_pool(
        candidate_manifest,
        expected_split=config.expected_split,
        verify_video_hashes=formal,
    )

    candidate_manifest_sha256 = file_sha256(candidate_manifest)
    run_identity = {
        "code_commit": code["commit"],
        "config_hash": config.config_hash,
        "candidate_manifest_sha256": candidate_manifest_sha256,
    }
    run_id = file_sha256(candidate_manifest)[:12] + "-" + config.config_hash[:12]
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"{code['commit'][:12]}-{run_id}"
    final = run_dir / "ranking.json"
    if run_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing ranking output: {run_dir}"
        )
    temporary_run_dir = Path(
        tempfile.mkdtemp(prefix=f".{run_dir.name}.attempt-", dir=output_root)
    )
    write_resolved_config(config, temporary_run_dir / "config.resolved.yaml")
    case_results = []
    for case in pool["cases"]:
        scores = []
        for candidate in case["candidates"]:
            cache_metadata = load_geometry_cache_metadata(
                geometry_cache_root,
                candidate["geometry_cache_key"],
            )
            prediction = load_geometry_cache(
                geometry_cache_root,
                candidate["geometry_cache_key"],
                expected_video_sha256=candidate["video_sha256"],
            )
            report = score_geometry(prediction, config.scorer)
            scores.append(
                CandidateScore(
                    candidate_id=candidate["candidate_id"],
                    seed=int(candidate["seed"]),
                    video_sha256=candidate["video_sha256"],
                    geometry_cache_key=candidate["geometry_cache_key"],
                    is_incumbent=bool(candidate["is_incumbent"]),
                    report=report,
                )
            )
            candidate["verified_cache"] = {
                "cache_key": candidate["geometry_cache_key"],
                "arrays_sha256": cache_metadata["arrays_sha256"],
                "provenance_sha256": cache_metadata["provenance_sha256"],
            }
        selected = select_candidate(scores, config.selection)
        case_results.append(
            {
                "case_id": case["case_id"],
                "scene_uid": case["scene_uid"],
                "split": case["split"],
                "pairing_id": case["pairing_id"],
                "candidate_pool_id": case["candidate_pool_id"],
                "random_of_k": {
                    "algorithm": "sha256-sorted-candidate-id-v1",
                    "control_seed": config.selection.random_control_seed,
                    "candidate_id": deterministic_random_candidate(
                        case,
                        control_seed=config.selection.random_control_seed,
                    ),
                },
                "verified_caches": {
                    candidate["candidate_id"]: candidate["verified_cache"]
                    for candidate in case["candidates"]
                },
                "selection": selected.to_dict(),
            }
        )

    output = {
        "status": "COMPLETE",
        "schema": "geometry-ranking-results-v1",
        "formal": formal,
        "verification": {
            "video_hashes": formal,
            "expected_git_commit": args.expected_git_commit,
        },
        "code": code,
        "config_hash": config.config_hash,
        "run_identity": run_identity,
        "candidate_manifest": str(candidate_manifest),
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "case_count": len(case_results),
        "cases": case_results,
    }
    temporary = temporary_run_dir / f".ranking.{uuid.uuid4().hex}.tmp.json"
    complete = temporary_run_dir / "ranking.json"
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(complete)
    os.rename(temporary_run_dir, run_dir)
    print(json.dumps({"output": str(final), "cases": len(case_results)}, indent=2))


if __name__ == "__main__":
    main()
