#!/usr/bin/env python3
"""Rank a frozen candidate pool using cached geometry and one resolved config."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_selection.cache import load_geometry_cache
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
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-video-hash-verification", action="store_true")
    args = parser.parse_args()

    config = load_offline_ranking_config(args.config)
    candidate_manifest = Path(config.candidate_manifest).resolve()
    geometry_cache_root = Path(config.geometry_cache_root).resolve()
    output_root = Path(config.output_root).resolve()
    code = git_identity(args.repo.resolve())
    if code["dirty"]:
        raise RuntimeError("formal ranking refuses to run from a dirty Git worktree")
    pool = load_and_validate_candidate_pool(
        candidate_manifest,
        expected_split=config.expected_split,
        verify_video_hashes=not args.skip_video_hash_verification,
    )

    run_dir = output_root / f"{code['commit'][:12]}-{config.config_hash[:12]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    final = run_dir / "ranking.json"
    if final.exists():
        raise FileExistsError(
            f"refusing to overwrite completed ranking output: {final}"
        )
    write_resolved_config(config, run_dir / "config.resolved.yaml")
    case_results = []
    for case in pool["cases"]:
        scores = []
        for candidate in case["candidates"]:
            prediction = load_geometry_cache(
                geometry_cache_root,
                candidate["geometry_cache_key"],
            )
            report = score_geometry(prediction, config.scorer)
            scores.append(
                CandidateScore(
                    candidate_id=candidate["candidate_id"],
                    seed=int(candidate["seed"]),
                    video_sha256=candidate["video_sha256"],
                    is_incumbent=bool(candidate["is_incumbent"]),
                    report=report,
                )
            )
        selected = select_candidate(scores, config.selection)
        case_results.append(
            {
                "case_id": case["case_id"],
                "scene_uid": case["scene_uid"],
                "split": case["split"],
                "pairing_id": case["pairing_id"],
                "candidate_pool_id": case["candidate_pool_id"],
                "random_of_k_candidate_id": deterministic_random_candidate(case),
                "selection": selected.to_dict(),
            }
        )

    output = {
        "status": "COMPLETE",
        "schema": "geometry-ranking-results-v1",
        "code": code,
        "config_hash": config.config_hash,
        "candidate_manifest": str(candidate_manifest),
        "candidate_manifest_sha256": file_sha256(candidate_manifest),
        "case_count": len(case_results),
        "cases": case_results,
    }
    temporary = run_dir / f".ranking.{uuid.uuid4().hex}.tmp.json"
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(final)
    print(json.dumps({"output": str(final), "cases": len(case_results)}, indent=2))


if __name__ == "__main__":
    main()
