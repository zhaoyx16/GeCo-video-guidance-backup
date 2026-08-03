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
from geometry_selection.protocol import (
    FORMAL_SPLIT_COUNTS,
    file_sha256 as protocol_file_sha256,
    validate_candidate_pool_against_protocol,
    validate_committed_file,
    validate_committed_test_release,
    validate_formal_protocol,
)
from geometry_selection.model_lock import load_model_lock
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
    parser.add_argument(
        "--artifact-mode",
        choices=("formal", "legacy-debug"),
        required=True,
    )
    parser.add_argument("--test-release", type=Path)
    parser.add_argument("--debug-skip-video-hash-verification", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_offline_ranking_config(config_path)

    def resolve_config_path(value: str) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else config_path.parent / path).resolve()

    candidate_manifest = resolve_config_path(config.candidate_manifest)
    protocol_manifest = resolve_config_path(config.protocol_manifest)
    model_lock_path = resolve_config_path(config.model_lock)
    dataset_root = resolve_config_path(config.dataset_root)
    geometry_cache_root = resolve_config_path(config.geometry_cache_root)
    output_root = resolve_config_path(config.output_root)
    code = git_identity(REPO_ROOT)
    if code["dirty"]:
        raise RuntimeError("formal ranking refuses to run from a dirty Git worktree")
    if code["commit"] != args.expected_git_commit:
        raise RuntimeError(
            f"code commit mismatch: expected {args.expected_git_commit}, got {code['commit']}"
        )
    formal = args.artifact_mode == "formal"
    if formal and args.debug_skip_video_hash_verification:
        parser.error("formal ranking cannot skip candidate video hash verification")
    if formal:
        validate_committed_file(config_path, REPO_ROOT, code["commit"])
        validate_committed_file(protocol_manifest, REPO_ROOT, code["commit"])
        validate_committed_file(model_lock_path, REPO_ROOT, code["commit"])
        protocol = validate_formal_protocol(protocol_manifest)
        if protocol_file_sha256(model_lock_path) != protocol["_meta"]["model_lock_sha256"]:
            raise ValueError("model lock digest differs from frozen protocol")
        model_lock = load_model_lock(model_lock_path)
        locked_geometry = model_lock["geometry_backbone"]
        config_geometry = {
            "checkpoint_sha256": config.geometry_checkpoint_sha256,
            "source_tree_sha256": config.geometry_source_tree_sha256,
            "source_commit": config.geometry_source_commit,
        }
        if any(locked_geometry.get(key) != value for key, value in config_geometry.items()):
            raise ValueError("ranking config geometry identity differs from model lock")
    else:
        model_lock = None
    pool = load_and_validate_candidate_pool(
        candidate_manifest,
        expected_split=config.expected_split,
        verify_video_hashes=formal,
        expected_case_count=FORMAL_SPLIT_COUNTS[config.expected_split] if formal else None,
    )
    validate_candidate_pool_against_protocol(
        pool,
        protocol_manifest,
        dataset_root,
        formal=formal,
        model_lock=model_lock,
    )

    candidate_manifest_sha256 = file_sha256(candidate_manifest)
    protocol_manifest_sha256 = protocol_file_sha256(protocol_manifest)
    if formal and config.expected_split == "test":
        if args.test_release is None:
            parser.error("formal test ranking requires --test-release")
        validate_committed_test_release(
            args.test_release,
            REPO_ROOT,
            code["commit"],
            {
                "schema": "geometry-test-release-v1",
                "phase": "ranking",
                "protocol_manifest_sha256": protocol_manifest_sha256,
                "ranking_config_hash": config.config_hash,
                "candidate_manifest_sha256": candidate_manifest_sha256,
            },
        )
    run_identity = {
        "code_commit": code["commit"],
        "config_hash": config.config_hash,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "protocol_manifest_sha256": protocol_manifest_sha256,
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
            if formal:
                geometry_identity = cache_metadata["provenance"].get(
                    "geometry_backbone", {}
                )
                expected_geometry = {
                    "checkpoint_sha256": config.geometry_checkpoint_sha256,
                    "source_tree_sha256": config.geometry_source_tree_sha256,
                    "source_commit": config.geometry_source_commit,
                }
                geometry_mismatches = {
                    key: (geometry_identity.get(key), expected)
                    for key, expected in expected_geometry.items()
                    if geometry_identity.get(key) != expected
                }
                if geometry_mismatches:
                    raise ValueError(
                        f"geometry backbone identity mismatch: {geometry_mismatches}"
                    )
                producer = cache_metadata["provenance"].get("producer", {})
                expected_sources = {
                    "dirty": False,
                    "adapter_sha256": file_sha256(
                        REPO_ROOT / "geometry_selection/backbones/vggt_omega.py"
                    ),
                    "extractor_sha256": file_sha256(
                        REPO_ROOT / "scripts/extract_vggt_omega_geometry.py"
                    ),
                    "preparer_sha256": file_sha256(
                        REPO_ROOT / "scripts/prepare_candidate_pool.py"
                    ),
                }
                producer_mismatches = {
                    key: (producer.get(key), expected)
                    for key, expected in expected_sources.items()
                    if producer.get(key) != expected
                }
                if producer_mismatches or not producer.get("commit"):
                    raise ValueError(
                        "geometry cache producer mismatch: "
                        f"{producer_mismatches or {'commit': 'missing'}}"
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
            "protocol_manifest_sha256": protocol_manifest_sha256,
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
