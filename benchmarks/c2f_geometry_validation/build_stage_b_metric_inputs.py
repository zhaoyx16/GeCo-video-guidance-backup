#!/usr/bin/env python3
"""Build immutable evaluator inputs for the locked 10-case Stage B experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path


VIDEO_CONTRACT = {"frames": 121, "width": 1280, "height": 704, "fps": 24}
INPUT_SCHEMA = "geometry-selection-five-metric-hippasus-input-lock-v2"
STAGE_B_SCOPE = "stage_b_development_subset_disjoint_from_frozen_test_validation_debug"
EXPECTED_OVERLAP = {"test": 0, "validation": 0, "debug": 0}
METHODS = {
    "B": "official_same_host_reference",
    "C": "c2f_k3_a0025",
    "G": "c2f_geometry_hard_gate",
    "U": "c2f_geometry_uniform_norm_control",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def atomic_readonly_json(path: Path, payload: dict) -> str:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha256_file(path)


def require_readonly(path: Path, expected_sha: str | None = None) -> Path:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"regular file required: {path}")
    if path.stat().st_mode & 0o222:
        raise ValueError(f"input must be read-only: {path}")
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise RuntimeError(f"SHA mismatch: {path}")
    return path


def readonly_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(root, 0o555)


def normalized_probe(metadata: dict, case_id: str) -> dict:
    generation = metadata.get("generation")
    probe = metadata.get("video_probe")
    if not isinstance(generation, dict) or not isinstance(probe, dict):
        raise ValueError(f"missing video contract for {case_id}")
    expected_generation = {
        "frames": 121,
        "width": 1280,
        "height": 704,
        "fps": 24,
        "steps": 50,
        "guidance_scale": 5.0,
        "negative_prompt": None,
    }
    if any(generation.get(key) != value for key, value in expected_generation.items()):
        raise ValueError(f"generation contract mismatch for {case_id}: {generation}")
    if (
        str(probe.get("nb_read_frames")) != "121"
        or probe.get("width") != 1280
        or probe.get("height") != 704
        or probe.get("r_frame_rate") != "24/1"
    ):
        raise ValueError(f"ffprobe contract mismatch for {case_id}: {probe}")
    return VIDEO_CONTRACT


def selected_existing_entries(lock_path: Path, cases: list[dict], method_id: str) -> tuple[list[dict], dict]:
    source_lock = read_json(require_readonly(lock_path))
    if (
        source_lock.get("schema") != INPUT_SCHEMA
        or source_lock.get("method_id") != method_id
        or source_lock.get("reserved_overlap_counts") != EXPECTED_OVERLAP
        or len(source_lock.get("entries", [])) != 25
    ):
        raise ValueError(f"unexpected frozen Dev25 input lock: {lock_path}")
    by_case = {entry["case_id"]: entry for entry in source_lock["entries"]}
    if len(by_case) != 25:
        raise ValueError(f"duplicate cases in {lock_path}")
    selected = []
    for case in cases:
        case_id = case["case_id"]
        entry = dict(by_case[case_id])
        if (
            entry.get("seed") != 0
            or entry.get("prompt") != case["text_prompt"]
            or entry.get("video_probe") != VIDEO_CONTRACT
        ):
            raise ValueError(f"frozen comparator mismatch for {method_id}/{case_id}")
        video = require_readonly(Path(entry["metric_video_path"]), entry["video_sha256"])
        require_readonly(Path(entry["metric_metadata_path"]), entry["metadata_sha256"])
        require_readonly(Path(entry["metric_complete_path"]), entry["complete_sha256"])
        if video.stat().st_mode & 0o222:
            raise ValueError(f"comparator video is writable: {video}")
        if method_id == METHODS["C"] and entry["video_sha256"] != case["c2f_video_sha256"]:
            raise RuntimeError(f"Stage B C2F SHA mismatch for {case_id}")
        selected.append(entry)
    return selected, source_lock


def build_existing_lock(
    output_path: Path,
    source_path: Path,
    source_payload: dict,
    entries: list[dict],
    stage_b_lock: Path,
    stage_b_lock_sha: str,
    schedule_sha: str,
) -> dict:
    method_id = source_payload["method_id"]
    lock = {
        **{key: value for key, value in source_payload.items() if key != "entries"},
        "scope": STAGE_B_SCOPE,
        "candidate_budget": {
            "candidate_count": 1,
            "selected_output_count": 1,
            "comparison_note": "locked Stage B seed-0 subset of existing Dev25 comparator",
        },
        "stage_b_lock": {"path": str(stage_b_lock), "sha256": stage_b_lock_sha},
        "source_full_dev25_lock": {"path": str(source_path.resolve()), "sha256": sha256_file(source_path)},
        "metric_schedule_sha256": schedule_sha,
        "entries": entries,
    }
    lock_sha = atomic_readonly_json(output_path, lock)
    return {"method_id": method_id, "path": str(output_path.resolve()), "sha256": lock_sha}


def build_generated_lock(
    output_path: Path,
    mirrors_root: Path,
    generation_root: Path,
    method_id: str,
    cases: list[dict],
    stage_b_lock: Path,
    stage_b_lock_sha: str,
    schedule_sha: str,
) -> dict:
    final_mirror = mirrors_root / method_id
    staging = mirrors_root / f".{method_id}.{uuid.uuid4().hex}.staging"
    videos_dir = staging / "videos"
    metadata_dir = staging / "metadata"
    complete_dir = staging / "complete"
    for directory in (videos_dir, metadata_dir, complete_dir):
        directory.mkdir(parents=True, mode=0o700)
    entries = []
    config_path: Path | None = None
    config_sha: str | None = None
    for ordinal, case in enumerate(cases):
        case_id = case["case_id"]
        source_dir = generation_root / method_id / case_id / "seed_0"
        source_video = source_dir / "video.mp4"
        source_metadata = source_dir / "metadata.json"
        source_complete = source_dir / "COMPLETE.json"
        if not all(path.is_file() and not path.is_symlink() for path in (source_video, source_metadata, source_complete)):
            raise FileNotFoundError(f"incomplete generated source: {source_dir}")
        metadata = read_json(source_metadata)
        complete = read_json(source_complete)
        video_sha = sha256_file(source_video)
        if (
            metadata.get("case_id") != case_id
            or metadata.get("method_id") != method_id
            or metadata.get("seed") != 0
            or metadata.get("prompt") != case["text_prompt"]
            or metadata.get("video_sha256") != video_sha
            or complete.get("status") != "complete"
            or complete.get("video_sha256") != video_sha
        ):
            raise RuntimeError(f"generation provenance mismatch for {method_id}/{case_id}")
        if metadata.get("image_sha256") != sha256_file(Path(case["conditioning_image"])):
            raise RuntimeError(f"conditioning image SHA mismatch for {method_id}/{case_id}")
        probe = normalized_probe(metadata, case_id)
        this_config = Path(metadata["config_path"]).resolve()
        this_config_sha = metadata["config_sha256"]
        if sha256_file(this_config) != this_config_sha:
            raise RuntimeError(f"generation config SHA mismatch for {method_id}/{case_id}")
        if config_path is None:
            config_path, config_sha = this_config, this_config_sha
        elif (this_config, this_config_sha) != (config_path, config_sha):
            raise RuntimeError(f"mixed generation configs for {method_id}")
        stem = f"{ordinal:03d}_{case_id}"
        target_video = videos_dir / f"{stem}.mp4"
        target_metadata = metadata_dir / f"{stem}.json"
        target_complete = complete_dir / stem
        for source, target in (
            (source_video, target_video),
            (source_metadata, target_metadata),
            (source_complete, target_complete),
        ):
            shutil.copy2(source, target)
            if sha256_file(source) != sha256_file(target):
                raise RuntimeError(f"copy SHA mismatch: {target}")
        entries.append(
            {
                "case_id": case_id,
                "seed": 0,
                "metric_video_path": str((final_mirror / "videos" / target_video.name).resolve()),
                "metric_metadata_path": str((final_mirror / "metadata" / target_metadata.name).resolve()),
                "metric_complete_path": str((final_mirror / "complete" / target_complete.name).resolve()),
                "video_sha256": sha256_file(target_video),
                "metadata_sha256": sha256_file(target_metadata),
                "complete_sha256": sha256_file(target_complete),
                "conditioning_image_sha256": metadata["image_sha256"],
                "prompt": metadata["prompt"],
                "video_probe": probe,
            }
        )
    staging.replace(final_mirror)
    readonly_tree(final_mirror)
    if config_path is None or config_sha is None:
        raise RuntimeError(f"no generated entries for {method_id}")
    lock = {
        "schema": INPUT_SCHEMA,
        "scope": STAGE_B_SCOPE,
        "evaluation_site": "Hippasus",
        "method": method_id,
        "method_id": method_id,
        "candidate_budget": {
            "candidate_count": 1,
            "selected_output_count": 1,
            "comparison_note": "locked paired seed-0 Stage B geometry-gating experiment",
        },
        "selection_policy": {
            "selection_uses_generated_outputs": False,
            "selection_uses_metrics": False,
            "motion_strata": ["forward", "forward_left", "forward_right", "lateral_left", "lateral_right"],
            "cases_per_stratum": 2,
            "stratum_quantile_ranks": [1, 3],
        },
        "stage_b_lock": {"path": str(stage_b_lock), "sha256": stage_b_lock_sha},
        "reserved_overlap_counts": EXPECTED_OVERLAP,
        "generation_config": str(config_path),
        "generation_config_sha256": config_sha,
        "metric_schedule_sha256": schedule_sha,
        "video_contract": VIDEO_CONTRACT,
        "mirror_root": str(final_mirror.resolve()),
        "entries": entries,
    }
    lock_sha = atomic_readonly_json(output_path, lock)
    return {"method_id": method_id, "path": str(output_path.resolve()), "sha256": lock_sha}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-b-lock", type=Path, required=True)
    parser.add_argument("--baseline-lock", type=Path, required=True)
    parser.add_argument("--c2f-lock", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--expected-schedule-sha256", required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite evaluator input root: {args.output_root}")
    if sha256_file(args.schedule) != args.expected_schedule_sha256:
        raise RuntimeError("metric schedule SHA mismatch")
    stage_b_lock = args.stage_b_lock.resolve()
    stage_b = read_json(stage_b_lock)
    stage_b_sha = sha256_file(stage_b_lock)
    cases = stage_b.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) != 10
        or stage_b.get("reserved_overlap_counts") != EXPECTED_OVERLAP
        or stage_b.get("selection_uses_generated_outputs") is not False
        or stage_b.get("selection_uses_metrics") is not False
    ):
        raise ValueError("invalid frozen Stage B lock")
    case_ids = [case.get("case_id") for case in cases]
    if len(set(case_ids)) != 10:
        raise ValueError("Stage B cases must be unique")

    args.output_root.mkdir(parents=True, mode=0o700)
    mirrors_root = args.output_root / "mirrors"
    locks_root = args.output_root / "locks"
    mirrors_root.mkdir(mode=0o700)
    locks_root.mkdir(mode=0o700)

    baseline_entries, baseline_payload = selected_existing_entries(
        args.baseline_lock, cases, METHODS["B"]
    )
    c2f_entries, c2f_payload = selected_existing_entries(args.c2f_lock, cases, METHODS["C"])
    lock_records = [
        build_existing_lock(
            locks_root / f"{METHODS['B']}.json",
            args.baseline_lock,
            baseline_payload,
            baseline_entries,
            stage_b_lock,
            stage_b_sha,
            args.expected_schedule_sha256,
        ),
        build_existing_lock(
            locks_root / f"{METHODS['C']}.json",
            args.c2f_lock,
            c2f_payload,
            c2f_entries,
            stage_b_lock,
            stage_b_sha,
            args.expected_schedule_sha256,
        ),
    ]
    for method_id in (METHODS["G"], METHODS["U"]):
        lock_records.append(
            build_generated_lock(
                locks_root / f"{method_id}.json",
                mirrors_root,
                args.generation_root,
                method_id,
                cases,
                stage_b_lock,
                stage_b_sha,
                args.expected_schedule_sha256,
            )
        )

    source_lre_eligibility = Path(
        "/vol/dissolve/yz10325/outputs/c2f_validation_0914/"
        "dev25_lre_locks_v1/baseline_lre_eligibility.json"
    )
    source_lre = read_json(require_readonly(source_lre_eligibility))
    eligible_ids = {
        unit["case_id"]
        for unit in source_lre.get("locked_units", {}).get("long_range_reprojection_error", [])
    }
    stage_b_lre_ids = sorted(set(case_ids) & eligible_ids)
    lre_status = {
        "source_eligibility_lock": {
            "path": str(source_lre_eligibility),
            "sha256": sha256_file(source_lre_eligibility),
        },
        "stage_b_baseline_eligible_case_ids": stage_b_lre_ids,
        "stage_b_baseline_eligible_count": len(stage_b_lre_ids),
        "decision": "not_evaluated_no_baseline_eligible_cases" if not stage_b_lre_ids else "evaluate_locked_subset",
    }
    lre_status_path = locks_root / "LRE_STATUS.json"
    lre_status_sha = atomic_readonly_json(lre_status_path, lre_status)

    ready = {
        "schema": "wan-c2f-geometry-stage-b-evaluator-inputs-ready-v1",
        "status": "ready",
        "case_count": 10,
        "stage_b_lock": {"path": str(stage_b_lock), "sha256": stage_b_sha},
        "reserved_overlap_counts": EXPECTED_OVERLAP,
        "metric_schedule": {
            "path": str(args.schedule.resolve()),
            "sha256": args.expected_schedule_sha256,
        },
        "locks": lock_records,
        "lre_status": {"path": str(lre_status_path.resolve()), "sha256": lre_status_sha, **lre_status},
    }
    ready_path = args.output_root / "READY.json"
    atomic_readonly_json(ready_path, ready)
    os.chmod(locks_root, 0o555)
    os.chmod(mirrors_root, 0o555)
    os.chmod(args.output_root, 0o555)
    print(json.dumps(ready, indent=2))


if __name__ == "__main__":
    main()
