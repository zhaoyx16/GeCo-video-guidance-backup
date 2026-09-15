#!/usr/bin/env python3
"""Materialize immutable evaluator inputs for paired C2F development runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import uuid
from pathlib import Path


VIDEO_CONTRACT = {"frames": 121, "width": 1280, "height": 704, "fps": 24}
INPUT_SCHEMA = "geometry-selection-five-metric-hippasus-input-lock-v2"


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


def atomic_write(path: Path, payload: dict) -> None:
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


def ordered_cases(manifest: dict) -> list[tuple[str, dict]]:
    records = [(case_id, case) for case_id, case in manifest.items() if not case_id.startswith("_")]
    return sorted(records, key=lambda item: item[1]["c2f_dev_selection"]["selection_order"])


def normalized_probe(metadata: dict, case_id: str) -> dict:
    probe = metadata.get("video_probe")
    generation = metadata.get("generation")
    if not isinstance(probe, dict) or not isinstance(generation, dict):
        raise ValueError(f"missing video contract for {case_id}")
    expected_generation = {"frames": 121, "width": 1280, "height": 704, "fps": 24}
    if any(generation.get(key) != value for key, value in expected_generation.items()):
        raise ValueError(f"generation contract mismatch for {case_id}")
    if (
        str(probe.get("nb_read_frames")) != "121"
        or probe.get("width") != 1280
        or probe.get("height") != 704
        or probe.get("r_frame_rate") != "24/1"
    ):
        raise ValueError(f"ffprobe contract mismatch for {case_id}: {probe}")
    return VIDEO_CONTRACT


def readonly_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(root, 0o555)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--expected-schedule-sha256", required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite evaluator input root: {args.output_root}")
    if sha256_file(args.schedule) != args.expected_schedule_sha256:
        raise RuntimeError("metric schedule SHA mismatch")

    configs = [(path.resolve(), read_json(path)) for path in args.config]
    selection_paths = []
    for config_path, config in configs:
        selection = Path(config["selection_manifest"])
        if not selection.is_absolute():
            selection = config_path.parents[2] / selection
        selection_paths.append(selection.resolve())
    if len(set(selection_paths)) != 1:
        raise RuntimeError("all paired methods must use the same selection manifest")
    selection_path = selection_paths[0]
    manifest = read_json(selection_path)
    expected_overlap = {"test": 0, "validation": 0, "debug": 0}
    if manifest.get("_meta", {}).get("reserved_overlap_counts") != expected_overlap:
        raise RuntimeError("selection manifest does not certify zero reserved-split overlap")
    cases = ordered_cases(manifest)
    if len(cases) != 25:
        raise RuntimeError(f"expected exactly 25 development cases, found {len(cases)}")

    args.output_root.mkdir(parents=True, mode=0o700)
    mirrors_root = args.output_root / "mirrors"
    locks_root = args.output_root / "locks"
    mirrors_root.mkdir(mode=0o700)
    locks_root.mkdir(mode=0o700)

    lock_records = []
    try:
        for config_path, config in configs:
            method_id = config["method_id"]
            final_mirror = mirrors_root / method_id
            if final_mirror.exists():
                raise FileExistsError(final_mirror)
            staging = mirrors_root / f".{method_id}.{uuid.uuid4().hex}.staging"
            videos_dir = staging / "videos"
            metadata_dir = staging / "metadata"
            complete_dir = staging / "complete"
            for directory in (videos_dir, metadata_dir, complete_dir):
                directory.mkdir(parents=True, mode=0o700)

            entries = []
            for ordinal, (case_id, case) in enumerate(cases):
                source_dir = args.generation_root / method_id / case_id / f"seed_{config['seed']}"
                source_video = source_dir / "video.mp4"
                source_metadata = source_dir / "metadata.json"
                source_complete = source_dir / "COMPLETE.json"
                if not all(path.is_file() and not path.is_symlink() for path in (source_video, source_metadata, source_complete)):
                    raise FileNotFoundError(f"incomplete generated source: {source_dir}")
                metadata = read_json(source_metadata)
                complete = read_json(source_complete)
                source_video_sha = sha256_file(source_video)
                if (
                    metadata.get("case_id") != case_id
                    or metadata.get("method_id") != method_id
                    or metadata.get("seed") != config["seed"]
                    or metadata.get("prompt") != case["text_prompt"]
                    or metadata.get("video_sha256") != source_video_sha
                    or complete.get("status") != "complete"
                    or complete.get("video_sha256") != source_video_sha
                ):
                    raise RuntimeError(f"generation provenance mismatch for {method_id}/{case_id}")
                probe = normalized_probe(metadata, case_id)
                stem = f"{ordinal:03d}_{case_id}"
                target_video = videos_dir / f"{stem}.mp4"
                target_metadata = metadata_dir / f"{stem}.json"
                target_complete = complete_dir / stem
                shutil.copy2(source_video, target_video)
                shutil.copy2(source_metadata, target_metadata)
                shutil.copy2(source_complete, target_complete)
                for source, target in (
                    (source_video, target_video),
                    (source_metadata, target_metadata),
                    (source_complete, target_complete),
                ):
                    if sha256_file(source) != sha256_file(target):
                        raise RuntimeError(f"copy SHA mismatch: {target}")
                entries.append(
                    {
                        "case_id": case_id,
                        "seed": config["seed"],
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

            lock = {
                "schema": INPUT_SCHEMA,
                "scope": "development_validation_disjoint_from_frozen_test_validation_debug",
                "evaluation_site": "Hippasus",
                "method": method_id,
                "method_id": method_id,
                "candidate_budget": {
                    "candidate_count": 1,
                    "selected_output_count": 1,
                    "comparison_note": "paired seed-0 same-host development validation",
                },
                "selection_policy": {
                    "selection_uses_generated_outputs": manifest["_meta"]["selection_uses_generated_outputs"],
                    "selection_uses_metrics": manifest["_meta"]["selection_uses_metrics"],
                    "strata": manifest["_meta"]["strata"],
                    "cases_per_stratum": manifest["_meta"]["cases_per_stratum"],
                },
                "selection_manifest": str(selection_path),
                "selection_manifest_sha256": sha256_file(selection_path),
                "reserved_overlap_counts": expected_overlap,
                "generation_config": str(config_path),
                "generation_config_sha256": sha256_file(config_path),
                "metric_schedule_sha256": args.expected_schedule_sha256,
                "video_contract": VIDEO_CONTRACT,
                "mirror_root": str(final_mirror.resolve()),
                "entries": entries,
            }
            lock_path = locks_root / f"{method_id}.json"
            atomic_write(lock_path, lock)
            lock_records.append(
                {"method_id": method_id, "path": str(lock_path.resolve()), "sha256": sha256_file(lock_path)}
            )
    except Exception:
        # Keep partial data for diagnosis but never publish READY.
        raise

    ready = {
        "schema": "wan-c2f-dev25-evaluator-inputs-ready-v1",
        "status": "ready",
        "case_count": 25,
        "selection_manifest_sha256": sha256_file(selection_path),
        "reserved_overlap_counts": expected_overlap,
        "metric_schedule_sha256": args.expected_schedule_sha256,
        "locks": lock_records,
    }
    ready_path = args.output_root / "READY.json"
    atomic_write(ready_path, ready)
    os.chmod(locks_root, 0o555)
    os.chmod(mirrors_root, 0o555)
    os.chmod(args.output_root, 0o555)
    print(json.dumps(ready, indent=2))


if __name__ == "__main__":
    main()
