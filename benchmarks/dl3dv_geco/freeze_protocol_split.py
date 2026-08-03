#!/usr/bin/env python3
"""Freeze deterministic scene-level debug/validation/test DL3DV splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from collections import Counter
from pathlib import Path


SCHEMA = "dl3dv-geometry-selection-v1"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_key(seed: int, scene_uid: str) -> str:
    return hashlib.sha256(f"dl3dv-split-v1:{seed}:{scene_uid}".encode()).hexdigest()


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as error:
        raise ValueError(f"{path} is outside dataset root {root}") from error


def resolve_source_path(value: str, source_directory: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else source_directory / path).resolve()


def stratified_order(prepared: list[tuple[str, dict]], seed: int) -> list[tuple[str, dict]]:
    by_motion: dict[str, list[tuple[str, dict]]] = {}
    for item in prepared:
        by_motion.setdefault(item[1]["motion_instruction"], []).append(item)
    for items in by_motion.values():
        items.sort(key=lambda item: deterministic_key(seed, item[1]["scene_uid"]))
    ordered = []
    active = sorted(by_motion)
    while active:
        next_active = []
        for motion in active:
            items = by_motion[motion]
            if items:
                ordered.append(items.pop(0))
            if items:
                next_active.append(motion)
        active = next_active
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260803)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--validation-count", type=int, default=100)
    parser.add_argument("--debug-count", type=int, default=3)
    args = parser.parse_args()
    if min(args.test_count, args.validation_count, args.debug_count) < 0:
        parser.error("split counts must be non-negative")
    required_count = args.test_count + args.validation_count + args.debug_count
    if required_count < 1:
        parser.error("at least one case is required")

    source_path = args.source_manifest.resolve()
    dataset_root = args.dataset_root.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise TypeError("source manifest must be a mapping")

    prepared = []
    scene_uids: set[str] = set()
    image_hashes: set[str] = set()
    for case_id, original in source.items():
        if case_id.startswith("_"):
            continue
        case = dict(original)
        required = {
            "text_prompt",
            "image_prompt",
            "scene_id",
            "motion_instruction",
            "pose_window",
            "pose_stats",
            "transforms_path",
        }
        missing = sorted(required - set(case))
        if missing:
            raise ValueError(f"source case {case_id} is missing fields: {missing}")
        image = resolve_source_path(case["image_prompt"], source_path.parent)
        transforms = resolve_source_path(case["transforms_path"], source_path.parent)
        if not image.is_file() or not transforms.is_file():
            raise FileNotFoundError(f"missing image/transforms for {case_id}: {image}, {transforms}")
        transforms_sha = sha256_file(transforms)
        image_sha = sha256_file(image)
        relative_scene = relative_to_root(transforms.parent, dataset_root)
        scene_uid = f"dl3dv:{relative_scene}"
        if scene_uid in scene_uids:
            raise ValueError(
                f"multiple clips from the same scene are not allowed in the frozen pool: {scene_uid}"
            )
        if image_sha in image_hashes:
            raise ValueError(f"conditioning image reused across cases: {image}")
        scene_uids.add(scene_uid)
        image_hashes.add(image_sha)
        case.update(
            {
                "dataset": "dl3dv",
                "scene_uid": scene_uid,
                "image_prompt": relative_to_root(image, dataset_root),
                "transforms_path": relative_to_root(transforms, dataset_root),
                "image_sha256": image_sha,
                "transforms_sha256": transforms_sha,
                "dataset_relative_image": relative_to_root(image, dataset_root),
                "dataset_relative_transforms": relative_to_root(transforms, dataset_root),
                "trajectory": {
                    "family": case["motion_instruction"],
                    "pose_window": case["pose_window"],
                    "stats": case["pose_stats"],
                },
            }
        )
        prepared.append((case_id, case))

    if len(prepared) < required_count:
        raise ValueError(f"need {required_count} scene-disjoint cases, found {len(prepared)}")
    selected = stratified_order(prepared, args.split_seed)[:required_count]
    boundaries = (
        ("test", 0, args.test_count),
        ("validation", args.test_count, args.test_count + args.validation_count),
        ("debug", args.test_count + args.validation_count, required_count),
    )
    split_cases: dict[str, dict] = {}
    for split, start, end in boundaries:
        for order_in_split, (case_id, case) in enumerate(selected[start:end]):
            case = dict(case)
            case["split"] = split
            case["split_order"] = order_in_split
            split_cases[case_id] = case

    split_counts = Counter(case["split"] for case in split_cases.values())
    motion_counts = {
        split: dict(
            Counter(
                case["motion_instruction"]
                for case in split_cases.values()
                if case["split"] == split
            )
        )
        for split in ("debug", "validation", "test")
    }
    payload = {
        "_meta": {
            "schema": SCHEMA,
            "source_manifest": str(source_path),
            "source_manifest_sha256": sha256_file(source_path),
            "dataset_root_at_freeze": str(dataset_root),
            "split_seed": args.split_seed,
            "split_algorithm": "motion-stratified round robin; sha256 within each family",
            "split_counts": dict(split_counts),
            "motion_counts": motion_counts,
            "scene_disjoint": True,
            "conditioning_image_disjoint": True,
            "test_policy": "held out until code and configuration are frozen",
            "candidate_seed_policy": {
                "version": "fixed-four-v1",
                "candidate_seeds": [0, 1, 2, 3],
                "incumbent_seed": 0,
            },
            "generation_profiles": {
                "Wan2.2-TI2V-5B": {
                    "steps": 50,
                    "frames": 121,
                    "height": 704,
                    "width": 1280,
                    "fps": 24,
                    "guidance_scale": 5.0,
                },
                "Cosmos-Predict2.5-2B-post": {
                    "steps": 36,
                    "frames": 93,
                    "height": 704,
                    "width": 1280,
                    "fps": 16,
                    "guidance_scale": 7.0,
                },
            },
        },
        **split_cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload["_meta"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
