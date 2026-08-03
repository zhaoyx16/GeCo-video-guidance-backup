#!/usr/bin/env python3
"""Build explicitly non-formal protocol/spec files for legacy GPU smoke videos."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_selection.protocol import DL3DV_PROTOCOL_SCHEMA, file_sha256
from geometry_selection.selection import CANDIDATE_SPEC_SCHEMA


GENERATION = {
    "steps": 50,
    "frames": 121,
    "height": 704,
    "width": 1280,
    "fps": 24,
    "guidance_scale": 5.0,
    "wan_negative_prompt_mode": "none",
}


def publish_json_no_clobber(payload: dict, output: Path) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, output)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite debug artifact: {output}") from error
    finally:
        temporary.unlink(missing_ok=True)


def within_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as error:
        raise ValueError(f"{path} is outside dataset root {root}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--protocol-output", type=Path, required=True)
    parser.add_argument("--spec-output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.case_ids) != 3:
        parser.error("debug smoke requires exactly three scene-disjoint cases")
    if len(set(args.case_ids)) != len(args.case_ids):
        parser.error("case IDs must be unique")
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("provide at least two unique candidate seeds")

    source_path = args.source_manifest.resolve()
    dataset_root = args.dataset_root.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    protocol_cases = {}
    spec_cases = []
    for order, case_id in enumerate(args.case_ids):
        if case_id not in source or case_id.startswith("_"):
            raise KeyError(f"unknown source case: {case_id}")
        source_case = source[case_id]
        image = Path(source_case["image_prompt"]).resolve()
        transforms = image.parent.parent / "transforms.json"
        if not image.is_file() or not transforms.is_file():
            raise FileNotFoundError(f"missing image/transforms for {case_id}")
        relative_scene = within_root(transforms.parent, dataset_root)
        protocol_cases[case_id] = {
            "scene_uid": f"dl3dv:{relative_scene}",
            "split": "debug",
            "split_order": order,
            "text_prompt": source_case["text_prompt"],
            "dataset_relative_image": within_root(image, dataset_root),
            "image_sha256": file_sha256(image),
            "dataset_relative_transforms": within_root(transforms, dataset_root),
            "transforms_sha256": file_sha256(transforms),
        }
        candidates = []
        for seed in args.seeds:
            video = (
                args.video_root.resolve()
                / case_id
                / f"baseline_seed{seed}_steps50_frames121.mp4"
            )
            if not video.is_file():
                raise FileNotFoundError(video)
            candidates.append(
                {
                    "candidate_id": f"seed-{seed}",
                    "seed": seed,
                    "video": str(video),
                    "is_incumbent": seed == args.seeds[0],
                }
            )
        spec_cases.append(
            {
                "case_id": case_id,
                "scene_uid": protocol_cases[case_id]["scene_uid"],
                "conditioning_image": str(image),
                "prompt": source_case["text_prompt"],
                "candidates": candidates,
            }
        )

    protocol = {
        "_meta": {
            "schema": DL3DV_PROTOCOL_SCHEMA,
            "formal_protocol": False,
            "purpose": "legacy-debug-smoke-only; forbidden for method selection",
            "source_manifest": str(source_path),
            "source_manifest_sha256": file_sha256(source_path),
            "split_counts": {"debug": 3},
            "candidate_seed_policy": {
                "version": "legacy-debug-explicit-v1",
                "candidate_seeds": args.seeds,
                "incumbent_seed": args.seeds[0],
            },
            "generation_profiles": {"Wan2.2-TI2V-5B": GENERATION},
        },
        **protocol_cases,
    }
    publish_json_no_clobber(protocol, args.protocol_output)
    spec = {
        "schema": CANDIDATE_SPEC_SCHEMA,
        "artifact_mode": "legacy-debug",
        "protocol_manifest_sha256": file_sha256(args.protocol_output),
        "split": "debug",
        "candidate_count": len(args.seeds),
        "backbone": "Wan2.2-TI2V-5B",
        "generation": GENERATION,
        "cases": spec_cases,
    }
    publish_json_no_clobber(spec, args.spec_output)
    print(
        json.dumps(
            {
                "formal": False,
                "protocol": str(args.protocol_output.resolve()),
                "spec": str(args.spec_output.resolve()),
                "cases": len(spec_cases),
                "candidates_per_case": len(args.seeds),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
