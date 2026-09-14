#!/usr/bin/env python3
"""Build deterministic visual-review artifacts for every frozen dev pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent


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


def ordered_cases(manifest: dict) -> list[tuple[str, dict]]:
    records = [(case_id, case) for case_id, case in manifest.items() if not case_id.startswith("_")]
    return sorted(records, key=lambda item: item[1]["c2f_dev_selection"]["selection_order"])


def verify_generated(video: Path, metadata_path: Path, complete_path: Path, case_id: str) -> dict:
    if not all(path.is_file() for path in (video, metadata_path, complete_path)):
        raise FileNotFoundError(f"incomplete generated artifact for {case_id}: {video.parent}")
    metadata = read_json(metadata_path)
    complete = read_json(complete_path)
    video_sha = sha256_file(video)
    if metadata.get("case_id") != case_id or metadata.get("video_sha256") != video_sha:
        raise ValueError(f"metadata/video mismatch for {case_id}")
    if complete.get("status") != "complete" or complete.get("video_sha256") != video_sha:
        raise ValueError(f"COMPLETE/video mismatch for {case_id}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--baseline-method", default="official_same_host_reference")
    parser.add_argument("--candidate-method", default="c2f_k3_a0025")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-indices", default="0,12,24,36,48,60,72,84,96,108,120")
    parser.add_argument("--tile-width", type=int, default=240)
    args = parser.parse_args()

    manifest = read_json(args.selection_manifest)
    expected_overlap = {"test": 0, "validation": 0, "debug": 0}
    if manifest.get("_meta", {}).get("reserved_overlap_counts") != expected_overlap:
        raise RuntimeError("selection manifest does not certify zero reserved-split overlap")

    cases = ordered_cases(manifest)
    if len(cases) != 25:
        raise RuntimeError(f"expected exactly 25 development cases, found {len(cases)}")

    index_records = []
    for ordinal, (case_id, case) in enumerate(cases):
        baseline_dir = args.generation_root / args.baseline_method / case_id / f"seed_{args.seed}"
        candidate_dir = args.generation_root / args.candidate_method / case_id / f"seed_{args.seed}"
        baseline_video = baseline_dir / "video.mp4"
        candidate_video = candidate_dir / "video.mp4"
        baseline_meta = verify_generated(
            baseline_video, baseline_dir / "metadata.json", baseline_dir / "COMPLETE.json", case_id
        )
        candidate_meta = verify_generated(
            candidate_video, candidate_dir / "metadata.json", candidate_dir / "COMPLETE.json", case_id
        )
        for key in (
            "prompt",
            "image_sha256",
            "seed",
            "generation",
            "hostname",
            "cuda_visible_devices",
            "torch_version",
        ):
            if baseline_meta.get(key) != candidate_meta.get(key):
                raise RuntimeError(f"paired {key} mismatch for {case_id}")

        pair_dir = args.output_root / case_id
        report_path = pair_dir / "visual_review.json"
        if report_path.is_file():
            report = read_json(report_path)
            if (
                report.get("baseline") != str(baseline_video.resolve())
                or report.get("candidate") != str(candidate_video.resolve())
                or report.get("frame_count") != 121
            ):
                raise RuntimeError(f"stale visual review for {case_id}")
            status = "verified-existing"
        else:
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "make_pair_visuals.py"),
                    "--baseline",
                    str(baseline_video),
                    "--candidate",
                    str(candidate_video),
                    "--output-dir",
                    str(pair_dir),
                    "--frame-indices",
                    args.frame_indices,
                    "--tile-width",
                    str(args.tile_width),
                ],
                check=True,
            )
            report = read_json(report_path)
            status = "created"
        index_records.append(
            {
                "ordinal": ordinal,
                "case_id": case_id,
                "motion_stratum": case["c2f_dev_selection"]["stratum"],
                "selection_score": case["pose_stats"]["selection_score"],
                "status": status,
                "mean_absolute_pixel_difference": report["mean_absolute_pixel_difference"],
                "hostname": candidate_meta["hostname"],
                "physical_gpu": candidate_meta["cuda_visible_devices"],
                "torch_version": candidate_meta["torch_version"],
                "contact_sheet": report["contact_sheet"],
                "side_by_side": report["side_by_side"],
                "visual_review": str(report_path.resolve()),
            }
        )

    payload = {
        "schema": "wan-c2f-dev25-visual-review-index-v1",
        "selection_manifest": str(args.selection_manifest.resolve()),
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "reserved_overlap_counts": expected_overlap,
        "baseline_method": args.baseline_method,
        "candidate_method": args.candidate_method,
        "frame_indices": [int(value) for value in args.frame_indices.split(",")],
        "pairs": index_records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    index_path = args.output_root / "VISUAL_REVIEW_INDEX.json"
    with tempfile.NamedTemporaryFile("w", dir=args.output_root, delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(index_path)
    print(json.dumps({"status": "complete", "pairs": len(index_records), "index": str(index_path)}, indent=2))


if __name__ == "__main__":
    main()
