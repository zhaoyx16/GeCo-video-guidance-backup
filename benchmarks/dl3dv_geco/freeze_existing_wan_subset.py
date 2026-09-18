#!/usr/bin/env python3
"""Freeze the DL3DV cases that already have paired Wan seed-0/1 baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_value(command: list[str], flag: str) -> str | None:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream: {path}")
    stream = streams[0]
    actual = {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": stream["avg_frame_rate"],
        "nb_frames": int(stream["nb_frames"]),
    }
    expected = {
        "width": 1280,
        "height": 704,
        "avg_frame_rate": "24/1",
        "nb_frames": 121,
    }
    if actual != expected:
        raise ValueError(f"Video probe mismatch for {path}: {actual} != {expected}")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-f",
            "null",
            "-",
        ],
        check=True,
    )
    return actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--wan-baseline-root", type=Path, required=True)
    parser.add_argument("--job-state-root", type=Path, required=True)
    parser.add_argument("--equivalence-report", type=Path, required=True)
    parser.add_argument("--validated-pipeline", type=Path, required=True)
    parser.add_argument("--bundle-input-checksums", type=Path, required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.source_manifest.read_text())
    equivalence = json.loads(args.equivalence_report.read_text())
    if (
        equivalence.get("passed") is not True
        or equivalence.get("official_repeat_valid") is not True
        or equivalence.get("candidate", {}).get("max_abs") != 0.0
    ):
        raise SystemExit("Wan guidance-off equivalence report is not an exact pass")
    validated_pipeline_sha256 = sha256_file(args.validated_pipeline)
    tested_pipeline_sha256 = (
        equivalence.get("candidate_meta", {})
        .get("implementation_source", {})
        .get("sha256")
    )
    if tested_pipeline_sha256 != validated_pipeline_sha256:
        raise SystemExit(
            "Equivalence artifact is not bound to the supplied validated pipeline"
        )
    input_checksums = json.loads(args.bundle_input_checksums.read_text())

    frozen = {}
    for case_id, case in source.items():
        if case_id.startswith("_") or case.get("dataset") != "dl3dv":
            continue
        videos = {
            str(seed): (
                args.wan_baseline_root
                / case_id
                / f"baseline_seed{seed}_steps50_frames121.mp4"
            )
            for seed in (0, 1)
        }
        if not all(path.is_file() for path in videos.values()):
            continue
        item = dict(case)
        image_path = Path(item["image_prompt"])
        if not image_path.is_absolute():
            image_path = args.source_manifest.resolve().parent / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing conditioning image: {image_path}")
        item["image_prompt"] = str(image_path.resolve())
        item["image_sha256"] = sha256_file(image_path)
        original_relative_image = case["image_prompt"]
        if input_checksums.get(original_relative_image) != item["image_sha256"]:
            raise ValueError(
                f"Packaged image checksum mismatch for {case_id}: {original_relative_image}"
            )
        item["existing_wan_baselines"] = {
            seed: {}
            for seed in videos
        }
        for seed, path in videos.items():
            records = sorted(
                args.job_state_root.glob(f"wan_{case_id}_seed{seed}_*.done.json")
            )
            if len(records) != 1:
                raise ValueError(
                    f"Expected one done record for {case_id} seed {seed}, found {records}"
                )
            record_path = records[0]
            record = json.loads(record_path.read_text())
            command = record.get("command", [])
            expected_command = {
                "--case": case_id,
                "--mode": "baseline",
                "--height": "704",
                "--width": "1280",
                "--frames": "121",
                "--steps": "50",
                "--fps": "24",
                "--seed": str(seed),
                "--model": args.expected_model,
            }
            mismatches = {
                flag: (command_value(command, flag), expected)
                for flag, expected in expected_command.items()
                if command_value(command, flag) != expected
            }
            if record.get("returncode") != 0 or record.get("status") != "done":
                mismatches["job_status"] = (
                    (record.get("returncode"), record.get("status")),
                    (0, "done"),
                )
            if Path(record.get("output", "")).resolve() != path.resolve():
                mismatches["output"] = (record.get("output"), str(path.resolve()))
            if Path(command_value(command, "--prompt_json") or "").resolve() != (
                args.source_manifest.resolve()
            ):
                mismatches["prompt_json"] = (
                    command_value(command, "--prompt_json"),
                    str(args.source_manifest.resolve()),
                )
            if mismatches:
                raise ValueError(
                    f"Job provenance mismatch for {case_id} seed {seed}: {mismatches}"
                )
            runner_path = Path(command[1])
            if not runner_path.is_file():
                raise FileNotFoundError(f"Recorded runner is missing: {runner_path}")
            runner_source = runner_path.read_text()
            if "guidance_scale=5.0" not in runner_source:
                raise ValueError(f"Baseline runner CFG is not the audited 5.0: {runner_path}")
            if "negative_prompt=" in runner_source:
                raise ValueError(
                    f"Baseline runner unexpectedly supplies a negative prompt: {runner_path}"
                )
            pipeline_path = (
                runner_path.parent
                / "external/guidance_wan/pipeline_wan_i2v_full_guided.py"
            )
            if not pipeline_path.is_file():
                raise FileNotFoundError(
                    f"Recorded bundle pipeline is missing: {pipeline_path}"
                )
            if sha256_file(pipeline_path) != validated_pipeline_sha256:
                raise ValueError(
                    f"Baseline pipeline differs from validated pipeline: {pipeline_path}"
                )
            item["existing_wan_baselines"][seed] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "video_probe": probe_video(path),
                "job_record": str(record_path.resolve()),
                "job_record_sha256": sha256_file(record_path),
                "runner": str(runner_path.resolve()),
                "runner_sha256": sha256_file(runner_path),
                "pipeline": str(pipeline_path.resolve()),
                "pipeline_sha256": sha256_file(pipeline_path),
                "model": command_value(command, "--model"),
                "elapsed_sec": record.get("elapsed_sec"),
            }
        frozen[case_id] = item

    if not frozen:
        raise SystemExit("No cases have both expected Wan baseline videos")

    split_counts = Counter(case["split"] for case in frozen.values())
    family_counts = Counter(case["trajectory"]["family"] for case in frozen.values())
    frozen["_benchmark"] = {
        "schema": "dl3dv-existing-wan-subset-v1",
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "wan_baseline_root": str(args.wan_baseline_root.resolve()),
        "job_state_root": str(args.job_state_root.resolve()),
        "equivalence_report": str(args.equivalence_report.resolve()),
        "equivalence_report_sha256": sha256_file(args.equivalence_report),
        "validated_pipeline": str(args.validated_pipeline.resolve()),
        "validated_pipeline_sha256": validated_pipeline_sha256,
        "bundle_input_checksums": str(args.bundle_input_checksums.resolve()),
        "bundle_input_checksums_sha256": sha256_file(args.bundle_input_checksums),
        "expected_model": args.expected_model,
        "num_cases": len(frozen),
        "num_videos": 2 * len(frozen),
        "seeds": [0, 1],
        "generation": {
            "steps": 50,
            "frames": 121,
            "height": 704,
            "width": 1280,
            "fps": 24,
            "guidance_scale": 5.0,
            "negative_prompt": None,
        },
        "split_counts": dict(split_counts),
        "motion_family_counts": dict(family_counts),
        "scene_ids": sorted({case["scene_id"] for case in frozen.values()}),
        "statistical_unit": "scene_id",
        "paired_methods_must_reuse_prompt_image_seed": True,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(frozen, indent=2) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(frozen) - 1,
                "videos": 2 * (len(frozen) - 1),
                "splits": dict(split_counts),
                "motion_families": dict(family_counts),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
