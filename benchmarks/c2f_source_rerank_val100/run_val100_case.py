#!/usr/bin/env python3
"""Run one frozen Val100 case for a configured Wan attention intervention."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_pipeline_class(path: Path):
    spec = importlib.util.spec_from_file_location("_frozen_val100_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WanImageToVideoPipeline


def git_identity() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
        ).strip()
    )
    return {"commit": commit, "dirty": dirty}


def ordered_cases(manifest: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    records = [(case_id, case) for case_id, case in manifest.items() if not case_id.startswith("_")]
    records.sort(key=lambda item: (int(item[1]["split_order"]), item[0]))
    orders = [int(case["split_order"]) for _, case in records]
    if len(records) != 100 or len(set(orders)) != 100:
        raise RuntimeError("frozen Val100 manifest must contain 100 unique split_order values")
    if any(case.get("protocol_split") != "validation" for _, case in records):
        raise RuntimeError("Val100 runner only accepts validation cases")
    return records


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,width,height,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    return json.loads(subprocess.check_output(command, text=True))["streams"][0]


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    manifest_path = args.manifest.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if config.get("schema") != "wan-c2f-val100-method-v1":
        raise RuntimeError("unexpected method config schema")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != config["manifest_sha256"]:
        raise RuntimeError("frozen Val100 manifest digest differs from method config")
    meta = manifest.get("_meta", {})
    if not meta.get("formal_protocol") or not meta.get("validation_only"):
        raise RuntimeError("manifest is not the frozen validation-only protocol")

    cases = ordered_cases(manifest)
    if not 0 <= args.case_index < len(cases):
        raise ValueError(f"case-index must be in [0,{len(cases) - 1}]")
    case_id, case = cases[args.case_index]
    image_path = (args.dataset_root / case["dataset_relative_image"]).resolve()
    if not image_path.is_file() or sha256_file(image_path) != case["image_sha256"]:
        raise RuntimeError(f"conditioning image is missing or changed: {image_path}")

    identity = git_identity()
    if args.expected_git_commit and identity["commit"] != args.expected_git_commit:
        raise RuntimeError("repository commit differs from --expected-git-commit")
    if identity["dirty"] and not args.allow_dirty:
        raise RuntimeError("refusing to run from a dirty worktree")

    pipeline_path = resolve_repo_path(config["pipeline_path"]).resolve()
    runner_path = Path(__file__).resolve()
    pipeline_sha = sha256_file(pipeline_path)
    runner_sha = sha256_file(runner_path)
    output_dir = (
        args.output_root.resolve()
        / config["method_id"]
        / f"{args.case_index:03d}_{case_id}"
        / f"seed_{int(config['seed'])}"
    )
    video_path = output_dir / "video.mp4"
    metadata_path = output_dir / "metadata.json"
    diagnostics_path = output_dir / "c2f_diagnostics.json"
    complete_path = output_dir / "COMPLETE.json"
    expected_complete = {
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": manifest_sha,
        "pipeline_sha256": pipeline_sha,
        "runner_sha256": runner_sha,
        "git_commit": identity["commit"],
    }
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if all(complete.get(key) == value for key, value in expected_complete.items()) and video_path.is_file():
            print(f"skip complete {config['method_id']} {case_id}")
            return
        raise RuntimeError(f"stale COMPLETE marker: {complete_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite partial output: {output_dir}")

    print("method:", config["method_id"])
    print("case:", case_id)
    print("case_index:", args.case_index)
    print("image:", image_path)
    print("pipeline_sha256:", pipeline_sha)
    print("runner_sha256:", runner_sha)
    if args.validate_only:
        print("validation-only gate passed")
        return

    output_dir.mkdir(parents=True, exist_ok=False)
    PipelineClass = load_pipeline_class(pipeline_path)
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    pipe = PipelineClass.from_pretrained(args.model, vae=vae, torch_dtype=torch.bfloat16).to(args.device)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    generation = config["generation"]
    method = config["method"]
    generator = torch.Generator(device=args.device).manual_seed(int(config["seed"]))
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    started = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            prompt=case["text_prompt"],
            negative_prompt=generation["negative_prompt"],
            image=Image.open(image_path).convert("RGB"),
            height=int(generation["height"]),
            width=int(generation["width"]),
            num_frames=int(generation["frames"]),
            num_inference_steps=int(generation["steps"]),
            guidance_scale=float(generation["guidance_scale"]),
            generator=generator,
            **method,
        )
    torch.cuda.synchronize(torch.device(args.device))
    wall_seconds = time.perf_counter() - started
    peak_memory_mib = torch.cuda.max_memory_allocated(torch.device(args.device)) / 1024**2

    temporary_video = output_dir / f".video.{uuid.uuid4().hex}.tmp.mp4"
    export_to_video(result.frames[0], str(temporary_video), fps=int(generation["fps"]))
    video_probe = probe_video(temporary_video)
    expected_probe = {
        "nb_read_frames": str(generation["frames"]),
        "width": int(generation["width"]),
        "height": int(generation["height"]),
        "r_frame_rate": f"{int(generation['fps'])}/1",
    }
    actual_probe = {
        "nb_read_frames": video_probe.get("nb_read_frames"),
        "width": int(video_probe["width"]),
        "height": int(video_probe["height"]),
        "r_frame_rate": video_probe["r_frame_rate"],
    }
    if actual_probe != expected_probe:
        raise RuntimeError(f"generated video probe mismatch: {actual_probe} != {expected_probe}")
    temporary_video.replace(video_path)

    diagnostics = getattr(pipe, "_last_c2f_diagnostics", None)
    if diagnostics is not None:
        atomic_json(diagnostics_path, diagnostics)
    metadata = {
        "schema": "wan-c2f-val100-output-v1",
        "method_id": config["method_id"],
        "case_id": case_id,
        "case_index": args.case_index,
        "seed": int(config["seed"]),
        "prompt": case["text_prompt"],
        "image_path": str(image_path),
        "image_sha256": case["image_sha256"],
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "pipeline": str(pipeline_path),
        "pipeline_sha256": pipeline_sha,
        "runner_sha256": runner_sha,
        "code_identity": identity,
        "model": str(args.model.resolve()),
        "generation": generation,
        "method": method,
        "wall_seconds": wall_seconds,
        "peak_memory_mib": peak_memory_mib,
        "video_probe": actual_probe,
        "video_sha256": sha256_file(video_path),
        "diagnostics_sha256": sha256_file(diagnostics_path) if diagnostics_path.is_file() else None,
    }
    atomic_json(metadata_path, metadata)
    atomic_json(
        complete_path,
        {
            "status": "complete",
            **expected_complete,
            "video_sha256": metadata["video_sha256"],
            "metadata_sha256": sha256_file(metadata_path),
            "diagnostics_sha256": metadata["diagnostics_sha256"],
        },
    )
    print(f"saved {video_path}")
    print(f"wall_seconds={wall_seconds:.2f} peak_memory_mib={peak_memory_mib:.1f}")


if __name__ == "__main__":
    main()
