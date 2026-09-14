#!/usr/bin/env python3
"""Run a frozen C2F configuration over one deterministic manifest shard."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan
from diffusers import WanImageToVideoPipeline as OfficialWanImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pipeline_class(path: Path):
    spec = importlib.util.spec_from_file_location("_frozen_c2f_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WanImageToVideoPipeline


def git_identity() -> dict:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
        ).strip()
    )
    return {"commit": commit, "dirty": dirty}


def ordered_cases(manifest: dict) -> list[tuple[str, dict]]:
    records = [(case_id, case) for case_id, case in manifest.items() if not case_id.startswith("_")]
    return sorted(
        records,
        key=lambda item: item[1].get("c2f_dev_selection", {}).get("selection_order", item[0]),
    )


def index_baselines(root: Path, seed: int) -> dict[str, tuple[Path, Path, dict]]:
    indexed: dict[str, tuple[Path, Path, dict]] = {}
    for metadata_path in root.rglob("metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        method = metadata.get("method", metadata.get("run_config", {}).get("method"))
        record_seed = metadata.get("seed", metadata.get("run_config", {}).get("seed"))
        case_id = metadata.get("case_id", metadata.get("run_config", {}).get("case_id"))
        if method != "baseline" or record_seed != seed or not case_id:
            continue
        video_path = metadata_path.parent / "video.mp4"
        if not video_path.is_file():
            continue
        if case_id in indexed:
            raise RuntimeError(f"multiple seed-{seed} baselines found for {case_id}")
        indexed[case_id] = (video_path, metadata_path, metadata)
    return indexed


def validate_baseline(
    case_id: str,
    case: dict,
    baseline: tuple[Path, Path, dict],
    config: dict,
) -> None:
    _, _, metadata = baseline
    generation = metadata.get("generation", {})
    expected = config["generation"]
    mismatches = {
        key: (generation.get(key), value)
        for key, value in expected.items()
        if generation.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"baseline generation mismatch for {case_id}: {mismatches}")
    if metadata.get("prompt") != case["text_prompt"]:
        raise RuntimeError(f"baseline prompt mismatch for {case_id}")
    image_path = Path(case["image_prompt"])
    if metadata.get("image_sha256") != sha256_file(image_path):
        raise RuntimeError(f"baseline conditioning image mismatch for {case_id}")


def probe_video(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate", "-of", "json", str(path),
    ]
    return json.loads(subprocess.check_output(command, text=True))["streams"][0]


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num_shards >= 1 and 0 <= shard_index < num_shards")
    if args.case_index is not None and (args.num_shards != 1 or args.shard_index != 0):
        parser.error("--case-index cannot be combined with sharding")

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha = sha256_file(config_path)
    source_path = Path(config["source_manifest"])
    if sha256_file(source_path) != config["source_manifest_sha256"]:
        raise RuntimeError("source manifest digest differs from frozen config")
    selection_path = Path(config["selection_manifest"])
    if not selection_path.is_absolute():
        selection_path = REPO_ROOT / selection_path
    manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    manifest_meta = manifest.get("_meta", {})
    if manifest_meta.get("source_manifest_sha256") != config["source_manifest_sha256"]:
        raise RuntimeError("selection manifest source digest differs from frozen config")
    overlap_counts = manifest_meta.get("reserved_overlap_counts")
    expected_overlap_counts = {split: 0 for split in config["required_zero_overlap_splits"]}
    if overlap_counts != expected_overlap_counts:
        raise RuntimeError(
            f"selection manifest lacks a passing reserved-split audit: "
            f"{overlap_counts} != {expected_overlap_counts}"
        )
    cases = ordered_cases(manifest)
    if args.case_index is not None:
        if not 0 <= args.case_index < len(cases):
            parser.error(f"case-index must be in [0, {len(cases) - 1}]")
        cases = [cases[args.case_index]]
    else:
        cases = [record for index, record in enumerate(cases) if index % args.num_shards == args.shard_index]

    baseline_index = index_baselines(Path(config["baseline_root"]), int(config["seed"]))
    for case_id, case in cases:
        if case_id not in baseline_index:
            raise RuntimeError(f"missing existing seed-{config['seed']} baseline for {case_id}")
        validate_baseline(case_id, case, baseline_index[case_id], config)

    pipeline_kind = config.get("pipeline_kind", "custom")
    if pipeline_kind not in {"custom", "official"}:
        raise ValueError("pipeline_kind must be 'custom' or 'official'")
    if pipeline_kind == "official":
        if config.get("method"):
            raise ValueError("official pipeline reference requires an empty method configuration")
        pipeline_path = Path(inspect.getfile(OfficialWanImageToVideoPipeline)).resolve()
        PipelineClass = OfficialWanImageToVideoPipeline
    else:
        pipeline_path = Path(config["pipeline_path"])
        if not pipeline_path.is_absolute():
            pipeline_path = REPO_ROOT / pipeline_path
        PipelineClass = load_pipeline_class(pipeline_path)
    pipeline_sha = sha256_file(pipeline_path)
    identity = git_identity()
    print("repo:", REPO_ROOT)
    print("git:", identity)
    print("config_sha256:", config_sha)
    print("selection_sha256:", sha256_file(selection_path))
    print("pipeline_sha256:", pipeline_sha)
    print("shard cases:", len(cases))
    if args.validate_only:
        print("validation-only gate passed")
        return

    model_path = config["model_path"]
    vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
    pipe = PipelineClass.from_pretrained(model_path, vae=vae, torch_dtype=torch.bfloat16).to(args.device)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    generation = config["generation"]
    method = config["method"]
    for ordinal, (case_id, case) in enumerate(cases, start=1):
        output_dir = args.output_root / config["method_id"] / case_id / f"seed_{config['seed']}"
        video_path = output_dir / "video.mp4"
        metadata_path = output_dir / "metadata.json"
        complete_path = output_dir / "COMPLETE.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        if complete_path.is_file():
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            expected = {
                "config_sha256": config_sha,
                "selection_manifest_sha256": sha256_file(selection_path),
                "pipeline_sha256": pipeline_sha,
            }
            if all(complete.get(key) == value for key, value in expected.items()) and video_path.is_file():
                print(f"[{ordinal}/{len(cases)}] skip complete {case_id}")
                continue
            raise RuntimeError(f"stale or mismatched COMPLETE output: {output_dir}")
        if video_path.exists() or metadata_path.exists():
            raise RuntimeError(f"refusing to overwrite partial output: {output_dir}")

        image_path = Path(case["image_prompt"])
        image_sha = sha256_file(image_path)
        baseline_video, baseline_metadata, _ = baseline_index[case_id]
        print(f"[{ordinal}/{len(cases)}] generate {case_id}", flush=True)
        generator = torch.Generator(device=args.device).manual_seed(int(config["seed"]))
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        started = time.perf_counter()
        with torch.inference_mode():
            output = pipe(
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
        export_to_video(output.frames[0], str(temporary_video), fps=int(generation["fps"]))
        probe = probe_video(temporary_video)
        expected_probe = {
            "nb_read_frames": str(generation["frames"]),
            "width": int(generation["width"]),
            "height": int(generation["height"]),
        }
        probe_mismatch = {key: (probe.get(key), value) for key, value in expected_probe.items() if probe.get(key) != value}
        if probe_mismatch:
            raise RuntimeError(f"generated video probe mismatch: {probe_mismatch}")
        temporary_video.replace(video_path)

        metadata = {
            "schema": "wan-c2f-generation-record-v1",
            "case_id": case_id,
            "case": case,
            "method_id": config["method_id"],
            "pipeline_kind": pipeline_kind,
            "seed": config["seed"],
            "prompt": case["text_prompt"],
            "image_path": str(image_path),
            "image_sha256": image_sha,
            "generation": generation,
            "method": method,
            "config_path": str(config_path),
            "config_sha256": config_sha,
            "source_manifest_sha256": config["source_manifest_sha256"],
            "selection_manifest": str(selection_path),
            "selection_manifest_sha256": sha256_file(selection_path),
            "pipeline_path": str(pipeline_path),
            "pipeline_sha256": pipeline_sha,
            "code_identity": identity,
            "baseline_video": str(baseline_video),
            "baseline_video_sha256": sha256_file(baseline_video),
            "baseline_metadata": str(baseline_metadata),
            "video_sha256": sha256_file(video_path),
            "video_probe": probe,
            "wall_seconds": wall_seconds,
            "peak_memory_mib": peak_memory_mib,
            "hostname": os.uname().nodename,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
        }
        atomic_json(metadata_path, metadata)
        atomic_json(
            complete_path,
            {
                "status": "complete",
                "config_sha256": config_sha,
                "selection_manifest_sha256": sha256_file(selection_path),
                "pipeline_sha256": pipeline_sha,
                "video_sha256": metadata["video_sha256"],
                "metadata_sha256": sha256_file(metadata_path),
            },
        )
        print(
            f"[{ordinal}/{len(cases)}] saved {video_path} "
            f"wall={wall_seconds:.1f}s peak={peak_memory_mib:.1f}MiB",
            flush=True,
        )
        if hasattr(pipe.vae, "clear_cache"):
            pipe.vae.clear_cache()
        del output
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
