#!/usr/bin/env python3
"""Replay frozen Dev25 trajectories and persist read-only C2F diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE = REPO_ROOT / "external/guidance_wan/pipeline_wan_i2v_c2f_forensics.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_pipeline_class(path: Path):
    spec = importlib.util.spec_from_file_location("_c2f_forensic_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WanImageToVideoPipeline


def git_identity() -> dict:
    commit = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True).strip()
    )
    return {"commit": commit, "dirty": dirty}


def ordered_cases(manifest: dict) -> list[tuple[str, dict]]:
    records = [(case_id, case) for case_id, case in manifest.items() if not case_id.startswith("_")]
    return sorted(
        records,
        key=lambda item: item[1].get("c2f_dev_selection", {}).get("selection_order", item[0]),
    )


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["baseline_observe", "c2f"], required=True)
    parser.add_argument("--pipeline-path", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--disable-diagnostics", action="store_true")
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--full-replay", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num_shards >= 1 and 0 <= shard_index < num_shards")
    if args.case_index is not None and (args.num_shards != 1 or args.shard_index != 0):
        parser.error("--case-index cannot be combined with sharding")
    if args.save_video and not args.full_replay:
        parser.error("--save-video requires --full-replay")
    if args.sample_size < 0:
        parser.error("--sample-size must be non-negative")

    config_path = args.config.resolve()
    lock_path = args.lock.resolve()
    pipeline_path = args.pipeline_path.resolve()
    config = read_json(config_path)
    lock = read_json(lock_path)
    if lock.get("schema") != "c2f-p0-forensics-lock-v1":
        raise RuntimeError("unexpected P0 lock schema")
    locked_config = lock["frozen_inputs"]["config"]
    if sha256_file(config_path) != locked_config["sha256"]:
        raise RuntimeError("config differs from P0 lock")

    selection_path = resolve_repo_path(config["selection_manifest"])
    manifest = read_json(selection_path)
    locked_manifest = lock["frozen_inputs"]["manifest"]
    if sha256_file(selection_path) != locked_manifest["sha256"]:
        raise RuntimeError("selection manifest differs from P0 lock")
    expected_overlap = {"test": 0, "validation": 0, "debug": 0}
    if manifest.get("_meta", {}).get("reserved_overlap_counts") != expected_overlap:
        raise RuntimeError("Dev25 overlaps a reserved split")

    outcome_by_case = {
        record["case_id"]: record for record in lock["cases_ranked_best_to_worst"]
    }
    cases = ordered_cases(manifest)
    if {case_id for case_id, _ in cases} != set(outcome_by_case):
        raise RuntimeError("P0 lock and selection manifest case sets differ")
    if args.case_index is not None:
        if not 0 <= args.case_index < len(cases):
            parser.error(f"case-index must be in [0, {len(cases) - 1}]")
        cases = [cases[args.case_index]]
    else:
        cases = [record for index, record in enumerate(cases) if index % args.num_shards == args.shard_index]

    identity = git_identity()
    if identity["dirty"] and not args.allow_dirty:
        raise RuntimeError("refusing to run from a dirty worktree; commit the forensic implementation first")
    if not pipeline_path.is_file():
        raise FileNotFoundError(pipeline_path)
    pipeline_sha = sha256_file(pipeline_path)
    diagnostics_enabled = not args.disable_diagnostics
    PipelineClass = load_pipeline_class(pipeline_path)
    call_parameters = inspect.signature(PipelineClass.__call__).parameters
    if diagnostics_enabled and "c2f_diagnostics" not in call_parameters:
        raise RuntimeError("selected pipeline does not support forensic diagnostics")

    print("repo:", REPO_ROOT)
    print("git:", identity)
    print("pipeline:", pipeline_path)
    print("pipeline_sha256:", pipeline_sha)
    print("mode:", args.mode)
    print("diagnostics:", diagnostics_enabled)
    print("full_replay:", args.full_replay)
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
    frozen_method = dict(config["method"])
    frozen_method["attn_avg_debug"] = False
    if args.mode == "baseline_observe":
        frozen_method["attn_avg_alpha"] = 0.0
    stop_after_step = int(config["method"]["attn_avg_end"])

    for ordinal, (case_id, case) in enumerate(cases, start=1):
        run_name = "diagnostic" if diagnostics_enabled else "reference"
        replay_name = "full" if args.full_replay else f"through_step_{stop_after_step}"
        output_dir = args.output_root / args.mode / run_name / replay_name / case_id / f"seed_{config['seed']}"
        complete_path = output_dir / "COMPLETE.json"
        record_path = output_dir / "replay.json"
        video_path = output_dir / "video.mp4"
        output_dir.mkdir(parents=True, exist_ok=True)
        if complete_path.is_file():
            complete = read_json(complete_path)
            expected = {
                "pipeline_sha256": pipeline_sha,
                "config_sha256": sha256_file(config_path),
                "lock_sha256": sha256_file(lock_path),
                "diagnostics_enabled": diagnostics_enabled,
            }
            if all(complete.get(key) == value for key, value in expected.items()):
                print(f"[{ordinal}/{len(cases)}] skip complete {case_id}")
                continue
            raise RuntimeError(f"stale output at {output_dir}")
        if record_path.exists() or video_path.exists():
            raise RuntimeError(f"refusing to overwrite partial output at {output_dir}")

        def stop_callback(pipeline, step_index, timestep, callback_kwargs):
            if step_index >= stop_after_step:
                pipeline._interrupt = True
            return callback_kwargs

        call_kwargs = {
            "prompt": case["text_prompt"],
            "negative_prompt": generation["negative_prompt"],
            "image": Image.open(case["image_prompt"]).convert("RGB"),
            "height": int(generation["height"]),
            "width": int(generation["width"]),
            "num_frames": int(generation["frames"]),
            "num_inference_steps": int(generation["steps"]),
            "guidance_scale": float(generation["guidance_scale"]),
            "generator": torch.Generator(device=args.device).manual_seed(int(config["seed"])),
            "output_type": "np" if args.save_video else "latent",
            **frozen_method,
        }
        if diagnostics_enabled:
            call_kwargs.update(
                c2f_diagnostics=True,
                c2f_diagnostics_sample_size=args.sample_size,
            )
        if not args.full_replay:
            call_kwargs.update(
                callback_on_step_end=stop_callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )

        print(f"[{ordinal}/{len(cases)}] replay {case_id}", flush=True)
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        started = time.perf_counter()
        with torch.inference_mode():
            output = pipe(**call_kwargs)
        torch.cuda.synchronize(torch.device(args.device))
        wall_seconds = time.perf_counter() - started
        peak_memory_mib = torch.cuda.max_memory_allocated(torch.device(args.device)) / 1024**2

        latent_digest = None
        if args.save_video:
            export_to_video(output.frames[0], str(video_path), fps=int(generation["fps"]))
        else:
            latent_digest = tensor_sha256(output.frames)

        diagnostics = getattr(pipe, "_last_c2f_diagnostics", None)
        if diagnostics_enabled and diagnostics is None:
            raise RuntimeError("diagnostic pipeline returned no diagnostics")
        if diagnostics is not None:
            diagnostics.update(
                case_id=case_id,
                replay_mode=args.mode,
                primary_outcome=outcome_by_case[case_id],
            )

        record = {
            "schema": "c2f-p0-replay-record-v1",
            "case_id": case_id,
            "case": case,
            "replay_mode": args.mode,
            "diagnostics_enabled": diagnostics_enabled,
            "partial_replay": not args.full_replay,
            "last_computed_step": stop_after_step if not args.full_replay else int(generation["steps"]) - 1,
            "latent_sha256": latent_digest,
            "video_path": str(video_path) if args.save_video else None,
            "primary_outcome": outcome_by_case[case_id],
            "generation": generation,
            "method": frozen_method,
            "sample_size": args.sample_size if diagnostics_enabled else 0,
            "pipeline_path": str(pipeline_path),
            "pipeline_sha256": pipeline_sha,
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "lock_path": str(lock_path),
            "lock_sha256": sha256_file(lock_path),
            "selection_manifest": str(selection_path),
            "selection_manifest_sha256": sha256_file(selection_path),
            "reserved_overlap_counts": expected_overlap,
            "code_identity": identity,
            "wall_seconds": wall_seconds,
            "peak_memory_mib": peak_memory_mib,
            "hostname": os.uname().nodename,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
            "diagnostics": diagnostics,
        }
        atomic_json(record_path, record)
        complete = {
            "status": "complete",
            "pipeline_sha256": pipeline_sha,
            "config_sha256": sha256_file(config_path),
            "lock_sha256": sha256_file(lock_path),
            "diagnostics_enabled": diagnostics_enabled,
            "record_sha256": sha256_file(record_path),
            "latent_sha256": latent_digest,
        }
        atomic_json(complete_path, complete)
        print(
            f"[{ordinal}/{len(cases)}] done wall={wall_seconds:.1f}s peak={peak_memory_mib:.1f}MiB "
            f"latent_sha256={latent_digest}",
            flush=True,
        )


if __name__ == "__main__":
    main()
