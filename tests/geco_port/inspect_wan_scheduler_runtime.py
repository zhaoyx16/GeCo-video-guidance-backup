#!/usr/bin/env python3
"""Read-only runtime preflight for the actual Wan scheduler used by adapted GeCo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import torch


def tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def cuda_runtime_identity(device: str) -> dict:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"Scheduler preflight requires CUDA, got {device!r}")
    index = 0 if resolved.index is None else resolved.index
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(index),
        "device_capability": list(torch.cuda.get_device_capability(index)),
    }


def canonical_scheduler_config(config: dict) -> dict:
    """Canonicalise the order-insensitive Diffusers default-value set for hashing."""
    canonical = dict(config)
    defaults = canonical.get("_use_default_values")
    if isinstance(defaults, list) and all(isinstance(value, str) for value in defaults):
        canonical["_use_default_values"] = sorted(defaults)
    return canonical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repo = args.repo.resolve()
    if git_output(repo, "status", "--porcelain"):
        raise RuntimeError("Scheduler preflight requires a clean repository.")
    code_commit = git_output(repo, "rev-parse", "HEAD")
    sys.path.insert(0, str(repo))
    import diffusers
    from benchmarks.dl3dv_geco.run_generation_case import build_pipeline
    from geometry_selection.online import flow_match_predicted_x0

    pipeline_args = SimpleNamespace(
        repo=repo,
        backbone="wan",
        method="adapted_geco",
        model=args.model,
        pipe_device=args.device,
        vae_device=None,
        metric_device=args.device,
        cross_device_grad_via_cpu=False,
        allow_split_vae=False,
        transformer_block_checkpointing=False,
    )
    pipe, vae_device = build_pipeline(pipeline_args)
    scheduler = pipe.scheduler
    scheduler.set_timesteps(50, device=args.device)
    if type(scheduler).__name__ != "UniPCMultistepScheduler":
        raise RuntimeError(
            "Frozen Wan adapted-GeCo preflight expects the official "
            f"UniPCMultistepScheduler, got {type(scheduler).__name__}."
        )
    if (
        scheduler.config.prediction_type != "flow_prediction"
        or not scheduler.config.predict_x0
        or scheduler.config.thresholding
    ):
        raise RuntimeError(
            "Frozen Wan scheduler must use unthresholded predict_x0 with flow_prediction; got "
            f"predict_x0={scheduler.config.predict_x0}, "
            f"prediction_type={scheduler.config.prediction_type!r}, "
            f"thresholding={scheduler.config.thresholding}."
        )
    if len(scheduler.sigmas) < 50:
        raise RuntimeError(f"Scheduler has only {len(scheduler.sigmas)} sigmas for 50 inference steps.")
    config = canonical_scheduler_config(dict(scheduler.config))
    sample = torch.tensor([1.25], device=args.device)
    model_output = torch.tensor([0.5], device=args.device)
    helper_errors = []
    scheduler_errors = []
    for step_index in range(50):
        # _init_step_index is also what scheduler.step() uses on its first call.
        scheduler._step_index = None
        scheduler._begin_index = None
        scheduler._init_step_index(scheduler.timesteps[step_index])
        if scheduler.step_index != step_index:
            raise RuntimeError(
                f"Scheduler failed to initialize at step {step_index}: {scheduler.step_index}"
            )
        sigma = scheduler.sigmas[step_index].to(args.device, dtype=torch.float32)
        x0 = flow_match_predicted_x0(scheduler, model_output, sample, step_index=step_index)
        expected_x0 = sample - sigma * model_output
        scheduler_x0 = scheduler.convert_model_output(model_output, sample=sample)
        helper_errors.append(float((x0 - expected_x0).abs().max().item()))
        scheduler_errors.append(float((x0 - scheduler_x0).abs().max().item()))
        if not torch.allclose(x0, scheduler_x0, atol=1e-6, rtol=1e-6):
            raise RuntimeError(
                f"flow_match_predicted_x0 disagrees with the actual scheduler at step {step_index}."
            )
    report = {
        "schema": "wan_scheduler_runtime_preflight_v1",
        "repo": str(repo),
        "code_commit": code_commit,
        "model": str(Path(args.model).resolve()),
        "flow_match_helper_source_sha256": file_sha256(repo / "geometry_selection/online.py"),
        "runtime": cuda_runtime_identity(args.device),
        "scheduler_class": type(scheduler).__name__,
        "scheduler_module": type(scheduler).__module__,
        "diffusers_version": diffusers.__version__,
        "scheduler_config": config,
        "scheduler_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str, separators=(",", ":")).encode()
        ).hexdigest(),
        "sigmas_length": int(len(scheduler.sigmas)),
        "timesteps_length": int(len(scheduler.timesteps)),
        "timesteps_sha256": tensor_sha256(scheduler.timesteps),
        "sigmas_sha256": tensor_sha256(scheduler.sigmas),
        "validated_step_indices": list(range(50)),
        "flow_x0_max_abs_error": max(helper_errors),
        "scheduler_x0_max_abs_error": max(scheduler_errors),
        "vae_device": str(vae_device),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise RuntimeError(f"Refusing to overwrite existing scheduler receipt: {args.output}")
        temporary = args.output.parent / f".{args.output.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(rendered + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            # link() is atomic and refuses an existing destination, unlike replace().
            os.link(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)
        print(json.dumps({"receipt": str(args.output), "sha256": hashlib.sha256((rendered + "\n").encode()).hexdigest()}))
    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
