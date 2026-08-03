#!/usr/bin/env python3
"""Generate one paired DL3DV benchmark sample with Wan or Cosmos.

Baseline runs use the official Diffusers pipeline. GeCo runs use the isolated
custom port after it has passed the runtime-equivalence gate. Every output has
a sidecar JSON containing enough provenance to reject stale or mismatched
paired comparisons.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib.util
import json
import os
import platform
import socket
import subprocess
import sys
import uuid
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_selection.protocol import (
    file_sha256 as protocol_file_sha256,
    resolve_protocol_image,
    resolve_protocol_transforms,
    validate_committed_file,
    validate_experiment_lock,
    validate_formal_protocol,
)
from geometry_selection.model_lock import load_model_lock, verify_generation_model
from geometry_selection.selection import validate_candidate_spec
from geometry_selection.video_probe import probe_video


WAN_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

COSMOS_NEGATIVE = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
    "pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color "
    "balance, washed out colors, choppy sequences, jerky movements, low frame rate, "
    "artifacting, color banding, unnatural transitions, outdated special effects, fake "
    "elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and "
    "flickering. Overall, the video is of poor quality."
)

PROFILES = {
    "wan": {
        "steps": 50,
        "frames": 121,
        "height": 704,
        "width": 1280,
        "fps": 24,
        "guidance_scale": 5.0,
    },
    "cosmos": {
        "steps": 36,
        "frames": 93,
        "height": 704,
        "width": 1280,
        "fps": 16,
        "guidance_scale": 7.0,
    },
}


def load_class(path: Path, class_name: str):
    spec = importlib.util.spec_from_file_location(f"_geco_benchmark_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_commit(model: str) -> str | None:
    parts = Path(model).resolve().parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    return parts[index + 1] if index + 1 < len(parts) else None


def model_identity(model: str) -> dict:
    path = Path(model).expanduser()
    identity = {
        "requested": model,
        "resolved": str(path.resolve()) if path.exists() else model,
        "snapshot_commit": snapshot_commit(model) if path.exists() else None,
    }
    if path.is_dir():
        records = []
        for candidate in sorted(path.rglob("*.json")):
            records.append(
                {
                    "path": str(candidate.relative_to(path)),
                    "sha256": sha256_file(candidate),
                }
            )
        identity["json_files"] = records
        identity["weight_files"] = [
            {
                "path": str(candidate.relative_to(path)),
                "size": candidate.stat().st_size,
            }
            for pattern in ("*.safetensors", "*.bin")
            for candidate in sorted(path.rglob(pattern))
        ]
    return identity


def mapped_fixed_frames(num_frames: int) -> list[int]:
    # Original GeCo uses 1-based 12,24,36,48 over 49 CogVideoX frames.
    fractions = (11 / 48, 23 / 48, 35 / 48, 47 / 48)
    return sorted({min(num_frames - 1, round(fraction * (num_frames - 1))) for fraction in fractions})


def adapted_geco_schedule(
    num_steps: int,
    start_fraction: float,
    end_fraction: float,
    repeats_per_step: int,
    learning_rate: float,
) -> tuple[list[int], list[float]]:
    """Create a one-update default schedule for the stop-gradient VGGT surrogate."""

    if not 0.0 <= start_fraction < end_fraction <= 1.0:
        raise ValueError("Guidance fractions must satisfy 0 <= start < end <= 1")
    if repeats_per_step < 1:
        raise ValueError("repeats_per_step must be positive")
    start = min(num_steps - 1, int(round(start_fraction * num_steps)))
    end = min(num_steps, max(start + 1, int(round(end_fraction * num_steps))))
    repeats = [0] * num_steps
    for index in range(start, end):
        repeats[index] = repeats_per_step
    return repeats, [learning_rate if repeat else 0.0 for repeat in repeats]


def load_case(
    manifest_path: Path,
    case_id: str | None,
    case_index: int | None,
) -> tuple[str, dict, dict]:
    payload = json.loads(manifest_path.read_text())
    metadata = payload.get("_meta", {})
    cases = [(key, value) for key, value in payload.items() if not key.startswith("_")]
    if case_id is not None:
        if case_id not in payload or case_id.startswith("_"):
            raise KeyError(f"Unknown case id: {case_id}")
        return case_id, payload[case_id], metadata
    if case_index is None or not 0 <= case_index < len(cases):
        raise IndexError(f"case_index must be in [0, {len(cases) - 1}]")
    selected_id, selected_case = cases[case_index]
    return selected_id, selected_case, metadata


def git_identity(repo: Path) -> dict:
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def load_metric(args: argparse.Namespace):
    from demo_guidance_fast_failure import _get_compute_dtype_for_vggt, make_motion_metric
    from uniflowmatch.models.ufm import UniFlowMatchConfidence
    from vggt.models.vggt import VGGT

    metric_device = torch.device(args.metric_device)
    compute_dtype = _get_compute_dtype_for_vggt(metric_device)
    vggt = VGGT.from_pretrained(args.vggt_model).to(metric_device).eval()
    ufm = (
        UniFlowMatchConfidence.from_pretrained(args.ufm_model)
        .to(dtype=torch.float32, device=metric_device)
        .eval()
    )
    vggt.requires_grad_(False)
    ufm.requires_grad_(False)
    metric = make_motion_metric(
        vggt,
        ufm,
        metric_device,
        compute_dtype,
        vggt_strategy="once",
        pair_mode="adjacent",
        ufm_scale=args.ufm_scale,
        cov_thresh=0.5,
        percentile_val=20,
        min_threshold=0.2,
        grad_through_vggt=False,
        debug_autograd=False,
    )

    def metric_on_device(frames_01):
        if (
            args.cross_device_grad_via_cpu
            and torch.is_grad_enabled()
            and frames_01.requires_grad
            and frames_01.device != metric_device
        ):
            frames_01 = frames_01.to("cpu").to(metric_device)
        else:
            frames_01 = frames_01.to(metric_device)
        return metric(frames_01)

    return metric_on_device


def place_vae(
    pipe,
    vae_device: str | None,
    pipe_device: str,
    *,
    backbone: str,
    method: str,
    allow_split_vae: bool,
) -> torch.device:
    requested = torch.device(vae_device or pipe_device)
    current = next(pipe.vae.parameters()).device
    if requested != current:
        if method == "baseline":
            raise ValueError("Official baseline pipelines do not support split-VAE placement")
        if not allow_split_vae:
            raise ValueError("Split-VAE placement requires --allow-split-vae and its runtime smoke test")
        if backbone == "wan" and not hasattr(pipe, "_get_geco_vae_device"):
            raise RuntimeError("Wan custom pipeline lacks the required VAE routing API")
        pipe.vae.to("cpu")
        if current.type == "cuda":
            with torch.cuda.device(current):
                torch.cuda.empty_cache()
        pipe.vae.to(requested)
    if hasattr(pipe, "_get_geco_vae_device"):
        pipe._geco_vae_device = requested
    actual = next(pipe.vae.parameters()).device
    if actual != requested:
        raise RuntimeError(f"VAE placement failed: requested={requested}, actual={actual}")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    return requested


def build_pipeline(args: argparse.Namespace):
    repo = args.repo.resolve()
    if args.backbone == "wan":
        from diffusers import AutoencoderKLWan

        if args.method == "baseline":
            from diffusers import WanImageToVideoPipeline as Pipeline
        else:
            Pipeline = load_class(
                repo / "external/guidance_wan/pipeline_wan_i2v_full_guided.py",
                "WanImageToVideoPipeline",
            )
        vae = AutoencoderKLWan.from_pretrained(
            args.model, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = Pipeline.from_pretrained(
            args.model, vae=vae, torch_dtype=torch.bfloat16
        ).to(args.pipe_device)
    else:
        if importlib.util.find_spec("cosmos_guardrail") is None:
            raise RuntimeError(
                "Formal Cosmos video generation requires cosmos_guardrail; "
                "the latent-only equivalence stub is intentionally not used here."
            )
        if args.method == "baseline":
            from diffusers import Cosmos2_5_PredictBasePipeline as Pipeline
        else:
            Pipeline = load_class(
                repo / "external/guidance_cosmos/pipeline_cosmos2_5_predict_guided.py",
                "Cosmos2_5_PredictBasePipeline",
            )
        pipe = Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(
            args.pipe_device
        )

    if args.method == "adapted_geco" and args.transformer_block_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
    vae_device = place_vae(
        pipe,
        args.vae_device,
        args.pipe_device,
        backbone=args.backbone,
        method=args.method,
        allow_split_vae=args.allow_split_vae,
    )
    return pipe, vae_device


def runtime_meta() -> dict:
    import diffusers
    import transformers

    return {
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=PROFILES, required=True)
    parser.add_argument("--method", choices=("baseline", "adapted_geco"), required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--protocol-mode",
        choices=("frozen", "legacy-debug"),
        required=True,
        help="Formal runs must explicitly select frozen; legacy-debug artifacts cannot enter formal pools.",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--expected-split", choices=("debug", "validation", "test"))
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--model-lock", type=Path)
    parser.add_argument("--model-verification-cache", type=Path)
    parser.add_argument("--experiment-lock", type=Path)
    parser.add_argument("--candidate-spec", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case-id")
    group.add_argument("--case-index", type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vggt-model", default="facebook/VGGT-1B")
    parser.add_argument("--ufm-model", default="infinity1096/UFM-Base")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--ufm-scale", type=float, default=0.25)
    parser.add_argument("--decode-spatial-scale", type=float, default=1.0)
    parser.add_argument("--max-relative-delta", type=float, default=0.0)
    parser.add_argument("--guidance-start-fraction", type=float, default=0.40)
    parser.add_argument("--guidance-end-fraction", type=float, default=0.84)
    parser.add_argument("--guidance-repeats", type=int, default=1)
    parser.add_argument("--guidance-lr", type=float, default=0.1)
    parser.add_argument(
        "--wan-negative-prompt-mode",
        choices=("none", "frozen"),
        default="none",
        help="Use none to match the existing 70 Wan baseline videos.",
    )
    parser.add_argument("--pipe-device", default="cuda:0")
    parser.add_argument("--vae-device")
    parser.add_argument("--metric-device", default="cuda:0")
    parser.add_argument("--cross-device-grad-via-cpu", action="store_true")
    parser.add_argument("--transformer-block-checkpointing", action="store_true")
    parser.add_argument("--allow-split-vae", action="store_true")
    parser.add_argument("--split-vae-smoke-report", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for key, value in PROFILES[args.backbone].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.method == "adapted_geco" and args.decode_spatial_scale != 1.0:
        parser.error("Formal adapted-GeCo benchmark requires --decode-spatial-scale 1.0")

    is_frozen_protocol = args.protocol_mode == "frozen"
    if is_frozen_protocol and args.method == "adapted_geco":
        parser.error(
            "formal adapted-GeCo runs are disabled until guidance hyperparameters "
            "and VGGT/UFM checkpoints are included in the frozen model/method lock"
        )
    if is_frozen_protocol and args.overwrite:
        parser.error("frozen protocol forbids --overwrite of completed generations")
    if is_frozen_protocol and args.repo.resolve() != REPO_ROOT.resolve():
        raise ValueError("formal generation --repo must be the repository executing this runner")
    if is_frozen_protocol:
        if args.expected_split is None:
            parser.error("frozen protocol requires --expected-split")
        protocol = validate_formal_protocol(args.manifest)
        protocol_meta = protocol["_meta"]
        split_cases = sorted(
            (
                (key, value)
                for key, value in protocol.items()
                if not key.startswith("_") and value["split"] == args.expected_split
            ),
            key=lambda item: item[1]["split_order"],
        )
        if args.case_id is not None:
            matches = [item for item in split_cases if item[0] == args.case_id]
            if len(matches) != 1:
                parser.error(f"case {args.case_id!r} is not in split {args.expected_split}")
            case_id, case = matches[0]
        else:
            if args.case_index is None or not 0 <= args.case_index < len(split_cases):
                parser.error(
                    f"case_index must be in [0, {len(split_cases) - 1}] for "
                    f"split {args.expected_split}"
                )
            case_id, case = split_cases[args.case_index]
    else:
        case_id, case, protocol_meta = load_case(
            args.manifest, args.case_id, args.case_index
        )
    code_identity = git_identity(REPO_ROOT if is_frozen_protocol else args.repo.resolve())
    locked_model_identity = None
    model_lock_sha256 = None
    experiment_lock_sha256 = None
    implementation_sha256 = None
    candidate_spec_sha256 = None
    planned_candidate = None
    if is_frozen_protocol:
        if args.dataset_root is None:
            parser.error("frozen protocol requires --dataset-root")
        if args.expected_git_commit is None:
            parser.error("frozen protocol requires --expected-git-commit")
        if code_identity["dirty"] or code_identity["commit"] != args.expected_git_commit:
            raise RuntimeError(
                f"generation code is not the frozen clean commit: {code_identity}"
            )
        validate_committed_file(args.manifest, REPO_ROOT, code_identity["commit"])
        profile_name = {
            "wan": "Wan2.2-TI2V-5B",
            "cosmos": "Cosmos-Predict2.5-2B-post",
        }[args.backbone]
        profile = protocol_meta["generation_profiles"][profile_name]
        profile_mismatches = {
            key: (getattr(args, key), expected)
            for key, expected in profile.items()
            if getattr(args, key) != expected
        }
        if profile_mismatches:
            raise ValueError(f"generation profile differs from protocol: {profile_mismatches}")
        if args.model_lock is None:
            parser.error("frozen protocol requires --model-lock")
        if args.model_verification_cache is None:
            parser.error("frozen protocol requires --model-verification-cache")
        validate_committed_file(args.model_lock, REPO_ROOT, code_identity["commit"])
        model_lock_sha256 = protocol_file_sha256(args.model_lock)
        if model_lock_sha256 != protocol_meta["model_lock_sha256"]:
            raise ValueError("model lock digest differs from frozen protocol")
        locked_model_identity = verify_generation_model(
            load_model_lock(args.model_lock),
            profile_name,
            Path(args.model),
            verification_cache=args.model_verification_cache,
        )
        if args.experiment_lock is None:
            parser.error("frozen protocol requires --experiment-lock")
        if args.candidate_spec is None:
            parser.error("frozen protocol requires --candidate-spec")
        validate_committed_file(args.candidate_spec, REPO_ROOT, code_identity["commit"])
        candidate_spec = json.loads(args.candidate_spec.read_text(encoding="utf-8"))
        validate_candidate_spec(candidate_spec, require_candidate_videos=False)
        candidate_spec_sha256 = protocol_file_sha256(args.candidate_spec)
        if candidate_spec["split"] != args.expected_split:
            raise ValueError("candidate spec split differs from generation split")
        if candidate_spec["backbone"] != profile_name:
            raise ValueError("candidate spec backbone differs from generation profile")
        if candidate_spec["protocol_manifest_sha256"] != protocol_file_sha256(args.manifest):
            raise ValueError("candidate spec protocol digest differs from manifest")
        matching_candidates = [
            candidate
            for planned_case in candidate_spec["cases"]
            if planned_case["case_id"] == case_id
            for candidate in planned_case["candidates"]
            if candidate["seed"] == args.seed
        ]
        if len(matching_candidates) != 1:
            raise ValueError("candidate spec does not contain exactly one matching case/seed")
        planned_candidate = matching_candidates[0]
        experiment_lock = validate_experiment_lock(
            args.experiment_lock,
            REPO_ROOT,
            code_identity["commit"],
            protocol_manifest_sha256=protocol_file_sha256(args.manifest),
            model_lock_sha256=model_lock_sha256,
            split=args.expected_split,
            backbone=profile_name,
            candidate_spec_sha256=candidate_spec_sha256,
            artifact_root=args.output_root,
        )
        experiment_lock_sha256 = protocol_file_sha256(args.experiment_lock)
        implementation_sha256 = experiment_lock["implementation_sha256"]
        allowed_seeds = protocol_meta["candidate_seed_policy"]["candidate_seeds"]
        if args.seed not in allowed_seeds:
            raise ValueError(f"seed {args.seed} is outside frozen policy {allowed_seeds}")
        image_path = resolve_protocol_image(case, args.dataset_root)
        resolve_protocol_transforms(case, args.dataset_root)
    else:
        image_path = Path(case["image_prompt"])
        if not image_path.is_absolute():
            image_path = args.manifest.resolve().parent / image_path
    if not image_path.is_file():
        raise FileNotFoundError(f"Conditioning image not found: {image_path}")
    prompt = case["text_prompt"]
    image_sha256 = sha256_file(image_path)
    manifest_sha256 = sha256_file(args.manifest)
    runner_sha256 = sha256_file(Path(__file__).resolve())
    pipeline_path = (
        args.repo.resolve()
        / (
            "external/guidance_wan/pipeline_wan_i2v_full_guided.py"
            if args.backbone == "wan"
            else "external/guidance_cosmos/pipeline_cosmos2_5_predict_guided.py"
        )
    )
    pipeline_sha256 = (
        sha256_file(pipeline_path) if args.method == "adapted_geco" else None
    )
    split_vae_report = None
    requested_vae_device = args.vae_device or args.pipe_device
    if torch.device(requested_vae_device) != torch.device(args.pipe_device):
        if not args.allow_split_vae or args.split_vae_smoke_report is None:
            parser.error(
                "Split VAE requires --allow-split-vae and --split-vae-smoke-report"
            )
        split_vae_report = json.loads(args.split_vae_smoke_report.read_text())
        expected_report = {
            "passed": True,
            "backbone": args.backbone,
            "pipeline_sha256": pipeline_sha256,
            "pipe_device": args.pipe_device,
            "vae_device": requested_vae_device,
            "metric_device": args.metric_device,
            "cross_device_grad_via_cpu": args.cross_device_grad_via_cpu,
        }
        mismatches = {
            key: (split_vae_report.get(key), expected)
            for key, expected in expected_report.items()
            if split_vae_report.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(f"Split-VAE smoke report mismatch: {mismatches}")

    fixed_frames = mapped_fixed_frames(args.frames)
    guidance_step, guidance_lr = adapted_geco_schedule(
        args.steps,
        args.guidance_start_fraction,
        args.guidance_end_fraction,
        args.guidance_repeats,
        args.guidance_lr,
    )
    negative_prompt = (
        WAN_NEGATIVE
        if args.backbone == "wan" and args.wan_negative_prompt_mode == "frozen"
        else COSMOS_NEGATIVE
        if args.backbone == "cosmos"
        else None
    )
    config_for_id = {
        "manifest_sha256": manifest_sha256,
        "case_id": case_id,
        "image_sha256": image_sha256,
        "prompt": prompt,
        "backbone": args.backbone,
        "method": args.method,
        "code_identity": code_identity,
        "seed": args.seed,
        "model": model_identity(args.model),
        "locked_model_identity": locked_model_identity,
        "model_lock_sha256": model_lock_sha256,
        "model_content_verified": bool(is_frozen_protocol),
        "experiment_lock_sha256": experiment_lock_sha256,
        "implementation_sha256": implementation_sha256,
        "runner_sha256": runner_sha256,
        "pipeline_sha256": pipeline_sha256,
        "vggt_model": model_identity(args.vggt_model)
        if args.method == "adapted_geco"
        else None,
        "ufm_model": model_identity(args.ufm_model)
        if args.method == "adapted_geco"
        else None,
        "steps": args.steps,
        "frames": args.frames,
        "height": args.height,
        "width": args.width,
        "fps": args.fps,
        "guidance_scale": args.guidance_scale,
        "negative_prompt": negative_prompt,
        "fixed_frames": fixed_frames,
        "guidance_step": guidance_step if args.method == "adapted_geco" else None,
        "guidance_lr": guidance_lr if args.method == "adapted_geco" else None,
        "ufm_scale": args.ufm_scale,
        "decode_spatial_scale": args.decode_spatial_scale,
        "max_relative_delta": args.max_relative_delta,
        "transformer_block_checkpointing": args.transformer_block_checkpointing,
        "split_vae_smoke_report_sha256": (
            sha256_file(args.split_vae_smoke_report)
            if args.split_vae_smoke_report is not None
            else None
        ),
        "devices": {
            "pipe": args.pipe_device,
            "vae": args.vae_device or args.pipe_device,
            "metric": args.metric_device if args.method == "adapted_geco" else None,
            "cross_device_grad_via_cpu": args.cross_device_grad_via_cpu,
            "allow_split_vae": args.allow_split_vae,
            "split_vae_smoke_report": (
                str(args.split_vae_smoke_report.resolve())
                if args.split_vae_smoke_report is not None
                else None
            ),
        },
    }
    run_id = hashlib.sha256(
        json.dumps(config_for_id, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    output_dir = (
        args.output_root
        / args.backbone
        / args.method
        / case_id
        / f"seed_{args.seed}"
        / f"run_{run_id}"
    )
    video_path = output_dir / "video.mp4"
    if is_frozen_protocol and Path(planned_candidate["video"]).resolve() != video_path.resolve():
        raise ValueError(
            "candidate spec video path differs from the deterministic generation output: "
            f"planned={planned_candidate['video']} actual={video_path}"
        )
    metadata_path = output_dir / "metadata.json"
    complete_path = output_dir / "COMPLETE"
    if complete_path.exists() and not args.overwrite:
        stored = json.loads(metadata_path.read_text())
        if stored.get("run_id") != run_id:
            raise RuntimeError(f"COMPLETE metadata run_id mismatch: {output_dir}")
        if not video_path.is_file() or stored.get("video_sha256") != sha256_file(video_path):
            raise RuntimeError(f"COMPLETE video integrity mismatch: {output_dir}")
        probe_video(
            video_path,
            frames=args.frames,
            height=args.height,
            width=args.width,
            fps=args.fps,
        )
        print(json.dumps({"status": "already_complete", "video": str(video_path)}, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    claim_path = output_dir / "RUNNING"
    try:
        claim_fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Another job owns this run directory: {output_dir}") from error
    os.write(
        claim_fd,
        json.dumps({"hostname": socket.gethostname(), "pid": os.getpid()}).encode(),
    )
    os.close(claim_fd)

    def release_claim() -> None:
        try:
            claim_path.unlink()
        except FileNotFoundError:
            pass

    atexit.register(release_claim)

    pipe, vae_device = build_pipeline(args)
    image = Image.open(image_path).convert("RGB")
    generator = torch.Generator(device=args.pipe_device).manual_seed(args.seed)

    common = {
        "image": image,
        "prompt": prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.frames,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
    }
    if args.backbone == "wan":
        common["negative_prompt"] = negative_prompt
    else:
        common.update(video=None, negative_prompt=negative_prompt)

    if args.method == "adapted_geco":
        if (
            torch.device(args.metric_device) != torch.device(args.pipe_device)
            and not args.cross_device_grad_via_cpu
        ):
            raise ValueError(
                "Cross-device metric guidance requires --cross-device-grad-via-cpu"
            )
        metric = load_metric(args)
        additional_inputs = {
            "residual_motion_metric": metric,
            "decode_spatial_scale": args.decode_spatial_scale,
            "cross_device_grad_via_cpu": args.cross_device_grad_via_cpu,
        }
        if args.backbone == "wan":
            additional_inputs["max_relative_delta"] = args.max_relative_delta
        else:
            additional_inputs.update(
                transformer_activation_checkpointing=False,
                vae_activation_checkpointing=True,
                vae_checkpoint_mode="whole",
                debug_guidance_gradient=False,
                debug_guidance_stages=False,
                debug_guidance_dump_dir=None,
            )
        common.update(
            fixed_frames=fixed_frames,
            guidance_step=guidance_step,
            guidance_lr=guidance_lr,
            loss_fn="residual_motion",
            additional_inputs=additional_inputs,
        )

    if args.method == "baseline":
        with torch.inference_mode():
            output = pipe(**common)
    else:
        output = pipe(**common)
    temporary_video = output_dir / f".video.{uuid.uuid4().hex}.tmp.mp4"
    export_to_video(output.frames[0], str(temporary_video), fps=args.fps)
    video_probe = probe_video(
        temporary_video,
        frames=args.frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
    )
    temporary_video.replace(video_path)
    video_sha256 = sha256_file(video_path)

    metadata = {
        "case_id": case_id,
        "case": case,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256,
        "protocol": {
            "is_frozen": is_frozen_protocol,
            "mode": args.protocol_mode,
            "split": case.get("split"),
            "expected_split": args.expected_split,
            "experiment_lock": (
                str(args.experiment_lock.resolve()) if args.experiment_lock else None
            ),
            "experiment_lock_sha256": experiment_lock_sha256,
        },
        "code_identity": code_identity,
        "image_path": str(image_path.resolve()),
        "image_sha256": image_sha256,
        "prompt": prompt,
        "backbone": args.backbone,
        "method": args.method,
        "run_id": run_id,
        "run_config": config_for_id,
        "run_config_sha256": hashlib.sha256(
            json.dumps(config_for_id, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "seed": args.seed,
        "model": model_identity(args.model),
        "locked_model_identity": locked_model_identity,
        "model_lock_sha256": model_lock_sha256,
        "model_content_verified": bool(is_frozen_protocol),
        "experiment_lock_sha256": experiment_lock_sha256,
        "implementation_sha256": implementation_sha256,
        "candidate_spec_sha256": candidate_spec_sha256,
        "vggt_model": model_identity(args.vggt_model)
        if args.method == "adapted_geco"
        else None,
        "ufm_model": model_identity(args.ufm_model)
        if args.method == "adapted_geco"
        else None,
        "generation": {
            "steps": args.steps,
            "frames": args.frames,
            "height": args.height,
            "width": args.width,
            "fps": args.fps,
            "guidance_scale": args.guidance_scale,
            "negative_prompt": negative_prompt,
            "wan_negative_prompt_mode": (
                args.wan_negative_prompt_mode if args.backbone == "wan" else None
            ),
        },
        "geco": {
            "enabled": args.method == "adapted_geco",
            "fixed_frames": fixed_frames,
            "guidance_step": guidance_step if args.method == "adapted_geco" else None,
            "guidance_lr": guidance_lr if args.method == "adapted_geco" else None,
            "ufm_scale": args.ufm_scale,
            "decode_spatial_scale": args.decode_spatial_scale,
            "max_relative_delta": args.max_relative_delta,
            "grad_through_vggt": False,
            "pair_mode": "adjacent",
            "time_travel": None,
            "schedule_source": (
                "Adapted single-repeat schedule for a flow-matching backbone; "
                "VGGT geometry is re-estimated with stop-gradient at every update."
            ),
            "guidance_start_fraction": args.guidance_start_fraction,
            "guidance_end_fraction": args.guidance_end_fraction,
            "guidance_repeats": args.guidance_repeats,
            "vggt_strategy_argument": "once",
            "vggt_is_cached_across_updates": False,
        },
        "devices": {
            "pipe": args.pipe_device,
            "vae": str(vae_device),
            "metric": args.metric_device if args.method == "adapted_geco" else None,
            "cross_device_grad_via_cpu": args.cross_device_grad_via_cpu,
            "allow_split_vae": args.allow_split_vae,
            "split_vae_smoke_report": (
                str(args.split_vae_smoke_report.resolve())
                if args.split_vae_smoke_report is not None
                else None
            ),
            "split_vae_smoke_report_sha256": (
                sha256_file(args.split_vae_smoke_report)
                if args.split_vae_smoke_report is not None
                else None
            ),
        },
        "runtime": runtime_meta(),
        "video": str(video_path),
        "video_sha256": video_sha256,
        "video_probe": video_probe,
        "runner_sha256": runner_sha256,
        "pipeline_sha256": pipeline_sha256,
    }
    temporary_metadata = output_dir / f".metadata.{uuid.uuid4().hex}.tmp.json"
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary_metadata.replace(metadata_path)
    temporary_complete = output_dir / f".complete.{uuid.uuid4().hex}.tmp"
    temporary_complete.write_text(f"{run_id}\n")
    temporary_complete.replace(complete_path)
    release_claim()
    print(json.dumps({"video": str(video_path), "metadata": str(metadata_path)}, indent=2))


if __name__ == "__main__":
    main()
