"""Controlled Wan x0 frame-MSE experiments on the audited RGB-GeCo pipeline.

This runner deliberately leaves the existing ``run_wan_geco_case_full.py`` arm
untouched.  It creates two paired arms from one frozen manifest:

* ``fg_only``: controlled sparse first/middle/last x0 frame-MSE.
* ``fg_geco``: the same x0 frame-MSE plus the current RGB GeCo residual loss.

Both arms run the same Wan sampler, seed, prompt, anchors, resolution, FPS, step
schedule, and latent-update schedule.  The only intended mathematical difference
is whether the RGB GeCo term is added to the frame-MSE objective.  These are
not claimed to be faithful reproductions of official Frame Guidance/VLO.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "guidance_wan"))

from frame_guidance_manifest import (
    FrameGuidanceManifestError,
    build_frame_guidance_time_contract,
    canonical_json_hash,
    load_frame_guidance_case,
)
from pipeline_wan_i2v_full_guided import WanImageToVideoPipeline


def remap_path(path: str) -> str:
    value = str(path)
    value = value.replace(
        "/data2/yz10325/experiments_videogpa_pilot",
        "/vol/dissolve/yz10325/experiments/experiments_videogpa_pilot",
    )
    return value.replace(
        "/data2/yz10325/experiments_videogpa",
        "/vol/dissolve/yz10325/experiments/experiments_videogpa",
    )


def parse_schedule(value: str, num_steps: int, parser: argparse.ArgumentParser) -> list[int]:
    schedule = [0] * num_steps
    seen_steps: set[int] = set()
    for entry in value.split(","):
        try:
            step_text, repeats_text = entry.strip().split(":", 1)
            step, repeats = int(step_text), int(repeats_text)
        except ValueError:
            parser.error("--guidance_schedule entries must use step:repeats, e.g. '25:1,26:1'.")
        if step < 0 or step >= num_steps:
            parser.error(f"Guidance step {step} must be in [0, {num_steps - 1}].")
        if repeats <= 0:
            parser.error(f"Guidance repeats for step {step} must be positive.")
        if step in seen_steps:
            parser.error(f"Guidance schedule repeats step {step}.")
        seen_steps.add(step)
        schedule[step] = repeats
    return schedule


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cuda_peak_stats(devices: list[str | None]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    if not torch.cuda.is_available():
        return stats
    for requested in sorted({str(device) for device in devices if device and str(device).startswith("cuda")}):
        device = torch.device(requested)
        stats[requested] = {
            "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
        }
    return stats


def reset_cuda_peak_stats(devices: list[str | None]) -> None:
    if not torch.cuda.is_available():
        return
    for requested in {str(device) for device in devices if device and str(device).startswith("cuda")}:
        torch.cuda.reset_peak_memory_stats(torch.device(requested))


def write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_anchor_images(case: dict[str, Any]) -> dict[int, Image.Image]:
    targets: dict[int, Image.Image] = {}
    for anchor in case["anchors"]:
        index = anchor["frame_index"]
        with Image.open(anchor["image_path"]) as image:
            targets[index] = image.convert("RGB").copy()
    return targets


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", required=True)
parser.add_argument("--manifest", required=True, help="JSON manifest containing prompt, condition, and first/middle/last anchors.")
parser.add_argument("--output_root", required=True)
parser.add_argument("--mode", choices=["fg_only", "fg_geco"], required=True)
parser.add_argument(
    "--pair_id",
    required=True,
    help="Shared immutable identifier used for both fg_only and fg_geco runs in a paired comparison.",
)
parser.add_argument("--model", default="/vol/dissolve/yz10325/checkpoints/Wan2.2-TI2V-5B-Diffusers")
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--frames", type=int, default=121)
parser.add_argument("--height", type=int, default=704)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--guidance_scale", type=float, default=5.0)
parser.add_argument(
    "--guidance_schedule",
    required=True,
    help=(
        "Comma-separated step:repeats schedule. It is intentionally explicit because the "
        "appropriate Wan window must be chosen from x0-prediction diagnostics, e.g. '25:1,26:1,27:1'."
    ),
)
parser.add_argument("--guidance_lr", type=float, required=True)
parser.add_argument("--frame_loss_weight", type=float, default=1.0)
parser.add_argument("--geco_loss_weight", type=float, default=1.0)
parser.add_argument("--ufm_scale", type=float, default=0.125)
parser.add_argument("--metric_device", default="cuda")
parser.add_argument("--pipe_device", default="cuda:0")
parser.add_argument("--vae_device", default=None)
parser.add_argument("--cross_device_grad_via_cpu", action="store_true")
parser.add_argument("--transformer_block_checkpointing", action="store_true")
parser.add_argument("--max_relative_delta", type=float, default=0.0)
parser.add_argument("--debug_x0_interval", type=int, default=0)
parser.add_argument("--debug_x0_dir", default=None)
parser.add_argument("--debug_guidance_consistency", action="store_true")
args = parser.parse_args()

if args.steps <= 0 or args.frames <= 1:
    parser.error("--steps must be positive and --frames must be greater than one.")
if args.guidance_lr <= 0:
    parser.error("--guidance_lr must be positive.")
if args.frame_loss_weight <= 0:
    parser.error("--frame_loss_weight must be positive.")
if args.mode == "fg_geco" and args.geco_loss_weight <= 0:
    parser.error("--geco_loss_weight must be positive for fg_geco.")

case = load_frame_guidance_case(args.manifest, args.case, path_mapper=remap_path)
try:
    anchor_time_contract = build_frame_guidance_time_contract(case, args.fps)
except FrameGuidanceManifestError as error:
    parser.error(str(error))
anchor_indices = [anchor["frame_index"] for anchor in case["anchors"]]
if max(anchor_indices) >= args.frames:
    parser.error(
        f"Anchor frame {max(anchor_indices)} is outside --frames={args.frames}. "
        "Use a manifest clip with matching generated-frame indices."
    )
if anchor_indices[0] != 0:
    parser.error("Frame Guidance requires a frame-0 condition anchor.")
guidance_step = parse_schedule(args.guidance_schedule, args.steps, parser)
guidance_lr = [args.guidance_lr if repeats else 0.0 for repeats in guidance_step]

anchor_images = load_anchor_images(case)
condition_image = anchor_images[0]

uses_geco = args.mode == "fg_geco"
loss_fn = "frame_residual_motion" if uses_geco else "frame"
guidance_variant = (
    "controlled_wan_x0_frame_mse_rgb_geco_flow_matching_variant"
    if uses_geco
    else "controlled_wan_x0_frame_mse_variant"
)
guidance_diagnostics: list[dict[str, Any]] = []
additional_inputs: dict[str, Any] = {
    "frame_guidance_targets": anchor_images,
    "frame_loss_weight": args.frame_loss_weight,
    "geco_loss_weight": args.geco_loss_weight,
    "decode_spatial_scale": 1.0,
    "max_relative_delta": args.max_relative_delta,
    "cross_device_grad_via_cpu": args.cross_device_grad_via_cpu,
    "debug_guidance_consistency": args.debug_guidance_consistency,
    "guidance_diagnostics": guidance_diagnostics,
}

if uses_geco:
    from demo_guidance_fast_failure import _get_compute_dtype_for_vggt, make_motion_metric
    from uniflowmatch.models.ufm import UniFlowMatchConfidence
    from vggt.models.vggt import VGGT

    metric_device = torch.device(args.metric_device)
    compute_dtype = _get_compute_dtype_for_vggt(metric_device)
    print("loading VGGT/UFM...")
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(metric_device).eval()
    ufm_model = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base").to(
        dtype=torch.float32, device=metric_device
    ).eval()
    for parameter in vggt_model.parameters():
        parameter.requires_grad_(False)
    for parameter in ufm_model.parameters():
        parameter.requires_grad_(False)

    residual_motion_metric = make_motion_metric(
        vggt_model,
        ufm_model,
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

    def residual_motion_metric_cross_gpu(frames_01: torch.Tensor) -> torch.Tensor:
        if (
            args.cross_device_grad_via_cpu
            and torch.is_grad_enabled()
            and frames_01.requires_grad
            and frames_01.device != metric_device
        ):
            frames_01 = frames_01.to("cpu").to(metric_device)
        else:
            frames_01 = frames_01.to(metric_device)
        return residual_motion_metric(frames_01)

    additional_inputs["residual_motion_metric"] = residual_motion_metric_cross_gpu

out_dir = Path(args.output_root).resolve() / args.case
out_dir.mkdir(parents=True, exist_ok=True)
anchor_provenance = [
    {
        "frame_index": anchor["frame_index"],
        "sha256": anchor["sha256"],
        "source_frame_index": anchor["source_frame_index"],
        "source_timestamp_seconds": anchor["source_timestamp_seconds"],
    }
    for anchor in case["anchors"]
]
pair_invariants = {
    "pair_id": args.pair_id,
    "case": args.case,
    "model": args.model,
    "prompt": case["text_prompt"],
    "condition_image_sha256": case["condition_image_sha256"],
    "anchors": anchor_provenance,
    "anchor_time_contract": anchor_time_contract,
    "steps": args.steps,
    "frames": args.frames,
    "height": args.height,
    "width": args.width,
    "fps": args.fps,
    "seed": args.seed,
    "guidance_scale": args.guidance_scale,
    "guidance_schedule": guidance_step,
    "guidance_lr": guidance_lr,
    "frame_loss_weight": args.frame_loss_weight,
    "decode_spatial_scale": 1.0,
    "max_relative_delta": args.max_relative_delta,
    "transformer_jacobian": "full",
    "selected_frame_temporal_slice": "wan_causal_predecessor_target_pair_v1",
    "scheduler": "FlowMatchEulerDiscreteScheduler",
    "time_travel_renoising": False,
    "transformer_block_checkpointing": args.transformer_block_checkpointing,
    "cross_device_grad_via_cpu": args.cross_device_grad_via_cpu,
}
pair_config_hash = canonical_json_hash(pair_invariants)
config_for_hash = {
    "case": args.case,
    "mode": args.mode,
    "pair_id": args.pair_id,
    "pair_config_hash": pair_config_hash,
    "guidance_variant": guidance_variant,
    "rgb_geco_component_variant": "rgb_geco_flow_matching_variant" if uses_geco else None,
    "loss_fn": loss_fn,
    "model": args.model,
    "prompt": case["text_prompt"],
    "anchors": anchor_provenance,
    "anchor_time_contract": anchor_time_contract,
    "steps": args.steps,
    "frames": args.frames,
    "height": args.height,
    "width": args.width,
    "fps": args.fps,
    "seed": args.seed,
    "guidance_scale": args.guidance_scale,
    "guidance_schedule": guidance_step,
    "guidance_lr": guidance_lr,
    "loss_weights": {
        "frame_loss_weight": args.frame_loss_weight,
        "geco_loss_weight": args.geco_loss_weight if uses_geco else None,
        "weights_normalized": False,
    },
    "ufm_scale": args.ufm_scale if uses_geco else None,
    "decode_spatial_scale": 1.0,
    "max_relative_delta": args.max_relative_delta,
    "transformer_jacobian": "full",
    "vae_decode_spatial_scale": 1.0,
    "selected_frame_temporal_slice": "wan_causal_predecessor_target_pair_v1",
    "selected_frame_full_decode_parity": "not_assumed; optional probe required",
    "scheduler": "FlowMatchEulerDiscreteScheduler",
    "time_travel_renoising": False,
    "transformer_block_checkpointing": args.transformer_block_checkpointing,
    "cross_device_grad_via_cpu": args.cross_device_grad_via_cpu,
}
config_hash = canonical_json_hash(config_for_hash)
run_stem = f"{args.mode}_seed{args.seed}_steps{args.steps}_frames{args.frames}_cfg{config_hash[:12]}"
out_path = out_dir / f"{run_stem}.mp4"
run_manifest_path = out_dir / f"{run_stem}.json"

run_record: dict[str, Any] = {
    "schema_version": 1,
    "status": "running",
    "created_unix_seconds": time.time(),
    "runner_git_commit": git_commit(),
    "runner_path": str(Path(__file__).resolve()),
    "output_path": str(out_path),
    "config_hash": config_hash,
    "config": config_for_hash,
    "manifest": case,
    "paired_run_provenance": {
        "pair_id": args.pair_id,
        "pair_config_hash": pair_config_hash,
        "expected_modes": ["fg_only", "fg_geco"],
        "anchor_time_contract": anchor_time_contract,
    },
    "devices": {
        "pipe_device": args.pipe_device,
        "vae_device": args.vae_device or args.pipe_device,
        "metric_device": args.metric_device if uses_geco else None,
    },
    "implementation_notes": [
        "This is a controlled Wan x0 frame-MSE variant, not a faithful Frame Guidance/VLO reproduction.",
        "Frame MSE is computed on decoded x0 predictions at nonzero anchor indices.",
        "Frame zero is the Wan I2V condition and is recorded but does not receive a latent gradient.",
        "The audited FlowMatch scheduler, x0 conversion, and full RGB GeCo residual path are unchanged.",
        "The RGB-GeCo arm is a controlled flow-matching variant, not a faithful original-GeCo reproduction: it omits time-travel/re-noising and uses a causal-VAE selected-frame approximation.",
        "Raw frame and GeCo losses use tunable multipliers; 1.0/1.0 is not normalization.",
    ],
}
write_json(run_manifest_path, run_record)

print("case:", args.case)
print("mode:", args.mode)
print("pair id:", args.pair_id)
print("pair config hash:", pair_config_hash)
print("condition image:", case["condition_image_path"])
print("anchor indices:", anchor_indices)
print("out:", out_path)
print("run manifest:", run_manifest_path)
print("guidance_step:", guidance_step)
print("prompt tail:", case["text_prompt"][-300:])

vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(args.model, vae=vae, torch_dtype=torch.bfloat16).to(args.pipe_device)
if args.transformer_block_checkpointing:
    pipe.transformer.enable_gradient_checkpointing()
if args.vae_device is not None:
    requested_vae_device = torch.device(args.vae_device)
    current_vae_device = next(pipe.vae.parameters()).device
    if requested_vae_device != current_vae_device:
        pipe.vae.to("cpu")
        if current_vae_device.type == "cuda":
            with torch.cuda.device(current_vae_device):
                torch.cuda.empty_cache()
        pipe.vae.to(requested_vae_device)
    pipe._geco_vae_device = requested_vae_device
else:
    pipe._geco_vae_device = torch.device(args.pipe_device)
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

if args.debug_x0_interval > 0:
    debug_x0_dir = args.debug_x0_dir or str(out_dir / f"debug_x0_{run_stem}")
    additional_inputs.update(
        {
            "debug_x0_interval": args.debug_x0_interval,
            "debug_x0_dir": debug_x0_dir,
            "debug_x0_frames": anchor_indices,
            "debug_x0_decode_spatial_scale": 1.0,
        }
    )

generator = torch.Generator(device=args.pipe_device).manual_seed(args.seed)
devices = [args.pipe_device, args.vae_device or args.pipe_device, args.metric_device if uses_geco else None]
reset_cuda_peak_stats(devices)
start_time = time.perf_counter()
try:
    output = pipe(
        prompt=case["text_prompt"],
        image=condition_image,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        fixed_frames=anchor_indices,
        guidance_step=guidance_step,
        guidance_lr=guidance_lr,
        loss_fn=loss_fn,
        additional_inputs=additional_inputs,
    )
    export_to_video(output.frames[0], str(out_path), fps=args.fps)
    run_record["status"] = "completed"
except BaseException as error:
    run_record["status"] = "failed"
    run_record["error"] = repr(error)
    run_record["traceback"] = traceback.format_exc()
    raise
finally:
    run_record["runtime_seconds"] = time.perf_counter() - start_time
    run_record["cuda_peak_bytes"] = cuda_peak_stats(devices)
    run_record["guidance_loss_diagnostics"] = guidance_diagnostics
    write_json(run_manifest_path, run_record)

print("saved:", out_path)
