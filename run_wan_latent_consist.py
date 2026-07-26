import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video

sys.path.insert(0, "/vol/dissolve/yz10325/repos/GeCo/external/guidance_wan")
from pipeline_wan_i2v_latent_consist import WanImageToVideoPipeline


parser = argparse.ArgumentParser()
parser.add_argument("--prompt", required=True)
parser.add_argument("--image_path", required=True)
parser.add_argument("--output_path", required=True)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--frames", type=int, default=81)
parser.add_argument("--height", type=int, default=704)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument(
    "--latent_consist_mode",
    choices=(
        "off",
        "same_position_l2",
        "matched_cosine",
        "match_prev",
        "static_match_prev",
        "match_prev_input",
        "static_match_prev_input",
        "static_track_prev_input",
        "static_anchor_match_prev_input",
        "static_tracklet_match_prev_input",
        "static_match_prev_kv",
        "static_qk_match_prev_kv",
        "qk_probe",
        "noise_ar1",
    ),
    default="off",
)
parser.add_argument("--latent_consist_lr", type=float, default=0.0)
parser.add_argument("--latent_consist_layer", type=int, default=15)
parser.add_argument("--latent_consist_start_step", type=int, default=5)
parser.add_argument("--latent_consist_end_step", type=int, default=45)
parser.add_argument("--latent_consist_match_radius", type=int, default=2)
parser.add_argument("--latent_consist_match_confidence", type=float, default=0.55)
parser.add_argument("--latent_consist_static_coherence", type=float, default=1.0)
parser.add_argument("--latent_consist_qk_require_mutual", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--latent_consist_anchor_confidence", type=float, default=0.45)
parser.add_argument("--latent_consist_tracklet_horizon", type=int, default=3)
parser.add_argument("--latent_consist_tracklet_confidence", type=float, default=0.40)
parser.add_argument("--latent_consist_descriptor_dim", type=int, default=64)
parser.add_argument("--latent_consist_cond_only", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--latent_consist_noise_rho", type=float, default=0.0)
parser.add_argument("--latent_consist_max_relative_delta", type=float, default=0.0)
parser.add_argument("--latent_consist_debug", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gradient_checkpointing", action="store_true")
parser.add_argument("--model_path", default="/vol/dissolve/yz10325/checkpoints/Wan2.2-TI2V-5B-Diffusers")
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--guidance_scale", type=float, default=5.0)
args = parser.parse_args()

out = Path(args.output_path)
out.parent.mkdir(parents=True, exist_ok=True)

print("model:", args.model_path)
print("image:", args.image_path)
print("output:", out)
print("steps/frames/size:", args.steps, args.frames, args.height, args.width)
print("latent_consist_mode:", args.latent_consist_mode)
print("latent_consist_lr:", args.latent_consist_lr)
print("latent_consist_layer:", args.latent_consist_layer)
print("latent_consist_steps:", args.latent_consist_start_step, args.latent_consist_end_step)
print("latent_consist_match:", args.latent_consist_match_radius, args.latent_consist_match_confidence)
print("latent_consist_static_coherence:", args.latent_consist_static_coherence)
print("latent_consist_qk_require_mutual:", args.latent_consist_qk_require_mutual)
print("latent_consist_anchor_confidence:", args.latent_consist_anchor_confidence)
print("latent_consist_tracklet:", args.latent_consist_tracklet_horizon, args.latent_consist_tracklet_confidence)
print("latent_consist_noise_rho:", args.latent_consist_noise_rho)
print("latent_consist_max_relative_delta:", args.latent_consist_max_relative_delta)
print("gradient_checkpointing:", args.gradient_checkpointing)
print("seed:", args.seed)

vae = AutoencoderKLWan.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(args.model_path, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

if args.gradient_checkpointing:
    if pipe.transformer is not None:
        pipe.transformer.enable_gradient_checkpointing()
    if getattr(pipe, "transformer_2", None) is not None:
        pipe.transformer_2.enable_gradient_checkpointing()

image = Image.open(args.image_path).convert("RGB")
generator = torch.Generator(device="cuda").manual_seed(args.seed)

torch.cuda.reset_peak_memory_stats()
start_time = time.perf_counter()
output = pipe(
    prompt=args.prompt,
    image=image,
    height=args.height,
    width=args.width,
    num_frames=args.frames,
    num_inference_steps=args.steps,
    guidance_scale=args.guidance_scale,
    generator=generator,
    latent_consist_mode=args.latent_consist_mode,
    latent_consist_lr=args.latent_consist_lr,
    latent_consist_layer=args.latent_consist_layer,
    latent_consist_start_step=args.latent_consist_start_step,
    latent_consist_end_step=args.latent_consist_end_step,
    latent_consist_match_radius=args.latent_consist_match_radius,
    latent_consist_match_confidence=args.latent_consist_match_confidence,
    latent_consist_static_coherence=args.latent_consist_static_coherence,
    latent_consist_qk_require_mutual=args.latent_consist_qk_require_mutual,
    latent_consist_anchor_confidence=args.latent_consist_anchor_confidence,
    latent_consist_tracklet_horizon=args.latent_consist_tracklet_horizon,
    latent_consist_tracklet_confidence=args.latent_consist_tracklet_confidence,
    latent_consist_descriptor_dim=args.latent_consist_descriptor_dim,
    latent_consist_cond_only=args.latent_consist_cond_only,
    latent_consist_noise_rho=args.latent_consist_noise_rho,
    latent_consist_max_relative_delta=args.latent_consist_max_relative_delta,
    latent_consist_debug=args.latent_consist_debug,
)
torch.cuda.synchronize()
wall_seconds = time.perf_counter() - start_time
peak_memory_mib = torch.cuda.max_memory_allocated() / 1024**2

export_to_video(output.frames[0], str(out), fps=args.fps)
print("saved:", out)
print(f"wall_seconds: {wall_seconds:.2f}")
print(f"peak_memory_mib: {peak_memory_mib:.1f}")

losses = getattr(pipe, "_latent_consist_losses", [])
gradient_stats = getattr(pipe, "_latent_consist_gradient_stats", [])
diagnostics = getattr(pipe, "_latent_consist_diagnostics", [])
metadata = vars(args).copy()
metadata.update(
    {
        "wall_seconds": wall_seconds,
        "peak_memory_mib": peak_memory_mib,
        "latent_consist_losses": losses,
        "latent_consist_gradient_stats": gradient_stats,
        "latent_consist_diagnostics": diagnostics,
    }
)
out.with_suffix(out.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")

print("latent_consist_losses:")
if not losses:
    print("  none")
else:
    for step, loss_value in losses:
        print(f"  step {step}: {loss_value:.6f}")

print("latent_consist_diagnostics:")
if not diagnostics:
    print("  none")
else:
    for item in diagnostics:
        print(" ", item)
