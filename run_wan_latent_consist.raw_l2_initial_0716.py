import argparse
import sys
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
parser.add_argument("--latent_consist_lr", type=float, default=0.01)
parser.add_argument("--latent_consist_layer", type=int, default=15)
parser.add_argument("--latent_consist_start_step", type=int, default=5)
parser.add_argument("--latent_consist_end_step", type=int, default=45)
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
print("latent_consist_lr:", args.latent_consist_lr)
print("latent_consist_layer:", args.latent_consist_layer)
print("latent_consist_steps:", args.latent_consist_start_step, args.latent_consist_end_step)
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

output = pipe(
    prompt=args.prompt,
    image=image,
    height=args.height,
    width=args.width,
    num_frames=args.frames,
    num_inference_steps=args.steps,
    guidance_scale=args.guidance_scale,
    generator=generator,
    latent_consist_lr=args.latent_consist_lr,
    latent_consist_layer=args.latent_consist_layer,
    latent_consist_start_step=args.latent_consist_start_step,
    latent_consist_end_step=args.latent_consist_end_step,
)

export_to_video(output.frames[0], str(out), fps=args.fps)
print("saved:", out)

losses = getattr(pipe, "_latent_consist_losses", [])
print("latent_consist_losses:")
if not losses:
    print("  none")
else:
    for step, loss_value in losses:
        print(f"  step {step}: {loss_value:.6f}")
