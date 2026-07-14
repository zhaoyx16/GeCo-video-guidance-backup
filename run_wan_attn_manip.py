import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video

sys.path.insert(0, "/vol/dissolve/yz10325/repos/GeCo/external/guidance_wan")
from pipeline_wan_i2v_attn_manip import WanImageToVideoPipeline


def parse_layers(text: str):
    if text is None or not text.strip():
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


parser = argparse.ArgumentParser()
parser.add_argument("--prompt", required=True)
parser.add_argument("--image_path", required=True)
parser.add_argument("--output_path", required=True)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--frames", type=int, default=81)
parser.add_argument("--height", type=int, default=704)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--attn_avg_alpha", type=float, default=0.3)
parser.add_argument("--attn_avg_layers", default="10,15,20")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_path", default="/vol/dissolve/yz10325/checkpoints/Wan2.2-TI2V-5B-Diffusers")
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--guidance_scale", type=float, default=5.0)
args = parser.parse_args()

attn_avg_layers = parse_layers(args.attn_avg_layers)
out = Path(args.output_path)
out.parent.mkdir(parents=True, exist_ok=True)

print("model:", args.model_path)
print("image:", args.image_path)
print("output:", out)
print("steps/frames/size:", args.steps, args.frames, args.height, args.width)
print("attn_avg_alpha:", args.attn_avg_alpha)
print("attn_avg_layers:", attn_avg_layers)
print("seed:", args.seed)

vae = AutoencoderKLWan.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(args.model_path, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

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
    attn_avg_alpha=args.attn_avg_alpha,
    attn_avg_layers=attn_avg_layers,
)

export_to_video(output.frames[0], str(out), fps=args.fps)
print("saved:", out)
