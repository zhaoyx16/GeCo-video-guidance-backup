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
from pipeline_wan_i2v_c2f_rope import WanImageToVideoPipeline


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
parser.add_argument(
    "--attn_avg_mode",
    choices=(
        "global", "local", "anchor", "match_prev", "c2f_match_prev", "query_match_prev",
        "key_match_prev", "kv_match_prev", "value_residual_prev", "c2f_value_residual_prev",
        "c2f_value_residual_anchor", "c2f_value_residual_memory", "c2f_rope_memory",
    ),
    default="c2f_rope_memory",
)
parser.add_argument("--attn_avg_start", type=int, default=0)
parser.add_argument("--attn_avg_end", type=int, default=None)
parser.add_argument("--attn_avg_temporal_radius", type=int, default=1)
parser.add_argument("--attn_avg_match_radius", type=int, default=1)
parser.add_argument("--attn_avg_match_confidence", type=float, default=0.0)
parser.add_argument("--attn_avg_match_mutual", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--attn_avg_descriptor_dim", type=int, default=64)
parser.add_argument("--attn_avg_coarse_factor", type=int, default=2)
parser.add_argument("--attn_avg_memory_lookback", type=int, default=3)
parser.add_argument("--attn_avg_cond_only", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--attn_avg_preserve_first_frame", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--attn_avg_debug", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_path", default="/vol/dissolve/yz10325/checkpoints/Wan2.2-TI2V-5B-Diffusers")
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--guidance_scale", type=float, default=5.0)
parser.add_argument("--negative_prompt", default=None)
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
print("attn_avg_mode:", args.attn_avg_mode)
print("attn_avg_step_interval:", (args.attn_avg_start, args.attn_avg_end))
print("attn_avg_match_mutual:", args.attn_avg_match_mutual)
print("attn_avg_coarse_factor:", args.attn_avg_coarse_factor)
print("attn_avg_memory_lookback:", args.attn_avg_memory_lookback)
print("attn_avg_cond_only:", args.attn_avg_cond_only)
print("attn_avg_preserve_first_frame:", args.attn_avg_preserve_first_frame)
print("seed:", args.seed)

vae = AutoencoderKLWan.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(args.model_path, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

image = Image.open(args.image_path).convert("RGB")
generator = torch.Generator(device="cuda").manual_seed(args.seed)

torch.cuda.reset_peak_memory_stats()
start_time = time.perf_counter()
output = pipe(
    prompt=args.prompt,
    negative_prompt=args.negative_prompt,
    image=image,
    height=args.height,
    width=args.width,
    num_frames=args.frames,
    num_inference_steps=args.steps,
    guidance_scale=args.guidance_scale,
    generator=generator,
    attn_avg_alpha=args.attn_avg_alpha,
    attn_avg_layers=attn_avg_layers,
    attn_avg_mode=args.attn_avg_mode,
    attn_avg_start=args.attn_avg_start,
    attn_avg_end=args.attn_avg_end,
    attn_avg_temporal_radius=args.attn_avg_temporal_radius,
    attn_avg_match_radius=args.attn_avg_match_radius,
    attn_avg_match_confidence=args.attn_avg_match_confidence,
    attn_avg_match_mutual=args.attn_avg_match_mutual,
    attn_avg_descriptor_dim=args.attn_avg_descriptor_dim,
    attn_avg_coarse_factor=args.attn_avg_coarse_factor,
    attn_avg_memory_lookback=args.attn_avg_memory_lookback,
    attn_avg_cond_only=args.attn_avg_cond_only,
    attn_avg_preserve_first_frame=args.attn_avg_preserve_first_frame,
    attn_avg_debug=args.attn_avg_debug,
)
torch.cuda.synchronize()
wall_seconds = time.perf_counter() - start_time
peak_memory_mib = torch.cuda.max_memory_allocated() / 1024**2

export_to_video(output.frames[0], str(out), fps=args.fps)
metadata = vars(args).copy()
metadata["attn_avg_layers"] = attn_avg_layers
metadata["wall_seconds"] = wall_seconds
metadata["peak_memory_mib"] = peak_memory_mib
out.with_suffix(out.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
print("saved:", out)
print(f"wall_seconds: {wall_seconds:.2f}")
print(f"peak_memory_mib: {peak_memory_mib:.1f}")
