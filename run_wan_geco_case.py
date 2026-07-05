import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video

sys.path.insert(0, "/vol/dissolve/yz10325/repos/GeCo/external/guidance_wan")
from pipeline_wan_i2v_guided import WanImageToVideoPipeline

def remap_path(p):
    s = str(p)
    s = s.replace("/data2/yz10325/experiments_videogpa_pilot", "/vol/dissolve/yz10325/experiments/experiments_videogpa_pilot")
    s = s.replace("/data2/yz10325/experiments_videogpa", "/vol/dissolve/yz10325/experiments/experiments_videogpa")
    return s

parser = argparse.ArgumentParser()
parser.add_argument("--case", required=True)
parser.add_argument("--prompt_json", required=True)
parser.add_argument("--output_root", required=True)
parser.add_argument("--mode", choices=["baseline", "guided"], required=True)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--frames", type=int, default=21)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--width", type=int, default=832)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--fps", type=int, default=16)
parser.add_argument("--fixed_frames", default="0,8,16,20")
parser.add_argument("--guidance_start", type=int, default=3)
parser.add_argument("--guidance_end", type=int, default=7)
parser.add_argument("--guidance_lr", type=float, default=0.05)
parser.add_argument("--ufm_scale", type=float, default=0.125)
parser.add_argument("--metric_device", default="cuda")
args = parser.parse_args()

model = "/vol/dissolve/yz10325/checkpoints/Wan2.2-TI2V-5B-Diffusers"
data = json.load(open(args.prompt_json))
item = data[args.case]

image_path = remap_path(item["image_prompt"])
prompt = item["text_prompt"]

out_dir = Path(args.output_root) / args.case
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / f"{args.mode}_seed{args.seed}_steps{args.steps}_frames{args.frames}.mp4"

print("case:", args.case)
print("mode:", args.mode)
print("image:", image_path)
print("out:", out)
print("prompt tail:", prompt[-300:])

additional_inputs = None
loss_fn = None
guidance_step = [0] * args.steps
guidance_lr = [0.0] * args.steps

if args.mode == "guided":
    from demo_guidance_fast_failure import _get_compute_dtype_for_vggt, make_motion_metric
    from vggt.models.vggt import VGGT
    from uniflowmatch.models.ufm import UniFlowMatchConfidence

    metric_device = torch.device(args.metric_device)
    compute_dtype = _get_compute_dtype_for_vggt(metric_device)

    print("loading VGGT/UFM...")
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(metric_device).eval()
    ufm_model = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base").to(dtype=torch.float32, device=metric_device).eval()
    for p in vggt_model.parameters():
        p.requires_grad_(False)
    for p in ufm_model.parameters():
        p.requires_grad_(False)

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

    loss_fn = "residual_motion"
    def residual_motion_metric_cross_gpu(frames_01):
        return residual_motion_metric(frames_01.to(metric_device))

    additional_inputs = {"residual_motion_metric": residual_motion_metric_cross_gpu}

    for i in range(args.guidance_start, min(args.guidance_end, args.steps)):
        guidance_step[i] = 1
        guidance_lr[i] = args.guidance_lr

fixed_frames = [int(x) for x in args.fixed_frames.split(",") if x.strip()]
print("fixed_frames 0-based:", fixed_frames)
print("guidance_step:", guidance_step)

vae = AutoencoderKLWan.from_pretrained(model, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(model, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

image = Image.open(image_path).convert("RGB")
generator = torch.Generator(device="cuda").manual_seed(args.seed)

output = pipe(
    prompt=prompt,
    image=image,
    height=args.height,
    width=args.width,
    num_frames=args.frames,
    num_inference_steps=args.steps,
    guidance_scale=5.0,
    generator=generator,
    fixed_frames=fixed_frames,
    guidance_step=guidance_step,
    guidance_lr=guidance_lr,
    loss_fn=loss_fn,
    additional_inputs=additional_inputs,
)

export_to_video(output.frames[0], str(out), fps=args.fps)
print("saved:", out)
