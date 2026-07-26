'''
run_wan_geco_case.py 不做 guidance。
它只是准备参数、模型、metric，然后把这些交给 custom pipeline。
'''

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video

sys.path.insert(0, "/vol/dissolve/yz10325/repos/GeCo/external/guidance_wan")
from pipeline_wan_i2v_full_guided import WanImageToVideoPipeline

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
parser.add_argument("--guidance_repeats", type=int, default=1)
parser.add_argument(
    "--guidance_schedule",
    default="",
    help=(
        "Optional comma-separated step:repeats entries, e.g. '16:3,18:3,20:2'. "
        "Overrides --guidance_start/--guidance_end/--guidance_repeats while preserving "
        "the same full-Jacobian guidance update at each requested repeat."
    ),
)
parser.add_argument("--ufm_scale", type=float, default=0.125)
parser.add_argument("--metric_device", default="cuda")
parser.add_argument("--pipe_device", default="cuda:0")
parser.add_argument("--vae_device", default=None)
parser.add_argument(
    "--transformer_block_checkpointing",
    action="store_true",
    help="Use Diffusers' native per-block activation checkpointing for the exact guided Transformer Jacobian.",
)
parser.add_argument(
    "--cross_device_grad_via_cpu",
    action="store_true",
    help="CPU-stage differentiable VAE-to-metric transfers for verified-correct multi-GPU VJPs.",
)
parser.add_argument("--decode_spatial_scale", type=float, default=1.0)
parser.add_argument("--max_relative_delta", type=float, default=0.0, help="Optional cap on mean absolute latent update as a fraction of mean abs latent, e.g. 0.002 for 0.2%.")
parser.add_argument("--debug_x0_interval", type=int, default=0, help="If >0, save decoded x0_pred frames every N denoising steps.")
parser.add_argument("--debug_x0_dir", default=None, help="Directory for x0_pred debug PNGs. Defaults under the case output directory.")
parser.add_argument("--debug_x0_frames", default="", help="Comma-separated frame indices to save for x0_pred debug. Defaults to fixed_frames.")
parser.add_argument("--debug_x0_decode_spatial_scale", type=float, default=1.0, help="Optional latent spatial scale for x0_pred debug decode only.")
parser.add_argument(
    "--debug_guidance_consistency",
    action="store_true",
    help="Print debug-only comparisons between the sampling and guidance predictions and VAE decode paths.",
)
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
guidance_step = [0] * args.steps     #初始化 guidance_step, 默认全是 0
guidance_lr = [0.0] * args.steps     #初始化 guidance_lr，默认全是 0

#64-104  guided 模式：加载 VGGT/UFM，构造 residual_motion_metric，设置 guidance schedule
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
        # Direct GPU-to-GPU CopyBackward is invalid on this host for the guidance VJP.
        # CPU staging preserves tensor values and the checked gradient path.
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

    additional_inputs = {
        "residual_motion_metric": residual_motion_metric_cross_gpu,
        "decode_spatial_scale": args.decode_spatial_scale,
        "max_relative_delta": args.max_relative_delta,
        "cross_device_grad_via_cpu": args.cross_device_grad_via_cpu,
        "debug_guidance_consistency": args.debug_guidance_consistency,
    }

    if args.guidance_schedule.strip():
        # A non-uniform schedule changes only update count/timing; the pipeline math is unchanged.
        seen_steps = set()
        for entry in args.guidance_schedule.split(","):
            try:
                step_text, repeats_text = entry.strip().split(":", 1)
                step, repeats = int(step_text), int(repeats_text)
            except ValueError:
                parser.error("--guidance_schedule entries must use step:repeats, e.g. 16:3,18:3,20:2")
            if step < 0 or step >= args.steps:
                parser.error(f"guidance schedule step {step} must be in [0, {args.steps - 1}]")
            if repeats <= 0:
                parser.error(f"guidance schedule repeats for step {step} must be positive")
            if step in seen_steps:
                parser.error(f"guidance schedule repeats step {step}")
            seen_steps.add(step)
            guidance_step[step] = repeats
            guidance_lr[step] = args.guidance_lr
    else:
        for i in range(args.guidance_start, min(args.guidance_end, args.steps)):
            guidance_step[i] = args.guidance_repeats
            guidance_lr[i] = args.guidance_lr

#107-111 打印 fixed_frames / guidance_step / scale / cap
fixed_frames = [int(x) for x in args.fixed_frames.split(",") if x.strip()]
if args.debug_x0_frames.strip():
    debug_x0_frames = [int(x) for x in args.debug_x0_frames.split(",") if x.strip()]
else:
    debug_x0_frames = list(fixed_frames)

if args.debug_x0_interval > 0:
    if additional_inputs is None:
        additional_inputs = {}
    debug_x0_dir = args.debug_x0_dir
    if debug_x0_dir is None:
        debug_x0_dir = str(out_dir / f"debug_x0_{args.mode}_seed{args.seed}_steps{args.steps}_frames{args.frames}")
    additional_inputs.update(
        {
            "debug_x0_interval": args.debug_x0_interval,
            "debug_x0_dir": debug_x0_dir,
            "debug_x0_frames": debug_x0_frames,
            "debug_x0_decode_spatial_scale": args.debug_x0_decode_spatial_scale,
        }
    )

print("fixed_frames 0-based:", fixed_frames)
print("guidance_step:", guidance_step)
print("max_relative_delta:", args.max_relative_delta)
print("decode_spatial_scale:", args.decode_spatial_scale)
print("cross_device_grad_via_cpu:", args.cross_device_grad_via_cpu)
print("transformer_block_checkpointing:", args.transformer_block_checkpointing)
print("debug_x0_interval:", args.debug_x0_interval)
print("debug_guidance_consistency:", args.debug_guidance_consistency)
if args.debug_x0_interval > 0:
    print("debug_x0_frames:", debug_x0_frames)
    print("debug_x0_dir:", additional_inputs["debug_x0_dir"])
    print("debug_x0_decode_spatial_scale:", args.debug_x0_decode_spatial_scale)

#114-117 加载 Wan VAE + custom Wan pipeline
vae = AutoencoderKLWan.from_pretrained(model, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(model, vae=vae, torch_dtype=torch.bfloat16).to(args.pipe_device)
if args.transformer_block_checkpointing:
    # Native Diffusers per-block checkpointing preserves the same Transformer
    # forward and Jacobian while releasing each block's activations until backward.
    pipe.transformer.enable_gradient_checkpointing()
if args.vae_device is not None:
    requested_vae_device = torch.device(args.vae_device)
    current_vae_device = next(pipe.vae.parameters()).device
    if requested_vae_device != current_vae_device:
        # Direct GPU-to-GPU module moves produced invalid VAE values on this host.
        # CPU staging preserves the unchanged VAE weights and its differentiable path.
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
print("pipe_device:", args.pipe_device)
print("vae_device:", pipe._geco_vae_device)

#120-137 调用 pipe，把 fixed_frames / guidance_step / guidance_lr / loss_fn / additional_inputs 传进去
image = Image.open(image_path).convert("RGB")
generator = torch.Generator(device=args.pipe_device).manual_seed(args.seed)

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
