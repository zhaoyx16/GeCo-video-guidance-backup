import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from diffusers.utils import export_to_video

sys.path.insert(0, "/vol/dissolve/yz10325/repos/GeCo/external/guidance_cosmos")
from pipeline_cosmos2_5_predict_guided import Cosmos2_5_PredictBasePipeline


def _cosmos_execution_device(self):
    try:
        return next(self.transformer.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


Cosmos2_5_PredictBasePipeline._execution_device = property(_cosmos_execution_device)


def remap_path(p):
    s = str(p)
    s = s.replace("/data2/yz10325/experiments_videogpa_pilot", "/vol/dissolve/yz10325/experiments/experiments_videogpa_pilot")
    s = s.replace("/data2/yz10325/experiments_videogpa", "/vol/dissolve/yz10325/experiments/experiments_videogpa")
    s = s.replace("/data2/yz10325/checkpoints", "/vol/dissolve/yz10325/checkpoints")
    return s


PROFILE_DEFAULTS = {
    "smoke": {
        "steps": 25,
        "frames": 21,
        "height": 256,
        "width": 448,
        "fps": 16,
        "fixed_frames": "0,4,8,12,16,20",
        "ufm_scale": 0.0625,
        "decode_spatial_scale": 1.0,
    },
    "model_default": {
        "steps": 36,
        "frames": 93,
        "height": 704,
        "width": 1280,
        "fps": 16,
        "fixed_frames": "0,24,48,72,92",
        "ufm_scale": 0.125,
        "decode_spatial_scale": 1.0,
    },
}


parser = argparse.ArgumentParser()
parser.add_argument("--case", required=True)
parser.add_argument("--prompt_json", required=True)
parser.add_argument("--output_root", required=True)
parser.add_argument("--mode", choices=["baseline", "guided"], required=True)
parser.add_argument("--profile", choices=PROFILE_DEFAULTS, default="smoke")
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--frames", type=int, default=None)
parser.add_argument("--height", type=int, default=None)
parser.add_argument("--width", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--fps", type=int, default=None)
parser.add_argument("--fixed_frames", default=None)
parser.add_argument("--guidance_start", type=int, default=8)
parser.add_argument("--guidance_end", type=int, default=12)
parser.add_argument("--guidance_lr", type=float, default=0.02)
parser.add_argument("--guidance_repeats", type=int, default=1)
parser.add_argument("--ufm_scale", type=float, default=None)
parser.add_argument("--metric_device", default="cuda")
parser.add_argument(
    "--vae_device",
    default=None,
    help="Optional device for Cosmos VAE encode/decode; use a separate CUDA device for three-GPU guidance.",
)
parser.add_argument("--decode_spatial_scale", type=float, default=None)
parser.add_argument(
    "--transformer_activation_checkpointing",
    action="store_true",
    help="Checkpoint the guided Transformer forward to reduce full-Jacobian activation memory.",
)
parser.add_argument(
    "--transformer_block_checkpointing",
    action="store_true",
    help="Use Diffusers' native per-block activation checkpointing for the exact guided Transformer Jacobian.",
)
parser.add_argument(
    "--vae_activation_checkpointing",
    action="store_true",
    help="Checkpoint differentiable VAE decode without changing the GeCo loss or decoded values.",
)
parser.add_argument(
    "--vae_checkpoint_mode",
    choices=("whole", "per_tile"),
    default="whole",
    help="Use exact native whole-VAE checkpointing (formal default) or the experimental per-tile memory fallback.",
)
parser.add_argument(
    "--cross_device_grad_via_cpu",
    action="store_true",
    help="CPU-stage differentiable VAE-to-metric transfers for verified-correct multi-GPU VJPs.",
)
parser.add_argument("--debug_x0_interval", type=int, default=0, help="Save decoded x0 predictions every N denoising steps.")
parser.add_argument("--debug_x0_dir", default=None, help="Directory for x0 debug PNGs; defaults below the case output.")
parser.add_argument("--debug_x0_frames", default="", help="Comma-separated x0 video-frame indices; defaults to fixed_frames.")
parser.add_argument(
    "--debug_guidance_gradient",
    action="store_true",
    help="Print raw and condition-masked guidance gradient norms without changing sampling.",
)
parser.add_argument(
    "--debug_guidance_stages",
    action="store_true",
    help="For a small diagnostic run, print VJP norms at x0, VAE, RGB, and latent stages.",
)
parser.add_argument(
    "--debug_guidance_dump_dir",
    default=None,
    help="Optional directory for selected VAE-input chunks in a stage-gradient diagnostic.",
)
args = parser.parse_args()
for name, value in PROFILE_DEFAULTS[args.profile].items():
    if getattr(args, name) is None:
        setattr(args, name, value)

model = "/vol/dissolve/yz10325/checkpoints/Cosmos-Predict2.5-2B-diffusers-base-post-trained"
data = json.load(open(args.prompt_json))
item = data[args.case]

image_path = remap_path(item["image_prompt"])
prompt = item["text_prompt"]

out_dir = Path(args.output_root) / args.case
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / f"{args.mode}_seed{args.seed}_steps{args.steps}_frames{args.frames}_{args.height}x{args.width}.mp4"

print("case:", args.case, flush=True)
print("mode:", args.mode, flush=True)
print("profile:", args.profile, flush=True)
print("image:", image_path, flush=True)
print("out:", out, flush=True)
print("prompt tail:", prompt[-300:], flush=True)

additional_inputs = None
loss_fn = None
guidance_step = [0] * args.steps
guidance_lr = [0.0] * args.steps
fixed_frames = [int(x) for x in args.fixed_frames.split(",") if x.strip()]
fixed_frames = [min(max(x, 0), args.frames - 1) for x in fixed_frames]

if args.mode == "guided":
    from demo_guidance_fast_failure import _get_compute_dtype_for_vggt, make_motion_metric
    from vggt.models.vggt import VGGT
    from uniflowmatch.models.ufm import UniFlowMatchConfidence

    metric_device = torch.device(args.metric_device)
    compute_dtype = _get_compute_dtype_for_vggt(metric_device)

    print("loading VGGT/UFM on", metric_device, flush=True)
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
        "transformer_activation_checkpointing": args.transformer_activation_checkpointing,
        "vae_activation_checkpointing": args.vae_activation_checkpointing,
        "vae_checkpoint_mode": args.vae_checkpoint_mode,
        "cross_device_grad_via_cpu": args.cross_device_grad_via_cpu,
        "debug_guidance_gradient": args.debug_guidance_gradient,
        "debug_guidance_stages": args.debug_guidance_stages,
        "debug_guidance_dump_dir": args.debug_guidance_dump_dir,
    }
    loss_fn = "residual_motion"
    for i in range(args.guidance_start, min(args.guidance_end, args.steps)):
        guidance_step[i] = args.guidance_repeats
        guidance_lr[i] = args.guidance_lr

if args.debug_x0_interval > 0:
    if additional_inputs is None:
        additional_inputs = {}
    if args.debug_x0_frames.strip():
        debug_x0_frames = [int(x) for x in args.debug_x0_frames.split(",") if x.strip()]
    else:
        debug_x0_frames = list(fixed_frames)
    debug_x0_dir = args.debug_x0_dir or str(
        out_dir / f"debug_x0_{args.mode}_seed{args.seed}_steps{args.steps}_frames{args.frames}"
    )
    additional_inputs.update(
        {
            "debug_x0_interval": args.debug_x0_interval,
            "debug_x0_dir": debug_x0_dir,
            "debug_x0_frames": debug_x0_frames,
        }
    )

print("fixed_frames 0-based:", fixed_frames, flush=True)
print("guidance_step:", guidance_step, flush=True)
print("decode_spatial_scale:", args.decode_spatial_scale, flush=True)
print("transformer_activation_checkpointing:", args.transformer_activation_checkpointing, flush=True)
print("vae_activation_checkpointing:", args.vae_activation_checkpointing, flush=True)
print("vae_checkpoint_mode:", args.vae_checkpoint_mode, flush=True)
print("cross_device_grad_via_cpu:", args.cross_device_grad_via_cpu, flush=True)
print("debug_guidance_gradient:", args.debug_guidance_gradient, flush=True)
print("debug_guidance_stages:", args.debug_guidance_stages, flush=True)
print("debug_guidance_dump_dir:", args.debug_guidance_dump_dir, flush=True)
print("debug_x0_interval:", args.debug_x0_interval, flush=True)
if args.debug_x0_interval > 0:
    print("debug_x0_frames:", additional_inputs["debug_x0_frames"], flush=True)
    print("debug_x0_dir:", additional_inputs["debug_x0_dir"], flush=True)
if args.mode == "guided" and args.decode_spatial_scale != 1.0:
    print("WARNING: decode_spatial_scale != 1.0 is an efficiency approximation, not full GeCo guidance.", flush=True)

pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda")
if args.transformer_block_checkpointing:
    # Native Diffusers per-block checkpointing preserves the same Transformer
    # forward and Jacobian while releasing each block's activations until backward.
    pipe.transformer.enable_gradient_checkpointing()
transformer_device = next(pipe.transformer.parameters()).device
if args.vae_device is not None:
    requested_vae_device = torch.device(args.vae_device)
    if requested_vae_device != transformer_device:
        # This host returns invalid values after a direct GPU-to-GPU module move.
        # Stage the unchanged VAE weights through CPU before placing the decoder on
        # its dedicated GPU. This changes placement only, not the model or loss.
        pipe.vae.to("cpu")
        if transformer_device.type == "cuda":
            with torch.cuda.device(transformer_device):
                torch.cuda.empty_cache()
        pipe.vae.to(requested_vae_device)
print("transformer_device:", transformer_device, flush=True)
print("vae_device:", next(pipe.vae.parameters()).device, flush=True)
print("transformer_block_checkpointing:", args.transformer_block_checkpointing, flush=True)
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

image = Image.open(image_path).convert("RGB")
generator = torch.Generator(device="cuda").manual_seed(args.seed)

output = pipe(
    image=image,
    video=None,
    prompt=prompt,
    negative_prompt="low quality, blurry, distorted geometry, flickering, object deformation",
    height=args.height,
    width=args.width,
    num_frames=args.frames,
    num_inference_steps=args.steps,
    guidance_scale=7.0,
    generator=generator,
    fixed_frames=fixed_frames,
    guidance_step=guidance_step,
    guidance_lr=guidance_lr,
    loss_fn=loss_fn,
    additional_inputs=additional_inputs,
)

export_to_video(output.frames[0], str(out), fps=args.fps)
print("saved:", out, flush=True)
