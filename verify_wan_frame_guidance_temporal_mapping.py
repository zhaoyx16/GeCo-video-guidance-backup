"""Optional full-decode parity probe for Wan Frame Guidance temporal slices.

This is intentionally not a unit test: loading the Wan VAE and decoding a full
video requires model assets and a GPU.  It measures, rather than assumes,
agreement between the predecessor/target pair used by controlled Frame Guidance
and the corresponding frame of a full causal VAE decode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan

from external.guidance_wan.pipeline_wan_i2v_full_guided import _wan_causal_frame_decode_plan


def parse_indices(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--anchor_frames must be a comma-separated list of integers.") from error


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", default="/vol/dissolve/yz10325/checkpoints/Wan2.2-TI2V-5B-Diffusers")
parser.add_argument("--device", default="cuda")
parser.add_argument("--height", type=int, default=256)
parser.add_argument("--width", type=int, default=448)
parser.add_argument("--frames", type=int, default=121)
parser.add_argument("--anchor_frames", type=parse_indices, default=[0, 60, 120])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--require_parity", action="store_true")
parser.add_argument("--mean_abs_tolerance", type=float, default=1e-4)
parser.add_argument("--max_abs_tolerance", type=float, default=1e-3)
parser.add_argument("--output_json", default=None)
args = parser.parse_args()

device = torch.device(args.device)
if device.type == "cuda" and not torch.cuda.is_available():
    parser.error("--device requests CUDA but CUDA is unavailable.")
if args.height <= 0 or args.width <= 0 or args.frames <= 0:
    parser.error("--height, --width, and --frames must be positive.")

vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.bfloat16).to(device).eval()
vae.enable_tiling()
temporal_scale = 4
if (args.frames - 1) % temporal_scale:
    parser.error("Wan frame count must follow 4n+1 for this causal temporal probe.")
if args.height % vae.spatial_compression_ratio or args.width % vae.spatial_compression_ratio:
    parser.error("Height and width must be divisible by the Wan VAE spatial compression ratio.")

latent_frames = 1 + (args.frames - 1) // temporal_scale
generator = torch.Generator(device=device).manual_seed(args.seed)
latents = torch.randn(
    1,
    vae.config.z_dim,
    latent_frames,
    args.height // vae.spatial_compression_ratio,
    args.width // vae.spatial_compression_ratio,
    device=device,
    dtype=vae.dtype,
    generator=generator,
)

with torch.inference_mode():
    full_decode = vae.decode(latents, return_dict=False)[0]
    rows: list[dict[str, float | int | list[int]]] = []
    for frame_index in args.anchor_frames:
        start, end, local_index = _wan_causal_frame_decode_plan(
            frame_index,
            num_frames=args.frames,
            latent_frames=latent_frames,
            temporal_scale=temporal_scale,
        )
        selected_decode = vae.decode(latents[:, :, start:end], return_dict=False)[0]
        if local_index >= selected_decode.shape[2]:
            raise RuntimeError(
                f"Selected decode for frame {frame_index} has {selected_decode.shape[2]} frames, "
                f"but mapping requires local index {local_index}."
            )
        difference = (full_decode[:, :, frame_index] - selected_decode[:, :, local_index]).float().abs()
        rows.append(
            {
                "frame_index": frame_index,
                "latent_slice": [start, end],
                "local_output_index": local_index,
                "mean_abs_difference": float(difference.mean().item()),
                "max_abs_difference": float(difference.max().item()),
            }
        )

result = {
    "model": args.model,
    "device": str(device),
    "frames": args.frames,
    "height": args.height,
    "width": args.width,
    "temporal_scale": temporal_scale,
    "results": rows,
}
print(json.dumps(result, indent=2, sort_keys=True))
if args.output_json:
    Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

if args.require_parity:
    failures = [
        row
        for row in rows
        if row["mean_abs_difference"] > args.mean_abs_tolerance
        or row["max_abs_difference"] > args.max_abs_tolerance
    ]
    if failures:
        raise SystemExit(f"Selected-slice/full-decode parity exceeded tolerance: {failures}")
