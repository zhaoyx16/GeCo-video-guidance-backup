#!/usr/bin/env python3
"""Verify per-tile VAE checkpointing against Diffusers' native tiled decode."""

import argparse
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan


REPO = Path("/vol/dissolve/yz10325/repos/GeCo")
sys.path.insert(0, str(REPO / "external" / "guidance_cosmos"))
from pipeline_cosmos2_5_predict_guided import _geco_tiled_decode_with_per_tile_checkpoint as cosmos_decode

sys.path.insert(0, str(REPO / "external" / "guidance_wan"))
from pipeline_wan_i2v_full_guided import _geco_tiled_decode_with_per_tile_checkpoint as wan_decode


def compare(name, decode_fn, vae, z_base):
    z_direct = z_base.detach().clone().requires_grad_(True)
    native = vae.decode(z_direct, return_dict=False)[0]
    native_loss = native.float().square().mean()
    native_grad = torch.autograd.grad(native_loss, z_direct)[0]

    vae.clear_cache()
    z_checkpoint = z_base.detach().clone().requires_grad_(True)
    checkpointed = decode_fn(vae, z_checkpoint)
    checkpointed_loss = checkpointed.float().square().mean()
    checkpointed_grad = torch.autograd.grad(checkpointed_loss, z_checkpoint)[0]

    output_diff = (native - checkpointed).float().abs()
    grad_diff = (native_grad - checkpointed_grad).float().abs()
    print(
        f"{name}: output_shape={tuple(native.shape)} "
        f"output_mean_abs={output_diff.mean().item():.8e} output_max_abs={output_diff.max().item():.8e} "
        f"grad_mean_abs={grad_diff.mean().item():.8e} grad_max_abs={grad_diff.max().item():.8e}"
    )
    torch.testing.assert_close(native, checkpointed, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(native_grad, checkpointed_grad, rtol=5e-4, atol=5e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--latent_frames", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
    vae.requires_grad_(False)
    vae.enable_tiling()
    latent_h = args.height // vae.spatial_compression_ratio
    latent_w = args.width // vae.spatial_compression_ratio
    torch.manual_seed(0)
    z = torch.randn(
        1,
        vae.config.z_dim,
        args.latent_frames,
        latent_h,
        latent_w,
        device=device,
        dtype=vae.dtype,
    )
    print(
        f"model={args.model} device={device} z={tuple(z.shape)} "
        f"tiling={vae.use_tiling} tile={vae.tile_sample_min_height}x{vae.tile_sample_min_width}"
    )
    compare("cosmos_helper", cosmos_decode, vae, z)
    compare("wan_helper", wan_decode, vae, z)
    print("PASS: per-tile checkpoint matches the native tiled VAE decode and latent gradient")


if __name__ == "__main__":
    main()
