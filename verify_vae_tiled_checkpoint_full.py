#!/usr/bin/env python3
"""Compare native and checkpointed Wan VAE tiled decode at production resolution."""

import argparse
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan


REPO = Path("/vol/dissolve/yz10325/repos/GeCo")
sys.path.insert(0, str(REPO / "external" / "guidance_cosmos"))
from pipeline_cosmos2_5_predict_guided import _geco_tiled_decode_with_per_tile_checkpoint


def tensor_stats(name: str, tensor: torch.Tensor) -> None:
    values = tensor.detach().float()
    print(
        f"{name}: shape={tuple(tensor.shape)} "
        f"mean_abs={values.abs().mean().item():.8e} "
        f"max_abs={values.abs().max().item():.8e}",
        flush=True,
    )


def decode_native(vae: AutoencoderKLWan, z: torch.Tensor) -> torch.Tensor:
    return vae.decode(z, return_dict=False)[0]


def decode_checkpointed(vae: AutoencoderKLWan, z: torch.Tensor) -> torch.Tensor:
    return _geco_tiled_decode_with_per_tile_checkpoint(vae, z)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--latent_frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--latent_path",
        default=None,
        help="Optional .pt file containing a saved guidance decode_chunk.",
    )
    parser.add_argument(
        "--grad_mode",
        action="store_true",
        help="Run the forward pass with a grad-carrying latent, as guidance does.",
    )
    parser.add_argument(
        "--checkpointed_only",
        action="store_true",
        help="Skip the native branch when its full activation graph cannot fit.",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Run one VJP from a scalar decoded-video loss back to the latent.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    vae = AutoencoderKLWan.from_pretrained(
        args.model, subfolder="vae", torch_dtype=torch.bfloat16
    ).to(device).eval()
    vae.requires_grad_(False)
    vae.enable_tiling()

    if args.latent_path is not None:
        payload = torch.load(args.latent_path, map_location="cpu", weights_only=False)
        z = payload["decode_chunk"].to(device=device, dtype=vae.dtype)
        print(
            f"loaded_chunk={args.latent_path} frame_index={payload.get('frame_index')} "
            f"latent_id={payload.get('latent_id')} lo={payload.get('lo')} hi={payload.get('hi')}",
            flush=True,
        )
    else:
        latent_h = args.height // vae.spatial_compression_ratio
        latent_w = args.width // vae.spatial_compression_ratio
        torch.manual_seed(args.seed)
        z = torch.randn(
            1,
            vae.config.z_dim,
            args.latent_frames,
            latent_h,
            latent_w,
            device=device,
            dtype=vae.dtype,
        )
    if args.grad_mode:
        z.requires_grad_(True)

    print(
        f"model={args.model} device={device} z={tuple(z.shape)} tiling={vae.use_tiling} "
        f"tile={vae.tile_sample_min_height}x{vae.tile_sample_min_width} grad_mode={args.grad_mode}",
        flush=True,
    )
    tensor_stats("z", z)

    native = None
    if not args.checkpointed_only:
        native = decode_native(vae, z)
        tensor_stats("native", native)
        vae.clear_cache()
    checkpointed = decode_checkpointed(vae, z)
    tensor_stats("checkpointed", checkpointed)

    if native is not None:
        diff = (native - checkpointed).float().abs()
        print(
            f"output_diff: mean_abs={diff.mean().item():.8e} max_abs={diff.max().item():.8e}",
            flush=True,
        )

    if args.backward:
        if not z.requires_grad:
            raise ValueError("--backward requires --grad_mode")
        loss = checkpointed.float().square().mean()
        grad = torch.autograd.grad(loss, z)[0]
        print(f"vjp_loss={loss.item():.8e}", flush=True)
        tensor_stats("dL_dz", grad)


if __name__ == "__main__":
    main()
