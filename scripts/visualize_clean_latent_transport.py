#!/usr/bin/env python3
"""Decode offline clean-latent transport targets without rerunning the VDM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from PIL import Image


def patchify(
    latent: torch.Tensor,
    token_grid: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    batch, channels = latent.shape[:2]
    tokens_t, tokens_h, tokens_w = token_grid
    patch_t, patch_h, patch_w = patch_size
    return (
        latent.float()
        .reshape(
            batch,
            channels,
            tokens_t,
            patch_t,
            tokens_h,
            patch_h,
            tokens_w,
            patch_w,
        )
        .permute(0, 2, 4, 6, 1, 3, 5, 7)
        .reshape(
            batch,
            tokens_t,
            tokens_h * tokens_w,
            channels * patch_t * patch_h * patch_w,
        )
    )


def unpatchify(
    tokens: torch.Tensor,
    channels: int,
    token_grid: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    batch = tokens.shape[0]
    tokens_t, tokens_h, tokens_w = token_grid
    patch_t, patch_h, patch_w = patch_size
    return (
        tokens.reshape(
            batch,
            tokens_t,
            tokens_h,
            tokens_w,
            channels,
            patch_t,
            patch_h,
            patch_w,
        )
        .permute(0, 4, 1, 5, 2, 6, 3, 7)
        .reshape(
            batch,
            channels,
            tokens_t * patch_t,
            tokens_h * patch_h,
            tokens_w * patch_w,
        )
    )


def build_variants(
    clean_latents: torch.Tensor,
    geometry_map: dict,
    lowpass_radius: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_grid = tuple(int(value) for value in geometry_map["metadata"]["token_grid"])
    patch_size = (
        clean_latents.shape[2] // token_grid[0],
        clean_latents.shape[3] // token_grid[1],
        clean_latents.shape[4] // token_grid[2],
    )
    clean_tokens = patchify(clean_latents, token_grid, patch_size)
    batch, tokens_t, spatial_tokens, transport_channels = clean_tokens.shape
    source_time = geometry_map["source_time"].long()
    source_index = geometry_map["source_index"].long()
    confidence = geometry_map["confidence"].float()
    memory_slots = confidence.shape[-1]
    all_sources = clean_tokens.reshape(batch, tokens_t * spatial_tokens, transport_channels)

    matched = torch.zeros_like(clean_tokens[:, 1:])
    gate = torch.zeros(tokens_t - 1, spatial_tokens, 1)
    for target_time in range(1, tokens_t):
        confidence_t = confidence[target_time - 1].reshape(spatial_tokens, memory_slots)
        confidence_sum = confidence_t.sum(dim=-1, keepdim=True)
        if not (confidence_sum > 0).any():
            continue
        source_flat = (
            source_time[target_time - 1].reshape(spatial_tokens, memory_slots)
            * spatial_tokens
            + source_index[target_time - 1].reshape(spatial_tokens, memory_slots)
        ).reshape(-1)
        source_values = all_sources[:, source_flat].reshape(
            batch,
            spatial_tokens,
            memory_slots,
            transport_channels,
        )
        weights = (confidence_t / confidence_sum.clamp_min(1e-8)).reshape(
            1,
            spatial_tokens,
            memory_slots,
            1,
        )
        matched[:, target_time - 1] = (weights * source_values).sum(dim=2)
        gate[target_time - 1] = (confidence_t.amax(dim=-1, keepdim=True) > 0).float()

    binary_gate = gate.reshape(1, tokens_t - 1, spatial_tokens, 1)
    full_target_tokens = clean_tokens.clone()
    full_target_tokens[:, 1:] = (
        clean_tokens[:, 1:] * (1.0 - binary_gate) + matched * binary_gate
    )

    tokens_h, tokens_w = token_grid[1:]
    residual = matched - clean_tokens[:, 1:]
    lowpass = torch.zeros_like(residual)
    kernel_size = 2 * lowpass_radius + 1
    for target_index in range(tokens_t - 1):
        gate_grid = gate[target_index, :, 0].reshape(1, 1, tokens_h, tokens_w)
        if not (gate_grid > 0).any():
            continue
        residual_grid = residual[:, target_index].reshape(
            batch,
            tokens_h,
            tokens_w,
            transport_channels,
        ).permute(0, 3, 1, 2)
        local_sum = F.avg_pool2d(
            residual_grid * gate_grid,
            kernel_size=kernel_size,
            stride=1,
            padding=lowpass_radius,
        )
        local_weight = F.avg_pool2d(
            gate_grid,
            kernel_size=kernel_size,
            stride=1,
            padding=lowpass_radius,
        )
        lowpass[:, target_index] = (
            local_sum / local_weight.clamp_min(1e-8)
        ).permute(0, 2, 3, 1).reshape(
            batch,
            spatial_tokens,
            transport_channels,
        )

    lowpass_tokens = clean_tokens.clone()
    lowpass_tokens[:, 1:] = clean_tokens[:, 1:] + lowpass * binary_gate
    channels = clean_latents.shape[1]
    return (
        clean_latents,
        unpatchify(full_target_tokens, channels, token_grid, patch_size),
        unpatchify(lowpass_tokens, channels, token_grid, patch_size),
    )


def decode_variant(
    vae: AutoencoderKLWan,
    normalized_latents: torch.Tensor,
    latent_frames: int,
) -> list[Image.Image]:
    normalized_latents = normalized_latents[:, :, :latent_frames].to(
        device=vae.device,
        dtype=vae.dtype,
    )
    latents_mean = (
        torch.tensor(vae.config.latents_mean, device=vae.device, dtype=vae.dtype)
        .view(1, vae.config.z_dim, 1, 1, 1)
    )
    latents_std = 1.0 / (
        torch.tensor(vae.config.latents_std, device=vae.device, dtype=vae.dtype)
        .view(1, vae.config.z_dim, 1, 1, 1)
    )
    decode_latents = normalized_latents / latents_std + latents_mean
    with torch.inference_mode():
        decoded = vae.decode(decode_latents, return_dict=False)[0]
    frames = (
        ((decoded[0].permute(1, 2, 3, 0).float() + 1.0) / 2.0)
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .cpu()
        .numpy()
    )
    return [Image.fromarray(frame) for frame in frames]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_latent_cache", required=True, type=Path)
    parser.add_argument("--geometry_map", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument(
        "--model_path",
        default="/vol/dissolve/yz10325/checkpoints/Wan2.2-TI2V-5B-Diffusers",
    )
    parser.add_argument("--last_video_frame", type=int, default=48)
    parser.add_argument("--lowpass_radius", type=int, default=2)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    cache = torch.load(args.clean_latent_cache, map_location="cpu", weights_only=False)
    geometry_map = torch.load(args.geometry_map, map_location="cpu", weights_only=False)
    clean_latents = cache["latents"].float()
    variants = build_variants(clean_latents, geometry_map, args.lowpass_radius)
    latent_frames = 1 + args.last_video_frame // 4

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKLWan.from_pretrained(
        args.model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to("cuda")
    vae.enable_tiling()
    vae.enable_slicing()

    labels = ("baseline_cache", "full_transport_a1", "lowpass_delta_a1")
    for label, variant in zip(labels, variants, strict=True):
        frames = decode_variant(vae, variant, latent_frames)
        export_to_video(frames, str(args.output_dir / f"{label}.mp4"), fps=args.fps)
        del frames
        torch.cuda.empty_cache()

    metadata = {
        "clean_latent_cache": str(args.clean_latent_cache),
        "geometry_map": str(args.geometry_map),
        "last_video_frame": args.last_video_frame,
        "lowpass_radius": args.lowpass_radius,
        "outputs": [f"{label}.mp4" for label in labels],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
