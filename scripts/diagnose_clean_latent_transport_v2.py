#!/usr/bin/env python3
"""Cheap final-latent counterfactuals for draft geometry transport.

This diagnostic deliberately avoids another Wan sampling pass. It loads the
baseline final latent cache, transports only the highest-confidence anchor for
each target token, unpatchifies the residual to the real latent grid, and then
decodes raw and spatially low-pass corrections at several strengths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from PIL import Image


def parse_floats(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one alpha")
    if any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("Every alpha must be positive")
    return values


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


def top_anchor_residual(
    clean_latents: torch.Tensor,
    geometry_map: dict,
    gate_mode: str,
    min_confidence: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
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
    expected_map_prefix = (tokens_t - 1, spatial_tokens)
    if source_time.shape[:2] != expected_map_prefix:
        raise ValueError(
            f"source_time shape {tuple(source_time.shape)} does not match "
            f"{expected_map_prefix} + [memory_slots]"
        )
    if source_index.shape != source_time.shape:
        raise ValueError("source_index and source_time shapes differ")
    if geometry_map["confidence"].shape[:3] != (
        tokens_t - 1,
        token_grid[1],
        token_grid[2],
    ):
        raise ValueError("confidence shape does not match the token grid")
    if geometry_map["confidence"].shape[-1] != source_time.shape[-1]:
        raise ValueError("confidence and source maps have different slot counts")

    confidence = geometry_map["confidence"].float().reshape(
        tokens_t - 1,
        spatial_tokens,
        -1,
    )

    top_confidence, top_slot = confidence.max(dim=-1)
    top_source_time = source_time.reshape_as(confidence).gather(
        -1,
        top_slot.unsqueeze(-1),
    ).squeeze(-1)
    top_source_index = source_index.reshape_as(confidence).gather(
        -1,
        top_slot.unsqueeze(-1),
    ).squeeze(-1)
    active = top_confidence >= min_confidence
    active &= top_confidence > 0
    top_source_time = torch.where(
        active,
        top_source_time,
        torch.zeros_like(top_source_time),
    )
    top_source_index = torch.where(
        active,
        top_source_index,
        torch.zeros_like(top_source_index),
    )
    if active.any():
        if top_source_time[active].min() < 0 or top_source_time[active].max() >= tokens_t:
            raise ValueError("Active source_time index lies outside the latent sequence")
        if (
            top_source_index[active].min() < 0
            or top_source_index[active].max() >= spatial_tokens
        ):
            raise ValueError("Active source_index lies outside the spatial token grid")

    all_sources = clean_tokens.reshape(
        batch,
        tokens_t * spatial_tokens,
        transport_channels,
    )
    source_flat = top_source_time * spatial_tokens + top_source_index
    matched = all_sources[:, source_flat.reshape(-1)].reshape(
        batch,
        tokens_t - 1,
        spatial_tokens,
        transport_channels,
    )
    residual_tokens = torch.zeros_like(clean_tokens)
    residual_tokens[:, 1:] = matched - clean_tokens[:, 1:]
    residual_latent = unpatchify(
        residual_tokens,
        clean_latents.shape[1],
        token_grid,
        patch_size,
    )

    token_gate = active.float()
    if gate_mode == "confidence":
        token_gate *= top_confidence
    gate_latent = token_gate.reshape(
        1,
        1,
        tokens_t - 1,
        token_grid[1],
        token_grid[2],
    )
    gate_latent = torch.cat(
        [torch.zeros_like(gate_latent[:, :, :1]), gate_latent],
        dim=2,
    )
    gate_latent = gate_latent.repeat_interleave(patch_size[0], dim=2)
    gate_latent = gate_latent.repeat_interleave(patch_size[1], dim=3)
    gate_latent = gate_latent.repeat_interleave(patch_size[2], dim=4)
    gate_latent = gate_latent.expand(batch, -1, -1, -1, -1).contiguous()

    stats = {
        "active_token_fraction": float(active.float().mean().item()),
        "mean_active_confidence": (
            float(top_confidence[active].mean().item()) if active.any() else 0.0
        ),
        "raw_residual_mean_abs_active": float(
            (
                residual_latent.abs()
                * gate_latent.gt(0)
            ).sum().item()
            / gate_latent.gt(0).expand_as(residual_latent).sum().clamp_min(1).item()
        ),
    }
    return residual_latent, gate_latent, stats


def normalized_spatial_lowpass(
    residual: torch.Tensor,
    gate: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    if radius <= 0:
        raise ValueError("lowpass radius must be positive")
    batch, channels, frames, height, width = residual.shape
    residual_2d = residual.permute(0, 2, 1, 3, 4).reshape(
        batch * frames,
        channels,
        height,
        width,
    )
    gate_2d = gate.permute(0, 2, 1, 3, 4).reshape(
        batch * frames,
        1,
        height,
        width,
    )
    kernel_size = 2 * radius + 1
    local_sum = F.avg_pool2d(
        residual_2d * gate_2d,
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )
    local_weight = F.avg_pool2d(
        gate_2d,
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )
    lowpass = local_sum / local_weight.clamp_min(1e-8)
    return lowpass.reshape(
        batch,
        frames,
        channels,
        height,
        width,
    ).permute(0, 2, 1, 3, 4)


def decode_variant(
    vae: AutoencoderKLWan,
    normalized_latents: torch.Tensor,
    latent_frames: int,
) -> list[Image.Image]:
    normalized_latents = normalized_latents[:, :, :latent_frames].to(
        device=vae.device,
        dtype=vae.dtype,
    )
    latents_mean = torch.tensor(
        vae.config.latents_mean,
        device=vae.device,
        dtype=vae.dtype,
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(
        vae.config.latents_std,
        device=vae.device,
        dtype=vae.dtype,
    ).view(1, vae.config.z_dim, 1, 1, 1)
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
    parser.add_argument("--alphas", type=parse_floats, default=[0.025, 0.05])
    parser.add_argument(
        "--gate_mode",
        choices=("confidence", "binary_support"),
        default="confidence",
    )
    parser.add_argument("--min_confidence", type=float, default=0.0)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    cache = torch.load(
        args.clean_latent_cache,
        map_location="cpu",
        weights_only=False,
    )
    geometry_map = torch.load(
        args.geometry_map,
        map_location="cpu",
        weights_only=False,
    )
    clean_latents = cache["latents"].float()
    residual, gate, stats = top_anchor_residual(
        clean_latents,
        geometry_map,
        args.gate_mode,
        args.min_confidence,
    )
    lowpass = normalized_spatial_lowpass(
        residual,
        gate,
        args.lowpass_radius,
    )
    if args.last_video_frame < 0:
        raise ValueError("last_video_frame must be non-negative")
    latent_frames = (
        1
        if args.last_video_frame == 0
        else (args.last_video_frame - 1) // 4 + 2
    )
    if latent_frames > clean_latents.shape[2]:
        raise ValueError(
            f"last_video_frame={args.last_video_frame} requires "
            f"{latent_frames} latent frames, but the cache has "
            f"{clean_latents.shape[2]}"
        )

    variants: list[tuple[str, torch.Tensor]] = [
        ("baseline_cache", clean_latents),
    ]
    for alpha in args.alphas:
        suffix = f"{alpha:.4f}".rstrip("0").rstrip(".").replace(".", "p")
        variants.append(
            (
                f"raw_top1_a{suffix}",
                clean_latents + alpha * residual * gate,
            )
        )
        variants.append(
            (
                f"lowpass_r{args.lowpass_radius}_top1_a{suffix}",
                clean_latents + alpha * lowpass * gate,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKLWan.from_pretrained(
        args.model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to("cuda")
    vae.enable_tiling()
    vae.enable_slicing()

    outputs = []
    for label, variant in variants:
        output_path = args.output_dir / f"{label}.mp4"
        frames = decode_variant(vae, variant, latent_frames)
        export_to_video(frames, str(output_path), fps=args.fps)
        outputs.append(str(output_path))
        del frames
        torch.cuda.empty_cache()

    manifest = {
        "clean_latent_cache": str(args.clean_latent_cache),
        "geometry_map": str(args.geometry_map),
        "last_video_frame": args.last_video_frame,
        "lowpass_radius": args.lowpass_radius,
        "alphas": args.alphas,
        "gate_mode": args.gate_mode,
        "min_confidence": args.min_confidence,
        "stats": stats,
        "outputs": outputs,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
