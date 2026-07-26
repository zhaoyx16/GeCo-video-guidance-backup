#!/usr/bin/env python3
"""Diagnose the full VAE-to-GeCo VJP for saved Cosmos x0 decode chunks."""

import argparse
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan


REPO = Path("/vol/dissolve/yz10325/repos/GeCo")
sys.path.insert(0, str(REPO / "external" / "guidance_cosmos"))
from pipeline_cosmos2_5_predict_guided import _geco_move_tensor, _geco_tiled_decode_with_per_tile_checkpoint

sys.path.insert(0, str(REPO))
from demo_guidance_fast_failure import _get_compute_dtype_for_vggt, make_motion_metric
from uniflowmatch.models.ufm import UniFlowMatchConfidence
from vggt.models.vggt import VGGT


def norm(tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return "None"
    return f"{tensor.detach().float().norm().item():.8e}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--chunks_dir", required=True)
    parser.add_argument(
        "--source_device",
        default=None,
        help="Optional separate source GPU to exercise the differentiable source-to-VAE bridge.",
    )
    parser.add_argument("--vae_device", default="cuda:0")
    parser.add_argument("--metric_device", default="cuda:1")
    parser.add_argument("--ufm_scale", type=float, default=0.125)
    args = parser.parse_args()

    vae_device = torch.device(args.vae_device)
    metric_device = torch.device(args.metric_device)
    source_device = torch.device(args.source_device) if args.source_device is not None else vae_device

    print("loading VAE", vae_device, flush=True)
    vae = AutoencoderKLWan.from_pretrained(
        args.model, subfolder="vae", torch_dtype=torch.bfloat16
    ).to(vae_device).eval()
    vae.requires_grad_(False)
    vae.enable_tiling()

    print("loading VGGT/UFM", metric_device, flush=True)
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(metric_device).eval()
    ufm_model = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base").to(
        dtype=torch.float32, device=metric_device
    ).eval()
    vggt_model.requires_grad_(False)
    ufm_model.requires_grad_(False)
    metric = make_motion_metric(
        vggt_model,
        ufm_model,
        metric_device,
        _get_compute_dtype_for_vggt(metric_device),
        vggt_strategy="once",
        pair_mode="adjacent",
        ufm_scale=args.ufm_scale,
        cov_thresh=0.5,
        percentile_val=20,
        min_threshold=0.2,
        grad_through_vggt=False,
        debug_autograd=False,
    )

    payloads = []
    for path in sorted(Path(args.chunks_dir).glob("*.pt")):
        payloads.append(torch.load(path, map_location="cpu", weights_only=False))
    if not payloads:
        raise FileNotFoundError(f"No .pt chunks in {args.chunks_dir}")
    payloads.sort(key=lambda item: int(item["frame_index"]))

    selected = []
    chunk_latents = []
    for payload in payloads:
        frame_index = int(payload["frame_index"])
        lo = int(payload["lo"])
        hi = int(payload["hi"])
        z = payload["decode_chunk"].to(device=source_device, dtype=vae.dtype).detach().requires_grad_(True)
        z_for_decode = _geco_move_tensor(
            z, vae_device, vae.dtype, via_cpu_for_grad=source_device != vae_device
        )
        decoded = _geco_tiled_decode_with_per_tile_checkpoint(vae, z_for_decode)
        target_frames = max(1, (hi - lo) * 4 + 1)
        if decoded.shape[2] != target_frames:
            decoded = torch.repeat_interleave(decoded, repeats=4, dim=2)[:, :, :target_frames]
        frames_01 = ((decoded.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0).clamp(0, 1)
        local_frame = min(max(frame_index - lo * 4, 0), frames_01.shape[1] - 1)
        selected.append(frames_01[:, local_frame : local_frame + 1])
        chunk_latents.append(z)
        print(
            f"frame={frame_index} source={source_device} vae={vae_device} "
            f"z_mean_abs={z.detach().float().abs().mean().item():.8e} "
            f"decoded_mean_abs={decoded.detach().float().abs().mean().item():.8e} "
            f"frames_min={frames_01.detach().float().min().item():.8e} "
            f"frames_max={frames_01.detach().float().max().item():.8e}",
            flush=True,
        )

    frames_01 = torch.cat(selected, dim=1)
    frames_metric = _geco_move_tensor(
        frames_01, metric_device, via_cpu_for_grad=True
    )
    score = metric(frames_metric)
    grads = torch.autograd.grad(
        score,
        [frames_metric, *chunk_latents],
        allow_unused=True,
        retain_graph=False,
    )
    print(
        f"score={score.item():.8e} requires_grad={score.requires_grad} "
        f"dscore_dRGB={norm(grads[0])} "
        + " ".join(f"dscore_dz{idx}={norm(grad)}" for idx, grad in enumerate(grads[1:])),
        flush=True,
    )


if __name__ == "__main__":
    main()
