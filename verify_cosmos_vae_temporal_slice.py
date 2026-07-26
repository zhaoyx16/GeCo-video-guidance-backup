import argparse

import torch
from diffusers import AutoencoderKLWan


DEFAULT_MODEL = "/vol/dissolve/yz10325/checkpoints/Cosmos-Predict2.5-2B-diffusers-base-post-trained"


def match_num_frames(video: torch.Tensor, target_num_frames: int, frames_per_latent: int) -> torch.Tensor:
    if target_num_frames <= 0 or video.shape[2] == target_num_frames:
        return video

    video = torch.repeat_interleave(video, repeats=frames_per_latent, dim=2)
    if video.shape[2] < target_num_frames:
        pad = video[:, :, -1:].repeat(1, 1, target_num_frames - video.shape[2], 1, 1)
        return torch.cat([video, pad], dim=2)
    return video[:, :, :target_num_frames]


def decode(vae: AutoencoderKLWan, latents: torch.Tensor) -> torch.Tensor:
    return vae.decode(latents, return_dict=False)[0]


def current_cosmos_frame(
    vae: AutoencoderKLWan,
    latents: torch.Tensor,
    frame_index: int,
    frames_per_latent: int,
) -> tuple[torch.Tensor, int, int, int]:
    latent_t = latents.shape[2]
    latent_id = min(max(frame_index // frames_per_latent, 0), latent_t - 1)
    lo = max(0, latent_id - 1)
    hi = min(latent_t - 1, latent_id + 1)
    decoded = decode(vae, latents[:, :, lo : hi + 1].contiguous())
    decoded = match_num_frames(decoded, max(1, (hi - lo) * frames_per_latent + 1), frames_per_latent)
    local_frame = min(max(frame_index - lo * frames_per_latent, 0), decoded.shape[2] - 1)
    return decoded[:, :, local_frame], latent_id, lo, hi


def cosmos_past_only_frame(
    vae: AutoencoderKLWan,
    latents: torch.Tensor,
    frame_index: int,
    frames_per_latent: int,
) -> tuple[torch.Tensor, int, int, int]:
    latent_t = latents.shape[2]
    if frame_index == 0:
        latent_id = 0
    else:
        latent_id = min(latent_t - 1, (frame_index - 1) // frames_per_latent + 1)
    lo = max(0, latent_id - 2)
    hi = latent_id
    decoded = decode(vae, latents[:, :, lo : hi + 1].contiguous())
    decoded = match_num_frames(decoded, max(1, (hi - lo) * frames_per_latent + 1), frames_per_latent)
    local_frame = min(max(frame_index - lo * frames_per_latent, 0), decoded.shape[2] - 1)
    return decoded[:, :, local_frame], latent_id, lo, hi


def current_wan_frame(
    vae: AutoencoderKLWan,
    latents: torch.Tensor,
    frame_index: int,
    frames_per_latent: int,
) -> tuple[torch.Tensor, int, int, int]:
    latent_t = latents.shape[2]
    if frame_index == 0:
        center_lat = 0
    else:
        center_lat = min(latent_t - 1, (frame_index - 1) // frames_per_latent + 1)
    chunk_start = max(0, center_lat - 1)
    chunk_end = min(latent_t, center_lat + 2)
    decoded = decode(vae, latents[:, :, chunk_start:chunk_end].contiguous())
    chunk_first_frame = 0 if chunk_start == 0 else 1 + (chunk_start - 1) * frames_per_latent
    local_frame = min(max(frame_index - chunk_first_frame, 0), decoded.shape[2] - 1)
    return decoded[:, :, local_frame], center_lat, chunk_start, chunk_end - 1


def error_stats(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
    diff = (reference.float() - candidate.float()).abs()
    return diff.mean().item(), diff.max().item()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare selected-frame VAE decode mappings against a full Cosmos VAE decode."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--frames", type=int, default=21)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed_frames", default="0,1,4,5,8,12,16,20")
    parser.add_argument("--enable_tiling", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.bfloat16)
    vae = vae.to(device).eval().requires_grad_(False)
    if args.enable_tiling:
        vae.enable_tiling()

    temporal_ratio = 2 ** sum(vae.temperal_downsample)
    spatial_ratio = 2 ** len(vae.temperal_downsample)
    if (args.frames - 1) % temporal_ratio != 0:
        raise ValueError(f"frames must satisfy (frames - 1) % {temporal_ratio} == 0")
    if args.height % spatial_ratio or args.width % spatial_ratio:
        raise ValueError(f"height and width must be divisible by {spatial_ratio}")

    latent_t = (args.frames - 1) // temporal_ratio + 1
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = torch.randn(
        (1, vae.config.z_dim, latent_t, args.height // spatial_ratio, args.width // spatial_ratio),
        generator=generator,
        device=device,
        dtype=vae.dtype,
    )

    requested_frames = [int(x) for x in args.fixed_frames.split(",") if x.strip()]
    requested_frames = [min(max(f, 0), args.frames - 1) for f in requested_frames]

    with torch.no_grad():
        full = match_num_frames(decode(vae, latents), args.frames, temporal_ratio)
        print(
            f"VAE output: raw temporal={decode(vae, latents).shape[2]}, matched temporal={full.shape[2]}, "
            f"latent_t={latent_t}, temporal_ratio={temporal_ratio}, tiling={args.enable_tiling}"
        )
        print(
            "frame,cosmos_symmetric_mean_abs,cosmos_symmetric_max_abs,"
            "cosmos_past_mean_abs,cosmos_past_max_abs,"
            "wan_mean_abs,wan_max_abs,cosmos_lat,wan_lat"
        )
        for frame_index in requested_frames:
            reference = full[:, :, frame_index]
            cosmos, cosmos_lat, _, _ = current_cosmos_frame(vae, latents, frame_index, temporal_ratio)
            cosmos_past, _, _, _ = cosmos_past_only_frame(vae, latents, frame_index, temporal_ratio)
            wan, wan_lat, _, _ = current_wan_frame(vae, latents, frame_index, temporal_ratio)
            cosmos_mean, cosmos_max = error_stats(reference, cosmos)
            cosmos_past_mean, cosmos_past_max = error_stats(reference, cosmos_past)
            wan_mean, wan_max = error_stats(reference, wan)
            print(
                f"{frame_index},{cosmos_mean:.8f},{cosmos_max:.8f},"
                f"{cosmos_past_mean:.8f},{cosmos_past_max:.8f},"
                f"{wan_mean:.8f},{wan_max:.8f},{cosmos_lat},{wan_lat}"
            )


if __name__ == "__main__":
    main()
