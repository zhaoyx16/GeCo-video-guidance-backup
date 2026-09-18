import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video

sys.path.insert(0, str(Path(__file__).resolve().parent / "external" / "guidance_wan"))
from geometry_provenance import build_generation_contract
from pipeline_wan_i2v_geometry_transport import WanImageToVideoPipeline


def parse_layers(text: str):
    if text is None or not text.strip():
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


parser = argparse.ArgumentParser()
parser.add_argument("--prompt", required=True)
parser.add_argument("--image_path", required=True)
parser.add_argument("--output_path", required=True)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--frames", type=int, default=81)
parser.add_argument("--height", type=int, default=704)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--attn_avg_alpha", type=float, default=0.0)
parser.add_argument("--attn_avg_layers", default="")
parser.add_argument(
    "--attn_avg_mode",
    choices=(
        "global", "local", "anchor", "match_prev", "c2f_match_prev", "query_match_prev",
        "key_match_prev", "kv_match_prev", "value_residual_prev", "c2f_value_residual_prev",
        "c2f_value_residual_anchor", "c2f_value_residual_memory",
    ),
    default="global",
)
parser.add_argument("--attn_avg_start", type=int, default=0)
parser.add_argument("--attn_avg_end", type=int, default=None)
parser.add_argument("--attn_avg_temporal_radius", type=int, default=1)
parser.add_argument("--attn_avg_match_radius", type=int, default=1)
parser.add_argument("--attn_avg_match_confidence", type=float, default=0.0)
parser.add_argument("--attn_avg_match_mutual", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--attn_avg_descriptor_dim", type=int, default=64)
parser.add_argument("--attn_avg_coarse_factor", type=int, default=2)
parser.add_argument("--attn_avg_memory_lookback", type=int, default=3)
parser.add_argument("--attn_avg_cond_only", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--attn_avg_preserve_first_frame", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--attn_avg_debug", action="store_true")
parser.add_argument("--geometry_transport_alpha", type=float, default=0.0)
parser.add_argument("--geometry_transport_layers", default="0")
parser.add_argument(
    "--geometry_transport_mode",
    choices=(
        "shadow",
        "value_residual",
        "free_space_value_residual",
        "kv_memory",
        "kv_transport",
        "qkv_transport",
        "query_key_steer",
        "projected_logit_margin",
        "sparse_geo_memory",
        "hidden_input_transport",
        "hidden_input_suppress",
    ),
    default="hidden_input_suppress",
)
parser.add_argument("--geometry_logit_margin", type=float, default=0.0)
parser.add_argument("--geometry_map_path", default=None)
parser.add_argument(
    "--geometry_min_active_frame_coverage",
    type=float,
    default=0.0,
    help=(
        "Abstain from geometry transport when the mean selected-token "
        "coverage over active mapped frames is below this threshold."
    ),
)
parser.add_argument(
    "--geometry_min_largest_component_share",
    type=float,
    default=0.0,
    help=(
        "Abstain when active geometry masks are spatially fragmented: the "
        "mean largest-component share over active frames must reach this value."
    ),
)
parser.add_argument(
    "--geometry_max_border_share",
    type=float,
    default=1.0,
    help=(
        "Abstain when too much of the geometry mask lies near token-grid "
        "boundaries, where normal disocclusion is common."
    ),
)
parser.add_argument("--geometry_border_width", type=int, default=2)
parser.add_argument("--geometry_anchor_step", type=int, default=30)
parser.add_argument("--geometry_transport_start", type=int, default=None)
parser.add_argument("--geometry_transport_end", type=int, default=None)
parser.add_argument("--geometry_frame_indices", default=None)
parser.add_argument("--geometry_memory_lookback", type=int, default=3)
parser.add_argument("--geometry_confidence_percentile", type=float, default=20.0)
parser.add_argument("--geometry_confidence_floor", type=float, default=0.2)
parser.add_argument("--geometry_depth_relative_threshold", type=float, default=0.15)
parser.add_argument("--geometry_transport_cond_only", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--geometry_transport_debug", action="store_true")
parser.add_argument("--geometry_debug_dir", default=None)
parser.add_argument(
    "--frozen_offline_suppression",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Enforce the reviewed protocol: precomputed format-v2 map, block 0 "
        "conditional-only hidden suppression, no online geometry or feature averaging."
    ),
)
parser.add_argument(
    "--frozen_free_space_transport",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Enforce reviewed ray-aligned observed-background value transport: "
        "format-v3 source-free-space map, block 0, conditional branch only."
    ),
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_path", default="/vol/dissolve/yz10325/checkpoints/Wan2.2-TI2V-5B-Diffusers")
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--guidance_scale", type=float, default=5.0)
parser.add_argument("--negative_prompt", default=None)
args = parser.parse_args()

if args.frozen_free_space_transport:
    args.frozen_offline_suppression = False
attn_avg_layers = parse_layers(args.attn_avg_layers)
geometry_transport_layers = parse_layers(args.geometry_transport_layers)
geometry_frame_indices = parse_layers(args.geometry_frame_indices)
if args.geometry_transport_alpha == 0.0:
    geometry_transport_layers = None
out = Path(args.output_path)
out.parent.mkdir(parents=True, exist_ok=True)

print("model:", args.model_path)
print("image:", args.image_path)
print("output:", out)
print("steps/frames/size:", args.steps, args.frames, args.height, args.width)
print("attn_avg_alpha:", args.attn_avg_alpha)
print("attn_avg_layers:", attn_avg_layers)
print("attn_avg_mode:", args.attn_avg_mode)
print("attn_avg_step_interval:", (args.attn_avg_start, args.attn_avg_end))
print("attn_avg_match_mutual:", args.attn_avg_match_mutual)
print("attn_avg_coarse_factor:", args.attn_avg_coarse_factor)
print("attn_avg_memory_lookback:", args.attn_avg_memory_lookback)
print("attn_avg_cond_only:", args.attn_avg_cond_only)
print("attn_avg_preserve_first_frame:", args.attn_avg_preserve_first_frame)
print("geometry_transport_alpha:", args.geometry_transport_alpha)
print("geometry_transport_layers:", geometry_transport_layers)
print("geometry_transport_mode:", args.geometry_transport_mode)
print("geometry_logit_margin:", args.geometry_logit_margin)
print("geometry_map_path:", args.geometry_map_path)
print(
    "geometry_min_active_frame_coverage:",
    args.geometry_min_active_frame_coverage,
)
print(
    "geometry_min_largest_component_share:",
    args.geometry_min_largest_component_share,
)
print("geometry_max_border_share:", args.geometry_max_border_share)
print("geometry_border_width:", args.geometry_border_width)
print("geometry_anchor_step:", args.geometry_anchor_step)
print("geometry_transport_interval:", (args.geometry_transport_start, args.geometry_transport_end))
print("geometry_frame_indices:", geometry_frame_indices)
print("geometry_memory_lookback:", args.geometry_memory_lookback)
print("geometry_debug_dir:", args.geometry_debug_dir)
print("frozen_offline_suppression:", args.frozen_offline_suppression)
print("frozen_free_space_transport:", args.frozen_free_space_transport)
print("seed:", args.seed)

vae = AutoencoderKLWan.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(args.model_path, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

image = Image.open(args.image_path).convert("RGB")
generator = torch.Generator(device="cuda").manual_seed(args.seed)
generation_contract = build_generation_contract(
    prompt=args.prompt,
    negative_prompt=args.negative_prompt,
    seed=args.seed,
    steps=args.steps,
    frames=args.frames,
    height=args.height,
    width=args.width,
    fps=args.fps,
    guidance_scale=args.guidance_scale,
    image_path=args.image_path,
    model_path=args.model_path,
)

torch.cuda.reset_peak_memory_stats()
start_time = time.perf_counter()
output = pipe(
    prompt=args.prompt,
    negative_prompt=args.negative_prompt,
    image=image,
    height=args.height,
    width=args.width,
    num_frames=args.frames,
    num_inference_steps=args.steps,
    guidance_scale=args.guidance_scale,
    generator=generator,
    attn_avg_alpha=args.attn_avg_alpha,
    attn_avg_layers=attn_avg_layers,
    attn_avg_mode=args.attn_avg_mode,
    attn_avg_start=args.attn_avg_start,
    attn_avg_end=args.attn_avg_end,
    attn_avg_temporal_radius=args.attn_avg_temporal_radius,
    attn_avg_match_radius=args.attn_avg_match_radius,
    attn_avg_match_confidence=args.attn_avg_match_confidence,
    attn_avg_match_mutual=args.attn_avg_match_mutual,
    attn_avg_descriptor_dim=args.attn_avg_descriptor_dim,
    attn_avg_coarse_factor=args.attn_avg_coarse_factor,
    attn_avg_memory_lookback=args.attn_avg_memory_lookback,
    attn_avg_cond_only=args.attn_avg_cond_only,
    attn_avg_preserve_first_frame=args.attn_avg_preserve_first_frame,
    attn_avg_debug=args.attn_avg_debug,
    geometry_transport_alpha=args.geometry_transport_alpha,
    geometry_transport_layers=geometry_transport_layers,
    geometry_transport_mode=args.geometry_transport_mode,
    geometry_logit_margin=args.geometry_logit_margin,
    geometry_map_path=args.geometry_map_path,
    geometry_min_active_frame_coverage=args.geometry_min_active_frame_coverage,
    geometry_min_largest_component_share=args.geometry_min_largest_component_share,
    geometry_max_border_share=args.geometry_max_border_share,
    geometry_border_width=args.geometry_border_width,
    geometry_anchor_step=args.geometry_anchor_step,
    geometry_transport_start=args.geometry_transport_start,
    geometry_transport_end=args.geometry_transport_end,
    geometry_frame_indices=geometry_frame_indices,
    geometry_memory_lookback=args.geometry_memory_lookback,
    geometry_confidence_percentile=args.geometry_confidence_percentile,
    geometry_confidence_floor=args.geometry_confidence_floor,
    geometry_depth_relative_threshold=args.geometry_depth_relative_threshold,
    geometry_transport_cond_only=args.geometry_transport_cond_only,
    geometry_transport_debug=args.geometry_transport_debug,
    geometry_debug_dir=args.geometry_debug_dir,
    geometry_frozen_offline_suppression=args.frozen_offline_suppression,
    geometry_frozen_free_space_transport=args.frozen_free_space_transport,
    geometry_expected_provenance=generation_contract,
)
torch.cuda.synchronize()
wall_seconds = time.perf_counter() - start_time
peak_memory_mib = torch.cuda.max_memory_allocated() / 1024**2

export_to_video(output.frames[0], str(out), fps=args.fps)
metadata = vars(args).copy()
metadata["attn_avg_layers"] = attn_avg_layers
metadata["geometry_transport_layers"] = geometry_transport_layers
metadata["geometry_frame_indices"] = geometry_frame_indices
metadata["wall_seconds"] = wall_seconds
metadata["peak_memory_mib"] = peak_memory_mib
metadata["generation_contract"] = generation_contract
out.with_suffix(out.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
print("saved:", out)
print(f"wall_seconds: {wall_seconds:.2f}")
print(f"peak_memory_mib: {peak_memory_mib:.1f}")
