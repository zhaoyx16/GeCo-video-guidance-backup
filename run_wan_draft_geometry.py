import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "external" / "guidance_wan"))
from pipeline_wan_i2v_draft_geometry import WanImageToVideoPipeline


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
parser.add_argument("--geometry_transport_layers", default="")
parser.add_argument(
    "--geometry_transport_mode",
    choices=(
        "shadow",
        "value_residual",
        "kv_memory",
        "attn_output_memory",
        "geometry_attention_bias",
        "geometry_attention_output_bias",
        "geometry_sparse_logit_bias",
        "geometry_virtual_token_bias",
        "geometry_surface_graph_logit_bias",
        "geometry_signed_surface_graph_logit_bias",
        "geometry_relative_edge_output",
        "cached_virtual_token_bias",
        "query_transport",
        "qk_transport",
        "cached_block_transport",
        "cached_value_transport",
        "cached_value_residual",
        "cached_value_memory",
        "cached_value_attention_blend",
        "cached_draft_value_delta",
        "cached_draft_value_delta_consensus",
        "cached_draft_output_delta",
        "cached_draft_output_delta_centered",
    ),
    default="shadow",
)
parser.add_argument("--draft_geometry_map_path", default=None)
parser.add_argument("--geometry_transport_start", type=int, default=None)
parser.add_argument("--geometry_transport_end", type=int, default=None)
parser.add_argument(
    "--geometry_transport_schedule",
    choices=("constant", "linear_decay", "cosine_decay"),
    default="cosine_decay",
)
parser.add_argument("--geometry_transport_min_confidence", type=float, default=0.0)
parser.add_argument(
    "--geometry_transport_gate_mode",
    choices=("confidence", "binary_support"),
    default="confidence",
)
parser.add_argument("--geometry_transport_null_logit", type=float, default=0.0)
parser.add_argument(
    "--geometry_transport_source_logit_bias",
    type=float,
    default=0.0,
    help=(
        "Additive bias for geometry-nominated source logits in "
        "geometry_attention_bias and geometry_attention_output_bias modes; "
        "negative values require stronger Q/K evidence than target self-attention."
    ),
)
parser.add_argument(
    "--geometry_transport_sparse_logit_boost",
    type=float,
    default=0.0,
    help=(
        "Positive sparse attention-logit boost for geometry-nominated source "
        "tokens in geometry_sparse_logit_bias mode."
    ),
)
parser.add_argument(
    "--geometry_transport_surface_depth_threshold",
    type=float,
    default=0.0,
    help=(
        "Maximum relative source-depth jump for surface-graph edges; "
        "0 disables depth filtering."
    ),
)
parser.add_argument(
    "--geometry_transport_surface_graph_scales",
    default="1",
    help=(
        "Comma-separated source-token offsets used to build cardinal "
        "same-surface graph edges. Scale 1 reproduces the original local graph."
    ),
)
parser.add_argument(
    "--geometry_transport_surface_boundary_logit_suppress",
    type=float,
    default=0.0,
    help=(
        "Positive magnitude of the negative logit bias applied to mapped "
        "neighbors across a source-depth boundary in signed surface-graph mode."
    ),
)
parser.add_argument(
    "--geometry_transport_max_relative_rms",
    type=float,
    default=0.0,
    help=(
        "Per-target-time RMS trust-region cap relative to the native attention "
        "output; 0 disables the cap."
    ),
)
parser.add_argument(
    "--geometry_transport_preserve_qk_rms",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument(
    "--geometry_transport_consensus_threshold",
    type=float,
    default=-1.0,
    help="Enable multi-view cached-V consensus gating at this cosine threshold; -1 disables it.",
)
parser.add_argument(
    "--geometry_transport_value_lowpass_radius",
    type=int,
    default=0,
    help="Spatial token radius for low-pass cached-V residuals; 0 preserves full-band values.",
)
parser.add_argument("--geometry_transport_cond_only", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--geometry_transport_debug", action="store_true")
parser.add_argument("--draft_feature_cache_out_path", default=None)
parser.add_argument("--draft_feature_cache_in_path", default=None)
parser.add_argument(
    "--draft_feature_cache_kind",
    choices=(
        "block_output",
        "attention_value",
        "attention_kv",
        "attention_value_delta",
    ),
    default="block_output",
)
parser.add_argument("--draft_vector_cache_out_path", default=None)
parser.add_argument("--draft_vector_cache_in_path", default=None)
parser.add_argument(
    "--draft_vector_anchor_outside_geometry",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument("--draft_vector_mask_spatial_dilation", type=int, default=1)
parser.add_argument("--draft_vector_mask_temporal_dilation", type=int, default=1)
parser.add_argument("--draft_vector_geometry_alpha", type=float, default=0.0)
parser.add_argument(
    "--draft_vector_geometry_gate_mode",
    choices=("confidence", "binary_support"),
    default="binary_support",
)
parser.add_argument("--draft_vector_geometry_debug", action="store_true")
parser.add_argument("--draft_vector_anchor_debug", action="store_true")
parser.add_argument("--draft_clean_latent_cache_out_path", default=None)
parser.add_argument("--draft_clean_latent_cache_in_path", default=None)
parser.add_argument("--draft_clean_latent_alpha", type=float, default=0.0)
parser.add_argument(
    "--draft_clean_latent_gate_mode",
    choices=("confidence", "binary_support"),
    default="confidence",
)
parser.add_argument(
    "--draft_clean_latent_transport_mode",
    choices=("patch_mean", "full_patch"),
    default="patch_mean",
)
parser.add_argument(
    "--draft_clean_latent_reference_mode",
    choices=(
        "source_to_current",
        "draft_delta",
        "draft_delta_centered",
        "draft_delta_highpass",
        "draft_delta_lowpass",
    ),
    default="source_to_current",
)
parser.add_argument("--draft_clean_latent_delta_quantile", type=float, default=0.0)
parser.add_argument("--draft_clean_latent_lowpass_radius", type=int, default=2)
parser.add_argument(
    "--draft_clean_latent_sigma_scaled",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument("--draft_clean_latent_debug", action="store_true")
parser.add_argument("--draft_geometry_condition_alpha", type=float, default=0.0)
parser.add_argument(
    "--draft_geometry_condition_gate_mode",
    choices=("confidence", "target_p90", "binary_support"),
    default="confidence",
)
parser.add_argument(
    "--draft_geometry_condition_spatial_dilation",
    type=int,
    default=0,
)
parser.add_argument(
    "--draft_geometry_condition_reference_mode",
    choices=("full_target", "lowpass_delta"),
    default="full_target",
)
parser.add_argument(
    "--draft_geometry_condition_lowpass_radius",
    type=int,
    default=2,
)
parser.add_argument("--draft_geometry_condition_start_step", type=int, default=0)
parser.add_argument("--draft_geometry_condition_ramp_end_step", type=int, default=0)
parser.add_argument(
    "--draft_geometry_condition_schedule",
    choices=("constant", "linear_ramp", "cosine_ramp"),
    default="constant",
)
parser.add_argument(
    "--draft_geometry_condition_final_blend",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument("--draft_geometry_condition_debug", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_path", default="/vol/dissolve/yz10325/checkpoints/Wan2.2-TI2V-5B-Diffusers")
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--guidance_scale", type=float, default=5.0)
parser.add_argument("--negative_prompt", default=None)
args = parser.parse_args()

attn_avg_layers = parse_layers(args.attn_avg_layers)
geometry_transport_layers = parse_layers(args.geometry_transport_layers)
geometry_transport_surface_graph_scales = parse_layers(
    args.geometry_transport_surface_graph_scales
)
out = Path(args.output_path)
out.parent.mkdir(parents=True, exist_ok=True)
image_sha256 = hashlib.sha256(Path(args.image_path).read_bytes()).hexdigest()
draft_feature_cache_fingerprint_payload = {
    "model_path": str(Path(args.model_path).resolve()),
    "prompt": args.prompt,
    "negative_prompt": args.negative_prompt,
    "image_sha256": image_sha256,
    "seed": args.seed,
    "steps": args.steps,
    "frames": args.frames,
    "height": args.height,
    "width": args.width,
    "guidance_scale": args.guidance_scale,
}
draft_feature_cache_fingerprint = hashlib.sha256(
    json.dumps(
        draft_feature_cache_fingerprint_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

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
print("draft_geometry_map_path:", args.draft_geometry_map_path)
print("geometry_transport_interval:", (args.geometry_transport_start, args.geometry_transport_end))
print("geometry_transport_schedule:", args.geometry_transport_schedule)
print("geometry_transport_min_confidence:", args.geometry_transport_min_confidence)
print("geometry_transport_gate_mode:", args.geometry_transport_gate_mode)
print("geometry_transport_null_logit:", args.geometry_transport_null_logit)
print(
    "geometry_transport_source_logit_bias:",
    args.geometry_transport_source_logit_bias,
)
print(
    "geometry_transport_sparse_logit_boost:",
    args.geometry_transport_sparse_logit_boost,
)
print(
    "geometry_transport_surface_depth_threshold:",
    args.geometry_transport_surface_depth_threshold,
)
print(
    "geometry_transport_surface_graph_scales:",
    geometry_transport_surface_graph_scales,
)
print(
    "geometry_transport_surface_boundary_logit_suppress:",
    args.geometry_transport_surface_boundary_logit_suppress,
)
print(
    "geometry_transport_max_relative_rms:",
    args.geometry_transport_max_relative_rms,
)
print(
    "geometry_transport_preserve_qk_rms:",
    args.geometry_transport_preserve_qk_rms,
)
print(
    "geometry_transport_consensus_threshold:",
    args.geometry_transport_consensus_threshold,
)
print(
    "geometry_transport_value_lowpass_radius:",
    args.geometry_transport_value_lowpass_radius,
)
print("draft_feature_cache_out_path:", args.draft_feature_cache_out_path)
print("draft_feature_cache_in_path:", args.draft_feature_cache_in_path)
print("draft_feature_cache_kind:", args.draft_feature_cache_kind)
print("draft_feature_cache_fingerprint:", draft_feature_cache_fingerprint)
print("draft_vector_cache_out_path:", args.draft_vector_cache_out_path)
print("draft_vector_cache_in_path:", args.draft_vector_cache_in_path)
print("draft_vector_anchor_outside_geometry:", args.draft_vector_anchor_outside_geometry)
print(
    "draft_vector_mask_dilation:",
    (args.draft_vector_mask_temporal_dilation, args.draft_vector_mask_spatial_dilation),
)
print("draft_vector_geometry_alpha:", args.draft_vector_geometry_alpha)
print("draft_vector_geometry_gate_mode:", args.draft_vector_geometry_gate_mode)
print("draft_clean_latent_cache_out_path:", args.draft_clean_latent_cache_out_path)
print("draft_clean_latent_cache_in_path:", args.draft_clean_latent_cache_in_path)
print("draft_clean_latent_alpha:", args.draft_clean_latent_alpha)
print("draft_clean_latent_gate_mode:", args.draft_clean_latent_gate_mode)
print("draft_clean_latent_transport_mode:", args.draft_clean_latent_transport_mode)
print("draft_clean_latent_reference_mode:", args.draft_clean_latent_reference_mode)
print("draft_clean_latent_delta_quantile:", args.draft_clean_latent_delta_quantile)
print("draft_clean_latent_lowpass_radius:", args.draft_clean_latent_lowpass_radius)
print("draft_clean_latent_sigma_scaled:", args.draft_clean_latent_sigma_scaled)
print("draft_geometry_condition_alpha:", args.draft_geometry_condition_alpha)
print("draft_geometry_condition_gate_mode:", args.draft_geometry_condition_gate_mode)
print(
    "draft_geometry_condition_spatial_dilation:",
    args.draft_geometry_condition_spatial_dilation,
)
print(
    "draft_geometry_condition_reference:",
    (
        args.draft_geometry_condition_reference_mode,
        args.draft_geometry_condition_lowpass_radius,
    ),
)
print(
    "draft_geometry_condition_step_schedule:",
    (
        args.draft_geometry_condition_start_step,
        args.draft_geometry_condition_ramp_end_step,
        args.draft_geometry_condition_schedule,
    ),
)
print(
    "draft_geometry_condition_final_blend:",
    args.draft_geometry_condition_final_blend,
)
print("seed:", args.seed)

vae = AutoencoderKLWan.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(args.model_path, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

image = Image.open(args.image_path).convert("RGB")
generator = torch.Generator(device="cuda").manual_seed(args.seed)

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
    draft_geometry_map_path=args.draft_geometry_map_path,
    geometry_transport_start=args.geometry_transport_start,
    geometry_transport_end=args.geometry_transport_end,
    geometry_transport_schedule=args.geometry_transport_schedule,
    geometry_transport_min_confidence=args.geometry_transport_min_confidence,
    geometry_transport_gate_mode=args.geometry_transport_gate_mode,
    geometry_transport_null_logit=args.geometry_transport_null_logit,
    geometry_transport_source_logit_bias=args.geometry_transport_source_logit_bias,
    geometry_transport_sparse_logit_boost=args.geometry_transport_sparse_logit_boost,
    geometry_transport_surface_depth_threshold=args.geometry_transport_surface_depth_threshold,
    geometry_transport_surface_graph_scales=geometry_transport_surface_graph_scales,
    geometry_transport_surface_boundary_logit_suppress=args.geometry_transport_surface_boundary_logit_suppress,
    geometry_transport_max_relative_rms=args.geometry_transport_max_relative_rms,
    geometry_transport_preserve_qk_rms=args.geometry_transport_preserve_qk_rms,
    geometry_transport_consensus_threshold=args.geometry_transport_consensus_threshold,
    geometry_transport_value_lowpass_radius=args.geometry_transport_value_lowpass_radius,
    geometry_transport_cond_only=args.geometry_transport_cond_only,
    geometry_transport_debug=args.geometry_transport_debug,
    draft_feature_cache_out_path=args.draft_feature_cache_out_path,
    draft_feature_cache_in_path=args.draft_feature_cache_in_path,
    draft_feature_cache_kind=args.draft_feature_cache_kind,
    draft_feature_cache_fingerprint=draft_feature_cache_fingerprint,
    draft_vector_cache_out_path=args.draft_vector_cache_out_path,
    draft_vector_cache_in_path=args.draft_vector_cache_in_path,
    draft_vector_anchor_outside_geometry=args.draft_vector_anchor_outside_geometry,
    draft_vector_mask_spatial_dilation=args.draft_vector_mask_spatial_dilation,
    draft_vector_mask_temporal_dilation=args.draft_vector_mask_temporal_dilation,
    draft_vector_geometry_alpha=args.draft_vector_geometry_alpha,
    draft_vector_geometry_gate_mode=args.draft_vector_geometry_gate_mode,
    draft_vector_geometry_debug=args.draft_vector_geometry_debug,
    draft_vector_anchor_debug=args.draft_vector_anchor_debug,
    draft_clean_latent_cache_out_path=args.draft_clean_latent_cache_out_path,
    draft_clean_latent_cache_in_path=args.draft_clean_latent_cache_in_path,
    draft_clean_latent_alpha=args.draft_clean_latent_alpha,
    draft_clean_latent_gate_mode=args.draft_clean_latent_gate_mode,
    draft_clean_latent_transport_mode=args.draft_clean_latent_transport_mode,
    draft_clean_latent_reference_mode=args.draft_clean_latent_reference_mode,
    draft_clean_latent_delta_quantile=args.draft_clean_latent_delta_quantile,
    draft_clean_latent_lowpass_radius=args.draft_clean_latent_lowpass_radius,
    draft_clean_latent_sigma_scaled=args.draft_clean_latent_sigma_scaled,
    draft_clean_latent_debug=args.draft_clean_latent_debug,
    draft_geometry_condition_alpha=args.draft_geometry_condition_alpha,
    draft_geometry_condition_gate_mode=args.draft_geometry_condition_gate_mode,
    draft_geometry_condition_spatial_dilation=args.draft_geometry_condition_spatial_dilation,
    draft_geometry_condition_reference_mode=args.draft_geometry_condition_reference_mode,
    draft_geometry_condition_lowpass_radius=args.draft_geometry_condition_lowpass_radius,
    draft_geometry_condition_start_step=args.draft_geometry_condition_start_step,
    draft_geometry_condition_ramp_end_step=args.draft_geometry_condition_ramp_end_step,
    draft_geometry_condition_schedule=args.draft_geometry_condition_schedule,
    draft_geometry_condition_final_blend=args.draft_geometry_condition_final_blend,
    draft_geometry_condition_debug=args.draft_geometry_condition_debug,
)
torch.cuda.synchronize()
wall_seconds = time.perf_counter() - start_time
peak_memory_mib = torch.cuda.max_memory_allocated() / 1024**2

export_to_video(output.frames[0], str(out), fps=args.fps)
metadata = vars(args).copy()
metadata["attn_avg_layers"] = attn_avg_layers
metadata["geometry_transport_layers"] = geometry_transport_layers
metadata["draft_feature_cache_fingerprint"] = draft_feature_cache_fingerprint
metadata["draft_feature_cache_fingerprint_payload"] = draft_feature_cache_fingerprint_payload
metadata["wall_seconds"] = wall_seconds
metadata["peak_memory_mib"] = peak_memory_mib
out.with_suffix(out.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
print("saved:", out)
print(f"wall_seconds: {wall_seconds:.2f}")
print(f"peak_memory_mib: {peak_memory_mib:.1f}")
