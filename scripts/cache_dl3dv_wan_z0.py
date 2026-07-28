#!/usr/bin/env python3
"""Cache scene-disjoint DL3DV clips as clean Wan VAE latents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import diffusers
import torch
from diffusers import AutoencoderKLWan
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from latent_geometry.data import (
    LATENT_DOMAIN_RAW_VAE_Z0,
    MANIFEST_FORMAT_VERSION,
    ProbeManifestRecord,
    save_clean_latent_record,
    tensor_sha256,
)
from latent_geometry.dl3dv import (
    choose_clip_starts,
    discover_dl3dv_scenes,
    latent_anchor_rgb_indices,
    preprocess_frame,
    source_content_sha256,
    transform_intrinsics_for_cover_crop,
)


def _parse_scene_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def _scene_slug(scene_uid: str) -> str:
    return scene_uid.replace("/", "__")


def _split_for_scene(scene_uid: str, val_scenes: set[str], test_scenes: set[str]) -> str:
    if scene_uid in val_scenes:
        return "val"
    if scene_uid in test_scenes:
        return "test"
    return "train"


def _extract_latent(encoder_output) -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise TypeError(f"Unsupported Wan VAE encoder output: {type(encoder_output)!r}")


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".safetensors", ".bin"}
    )
    if not files:
        raise FileNotFoundError(f"No VAE config/weight files found below {root}")
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, object]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--short")),
    }


def _encode_clip(vae, frames: list[Path], width: int, height: int, device: torch.device) -> torch.Tensor:
    video = torch.stack(
        [preprocess_frame(Image.open(path), width, height) for path in frames],
        dim=1,
    ).unsqueeze(0)
    video = video.to(device=device, dtype=vae.dtype)
    with torch.inference_mode():
        latent = _extract_latent(vae.encode(video))
    return latent.squeeze(0).float().cpu()


def _manifest_pairs(num_latents: int, pair_gaps: list[int]) -> list[tuple[int, int]]:
    pairs = []
    for gap in pair_gaps:
        if gap <= 0:
            raise ValueError("pair gaps must be positive")
        pairs.extend((source, source + gap) for source in range(num_latents - gap))
    if not pairs:
        raise ValueError("No valid latent pairs; reduce --pair-gaps or increase --clip-frames")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dl3dv-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Metadata time unit only; the DL3DV source FPS is not verified",
    )
    parser.add_argument("--clip-frames", type=int, default=17)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--clip-stride", type=int, default=16)
    parser.add_argument("--max-clips-per-scene", type=int, default=24)
    parser.add_argument("--pair-gaps", default="1")
    parser.add_argument("--val-scenes", required=True)
    parser.add_argument("--test-scenes", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.output_root} is not empty; pass --overwrite or choose a new output")
    args.output_root.mkdir(parents=True, exist_ok=True)
    cache_root = args.output_root / "cache"
    manifest_path = args.output_root / "manifest.jsonl"
    val_scenes = _parse_scene_set(args.val_scenes)
    test_scenes = _parse_scene_set(args.test_scenes)
    if val_scenes & test_scenes:
        raise SystemExit("Validation and test scene sets overlap")

    scenes = discover_dl3dv_scenes(args.dl3dv_root)
    known = {scene.scene_uid for scene in scenes}
    unknown = (val_scenes | test_scenes) - known
    if unknown:
        raise SystemExit(f"Unknown split scenes: {sorted(unknown)}; available: {sorted(known)}")
    if not val_scenes or not test_scenes:
        raise SystemExit("At least one explicit validation and test scene are required")
    split_counts = {
        split: sum(_split_for_scene(scene.scene_uid, val_scenes, test_scenes) == split for scene in scenes)
        for split in ("train", "val", "test")
    }
    if min(split_counts.values()) == 0:
        raise SystemExit(f"Every split must contain a source scene, got {split_counts}")

    device = torch.device(args.device)
    vae = AutoencoderKLWan.from_pretrained(
        str(args.model), subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    vae.enable_tiling()
    vae.enable_slicing()
    temporal_ratio = int(vae.config.scale_factor_temporal)
    anchor_indices = latent_anchor_rgb_indices(args.clip_frames, temporal_ratio)
    pair_gaps = [int(value) for value in args.pair_gaps.split(",") if value.strip()]
    pair_indices = _manifest_pairs(len(anchor_indices), pair_gaps)
    model_revision = _tree_sha256(args.model / "vae")
    git_state = _git_state()

    manifest_rows: list[dict] = []
    extraction_summary = {
        "model": str(args.model),
        "vae_tree_sha256": model_revision,
        "git": git_state,
        "torch_version": torch.__version__,
        "diffusers_version": diffusers.__version__,
        "height": args.height,
        "width": args.width,
        "fps": args.fps,
        "clip_frames": args.clip_frames,
        "frame_step": args.frame_step,
        "clip_stride": args.clip_stride,
        "max_clips_per_scene": args.max_clips_per_scene,
        "pair_gaps": pair_gaps,
        "temporal_ratio": temporal_ratio,
        "scene_splits": {},
        "clips": 0,
        "pairs": 0,
        "started_unix": time.time(),
    }

    for scene in scenes:
        split = _split_for_scene(scene.scene_uid, val_scenes, test_scenes)
        extraction_summary["scene_splits"][scene.scene_uid] = split
        starts = choose_clip_starts(
            len(scene.frame_paths),
            clip_frames=args.clip_frames,
            frame_step=args.frame_step,
            clip_stride=args.clip_stride,
            max_clips=args.max_clips_per_scene,
        )
        for clip_number, start in enumerate(starts):
            rgb_indices = [start + offset * args.frame_step for offset in range(args.clip_frames)]
            frame_paths = [scene.frame_paths[index] for index in rgb_indices]
            frame_ids = scene.frame_ids[rgb_indices].clone()
            poses = scene.camera_poses_w2c[rgb_indices].clone()
            first_image = Image.open(frame_paths[0])
            intrinsics = transform_intrinsics_for_cover_crop(
                scene.intrinsics[rgb_indices],
                calibration_width=scene.source_width,
                calibration_height=scene.source_height,
                image_width=first_image.width,
                image_height=first_image.height,
                target_width=args.width,
                target_height=args.height,
            )
            clip_uid = f"{scene.scene_uid}/start-{int(frame_ids[0]):05d}/n-{args.clip_frames}/step-{args.frame_step}"
            cache_id = f"{_scene_slug(scene.scene_uid)}__clip_{clip_number:04d}_{int(frame_ids[0]):05d}"
            cache_path = cache_root / _scene_slug(scene.scene_uid) / f"{cache_id}.pt"

            if cache_path.exists() and not args.overwrite:
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                z0 = payload["z0"]
                binding = {
                    "source_dataset": payload["source_provenance"]["source_dataset"],
                    "source_scene_uid": payload["source_provenance"]["source_scene_uid"],
                    "source_clip_uid": payload["source_provenance"]["source_clip_uid"],
                    "source_content_sha256": payload["source_provenance"]["source_content_sha256"],
                    "cache_sha256": payload["cache_sha256"],
                }
            else:
                z0 = _encode_clip(vae, frame_paths, args.width, args.height, device)
                if z0.shape[1] != len(anchor_indices):
                    raise RuntimeError(
                        f"Wan VAE temporal size {z0.shape[1]} does not match anchors {len(anchor_indices)}"
                    )
                source_digest = source_content_sha256(frame_paths)
                source_provenance = {
                    "source_dataset": "DL3DV-ALL-960P",
                    "source_scene_uid": scene.scene_uid,
                    "source_clip_uid": clip_uid,
                    "source_content_sha256": source_digest,
                    "source_frame_ids_sha256": tensor_sha256(frame_ids),
                    "source_pose_sha256": tensor_sha256(poses),
                }
                temporal_mapping = {
                    "mapping_type": "latent_anchor_frame_id",
                    "anchor_rule": "wan_causal_group_endpoint_anchor",
                    "is_causal": True,
                    "temporal_compression_ratio": temporal_ratio,
                    "latent_to_frame_ids": frame_ids[anchor_indices].clone(),
                }
                latent_spec = {
                    "domain": LATENT_DOMAIN_RAW_VAE_Z0,
                    "model_family": "wan",
                    "vae": {
                        "identifier": f"{args.model}/vae",
                        "revision": model_revision,
                        "latent_scaling": "raw_autoencoderklwan_posterior_mode_no_mean_std_normalization",
                        "latents_mean": [float(value) for value in vae.config.latents_mean],
                        "latents_std": [float(value) for value in vae.config.latents_std],
                        "tiling": True,
                        "slicing": True,
                        "torch_version": torch.__version__,
                        "diffusers_version": diffusers.__version__,
                        "extractor_git_commit": git_state["commit"],
                        "extractor_git_dirty": git_state["dirty"],
                    },
                    "preprocessing": {
                        "image_normalization": "RGB float [-1,1], resize-cover then center-crop",
                        "height": args.height,
                        "width": args.width,
                        "fps": args.fps,
                        "frame_sampling": (
                            f"ordered DL3DV frame-index units, source step={args.frame_step}; "
                            "source capture FPS not verified"
                        ),
                        "source_fps_verified": False,
                        "source_camera_model": scene.camera_model,
                        "source_distortion_k1_k2_p1_p2": list(scene.distortion),
                        "distortion_handling": (
                            "source pixels retained; distortion recorded but no undistortion applied"
                        ),
                        "camera_axes": "opencv_x_right_y_down_z_forward",
                    },
                    "temporal_mapping_type": "latent_anchor_frame_id",
                    "scheduler": None,
                    "condition": None,
                }
                binding = save_clean_latent_record(
                    cache_path,
                    record_id=cache_id,
                    scene_id=scene.scene_uid,
                    z0=z0,
                    camera_poses_w2c=poses,
                    intrinsics=intrinsics,
                    frame_ids=frame_ids,
                    source_provenance=source_provenance,
                    latent_spec=latent_spec,
                    temporal_mapping=temporal_mapping,
                    pose_spec={
                        "convention": "world_to_camera",
                        "transform_type": "SE3",
                        "coordinate_system": "right_handed",
                        "camera_axes": "opencv_x_right_y_down_z_forward",
                    },
                )

            for source_latent, target_latent in pair_indices:
                source_rgb = anchor_indices[source_latent]
                target_rgb = anchor_indices[target_latent]
                record = ProbeManifestRecord(
                    record_id=f"{cache_id}__pair_{source_latent:02d}_{target_latent:02d}",
                    scene_id=scene.scene_uid,
                    source_dataset=binding["source_dataset"],
                    source_scene_uid=binding["source_scene_uid"],
                    source_clip_uid=binding["source_clip_uid"],
                    source_content_sha256=binding["source_content_sha256"],
                    cache_sha256=binding["cache_sha256"],
                    split=split,
                    cache_path=cache_path,
                    source_pose_index=source_rgb,
                    target_pose_index=target_rgb,
                    source_latent_index=source_latent,
                    target_latent_index=target_latent,
                    static_scene=True,
                )
                manifest_rows.append(record.to_json_dict(args.output_root))
            extraction_summary["clips"] += 1
            extraction_summary["pairs"] += len(pair_indices)
            print(
                f"cached split={split} scene={scene.scene_uid} clip={clip_number + 1}/{len(starts)} "
                f"z0={tuple(z0.shape)} pairs={len(pair_indices)}",
                flush=True,
            )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            row["format_version"] = MANIFEST_FORMAT_VERSION
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    extraction_summary["finished_unix"] = time.time()
    (args.output_root / "extraction_summary.json").write_text(
        json.dumps(extraction_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"manifest: {manifest_path}")
    print(f"summary: {args.output_root / 'extraction_summary.json'}")


if __name__ == "__main__":
    main()
