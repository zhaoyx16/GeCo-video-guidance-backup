"""Deterministic tiny data generation for unit tests and smoke training only."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import torch

from .data import (
    LATENT_DOMAIN_RAW_VAE_Z0,
    MANIFEST_FORMAT_VERSION,
    save_clean_latent_record,
    tensor_sha256,
)


def _rotation_z(angle: float) -> torch.Tensor:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )


def create_synthetic_probe_manifest(
    root: str | Path,
    *,
    train_scenes: int = 2,
    val_scenes: int = 1,
    records_per_scene: int = 4,
    channels: int = 5,
) -> Path:
    """Create a toy scene-disjoint manifest with latent-readable pose labels.

    It is intentionally synthetic and only supports CPU smoke tests.  Rotation
    and direction are encoded in the spatial mean of z0, so a linear probe has
    a learnable signal and can verify the end-to-end data/training contract.
    """
    root = Path(root)
    cache_root = root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "synthetic_manifest.jsonl"
    generator = torch.Generator(device="cpu").manual_seed(1234)
    records: list[dict[str, object]] = []
    scene_specs = [("train", train_scenes), ("val", val_scenes)]
    global_index = 0
    for split, count in scene_specs:
        for scene_number in range(count):
            scene_id = f"{split}_scene_{scene_number:02d}"
            for record_number in range(records_per_scene):
                value = global_index / max(1, train_scenes * records_per_scene + val_scenes * records_per_scene - 1)
                angle = -0.45 + 0.9 * value
                direction = torch.tensor([math.cos(angle), math.sin(angle), 0.35], dtype=torch.float32)
                direction = direction / torch.linalg.vector_norm(direction)
                z0 = torch.randn((channels, 2, 4, 4), generator=generator) * 0.01
                z0[0].fill_(math.cos(angle))
                z0[1].fill_(math.sin(angle))
                z0[2].fill_(float(direction[0]))
                z0[3].fill_(float(direction[1]))
                z0[4].fill_(float(direction[2]))

                poses = torch.eye(4, dtype=torch.float32).repeat(2, 1, 1)
                poses[1, :3, :3] = _rotation_z(angle)
                poses[1, :3, 3] = direction * (1.0 + value)
                intrinsics = torch.eye(3, dtype=torch.float32).repeat(2, 1, 1)
                frame_ids = torch.tensor([0, 1], dtype=torch.long)
                record_id = f"{scene_id}_clip_{record_number:03d}"
                cache_path = cache_root / f"{record_id}.pt"
                source_provenance = {
                    "source_dataset": "synthetic-test-only",
                    "source_scene_uid": f"synthetic/{scene_id}",
                    "source_clip_uid": f"synthetic/{record_id}",
                    "source_content_sha256": hashlib.sha256(record_id.encode("utf-8")).hexdigest(),
                    "source_frame_ids_sha256": tensor_sha256(frame_ids),
                    "source_pose_sha256": tensor_sha256(poses),
                }
                temporal_mapping = {
                    "mapping_type": "latent_anchor_frame_id",
                    "anchor_rule": "synthetic_identity",
                    "is_causal": True,
                    "temporal_compression_ratio": 1,
                    "latent_to_frame_ids": frame_ids.clone(),
                }
                latent_spec = {
                    "domain": LATENT_DOMAIN_RAW_VAE_Z0,
                    "model_family": "wan",
                    "vae": {
                        "identifier": "synthetic-test-only",
                        "revision": "test",
                        "latent_scaling": "raw_encoder_output",
                    },
                    "preprocessing": {
                        "image_normalization": "synthetic",
                        "height": 32,
                        "width": 32,
                        "fps": 24.0,
                        "frame_sampling": "synthetic_identity",
                    },
                    "temporal_mapping_type": "latent_anchor_frame_id",
                    "scheduler": None,
                    "condition": None,
                }
                binding = save_clean_latent_record(
                    cache_path,
                    record_id=record_id,
                    scene_id=scene_id,
                    z0=z0,
                    camera_poses_w2c=poses,
                    intrinsics=intrinsics,
                    frame_ids=frame_ids,
                    source_provenance=source_provenance,
                    latent_spec=latent_spec,
                    temporal_mapping=temporal_mapping,
                )
                records.append(
                    {
                        "format_version": MANIFEST_FORMAT_VERSION,
                        "record_id": record_id,
                        "scene_id": scene_id,
                        "source_dataset": binding["source_dataset"],
                        "source_scene_uid": binding["source_scene_uid"],
                        "source_clip_uid": binding["source_clip_uid"],
                        "cache_sha256": binding["cache_sha256"],
                        "split": split,
                        "cache_path": str(cache_path.relative_to(root)),
                        "source_pose_index": 0,
                        "target_pose_index": 1,
                        "source_latent_index": 0,
                        "target_latent_index": 1,
                        "static_scene": True,
                    }
                )
                global_index += 1
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return manifest_path
