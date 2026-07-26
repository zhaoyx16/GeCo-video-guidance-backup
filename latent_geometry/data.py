"""Manifest-aware cached latent data for geometry probes.

This code deliberately caches only clean VAE latents (z0).  No VAE or video
diffusion model is loaded here.  Noisy z_t examples are generated online from a
deterministic flow-matching style interpolation schedule.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .geometry import make_relative_pose_target


CACHE_FORMAT_VERSION = 1
MANIFEST_FORMAT_VERSION = 1
_ALLOWED_SPLITS = {"train", "val", "test"}


class ManifestError(ValueError):
    """Raised when a manifest would permit an invalid or leaky experiment."""


@dataclass(frozen=True)
class ProbeManifestRecord:
    """One ordered pair from a clean latent clip.

    Pose indices address ``camera_poses_w2c`` in the cache.  Latent indices
    address the temporal axis of cached ``z0``.  Keeping both explicit avoids
    silently assuming that a VAE temporal index equals an RGB frame index.
    """

    record_id: str
    scene_id: str
    split: str
    cache_path: Path
    source_pose_index: int
    target_pose_index: int
    source_latent_index: int
    target_latent_index: int
    static_scene: bool
    source_dataset: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], manifest_root: Path) -> "ProbeManifestRecord":
        required = {
            "format_version",
            "record_id",
            "scene_id",
            "split",
            "cache_path",
            "source_pose_index",
            "target_pose_index",
            "source_latent_index",
            "target_latent_index",
            "static_scene",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ManifestError(f"Manifest record is missing required keys: {missing}")
        if int(payload["format_version"]) != MANIFEST_FORMAT_VERSION:
            raise ManifestError(
                f"Unsupported manifest format {payload['format_version']}; "
                f"expected {MANIFEST_FORMAT_VERSION}"
            )
        split = str(payload["split"])
        if split not in _ALLOWED_SPLITS:
            raise ManifestError(f"split must be one of {sorted(_ALLOWED_SPLITS)}, got {split!r}")
        cache_path = Path(str(payload["cache_path"]))
        if not cache_path.is_absolute():
            cache_path = manifest_root / cache_path
        indices = (
            "source_pose_index",
            "target_pose_index",
            "source_latent_index",
            "target_latent_index",
        )
        for key in indices:
            if int(payload[key]) < 0:
                raise ManifestError(f"{key} must be non-negative, got {payload[key]!r}")
        if not isinstance(payload["static_scene"], bool):
            raise ManifestError("static_scene must be a JSON boolean")
        return cls(
            record_id=str(payload["record_id"]),
            scene_id=str(payload["scene_id"]),
            split=split,
            cache_path=cache_path,
            source_pose_index=int(payload["source_pose_index"]),
            target_pose_index=int(payload["target_pose_index"]),
            source_latent_index=int(payload["source_latent_index"]),
            target_latent_index=int(payload["target_latent_index"]),
            static_scene=bool(payload["static_scene"]),
            source_dataset=(None if payload.get("source_dataset") is None else str(payload["source_dataset"])),
        )

    def to_json_dict(self, manifest_root: Path) -> dict[str, Any]:
        payload = asdict(self)
        cache_path = self.cache_path
        try:
            payload["cache_path"] = str(cache_path.relative_to(manifest_root))
        except ValueError:
            payload["cache_path"] = str(cache_path)
        payload["format_version"] = MANIFEST_FORMAT_VERSION
        return payload


def validate_scene_disjoint_splits(records: Sequence[ProbeManifestRecord]) -> None:
    """Reject manifests where the same scene appears in more than one split."""
    scene_splits: dict[str, str] = {}
    for record in records:
        existing = scene_splits.setdefault(record.scene_id, record.split)
        if existing != record.split:
            raise ManifestError(
                "Scene-disjoint split violation: "
                f"scene_id={record.scene_id!r} appears in both {existing!r} and {record.split!r}"
            )


def load_manifest_records(manifest_path: str | Path) -> list[ProbeManifestRecord]:
    """Read a JSONL manifest, enforce unique record IDs and scene-disjoint splits."""
    path = Path(manifest_path)
    records: list[ProbeManifestRecord] = []
    seen_record_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ManifestError(f"Invalid JSON at {path}:{line_number}: {error.msg}") from error
            record = ProbeManifestRecord.from_mapping(payload, path.parent)
            if record.record_id in seen_record_ids:
                raise ManifestError(f"Duplicate record_id in manifest: {record.record_id!r}")
            seen_record_ids.add(record.record_id)
            records.append(record)
    if not records:
        raise ManifestError(f"Manifest contains no records: {path}")
    validate_scene_disjoint_splits(records)
    return records


def save_clean_latent_record(
    path: str | Path,
    *,
    record_id: str,
    scene_id: str,
    z0: torch.Tensor,
    camera_poses_w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    model_family: str = "wan",
    vae_identifier: str = "",
    frame_ids: torch.Tensor | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write the versioned clean-latent cache format used by this branch.

    ``z0`` must be a CPU or GPU tensor shaped ``[C, T_latent, H_latent,
    W_latent]``.  It is copied to CPU before saving.  The VAE encoding details
    are recorded as metadata so that a later extractor can make the cache
    traceable without this module loading a VAE.
    """
    path = Path(path)
    if z0.ndim != 4:
        raise ValueError(f"z0 must have shape [C, T, H, W], got {tuple(z0.shape)}")
    if camera_poses_w2c.ndim != 3 or camera_poses_w2c.shape[-2:] != (4, 4):
        raise ValueError(
            "camera_poses_w2c must have shape [num_frames, 4, 4], "
            f"got {tuple(camera_poses_w2c.shape)}"
        )
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics must have shape [num_frames, 3, 3], got {tuple(intrinsics.shape)}")
    if intrinsics.shape[0] != camera_poses_w2c.shape[0]:
        raise ValueError("intrinsics and camera_poses_w2c must have the same frame count")
    if frame_ids is None:
        frame_ids = torch.arange(camera_poses_w2c.shape[0], dtype=torch.long)
    if frame_ids.ndim != 1 or frame_ids.shape[0] != camera_poses_w2c.shape[0]:
        raise ValueError("frame_ids must have shape [num_frames]")

    cache_metadata = dict(metadata or {})
    cache_metadata.update(
        {
            "record_id": record_id,
            "scene_id": scene_id,
            "model_family": model_family,
            "vae_identifier": vae_identifier,
            "z0_layout": "C,T,H,W",
            "pose_convention": "world_to_camera",
            "translation_target": "unit_direction_in_target_camera",
        }
    )
    payload = {
        "format_version": CACHE_FORMAT_VERSION,
        "z0": z0.detach().to(device="cpu").contiguous(),
        "camera_poses_w2c": camera_poses_w2c.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        "intrinsics": intrinsics.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        "frame_ids": frame_ids.detach().to(device="cpu", dtype=torch.long).contiguous(),
        "metadata": cache_metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_clean_latent_record(path: Path, expected_model_family: str | None) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except FileNotFoundError as error:
        raise ManifestError(f"Cached latent file does not exist: {path}") from error
    if not isinstance(payload, dict):
        raise ManifestError(f"Cached latent record must be a dict: {path}")
    required = {"format_version", "z0", "camera_poses_w2c", "intrinsics", "frame_ids", "metadata"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ManifestError(f"Cached latent record {path} is missing keys: {missing}")
    if int(payload["format_version"]) != CACHE_FORMAT_VERSION:
        raise ManifestError(
            f"Unsupported cached latent format {payload['format_version']} in {path}; "
            f"expected {CACHE_FORMAT_VERSION}"
        )
    z0 = payload["z0"]
    poses = payload["camera_poses_w2c"]
    intrinsics = payload["intrinsics"]
    frame_ids = payload["frame_ids"]
    metadata = payload["metadata"]
    if not isinstance(z0, torch.Tensor) or z0.ndim != 4:
        raise ManifestError(f"z0 in {path} must be a tensor [C, T, H, W]")
    if not torch.is_floating_point(z0) or not torch.isfinite(z0).all():
        raise ManifestError(f"z0 in {path} must be finite floating-point data")
    if not isinstance(poses, torch.Tensor) or poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ManifestError(f"camera_poses_w2c in {path} must be [num_frames, 4, 4]")
    if not isinstance(intrinsics, torch.Tensor) or intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ManifestError(f"intrinsics in {path} must be [num_frames, 3, 3]")
    if poses.shape[0] != intrinsics.shape[0] or poses.shape[0] != frame_ids.shape[0]:
        raise ManifestError(f"Pose, intrinsics and frame_ids counts disagree in {path}")
    if not isinstance(metadata, dict):
        raise ManifestError(f"metadata in {path} must be a dict")
    if metadata.get("pose_convention") != "world_to_camera":
        raise ManifestError(
            f"Cached record {path} must explicitly use pose_convention='world_to_camera'"
        )
    if expected_model_family is not None and metadata.get("model_family") != expected_model_family:
        raise ManifestError(
            f"Cached record {path} is for model_family={metadata.get('model_family')!r}, "
            f"expected {expected_model_family!r}"
        )
    return payload


@dataclass(frozen=True)
class LinearFlowNoiseSchedule:
    """Online noising for a flow-matching-style latent probe input.

    ``z_t = (1 - t) z0 + t epsilon``.  This is a deliberately small,
    model-agnostic probe distribution, not a replacement for Wan's scheduler.
    The future x0-prediction probe must use actual scheduler/model outputs.
    """

    min_timestep: float = 0.05
    max_timestep: float = 0.75

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_timestep <= self.max_timestep <= 1.0:
            raise ValueError("Noise timesteps must satisfy 0 <= min <= max <= 1")

    def sample(self, z0: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        timestep = torch.empty((), dtype=torch.float32).uniform_(
            self.min_timestep, self.max_timestep, generator=generator
        )
        noise = torch.randn(z0.shape, generator=generator, dtype=z0.dtype, device="cpu")
        zt = (1.0 - timestep.to(dtype=z0.dtype)) * z0 + timestep.to(dtype=z0.dtype) * noise
        return zt, timestep, noise


def _stable_seed(base_seed: int, epoch: int, record_id: str) -> int:
    value = f"{base_seed}|{epoch}|{record_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], byteorder="little") % (2**63 - 1)


class CachedLatentDataset(Dataset[dict[str, Any]]):
    """Read clean latent caches and create deterministic z_t samples online.

    The manifest is validated globally before filtering by split, so leakage is
    caught even if the caller constructs separate train and validation datasets.
    Each item returns both clean ``z0`` and online-sampled ``zt`` for the same
    ordered latent pair; callers select the input explicitly in their trainer.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        noise_schedule: LinearFlowNoiseSchedule | None = None,
        base_seed: int = 0,
        expected_model_family: str | None = "wan",
        require_static_scene: bool = True,
    ) -> None:
        if split not in _ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {sorted(_ALLOWED_SPLITS)}")
        all_records = load_manifest_records(manifest_path)
        records = [record for record in all_records if record.split == split]
        if require_static_scene:
            records = [record for record in records if record.static_scene]
        if not records:
            raise ManifestError(f"No records available for split={split!r}")
        self.records = records
        self.noise_schedule = noise_schedule or LinearFlowNoiseSchedule()
        self.base_seed = int(base_seed)
        self.expected_model_family = expected_model_family
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Change deterministic online-noise draws without changing data splits."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        payload = _load_clean_latent_record(record.cache_path, self.expected_model_family)
        z0_full = payload["z0"].float()
        num_latent_frames = z0_full.shape[1]
        if record.source_latent_index >= num_latent_frames or record.target_latent_index >= num_latent_frames:
            raise ManifestError(
                f"Latent pair ({record.source_latent_index}, {record.target_latent_index}) is outside "
                f"cached z0 temporal size {num_latent_frames} for {record.record_id!r}"
            )
        z0_pair = z0_full[:, [record.source_latent_index, record.target_latent_index]].contiguous()
        pose_target = make_relative_pose_target(
            payload["camera_poses_w2c"].float(),
            record.source_pose_index,
            record.target_pose_index,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_seed(self.base_seed, self.epoch, record.record_id))
        zt_pair, timestep, _ = self.noise_schedule.sample(z0_pair, generator)
        return {
            "z0": z0_pair,
            "zt": zt_pair,
            "timestep": timestep,
            "rotation_6d": pose_target["rotation_6d"].float(),
            "translation_direction": pose_target["translation_direction"].float(),
            "translation_valid": pose_target["translation_valid"],
            "record_id": record.record_id,
            "scene_id": record.scene_id,
            "pair_indices": torch.tensor(
                [
                    record.source_pose_index,
                    record.target_pose_index,
                    record.source_latent_index,
                    record.target_latent_index,
                ],
                dtype=torch.long,
            ),
        }
