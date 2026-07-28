"""Strict, manifest-aware cached-latent data for geometry probes.

This probe-only package deliberately does not load a VAE or a video diffusion
model.  It accepts a verifiable cache of clean Wan VAE latents (``z0``), creates
online probe ``zt`` samples, and rejects provenance, temporal-mapping, or latent
domain mismatches before training starts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .geometry import make_relative_pose_target, validate_world_to_camera_se3


CACHE_FORMAT_VERSION = 3
MANIFEST_FORMAT_VERSION = 3
LATENT_DOMAIN_RAW_VAE_Z0 = "raw_vae_z0"
LATENT_DOMAIN_NORMALIZED_DIFFUSION_Z = "normalized_diffusion_z"
LATENT_DOMAIN_X0_PRED = "x0_pred"
_LATENT_DOMAINS = {
    LATENT_DOMAIN_RAW_VAE_Z0,
    LATENT_DOMAIN_NORMALIZED_DIFFUSION_Z,
    LATENT_DOMAIN_X0_PRED,
}
_ALLOWED_SPLITS = {"train", "val", "test"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """Raised when a probe manifest/cache would be ambiguous, leaky, or invalid."""


def _require_nonempty_string(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{context} requires non-empty string {key!r}")
    return value


def _require_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ManifestError(f"{context} must be a lowercase 64-character SHA256 hex digest")
    return value


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ManifestError("Cache metadata must contain JSON-serializable values") from error


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and exact CPU bytes for cache binding."""
    if not isinstance(tensor, torch.Tensor):
        raise ManifestError("Cannot hash a non-tensor value")
    contiguous = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _source_scene_key(source_dataset: str, source_scene_uid: str) -> str:
    return f"{source_dataset}::{source_scene_uid}"


@dataclass(frozen=True)
class ProbeManifestRecord:
    """An ordered latent/camera pair with immutable source binding fields.

    ``scene_id`` is human-readable only.  Split safety is enforced using the
    dataset-qualified ``source_scene_uid``, ``source_clip_uid``, verified
    source-content digest, and the actual cache digest, so renaming ``scene_id``
    cannot move the same source across train/val/test.
    """

    record_id: str
    scene_id: str
    source_dataset: str
    source_scene_uid: str
    source_clip_uid: str
    source_content_sha256: str
    cache_sha256: str
    split: str
    cache_path: Path
    source_pose_index: int
    target_pose_index: int
    source_latent_index: int
    target_latent_index: int
    static_scene: bool

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], manifest_root: Path) -> "ProbeManifestRecord":
        required = {
            "format_version",
            "record_id",
            "scene_id",
            "source_dataset",
            "source_scene_uid",
            "source_clip_uid",
            "source_content_sha256",
            "cache_sha256",
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
                f"Unsupported manifest format {payload['format_version']}; expected {MANIFEST_FORMAT_VERSION}"
            )
        split = str(payload["split"])
        if split not in _ALLOWED_SPLITS:
            raise ManifestError(f"split must be one of {sorted(_ALLOWED_SPLITS)}, got {split!r}")
        cache_path = Path(str(payload["cache_path"]))
        if not cache_path.is_absolute():
            cache_path = manifest_root / cache_path
        for key in ("source_pose_index", "target_pose_index", "source_latent_index", "target_latent_index"):
            if int(payload[key]) < 0:
                raise ManifestError(f"{key} must be non-negative, got {payload[key]!r}")
        if not isinstance(payload["static_scene"], bool):
            raise ManifestError("static_scene must be a JSON boolean")
        return cls(
            record_id=_require_nonempty_string(payload, "record_id", "manifest record"),
            scene_id=_require_nonempty_string(payload, "scene_id", "manifest record"),
            source_dataset=_require_nonempty_string(payload, "source_dataset", "manifest record"),
            source_scene_uid=_require_nonempty_string(payload, "source_scene_uid", "manifest record"),
            source_clip_uid=_require_nonempty_string(payload, "source_clip_uid", "manifest record"),
            source_content_sha256=_require_sha256(
                payload["source_content_sha256"], "manifest source_content_sha256"
            ),
            cache_sha256=_require_sha256(payload["cache_sha256"], "manifest cache_sha256"),
            split=split,
            cache_path=cache_path,
            source_pose_index=int(payload["source_pose_index"]),
            target_pose_index=int(payload["target_pose_index"]),
            source_latent_index=int(payload["source_latent_index"]),
            target_latent_index=int(payload["target_latent_index"]),
            static_scene=bool(payload["static_scene"]),
        )

    def to_json_dict(self, manifest_root: Path) -> dict[str, Any]:
        payload = asdict(self)
        try:
            payload["cache_path"] = str(self.cache_path.relative_to(manifest_root))
        except ValueError:
            payload["cache_path"] = str(self.cache_path)
        payload["format_version"] = MANIFEST_FORMAT_VERSION
        return payload


def validate_scene_disjoint_splits(records: Sequence[ProbeManifestRecord]) -> None:
    """Reject source/cache reuse across splits, even under renamed scene labels."""
    split_maps: tuple[tuple[str, dict[str, str], Any], ...] = (
        ("scene_id", {}, lambda record: record.scene_id),
        ("source_scene_uid", {}, lambda record: _source_scene_key(record.source_dataset, record.source_scene_uid)),
        ("source_clip_uid", {}, lambda record: f"{record.source_dataset}::{record.source_clip_uid}"),
        ("source_content_sha256", {}, lambda record: record.source_content_sha256),
        ("cache_sha256", {}, lambda record: record.cache_sha256),
    )
    for label, split_map, key_fn in split_maps:
        for record in records:
            identity = key_fn(record)
            existing = split_map.setdefault(identity, record.split)
            if existing != record.split:
                raise ManifestError(
                    "Scene-disjoint split violation: "
                    f"{label}={identity!r} appears in both {existing!r} and {record.split!r}"
                )


def load_manifest_records(manifest_path: str | Path) -> list[ProbeManifestRecord]:
    """Read JSONL manifest and run cheap, manifest-only split validation."""
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


def _validate_pose_spec(pose_spec: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(pose_spec, Mapping):
        raise ManifestError("pose_spec must be a mapping")
    result = dict(pose_spec)
    if result.get("convention") != "world_to_camera":
        raise ManifestError("pose_spec.convention must be 'world_to_camera'")
    if result.get("transform_type") != "SE3":
        raise ManifestError("pose_spec.transform_type must be 'SE3'")
    if result.get("coordinate_system") != "right_handed":
        raise ManifestError("pose_spec.coordinate_system must be 'right_handed'")
    return result


def _validate_source_provenance(
    source_provenance: Mapping[str, Any],
    frame_ids: torch.Tensor,
    camera_poses_w2c: torch.Tensor,
) -> dict[str, Any]:
    if not isinstance(source_provenance, Mapping):
        raise ManifestError("source_provenance must be a mapping")
    result = dict(source_provenance)
    required_strings = ("source_dataset", "source_scene_uid", "source_clip_uid")
    for key in required_strings:
        _require_nonempty_string(result, key, "source_provenance")
    for key in ("source_content_sha256", "source_frame_ids_sha256", "source_pose_sha256"):
        _require_sha256(result.get(key), f"source_provenance.{key}")
    if result["source_frame_ids_sha256"] != tensor_sha256(frame_ids):
        raise ManifestError("source_provenance.source_frame_ids_sha256 does not match cached frame_ids")
    if result["source_pose_sha256"] != tensor_sha256(camera_poses_w2c):
        raise ManifestError("source_provenance.source_pose_sha256 does not match cached camera_poses_w2c")
    return result


def _validate_temporal_mapping(
    temporal_mapping: Mapping[str, Any],
    z0: torch.Tensor,
    frame_ids: torch.Tensor,
) -> dict[str, Any]:
    if not isinstance(temporal_mapping, Mapping):
        raise ManifestError("temporal_mapping must be a mapping")
    result = dict(temporal_mapping)
    if result.get("mapping_type") != "latent_anchor_frame_id":
        raise ManifestError("temporal_mapping.mapping_type must be 'latent_anchor_frame_id'")
    if not isinstance(result.get("anchor_rule"), str) or not result["anchor_rule"].strip():
        raise ManifestError("temporal_mapping.anchor_rule must be a non-empty string")
    if not isinstance(result.get("is_causal"), bool):
        raise ManifestError("temporal_mapping.is_causal must be a boolean")
    if not isinstance(result.get("temporal_compression_ratio"), int) or result["temporal_compression_ratio"] < 1:
        raise ManifestError("temporal_mapping.temporal_compression_ratio must be a positive integer")
    anchor_frame_ids = result.get("latent_to_frame_ids")
    if not isinstance(anchor_frame_ids, torch.Tensor) or anchor_frame_ids.ndim != 1:
        raise ManifestError("temporal_mapping.latent_to_frame_ids must be a tensor [T_latent]")
    anchor_frame_ids = anchor_frame_ids.to(device="cpu", dtype=torch.long).contiguous()
    if anchor_frame_ids.numel() != z0.shape[1]:
        raise ManifestError(
            "temporal_mapping.latent_to_frame_ids length must equal z0 temporal size; "
            f"got {anchor_frame_ids.numel()} versus {z0.shape[1]}"
        )
    if frame_ids.ndim != 1 or frame_ids.dtype not in (torch.int32, torch.int64):
        raise ManifestError("frame_ids must be a one-dimensional integer tensor")
    frame_ids = frame_ids.to(device="cpu", dtype=torch.long)
    if frame_ids.unique().numel() != frame_ids.numel():
        raise ManifestError("frame_ids must be unique so latent anchors map unambiguously to camera poses")
    available_frame_ids = set(int(value) for value in frame_ids.tolist())
    missing = sorted(set(int(value) for value in anchor_frame_ids.tolist()).difference(available_frame_ids))
    if missing:
        raise ManifestError(
            "temporal_mapping.latent_to_frame_ids contains anchors absent from cached frame_ids: "
            f"{missing[:8]}"
        )
    result["latent_to_frame_ids"] = anchor_frame_ids
    return result


def _validate_scheduler_spec(scheduler: Any) -> None:
    if not isinstance(scheduler, Mapping):
        raise ManifestError("latent_spec.scheduler must be a mapping for this latent domain")
    for key in ("identifier", "config_sha256", "timestep_parameterization"):
        _require_nonempty_string(scheduler, key, "latent_spec.scheduler")
    _require_sha256(scheduler["config_sha256"], "latent_spec.scheduler.config_sha256")


def _validate_condition_spec(condition: Any) -> None:
    if not isinstance(condition, Mapping):
        raise ManifestError("latent_spec.condition must be a mapping for x0_pred")
    for key in ("model_identifier", "model_revision", "prompt_sha256", "conditioning_image_sha256", "cfg_scale"):
        if key not in condition:
            raise ManifestError(f"latent_spec.condition is missing {key!r}")
    _require_nonempty_string(condition, "model_identifier", "latent_spec.condition")
    _require_nonempty_string(condition, "model_revision", "latent_spec.condition")
    _require_sha256(condition["prompt_sha256"], "latent_spec.condition.prompt_sha256")
    _require_sha256(condition["conditioning_image_sha256"], "latent_spec.condition.conditioning_image_sha256")
    if not isinstance(condition["cfg_scale"], (int, float)):
        raise ManifestError("latent_spec.condition.cfg_scale must be numeric")


def _validate_latent_spec(latent_spec: Mapping[str, Any], temporal_mapping: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(latent_spec, Mapping):
        raise ManifestError("latent_spec must be a mapping")
    result = copy.deepcopy(dict(latent_spec))
    domain = result.get("domain")
    if domain not in _LATENT_DOMAINS:
        raise ManifestError(f"latent_spec.domain must be one of {sorted(_LATENT_DOMAINS)}, got {domain!r}")
    _require_nonempty_string(result, "model_family", "latent_spec")
    vae = result.get("vae")
    if not isinstance(vae, Mapping):
        raise ManifestError("latent_spec.vae must be a mapping")
    for key in ("identifier", "revision", "latent_scaling"):
        _require_nonempty_string(vae, key, "latent_spec.vae")
    if domain == LATENT_DOMAIN_RAW_VAE_Z0:
        for key in ("latents_mean", "latents_std"):
            values = vae.get(key)
            if not isinstance(values, list) or not values or not all(
                isinstance(value, (int, float)) for value in values
            ):
                raise ManifestError(f"raw_vae_z0 latent_spec.vae.{key} must be a non-empty numeric list")
        if len(vae["latents_mean"]) != len(vae["latents_std"]):
            raise ManifestError("latent_spec.vae latents_mean and latents_std lengths must match")
        if not all(math.isfinite(float(value)) for value in vae["latents_mean"]):
            raise ManifestError("latent_spec.vae.latents_mean values must be finite")
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in vae["latents_std"]):
            raise ManifestError("latent_spec.vae.latents_std values must be finite and positive")
    preprocessing = result.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise ManifestError("latent_spec.preprocessing must be a mapping")
    for key in ("image_normalization", "height", "width", "fps", "frame_sampling"):
        if key not in preprocessing:
            raise ManifestError(f"latent_spec.preprocessing is missing {key!r}")
    if not isinstance(preprocessing["height"], int) or preprocessing["height"] <= 0:
        raise ManifestError("latent_spec.preprocessing.height must be a positive integer")
    if not isinstance(preprocessing["width"], int) or preprocessing["width"] <= 0:
        raise ManifestError("latent_spec.preprocessing.width must be a positive integer")
    if not isinstance(preprocessing["fps"], (int, float)) or preprocessing["fps"] <= 0:
        raise ManifestError("latent_spec.preprocessing.fps must be positive")
    if result.get("temporal_mapping_type") != temporal_mapping["mapping_type"]:
        raise ManifestError("latent_spec.temporal_mapping_type must match temporal_mapping.mapping_type")

    scheduler = result.get("scheduler")
    condition = result.get("condition")
    if domain == LATENT_DOMAIN_RAW_VAE_Z0:
        if scheduler is not None or condition is not None:
            raise ManifestError("raw_vae_z0 cache must use scheduler=None and condition=None")
    elif domain == LATENT_DOMAIN_NORMALIZED_DIFFUSION_Z:
        _validate_scheduler_spec(scheduler)
        if condition is not None and not isinstance(condition, Mapping):
            raise ManifestError("normalized_diffusion_z condition must be null or a mapping")
    else:
        _validate_scheduler_spec(scheduler)
        _validate_condition_spec(condition)
    return result


def compute_cache_sha256(
    *,
    z0: torch.Tensor,
    camera_poses_w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    frame_ids: torch.Tensor,
    source_provenance: Mapping[str, Any],
    pose_spec: Mapping[str, Any],
    latent_spec: Mapping[str, Any],
    temporal_mapping: Mapping[str, Any],
) -> str:
    """Compute immutable cache identity from tensors and validated metadata."""
    digest = hashlib.sha256()
    digest.update(b"latent_geometry_cache_v3")
    for name, tensor in (
        ("z0", z0),
        ("camera_poses_w2c", camera_poses_w2c),
        ("intrinsics", intrinsics),
        ("frame_ids", frame_ids),
        ("latent_to_frame_ids", temporal_mapping["latent_to_frame_ids"]),
    ):
        digest.update(name.encode("ascii"))
        digest.update(tensor_sha256(tensor).encode("ascii"))
    temporal_mapping_metadata = {
        key: value for key, value in temporal_mapping.items() if key != "latent_to_frame_ids"
    }
    for name, metadata in (
        ("source_provenance", source_provenance),
        ("pose_spec", pose_spec),
        ("latent_spec", latent_spec),
        ("temporal_mapping", temporal_mapping_metadata),
    ):
        digest.update(name.encode("ascii"))
        digest.update(_canonical_json_bytes(dict(metadata)))
    return digest.hexdigest()


def save_clean_latent_record(
    path: str | Path,
    *,
    record_id: str,
    scene_id: str,
    z0: torch.Tensor,
    camera_poses_w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    frame_ids: torch.Tensor,
    source_provenance: Mapping[str, Any],
    latent_spec: Mapping[str, Any],
    temporal_mapping: Mapping[str, Any],
    pose_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one versioned raw-z0 cache and return its immutable binding info.

    This is a cache writer, not a Wan/VAE extractor.  Callers must obtain the
    tensors and source-content digest themselves.  The strict metadata contract
    makes that extraction step traceable when it is added later.
    """
    path = Path(path)
    if z0.ndim != 4 or not torch.is_floating_point(z0) or not torch.isfinite(z0).all():
        raise ValueError("z0 must be a finite floating tensor [C, T_latent, H_latent, W_latent]")
    if camera_poses_w2c.ndim != 3 or camera_poses_w2c.shape[-2:] != (4, 4):
        raise ValueError("camera_poses_w2c must have shape [num_frames, 4, 4]")
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape [num_frames, 3, 3]")
    if intrinsics.shape[0] != camera_poses_w2c.shape[0]:
        raise ValueError("intrinsics and camera_poses_w2c must have the same frame count")
    if frame_ids.ndim != 1 or frame_ids.shape[0] != camera_poses_w2c.shape[0]:
        raise ValueError("frame_ids must have shape [num_frames]")

    z0_cpu = z0.detach().to(device="cpu").contiguous()
    poses_cpu = camera_poses_w2c.detach().to(device="cpu", dtype=torch.float32).contiguous()
    intrinsics_cpu = intrinsics.detach().to(device="cpu", dtype=torch.float32).contiguous()
    frame_ids_cpu = frame_ids.detach().to(device="cpu", dtype=torch.long).contiguous()
    validate_world_to_camera_se3(poses_cpu)
    validated_pose_spec = _validate_pose_spec(
        pose_spec
        or {
            "convention": "world_to_camera",
            "transform_type": "SE3",
            "coordinate_system": "right_handed",
        }
    )
    validated_temporal_mapping = _validate_temporal_mapping(temporal_mapping, z0_cpu, frame_ids_cpu)
    validated_source_provenance = _validate_source_provenance(source_provenance, frame_ids_cpu, poses_cpu)
    validated_latent_spec = _validate_latent_spec(latent_spec, validated_temporal_mapping)
    cache_sha256 = compute_cache_sha256(
        z0=z0_cpu,
        camera_poses_w2c=poses_cpu,
        intrinsics=intrinsics_cpu,
        frame_ids=frame_ids_cpu,
        source_provenance=validated_source_provenance,
        pose_spec=validated_pose_spec,
        latent_spec=validated_latent_spec,
        temporal_mapping=validated_temporal_mapping,
    )
    payload = {
        "format_version": CACHE_FORMAT_VERSION,
        "record_id": record_id,
        "scene_id": scene_id,
        "z0": z0_cpu,
        "camera_poses_w2c": poses_cpu,
        "intrinsics": intrinsics_cpu,
        "frame_ids": frame_ids_cpu,
        "source_provenance": validated_source_provenance,
        "pose_spec": validated_pose_spec,
        "latent_spec": validated_latent_spec,
        "temporal_mapping": validated_temporal_mapping,
        "cache_sha256": cache_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "record_id": record_id,
        "scene_id": scene_id,
        "source_dataset": validated_source_provenance["source_dataset"],
        "source_scene_uid": validated_source_provenance["source_scene_uid"],
        "source_clip_uid": validated_source_provenance["source_clip_uid"],
        "source_content_sha256": validated_source_provenance["source_content_sha256"],
        "cache_sha256": cache_sha256,
    }


def _load_clean_latent_record(
    path: Path,
    *,
    expected_model_family: str | None,
    expected_latent_domain: str,
    verify_cache_hash: bool,
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except FileNotFoundError as error:
        raise ManifestError(f"Cached latent file does not exist: {path}") from error
    if not isinstance(payload, dict):
        raise ManifestError(f"Cached latent record must be a dict: {path}")
    required = {
        "format_version",
        "record_id",
        "scene_id",
        "z0",
        "camera_poses_w2c",
        "intrinsics",
        "frame_ids",
        "source_provenance",
        "pose_spec",
        "latent_spec",
        "temporal_mapping",
        "cache_sha256",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ManifestError(f"Cached latent record {path} is missing keys: {missing}")
    if int(payload["format_version"]) != CACHE_FORMAT_VERSION:
        raise ManifestError(
            f"Unsupported cache format {payload['format_version']} in {path}; expected {CACHE_FORMAT_VERSION}"
        )
    z0 = payload["z0"]
    poses = payload["camera_poses_w2c"]
    intrinsics = payload["intrinsics"]
    frame_ids = payload["frame_ids"]
    if not isinstance(z0, torch.Tensor) or z0.ndim != 4 or not torch.is_floating_point(z0) or not torch.isfinite(z0).all():
        raise ManifestError(f"z0 in {path} must be finite float tensor [C, T, H, W]")
    if not isinstance(poses, torch.Tensor):
        raise ManifestError(f"camera_poses_w2c in {path} must be a tensor")
    if not isinstance(intrinsics, torch.Tensor) or intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ManifestError(f"intrinsics in {path} must be [num_frames, 3, 3]")
    if not isinstance(frame_ids, torch.Tensor) or frame_ids.ndim != 1:
        raise ManifestError(f"frame_ids in {path} must be a one-dimensional tensor")
    if poses.shape[0] != intrinsics.shape[0] or poses.shape[0] != frame_ids.shape[0]:
        raise ManifestError(f"Pose, intrinsics, and frame_ids counts disagree in {path}")
    poses = poses.float().contiguous()
    intrinsics = intrinsics.float().contiguous()
    frame_ids = frame_ids.long().contiguous()
    validate_world_to_camera_se3(poses)
    pose_spec = _validate_pose_spec(payload["pose_spec"])
    temporal_mapping = _validate_temporal_mapping(payload["temporal_mapping"], z0, frame_ids)
    source_provenance = _validate_source_provenance(payload["source_provenance"], frame_ids, poses)
    latent_spec = _validate_latent_spec(payload["latent_spec"], temporal_mapping)
    if expected_model_family is not None and latent_spec["model_family"] != expected_model_family:
        raise ManifestError(
            f"Cached record {path} is model_family={latent_spec['model_family']!r}, "
            f"expected {expected_model_family!r}"
        )
    if latent_spec["domain"] != expected_latent_domain:
        raise ManifestError(
            f"Cached record {path} has latent domain={latent_spec['domain']!r}, "
            f"expected {expected_latent_domain!r}"
        )
    cache_sha256 = _require_sha256(payload["cache_sha256"], f"cache_sha256 in {path}")
    if verify_cache_hash:
        expected_hash = compute_cache_sha256(
            z0=z0,
            camera_poses_w2c=poses,
            intrinsics=intrinsics,
            frame_ids=frame_ids,
            source_provenance=source_provenance,
            pose_spec=pose_spec,
            latent_spec=latent_spec,
            temporal_mapping=temporal_mapping,
        )
        if cache_sha256 != expected_hash:
            raise ManifestError(f"Cache SHA256 mismatch for {path}; cache metadata or tensor data was modified")
    payload["z0"] = z0.float().contiguous()
    payload["camera_poses_w2c"] = poses
    payload["intrinsics"] = intrinsics
    payload["frame_ids"] = frame_ids
    payload["source_provenance"] = source_provenance
    payload["pose_spec"] = pose_spec
    payload["latent_spec"] = latent_spec
    payload["temporal_mapping"] = temporal_mapping
    return payload


def _validate_manifest_cache_binding(record: ProbeManifestRecord, payload: Mapping[str, Any]) -> None:
    provenance = payload["source_provenance"]
    expected = {
        "source_dataset": record.source_dataset,
        "source_scene_uid": record.source_scene_uid,
        "source_clip_uid": record.source_clip_uid,
        "source_content_sha256": record.source_content_sha256,
        "cache_sha256": record.cache_sha256,
    }
    actual = {
        "source_dataset": provenance["source_dataset"],
        "source_scene_uid": provenance["source_scene_uid"],
        "source_clip_uid": provenance["source_clip_uid"],
        "source_content_sha256": provenance["source_content_sha256"],
        "cache_sha256": payload["cache_sha256"],
    }
    for key, expected_value in expected.items():
        if actual[key] != expected_value:
            raise ManifestError(
                f"Manifest/cache provenance mismatch for {record.record_id!r}: "
                f"{key}={expected_value!r} in manifest, {actual[key]!r} in cache"
            )


def _validate_record_temporal_mapping(record: ProbeManifestRecord, payload: Mapping[str, Any]) -> None:
    frame_ids = payload["frame_ids"]
    mapping = payload["temporal_mapping"]["latent_to_frame_ids"]
    num_poses = frame_ids.numel()
    num_latents = mapping.numel()
    if record.source_pose_index >= num_poses or record.target_pose_index >= num_poses:
        raise ManifestError(f"Pose index is outside cached pose count {num_poses} for {record.record_id!r}")
    if record.source_latent_index >= num_latents or record.target_latent_index >= num_latents:
        raise ManifestError(f"Latent index is outside cached latent count {num_latents} for {record.record_id!r}")
    expected_source_frame = int(mapping[record.source_latent_index].item())
    expected_target_frame = int(mapping[record.target_latent_index].item())
    actual_source_frame = int(frame_ids[record.source_pose_index].item())
    actual_target_frame = int(frame_ids[record.target_pose_index].item())
    if expected_source_frame != actual_source_frame or expected_target_frame != actual_target_frame:
        raise ManifestError(
            "Manifest RGB-pose to latent mapping mismatch for "
            f"{record.record_id!r}: latent anchors ({expected_source_frame}, {expected_target_frame}) "
            f"do not equal pose frame IDs ({actual_source_frame}, {actual_target_frame})"
        )


@dataclass(frozen=True)
class LinearFlowNoiseSchedule:
    """Probe-only online noising: ``zt = (1 - t) z0 + t epsilon``.

    It is not a replacement for Wan's scheduler.  This branch only accepts
    cached ``raw_vae_z0``; future ``x0_pred`` support requires a separate,
    scheduler-aware extraction branch.
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
    """Read verified raw-z0 cache pairs and construct deterministic online zt.

    Dataset construction validates *all* manifest cache bindings, not only the
    requested split.  This catches a malicious or accidental cross-split reuse
    where a cache is relabelled with a different manifest scene string.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        noise_schedule: LinearFlowNoiseSchedule | None = None,
        base_seed: int = 0,
        expected_model_family: str | None = "wan",
        expected_latent_domain: str = LATENT_DOMAIN_RAW_VAE_Z0,
        require_static_scene: bool = True,
    ) -> None:
        if split not in _ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {sorted(_ALLOWED_SPLITS)}")
        if expected_latent_domain != LATENT_DOMAIN_RAW_VAE_Z0:
            raise ValueError(
                "This foundation only trains z0/online-zt probes from raw_vae_z0 caches. "
                "Use a separate extractor/branch before probing x0_pred."
            )
        all_records = load_manifest_records(manifest_path)
        cache_payloads: dict[Path, dict[str, Any]] = {}
        for record in all_records:
            payload = cache_payloads.get(record.cache_path)
            if payload is None:
                payload = _load_clean_latent_record(
                    record.cache_path,
                    expected_model_family=expected_model_family,
                    expected_latent_domain=expected_latent_domain,
                    verify_cache_hash=True,
                )
                cache_payloads[record.cache_path] = payload
            _validate_manifest_cache_binding(record, payload)
            _validate_record_temporal_mapping(record, payload)

        records = [record for record in all_records if record.split == split]
        if require_static_scene:
            records = [record for record in records if record.static_scene]
        if not records:
            raise ManifestError(f"No records available for split={split!r}")
        self.records = records
        self.noise_schedule = noise_schedule or LinearFlowNoiseSchedule()
        self.base_seed = int(base_seed)
        self.expected_model_family = expected_model_family
        self.expected_latent_domain = expected_latent_domain
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Change deterministic online-noise draws without changing data splits."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        payload = _load_clean_latent_record(
            record.cache_path,
            expected_model_family=self.expected_model_family,
            expected_latent_domain=self.expected_latent_domain,
            verify_cache_hash=False,
        )
        _validate_manifest_cache_binding(record, payload)
        _validate_record_temporal_mapping(record, payload)
        z0_full = payload["z0"]
        z0_pair = z0_full[:, [record.source_latent_index, record.target_latent_index]].contiguous()
        vae_spec = payload["latent_spec"]["vae"]
        if len(vae_spec["latents_mean"]) != z0_pair.shape[0]:
            raise ManifestError(
                f"Wan latent normalization has {len(vae_spec['latents_mean'])} channels, "
                f"but cache contains {z0_pair.shape[0]}"
            )
        latent_mean = torch.tensor(vae_spec["latents_mean"], dtype=z0_pair.dtype).view(-1, 1, 1, 1)
        latent_std = torch.tensor(vae_spec["latents_std"], dtype=z0_pair.dtype).view(-1, 1, 1, 1)
        diffusion_z0_pair = (z0_pair - latent_mean) / latent_std
        pose_target = make_relative_pose_target(
            payload["camera_poses_w2c"], record.source_pose_index, record.target_pose_index
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_seed(self.base_seed, self.epoch, record.record_id))
        # Wan flow matching operates on channel-normalized VAE latents.
        zt_pair, timestep, _ = self.noise_schedule.sample(diffusion_z0_pair, generator)
        return {
            "z0": z0_pair,
            "z0_timestep": torch.zeros((), dtype=torch.float32),
            "diffusion_z0": diffusion_z0_pair,
            "diffusion_z0_timestep": torch.zeros((), dtype=torch.float32),
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
