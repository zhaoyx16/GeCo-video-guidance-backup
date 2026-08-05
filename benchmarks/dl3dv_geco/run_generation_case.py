#!/usr/bin/env python3
"""Generate one paired DL3DV benchmark sample with Wan or Cosmos.

Baseline runs use the official Diffusers pipeline. GeCo runs use the isolated
custom port after it has passed the runtime-equivalence gate. Every output has
a sidecar JSON containing enough provenance to reject stale or mismatched
paired comparisons.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib.util
import json
import math
import os
import platform
import socket
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import export_to_video
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_selection.protocol import (
    file_sha256 as protocol_file_sha256,
    resolve_protocol_image,
    resolve_protocol_transforms,
    validate_committed_file,
    validate_experiment_lock,
    validate_formal_protocol,
)
from geometry_selection.model_lock import load_model_lock, verify_generation_model
from geometry_selection.generation_lock import GenerationRunLock
from geometry_selection.selection import validate_candidate_spec
from geometry_selection.video_probe import probe_video
from geometry_selection.online import OnlineGeometrySelectionController
from geometry_selection.online_config import (
    OnlineSelectionRunConfig,
    load_online_selection_config,
)
from geometry_selection.online_vggt import (
    OnlinePoseGraphScorerConfig,
    OnlineVGGTPoseGraphScorer,
)
from geometry_selection.backbones.vggt_omega import VGGTOmegaAdapter


WAN_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

COSMOS_NEGATIVE = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
    "pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color "
    "balance, washed out colors, choppy sequences, jerky movements, low frame rate, "
    "artifacting, color banding, unnatural transitions, outdated special effects, fake "
    "elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and "
    "flickering. Overall, the video is of poor quality."
)

PROFILES = {
    "wan": {
        "steps": 50,
        "frames": 121,
        "height": 704,
        "width": 1280,
        "fps": 24,
        "guidance_scale": 5.0,
    },
    "cosmos": {
        "steps": 36,
        "frames": 93,
        "height": 704,
        "width": 1280,
        "fps": 16,
        "guidance_scale": 7.0,
    },
}


SPLIT_VAE_RUNTIME_RECEIPT_SCHEMA = "wan_split_vae_runtime_receipt_v1"
MODEL_CONTENT_MANIFEST_SCHEMA = "geco-model-content-manifest-v1"
SPLIT_VAE_BOOTSTRAP_PROFILE = {
    "steps": 50,
    "frames": 121,
    "height": 704,
    "width": 1280,
    "fps": 24,
    "guidance_scale": 5.0,
    "wan_negative_prompt_mode": "none",
}


def load_class(path: Path, class_name: str):
    spec = importlib.util.spec_from_file_location(f"_geco_benchmark_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_commit(model: str) -> str | None:
    parts = Path(model).resolve().parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    return parts[index + 1] if index + 1 < len(parts) else None


def model_identity(model: str) -> dict:
    path = Path(model).expanduser()
    identity = {
        "requested": model,
        "resolved": str(path.resolve()) if path.exists() else model,
        "snapshot_commit": snapshot_commit(model) if path.exists() else None,
    }
    if path.is_dir():
        records = []
        for candidate in sorted(path.rglob("*.json")):
            records.append(
                {
                    "path": str(candidate.relative_to(path)),
                    "sha256": sha256_file(candidate),
                }
            )
        identity["json_files"] = records
        identity["weight_files"] = [
            {
                "path": str(candidate.relative_to(path)),
                "size": candidate.stat().st_size,
            }
            for pattern in ("*.safetensors", "*.bin")
            for candidate in sorted(path.rglob(pattern))
        ]
    return identity


def require_sha256(value: str, *, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"{field} must be a lower-case SHA-256 digest")


def canonical_json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def validate_immutable_model_tree(root: Path, records: list[dict], *, role: str) -> None:
    """Re-hash every file and reject writable/symlinked tree changes before loading."""
    expected = {}
    for record in records:
        relative = record.get("relative_path")
        size_bytes = record.get("size_bytes")
        digest = record.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not isinstance(digest, str)
            or relative in expected
        ):
            raise RuntimeError(f"model-content manifest file record is invalid for {role}")
        expected[relative] = {"size_bytes": size_bytes, "sha256": digest}
    observed = {}
    for directory, directory_names, filenames in os.walk(str(root), followlinks=False):
        directory_path = Path(directory)
        if directory_path.is_symlink() or os.access(directory_path, os.W_OK):
            raise RuntimeError(f"{role} frozen model tree contains writable/symlink directory")
        for directory_name in directory_names:
            if (directory_path / directory_name).is_symlink():
                raise RuntimeError(f"{role} frozen model tree contains a symlink directory")
        for filename in filenames:
            child = directory_path / filename
            if child.is_symlink() or not child.is_file() or os.access(child, os.W_OK):
                raise RuntimeError(f"{role} frozen model tree contains writable/symlink file")
            relative = child.relative_to(root).as_posix()
            observed[relative] = {
                "size_bytes": child.stat().st_size,
                "sha256": sha256_file(child),
            }
    if observed != expected:
        raise RuntimeError(f"{role} frozen model file set or sizes differ from content manifest")


def load_verified_model_content_manifest(
    path: Path,
    *,
    expected_sha256: str,
    requested_models: dict[str, str],
) -> dict:
    """Bind every adapted-GeCo model path to a full, immutable content manifest.

    The manifest producer hashes every model byte before publication.  Runtime
    validation then requires the published SHA, the expected three roles, the
    exact resolved model roots, and non-writable roots before any loader runs.
    """

    require_sha256(expected_sha256, field="expected model-content manifest SHA")
    manifest_path = path.resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"model-content manifest is missing: {manifest_path}")
    if sha256_file(manifest_path) != expected_sha256:
        raise RuntimeError("model-content manifest SHA mismatch")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != MODEL_CONTENT_MANIFEST_SCHEMA:
        raise RuntimeError("model-content manifest schema mismatch")
    content_sha256 = payload.get("content_sha256")
    if not isinstance(content_sha256, str):
        raise RuntimeError("model-content manifest has no content SHA")
    require_sha256(content_sha256, field="model-content manifest content SHA")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("model-content manifest artifacts are invalid")
    content_body = {
        "schema": MODEL_CONTENT_MANIFEST_SCHEMA,
        "artifacts": [
            {
                "name": artifact.get("name"),
                "file_count": artifact.get("file_count"),
                "total_bytes": artifact.get("total_bytes"),
                "files": artifact.get("files"),
            }
            for artifact in artifacts
            if isinstance(artifact, dict)
        ],
    }
    if hashlib.sha256(canonical_json_bytes(content_body)).hexdigest() != content_sha256:
        raise RuntimeError("model-content manifest aggregate content SHA mismatch")
    by_role = {artifact.get("name"): artifact for artifact in artifacts if isinstance(artifact, dict)}
    if set(by_role) != {"wan", "vggt_omega", "ufm"}:
        raise RuntimeError("model-content manifest must contain exactly Wan, VGGT-Omega, and UFM")
    verified_roots = {}
    for role, requested in requested_models.items():
        artifact = by_role[role]
        raw_root = artifact.get("resolved_root")
        if not isinstance(raw_root, str) or not raw_root:
            raise RuntimeError(f"model-content manifest root is invalid for {role}")
        root = Path(raw_root).resolve()
        requested_path = Path(requested).expanduser().resolve()
        if requested_path != root:
            raise RuntimeError(
                f"{role} model path differs from frozen manifest root: "
                f"{requested_path} != {root}"
            )
        if not root.is_dir() or os.access(root, os.W_OK):
            raise RuntimeError(f"{role} frozen model root is missing or writable: {root}")
        files = artifact.get("files")
        if not isinstance(files, list) or not files:
            raise RuntimeError(f"model-content manifest has no file records for {role}")
        validate_immutable_model_tree(root, files, role=role)
        verified_roots[role] = str(root)
    return {
        "path": str(manifest_path),
        "sha256": expected_sha256,
        "content_sha256": content_sha256,
        "verified_roots": verified_roots,
    }


def mapped_fixed_frames(num_frames: int) -> list[int]:
    # Original GeCo uses 1-based 12,24,36,48 over 49 CogVideoX frames.
    fractions = (11 / 48, 23 / 48, 35 / 48, 47 / 48)
    return sorted({min(num_frames - 1, round(fraction * (num_frames - 1))) for fraction in fractions})


def resolve_adapted_geco_schedule(
    spec: dict,
    schedule_id: str,
) -> tuple[list[int], list[int], list[float], dict]:
    """Resolve one pre-registered Wan adaptation; no CLI schedule is permitted."""

    adapted = spec["adapted_geco"]
    schedule = adapted["schedule_candidates"][schedule_id]
    fixed_frames = list(adapted["fixed_frame_indices"])
    guidance_step = list(schedule["guidance_step"])
    learning_rate = schedule["guidance_learning_rate"]
    guidance_lr = [learning_rate if repeats else 0.0 for repeats in guidance_step]
    return fixed_frames, guidance_step, guidance_lr, {
        "id": schedule_id,
        "description": schedule["description"],
        "updates": schedule["updates"],
        "time_travel": adapted["time_travel"],
        "time_travel_note": adapted["time_travel_note"],
    }


def load_case(
    manifest_path: Path,
    case_id: str | None,
    case_index: int | None,
) -> tuple[str, dict, dict]:
    payload = json.loads(manifest_path.read_text())
    metadata = payload.get("_meta", {})
    cases = [(key, value) for key, value in payload.items() if not key.startswith("_")]
    if case_id is not None:
        if case_id not in payload or case_id.startswith("_"):
            raise KeyError(f"Unknown case id: {case_id}")
        return case_id, payload[case_id], metadata
    if case_index is None or not 0 <= case_index < len(cases):
        raise IndexError(f"case_index must be in [0, {len(cases) - 1}]")
    selected_id, selected_case = cases[case_index]
    return selected_id, selected_case, metadata


def git_identity(repo: Path) -> dict:
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def load_metric(args: argparse.Namespace):
    from demo_guidance_fast_failure import _get_compute_dtype_for_vggt, make_motion_metric
    from uniflowmatch.models.ufm import UniFlowMatchConfidence
    from vggt.models.vggt import VGGT

    metric_device = torch.device(args.metric_device)
    compute_dtype = _get_compute_dtype_for_vggt(metric_device)
    vggt = VGGT.from_pretrained(args.vggt_model).to(metric_device).eval()
    ufm = (
        UniFlowMatchConfidence.from_pretrained(args.ufm_model)
        .to(dtype=torch.float32, device=metric_device)
        .eval()
    )
    vggt.requires_grad_(False)
    ufm.requires_grad_(False)
    metric = make_motion_metric(
        vggt,
        ufm,
        metric_device,
        compute_dtype,
        vggt_strategy="once",
        pair_mode="adjacent",
        ufm_scale=args.ufm_scale,
        cov_thresh=0.5,
        percentile_val=20,
        min_threshold=0.2,
        grad_through_vggt=False,
        debug_autograd=False,
    )

    def metric_on_device(frames_01):
        if (
            args.cross_device_grad_via_cpu
            and torch.is_grad_enabled()
            and frames_01.requires_grad
            and frames_01.device != metric_device
        ):
            frames_01 = frames_01.to("cpu").to(metric_device)
        else:
            frames_01 = frames_01.to(metric_device)
        return metric(frames_01)

    return metric_on_device


def place_vae(
    pipe,
    vae_device: str | None,
    pipe_device: str,
    *,
    backbone: str,
    method: str,
    allow_split_vae: bool,
) -> torch.device:
    requested = torch.device(vae_device or pipe_device)
    current = next(pipe.vae.parameters()).device
    if requested != current:
        if method == "baseline":
            raise ValueError("Official baseline pipelines do not support split-VAE placement")
        if not allow_split_vae:
            raise ValueError("Split-VAE placement requires --allow-split-vae and its runtime smoke test")
        if backbone == "wan" and not hasattr(pipe, "_get_geco_vae_device"):
            raise RuntimeError("Wan custom pipeline lacks the required VAE routing API")
        pipe.vae.to("cpu")
        if current.type == "cuda":
            with torch.cuda.device(current):
                torch.cuda.empty_cache()
        pipe.vae.to(requested)
    if hasattr(pipe, "_get_geco_vae_device"):
        pipe._geco_vae_device = requested
    actual = next(pipe.vae.parameters()).device
    if actual != requested:
        raise RuntimeError(f"VAE placement failed: requested={requested}, actual={actual}")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    return requested


def _resolve_online_path(value: str, *, config_path: Path, repo: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    from_config = (config_path.parent / path).resolve()
    if from_config.exists():
        return from_config
    return (repo / path).resolve()


def make_wan_provisional_keyframe_decoder(pipe, expected_frames: int):
    """Decode a predicted clean Wan latent without exporting a temporary video."""

    def decode_keyframes(x0: torch.Tensor, indices, output_dir: Path) -> dict[int, Path]:
        if x0.ndim != 5 or x0.shape[0] != 1:
            raise ValueError("online Wan provisional decode requires one [1,C,T,H,W] latent")
        if output_dir.exists():
            if any(output_dir.iterdir()):
                raise FileExistsError(f"provisional frame directory is not empty: {output_dir}")
        else:
            output_dir.mkdir(parents=True)
        vae_device = next(pipe.vae.parameters()).device
        vae_dtype = pipe.vae.dtype
        latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(
            1, pipe.vae.config.z_dim, 1, 1, 1
        ).to(vae_device, vae_dtype)
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
            1, pipe.vae.config.z_dim, 1, 1, 1
        ).to(vae_device, vae_dtype)
        with torch.inference_mode():
            vae_latents = x0.to(device=vae_device, dtype=vae_dtype)
            vae_latents = vae_latents / latents_std + latents_mean
            decoded = pipe.vae.decode(vae_latents, return_dict=False)[0]
            processed = pipe.video_processor.postprocess_video(decoded, output_type="np")
        if hasattr(pipe.vae, "clear_cache"):
            pipe.vae.clear_cache()
        frames = processed[0] if isinstance(processed, (list, tuple)) else processed
        if isinstance(frames, torch.Tensor):
            frames = frames.detach().cpu().numpy()
        frames = np.asarray(frames)
        if frames.ndim == 5 and frames.shape[0] == 1:
            frames = frames[0]
        if frames.ndim != 4:
            raise ValueError(f"unexpected decoded provisional video shape: {frames.shape}")
        if frames.shape[-1] != 3 and frames.shape[0] == 3:
            frames = np.moveaxis(frames, 0, -1)
        if frames.shape[-1] != 3 or frames.shape[0] != expected_frames:
            raise ValueError(
                "decoded provisional video does not match expected [T,H,W,3]: "
                f"{frames.shape}, expected T={expected_frames}"
            )
        if frames.dtype != np.uint8:
            scale = 255.0 if float(np.nanmax(frames)) <= 1.0 else 1.0
            frames = np.clip(frames * scale, 0.0, 255.0).astype(np.uint8)
        paths: dict[int, Path] = {}
        for index in indices:
            if not 0 <= int(index) < expected_frames:
                raise IndexError(f"provisional keyframe index out of range: {index}")
            path = output_dir / f"frame_{int(index):06d}.png"
            Image.fromarray(np.ascontiguousarray(frames[int(index)])).save(path)
            paths[int(index)] = path
        return paths

    return decode_keyframes


def build_online_vggt_adapter(
    args: argparse.Namespace,
    config: OnlineSelectionRunConfig,
    config_path: Path,
) -> VGGTOmegaAdapter:
    if args.backbone != "wan":
        raise ValueError("online geometry selection is currently implemented only for Wan")
    source_root = _resolve_online_path(
        config.vggt_omega.source_root,
        config_path=config_path,
        repo=args.repo.resolve(),
    )
    checkpoint = _resolve_online_path(
        config.vggt_omega.checkpoint,
        config_path=config_path,
        repo=args.repo.resolve(),
    )
    return VGGTOmegaAdapter(
        source_root=source_root,
        checkpoint=checkpoint,
        device=config.vggt_omega.device,
        image_resolution=config.vggt_omega.image_resolution,
        preprocessing_mode=config.vggt_omega.preprocessing_mode,
    )


def build_online_geometry_selector(
    args: argparse.Namespace,
    pipe,
    config: OnlineSelectionRunConfig,
    config_path: Path,
    output_dir: Path,
    adapter: VGGTOmegaAdapter,
    geometry_identity: dict,
):
    scorer = OnlineVGGTPoseGraphScorer(
        adapter=adapter,
        decode_keyframes=make_wan_provisional_keyframe_decoder(pipe, args.frames),
        work_root=output_dir / "online_provisionals",
        config=OnlinePoseGraphScorerConfig(
            total_frames=args.frames,
            extraction=config.geometry_extraction,
            scorer=config.scorer,
            graph_score=config.graph_score,
            retain_frames=config.provisional.retain_frames,
        ),
    )
    controller = OnlineGeometrySelectionController(
        config.branch,
        config.selection,
        scorer,
    )
    return controller, {
        "config": config.resolved_dict(),
        "config_hash": config.config_hash,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "geometry_backbone": geometry_identity,
    }


def build_pipeline(args: argparse.Namespace):
    repo = args.repo.resolve()
    if args.backbone == "wan":
        from diffusers import AutoencoderKLWan

        if args.method == "baseline":
            from diffusers import WanImageToVideoPipeline as Pipeline
        else:
            Pipeline = load_class(
                repo / "external/guidance_wan/pipeline_wan_i2v_full_guided.py",
                "WanImageToVideoPipeline",
            )
        vae = AutoencoderKLWan.from_pretrained(
            args.model, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = Pipeline.from_pretrained(
            args.model, vae=vae, torch_dtype=torch.bfloat16
        ).to(args.pipe_device)
    else:
        if importlib.util.find_spec("cosmos_guardrail") is None:
            raise RuntimeError(
                "Formal Cosmos video generation requires cosmos_guardrail; "
                "the latent-only equivalence stub is intentionally not used here."
            )
        if args.method == "baseline":
            from diffusers import Cosmos2_5_PredictBasePipeline as Pipeline
        else:
            Pipeline = load_class(
                repo / "external/guidance_cosmos/pipeline_cosmos2_5_predict_guided.py",
                "Cosmos2_5_PredictBasePipeline",
            )
        pipe = Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(
            args.pipe_device
        )

    transformer_block_checkpointing_actual = False
    if args.method == "adapted_geco" and args.transformer_block_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
        transformer_block_checkpointing_actual = bool(
            getattr(pipe.transformer, "is_gradient_checkpointing", False)
        )
        if not transformer_block_checkpointing_actual:
            raise RuntimeError("Wan transformer did not enable gradient checkpointing")
    vae_device = place_vae(
        pipe,
        args.vae_device,
        args.pipe_device,
        backbone=args.backbone,
        method=args.method,
        allow_split_vae=args.allow_split_vae,
    )
    return pipe, vae_device, transformer_block_checkpointing_actual


def runtime_meta() -> dict:
    import diffusers
    import transformers

    return {
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ],
    }


def cuda_runtime_identity(device: str) -> dict:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"Runtime certification requires CUDA, got {device!r}")
    index = 0 if resolved.index is None else resolved.index
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(index),
        "device_capability": list(torch.cuda.get_device_capability(index)),
    }


def scheduler_sampling_trace(scheduler) -> dict:
    """Capture the actual 50-step scheduler grid used by this generation."""

    def values(name: str) -> list[float | int] | None:
        raw = getattr(scheduler, name, None)
        if raw is None:
            return None
        if isinstance(raw, torch.Tensor):
            raw = raw.detach().to("cpu").tolist()
        if not isinstance(raw, list):
            return None
        return [value.item() if hasattr(value, "item") else value for value in raw]

    def tensor_hash(name: str) -> str | None:
        raw = getattr(scheduler, name, None)
        if not isinstance(raw, torch.Tensor):
            return None
        payload = raw.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()

    return {
        "timesteps": values("timesteps"),
        "sigmas": values("sigmas"),
        "timesteps_length": len(getattr(scheduler, "timesteps", ())),
        "sigmas_length": len(getattr(scheduler, "sigmas", ())),
        "timesteps_sha256": tensor_hash("timesteps"),
        "sigmas_sha256": tensor_hash("sigmas"),
    }


def validate_scheduler_sampling_trace(trace: dict, preflight: dict) -> None:
    """Require this run's scheduler grid to equal the SHA-bound preflight grid."""

    expected = {
        key: preflight.get(key)
        for key in (
            "timesteps_length",
            "sigmas_length",
            "timesteps_sha256",
            "sigmas_sha256",
        )
    }
    actual = {key: trace.get(key) for key in expected}
    if (
        not isinstance(trace.get("timesteps"), list)
        or not isinstance(trace.get("sigmas"), list)
        or not all(math.isfinite(float(value)) for value in trace["timesteps"] + trace["sigmas"])
        or any(value is None for value in expected.values())
        or actual != expected
    ):
        raise RuntimeError(
            f"Scheduler sampling grid differs from preflight: expected={expected}, actual={actual}"
        )


def canonical_scheduler_config(config: dict) -> dict:
    """Canonicalise the order-insensitive Diffusers default-value set for hashing."""
    canonical = dict(config)
    defaults = canonical.get("_use_default_values")
    if isinstance(defaults, list) and all(isinstance(value, str) for value in defaults):
        canonical["_use_default_values"] = sorted(defaults)
    return canonical


def runtime_certification_report(
    events: list[dict],
    *,
    guidance_step: list[int],
    fixed_frames: list[int],
    prediction_max_abs: float,
    x0_max_abs: float,
    vae_max_abs: float,
    scheduler_config: dict,
    scheduler_identity: dict,
    scheduler_preflight: dict,
    specification_sha256: str,
    scheduler_sampling_trace: dict,
    adapted_geco_schedule: dict,
    guidance_lr: list[float],
) -> dict:
    """Validate non-mutating traces from the actual full guidance path."""

    expected_updates = sum(guidance_step)
    expected_prediction_checks = sum(repeats > 0 for repeats in guidance_step)
    expected_vae_checks = expected_updates * len(fixed_frames)
    expected_update_pairs = {
        (step_index, repeat_index)
        for step_index, repeats in enumerate(guidance_step)
        for repeat_index in range(repeats)
    }
    expected_prediction_pairs = {
        (step_index, 0)
        for step_index, repeats in enumerate(guidance_step)
        if repeats > 0
    }
    expected_vae_triples = {
        (step_index, repeat_index, frame_index)
        for step_index, repeat_index in expected_update_pairs
        for frame_index in fixed_frames
    }
    expected_final_update_pairs = {
        (step_index, repeats - 1)
        for step_index, repeats in enumerate(guidance_step)
        if repeats > 0
    }
    grouped = {
        kind: [event for event in events if event.get("kind") == kind]
        for kind in (
            "prediction",
            "vae",
            "update",
            "scheduler_recompute",
            "scheduler_step_input",
        )
    }
    failures: list[str] = []
    if len(grouped["prediction"]) != expected_prediction_checks:
        failures.append("prediction-check count mismatch")
    if len(grouped["vae"]) != expected_vae_checks:
        failures.append("VAE-check count mismatch")
    if len(grouped["update"]) != expected_updates:
        failures.append("guidance-update count mismatch")
    if len(grouped["scheduler_recompute"]) != len(expected_final_update_pairs):
        failures.append("scheduler-recompute count mismatch")
    if len(grouped["scheduler_step_input"]) != len(expected_final_update_pairs):
        failures.append("scheduler-step-input count mismatch")

    observed_update_pairs = {
        (event.get("step_index"), event.get("repeat_index"))
        for event in grouped["update"]
    }
    observed_prediction_pairs = {
        (event.get("step_index"), event.get("repeat_index"))
        for event in grouped["prediction"]
    }
    observed_vae_triples = {
        (event.get("step_index"), event.get("repeat_index"), event.get("frame_index"))
        for event in grouped["vae"]
    }
    observed_recompute_pairs = {
        (event.get("step_index"), event.get("repeat_index"))
        for event in grouped["scheduler_recompute"]
    }
    observed_scheduler_input_pairs = {
        (event.get("step_index"), event.get("repeat_index"))
        for event in grouped["scheduler_step_input"]
    }
    if observed_update_pairs != expected_update_pairs:
        failures.append("guidance-update coverage mismatch")
    if observed_prediction_pairs != expected_prediction_pairs:
        failures.append("prediction-check coverage mismatch")
    if observed_vae_triples != expected_vae_triples:
        failures.append("VAE-check coverage mismatch")
    if observed_recompute_pairs != expected_final_update_pairs:
        failures.append("scheduler-recompute coverage mismatch")
    if observed_scheduler_input_pairs != expected_final_update_pairs:
        failures.append("scheduler-step-input coverage mismatch")

    def finite(event: dict, *keys: str) -> bool:
        try:
            return all(math.isfinite(float(event[key])) for key in keys)
        except (KeyError, TypeError, ValueError):
            return False

    prediction_finite = all(
        finite(
            event,
            "sampling_finite",
            "guidance_finite",
            "pred_mean_abs_diff",
            "pred_max_abs_diff",
            "x0_mean_abs_diff",
            "x0_max_abs_diff",
        )
        and event["sampling_finite"] == 1.0
        and event["guidance_finite"] == 1.0
        for event in grouped["prediction"]
    )
    vae_finite = all(finite(event, "mean_abs_diff", "max_abs_diff") for event in grouped["vae"])
    update_finite = all(
        finite(
            event,
            "loss",
            "grad_norm",
            "latent_delta",
            "relative_delta",
            "loss_wrt_noise_norm",
            "transformer_loss_vjp_norm",
            "latent_before_finite",
            "latent_after_finite",
        )
        and event.get("frames_requires_grad") is True
        and event.get("score_requires_grad") is True
        and event.get("noise_pred_requires_grad") is True
        and event["loss_wrt_noise_norm"] > 0.0
        and event["transformer_loss_vjp_norm"] > 0.0
        and event["grad_norm"] > 0.0
        and event["latent_delta"] > 0.0
        and event["latent_before_finite"] == 1.0
        and event["latent_after_finite"] == 1.0
        for event in grouped["update"]
    )
    scheduler_recompute_finite = all(
        finite(event, "prediction_finite", "prediction_mean_abs", "latents_finite")
        and event["prediction_finite"] == 1.0
        and event["latents_finite"] == 1.0
        and isinstance(event.get("latents_sha256"), str)
        and isinstance(event.get("model_output_sha256"), str)
        for event in grouped["scheduler_recompute"]
    )
    scheduler_input_hashed = all(
        isinstance(event.get("latents_sha256"), str)
        and isinstance(event.get("model_output_sha256"), str)
        and finite(event, "latents_finite")
        and event["latents_finite"] == 1.0
        for event in grouped["scheduler_step_input"]
    )
    if not prediction_finite:
        failures.append("prediction trace contains non-finite values")
    if not vae_finite:
        failures.append("VAE trace contains non-finite values")
    if not update_finite:
        failures.append("guidance trace lacks a finite nonzero loss-to-transformer-to-latent path")
    if not scheduler_recompute_finite:
        failures.append("post-update scheduler recomputation is invalid")
    if not scheduler_input_hashed:
        failures.append("scheduler step input is not bound to trace hashes")

    updates = {
        (event["step_index"], event["repeat_index"]): event
        for event in grouped["update"]
    }
    recomputes = {
        (event["step_index"], event["repeat_index"]): event
        for event in grouped["scheduler_recompute"]
    }
    scheduler_inputs = {
        (event["step_index"], event["repeat_index"]): event
        for event in grouped["scheduler_step_input"]
    }
    for pair in expected_final_update_pairs:
        update = updates[pair]
        recompute = recomputes[pair]
        scheduler_input = scheduler_inputs[pair]
        if update.get("latent_before_sha256") == update.get("latent_after_sha256"):
            failures.append(f"guidance update did not change latent at {pair}")
        if update.get("latent_after_sha256") != recompute.get("latents_sha256"):
            failures.append(f"recompute did not use updated latent at {pair}")
        if (
            recompute.get("latents_sha256") != scheduler_input.get("latents_sha256")
            or recompute.get("model_output_sha256")
            != scheduler_input.get("model_output_sha256")
        ):
            failures.append(f"scheduler step did not receive recomputed prediction at {pair}")

    def maximum(kind: str, key: str) -> float | None:
        values = [float(event[key]) for event in grouped[kind] if key in event]
        return max(values) if values else None

    observed = {
        "prediction_max_abs_diff": maximum("prediction", "pred_max_abs_diff"),
        "x0_max_abs_diff": maximum("prediction", "x0_max_abs_diff"),
        "vae_max_abs_diff": maximum("vae", "max_abs_diff"),
    }
    if observed["prediction_max_abs_diff"] is None or observed["prediction_max_abs_diff"] > prediction_max_abs:
        failures.append("prediction agreement exceeds threshold")
    if observed["x0_max_abs_diff"] is None or observed["x0_max_abs_diff"] > x0_max_abs:
        failures.append("x0 agreement exceeds threshold")
    if observed["vae_max_abs_diff"] is None or observed["vae_max_abs_diff"] > vae_max_abs:
        failures.append("VAE agreement exceeds threshold")
    scheduler_json = json.dumps(
        canonical_scheduler_config(scheduler_config),
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return {
        "schema": "wan_geco_runtime_certificate_v1",
        "status": "passed" if not failures else "failed",
        "expected": {
            "guidance_updates": expected_updates,
            "prediction_checks": expected_prediction_checks,
            "vae_checks": expected_vae_checks,
            "scheduler_recomputes": len(expected_final_update_pairs),
            "scheduler_step_inputs": len(expected_final_update_pairs),
        },
        "observed": {
            "guidance_updates": len(grouped["update"]),
            "prediction_checks": len(grouped["prediction"]),
            "vae_checks": len(grouped["vae"]),
            "scheduler_recomputes": len(grouped["scheduler_recompute"]),
            "scheduler_step_inputs": len(grouped["scheduler_step_input"]),
            **observed,
        },
        "thresholds": {
            "prediction_max_abs": prediction_max_abs,
            "x0_max_abs": x0_max_abs,
            "vae_max_abs": vae_max_abs,
        },
        "scheduler_config_sha256": hashlib.sha256(scheduler_json.encode()).hexdigest(),
        "scheduler": scheduler_identity,
        "scheduler_preflight": scheduler_preflight,
        "scheduler_sampling_trace": scheduler_sampling_trace,
        "adapted_geco_schedule": adapted_geco_schedule,
        "fixed_frames": fixed_frames,
        "guidance_step": guidance_step,
        "guidance_lr": guidance_lr,
        "specification_sha256": specification_sha256,
        "events": events,
        "failures": failures,
    }


def load_runtime_certification_spec(repo: Path) -> tuple[Path, dict, str]:
    """Load the committed, non-overridable Wan runtime-certificate contract."""

    path = repo / "benchmarks/dl3dv_geco/specs/wan_geco_runtime_certificate_v1.json"
    if not path.is_file():
        raise RuntimeError(f"Missing committed runtime-certification spec: {path}")
    spec = json.loads(path.read_text())
    required = {
        "schema": "wan_geco_runtime_certificate_spec_v1",
        "backbone": "wan",
        "method": "adapted_geco",
    }
    if any(spec.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"Invalid runtime-certification spec identity: {path}")
    scheduler = spec.get("scheduler")
    scheduler_preflight = spec.get("scheduler_preflight")
    adapted_geco = spec.get("adapted_geco")
    thresholds = spec.get("thresholds")
    if (
        not isinstance(scheduler, dict)
        or not isinstance(scheduler_preflight, dict)
        or not isinstance(adapted_geco, dict)
        or not isinstance(thresholds, dict)
    ):
        raise RuntimeError(f"Invalid runtime-certification spec structure: {path}")
    if (
        scheduler_preflight.get("schema") != "wan_scheduler_runtime_preflight_v1"
        or scheduler_preflight.get("steps") != 50
        or not isinstance(scheduler_preflight.get("max_abs_error"), (int, float))
        or not math.isfinite(float(scheduler_preflight["max_abs_error"]))
        or float(scheduler_preflight["max_abs_error"]) < 0.0
    ):
        raise RuntimeError(f"Invalid runtime-certification preflight contract: {path}")
    if not all(
        isinstance(thresholds.get(key), (int, float))
        and math.isfinite(float(thresholds[key]))
        and float(thresholds[key]) >= 0.0
        for key in ("prediction_max_abs", "x0_max_abs", "vae_max_abs")
    ):
        raise RuntimeError(f"Invalid runtime-certification thresholds: {path}")
    required_geco = {
        "backbone": "wan",
        "frames": 121,
        "fixed_frame_indices": [28, 58, 88, 118],
        "ufm_scale": 0.25,
        "decode_spatial_scale": 1.0,
        "max_relative_delta": 0.0,
        "pair_mode": "adjacent",
        "vggt_strategy": "once",
        "time_travel": None,
    }
    if any(adapted_geco.get(key) != value for key, value in required_geco.items()):
        raise RuntimeError(f"Invalid frozen adapted-GeCo schedule: {path}")
    fixed_frames = adapted_geco.get("fixed_frame_indices")
    if (
        not isinstance(fixed_frames, list)
        or not all(type(index) is int for index in fixed_frames)
        or fixed_frames != sorted(set(fixed_frames))
    ):
        raise RuntimeError(f"Invalid adapted-GeCo index types/order: {path}")
    if type(adapted_geco.get("frames")) is not int:
        raise RuntimeError(f"Invalid adapted-GeCo frame-count type: {path}")
    float_fields = (
        "ufm_scale",
        "decode_spatial_scale",
        "max_relative_delta",
    )
    if any(
        type(adapted_geco.get(key)) is not float
        or not math.isfinite(adapted_geco[key])
        for key in float_fields
    ):
        raise RuntimeError(f"Invalid adapted-GeCo floating-point type/value: {path}")
    if not isinstance(adapted_geco.get("time_travel_note"), str):
        raise RuntimeError(f"Missing adapted-GeCo time-travel note: {path}")
    schedules = adapted_geco.get("schedule_candidates")
    expected_schedules = {
        "conservative_late_22": ([0] * 20 + [1] * 22 + [0] * 8, 0.1, 22),
        "medium_shape_64": ([0] * 3 + [2] * 17 + [1] * 30, 0.1, 64),
        "geco_repeat_shape_111": ([0] * 3 + [3] * 17 + [2] * 30, 3.0, 111),
    }
    if not isinstance(schedules, dict) or set(schedules) != set(expected_schedules):
        raise RuntimeError(f"Invalid pre-registered adapted-GeCo schedules: {path}")
    for schedule_id, (expected_steps, expected_lr, expected_updates) in expected_schedules.items():
        schedule = schedules[schedule_id]
        if (
            not isinstance(schedule, dict)
            or schedule.get("guidance_step") != expected_steps
            or type(schedule.get("guidance_learning_rate")) is not float
            or schedule.get("guidance_learning_rate") != expected_lr
            or type(schedule.get("updates")) is not int
            or schedule.get("updates") != expected_updates
            or sum(schedule["guidance_step"]) != schedule["updates"]
            or not isinstance(schedule.get("description"), str)
        ):
            raise RuntimeError(f"Invalid adapted-GeCo schedule {schedule_id!r}: {path}")
    return path, spec, sha256_file(path)


def validate_runtime_certification_geco_schedule(
    args: argparse.Namespace,
    spec: dict,
    *,
    schedule_id: str,
    fixed_frames: list[int],
    guidance_step: list[int],
    guidance_lr: list[float],
) -> None:
    """Reject any adapted-GeCo run that drifts from its registered schedule."""

    adapted = spec["adapted_geco"]
    expected_schedule = adapted["schedule_candidates"][schedule_id]
    observed = {
        "backbone": args.backbone,
        "frames": args.frames,
        "fixed_frame_indices": fixed_frames,
        "guidance_step": guidance_step,
        "guidance_learning_rate": [
            expected_schedule["guidance_learning_rate"] if repeats else 0.0
            for repeats in guidance_step
        ],
        "ufm_scale": args.ufm_scale,
        "decode_spatial_scale": args.decode_spatial_scale,
        "max_relative_delta": args.max_relative_delta,
        "pair_mode": "adjacent",
        "vggt_strategy": "once",
        "time_travel": None,
    }
    expected = {
        "backbone": adapted["backbone"],
        "frames": adapted["frames"],
        "fixed_frame_indices": adapted["fixed_frame_indices"],
        "guidance_step": expected_schedule["guidance_step"],
        "guidance_learning_rate": [
            expected_schedule["guidance_learning_rate"] if repeats else 0.0
            for repeats in expected_schedule["guidance_step"]
        ],
        "ufm_scale": adapted["ufm_scale"],
        "decode_spatial_scale": adapted["decode_spatial_scale"],
        "max_relative_delta": adapted["max_relative_delta"],
        "pair_mode": adapted["pair_mode"],
        "vggt_strategy": adapted["vggt_strategy"],
        "time_travel": adapted["time_travel"],
    }
    mismatch = {
        key: {"expected": value, "actual": observed[key]}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if guidance_lr != observed["guidance_learning_rate"]:
        mismatch["guidance_learning_rate"] = {
            "expected": observed["guidance_learning_rate"],
            "actual": guidance_lr,
        }
    if mismatch:
        raise RuntimeError(f"Adapted-GeCo schedule differs from frozen certificate spec: {mismatch}")


def validate_runtime_certification_scheduler(scheduler, spec: dict, *, repo: Path) -> dict:
    """Bind a certificate to the scheduler actually loaded by the adapted pipeline."""

    import diffusers

    expected = spec["scheduler"]
    config = canonical_scheduler_config(dict(scheduler.config))
    config_json = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    actual = {
        "class_name": type(scheduler).__name__,
        "module": type(scheduler).__module__,
        "diffusers_version": diffusers.__version__,
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "prediction_type": config.get("prediction_type"),
        "predict_x0": config.get("predict_x0"),
        "thresholding": config.get("thresholding"),
        "flow_match_helper_source_sha256": sha256_file(repo / "geometry_selection/online.py"),
    }
    expected_fields = (
        "class_name",
        "module",
        "diffusers_version",
        "config_sha256",
        "prediction_type",
        "predict_x0",
        "thresholding",
    )
    mismatch = {
        key: {"expected": expected.get(key), "actual": actual[key]}
        for key in expected_fields
        if expected.get(key) != actual[key]
    }
    if expected.get("clean_prediction_rule") != "x_t - sigma_t * model_output":
        mismatch["clean_prediction_rule"] = {
            "expected": "x_t - sigma_t * model_output",
            "actual": expected.get("clean_prediction_rule"),
        }
    if spec.get("flow_match_helper_source_sha256") != actual["flow_match_helper_source_sha256"]:
        mismatch["flow_match_helper_source_sha256"] = {
            "expected": spec.get("flow_match_helper_source_sha256"),
            "actual": actual["flow_match_helper_source_sha256"],
        }
    if mismatch:
        raise RuntimeError(f"Runtime-certification scheduler contract mismatch: {mismatch}")
    return actual


def validate_scheduler_preflight_receipt(
    receipt_path: Path,
    expected_sha256: str,
    *,
    code_commit: str,
    model: Path,
    steps: int,
    scheduler_identity: dict,
    runtime_identity: dict,
    spec: dict,
) -> dict:
    """Bind the certificate to a clean-endpoint preflight run in this exact code state."""

    receipt_path = receipt_path.resolve()
    if not receipt_path.is_file():
        raise RuntimeError(f"Missing scheduler preflight receipt: {receipt_path}")
    actual_sha256 = sha256_file(receipt_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError("Scheduler preflight receipt SHA mismatch")
    receipt = json.loads(receipt_path.read_text())
    preflight = spec["scheduler_preflight"]
    expected_fields = {
        "schema": preflight["schema"],
        "code_commit": code_commit,
        "model": str(Path(model).resolve()),
        "scheduler_class": scheduler_identity["class_name"],
        "scheduler_module": scheduler_identity["module"],
        "diffusers_version": scheduler_identity["diffusers_version"],
        "scheduler_config_sha256": scheduler_identity["config_sha256"],
        "flow_match_helper_source_sha256": scheduler_identity[
            "flow_match_helper_source_sha256"
        ],
        "runtime": runtime_identity,
    }
    mismatch = {
        key: {"expected": value, "actual": receipt.get(key)}
        for key, value in expected_fields.items()
        if receipt.get(key) != value
    }
    if receipt.get("validated_step_indices") != list(range(steps)):
        mismatch["validated_step_indices"] = {
            "expected": list(range(steps)),
            "actual": receipt.get("validated_step_indices"),
        }
    for key in ("flow_x0_max_abs_error", "scheduler_x0_max_abs_error"):
        value = receipt.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) > float(
            preflight["max_abs_error"]
        ):
            mismatch[key] = {
                "expected_max": preflight["max_abs_error"],
                "actual": value,
            }
    if mismatch:
        raise RuntimeError(f"Scheduler preflight receipt contract mismatch: {mismatch}")
    trace_fields = ("timesteps_length", "sigmas_length", "timesteps_sha256", "sigmas_sha256")
    if (
        type(receipt.get("timesteps_length")) is not int
        or type(receipt.get("sigmas_length")) is not int
        or receipt["timesteps_length"] != steps
        or receipt["sigmas_length"] < steps
        or any(not isinstance(receipt.get(key), str) or len(receipt[key]) != 64 for key in trace_fields[2:])
    ):
        raise RuntimeError("Scheduler preflight receipt has no valid timestep/sigma grid binding")
    return {
        "path": str(receipt_path),
        "sha256": actual_sha256,
        "code_commit": receipt["code_commit"],
        "model": receipt["model"],
        "flow_x0_max_abs_error": receipt["flow_x0_max_abs_error"],
        "scheduler_x0_max_abs_error": receipt["scheduler_x0_max_abs_error"],
        **{key: receipt[key] for key in trace_fields},
    }


def verify_completed_runtime_certificate(
    stored_metadata: dict,
    *,
    output_dir: Path,
    specification_sha256: str,
    adapted_geco_schedule: dict,
    fixed_frames: list[int],
    guidance_step: list[int],
    guidance_lr: list[float],
) -> None:
    """Verify that a completed certified run still has its bound receipt."""

    recorded = stored_metadata.get("runtime_certification")
    if not isinstance(recorded, dict):
        raise RuntimeError(f"COMPLETE runtime-certification receipt missing: {output_dir}")
    certificate_path = output_dir / "runtime_certification.json"
    if recorded.get("path") != str(certificate_path.resolve()):
        raise RuntimeError(f"COMPLETE runtime-certification path mismatch: {output_dir}")
    if recorded.get("status") != "passed":
        raise RuntimeError(f"COMPLETE runtime-certification status is not passed: {output_dir}")
    if recorded.get("schema") != "wan_geco_runtime_certificate_v1":
        raise RuntimeError(f"COMPLETE runtime-certification schema mismatch: {output_dir}")
    if recorded.get("specification_sha256") != specification_sha256:
        raise RuntimeError(f"COMPLETE runtime-certification specification mismatch: {output_dir}")
    if not certificate_path.is_file():
        raise RuntimeError(f"COMPLETE runtime-certification receipt missing: {output_dir}")
    if recorded.get("sha256") != sha256_file(certificate_path):
        raise RuntimeError(f"COMPLETE runtime-certification receipt SHA mismatch: {output_dir}")
    receipt = json.loads(certificate_path.read_text())
    if (
        receipt.get("schema") != recorded["schema"]
        or receipt.get("status") != recorded["status"]
        or receipt.get("specification_sha256") != specification_sha256
        or receipt.get("scheduler_preflight") != recorded.get("scheduler_preflight")
    ):
        raise RuntimeError(f"COMPLETE runtime-certification receipt content mismatch: {output_dir}")
    if (
        receipt.get("adapted_geco_schedule") != adapted_geco_schedule
        or receipt.get("fixed_frames") != fixed_frames
        or receipt.get("guidance_step") != guidance_step
        or receipt.get("guidance_lr") != guidance_lr
    ):
        raise RuntimeError(f"COMPLETE runtime-certification schedule mismatch: {output_dir}")


def split_vae_contract(
    args: argparse.Namespace,
    *,
    code_identity: dict,
    runner_sha256: str,
    pipeline_sha256: str,
    model_content_manifest: dict,
) -> dict:
    """Split-VAE execution contract shared by all pre-registered schedules."""

    return {
        "backbone": args.backbone,
        "method": args.method,
        "code_commit": code_identity["commit"],
        "runner_sha256": runner_sha256,
        "pipeline_sha256": pipeline_sha256,
        "runtime_certification_spec_sha256": args.expected_runtime_certification_spec_sha256,
        "scheduler_preflight_sha256": args.expected_scheduler_preflight_sha256,
        "profile": {
            "steps": args.steps,
            "frames": args.frames,
            "height": args.height,
            "width": args.width,
            "fps": args.fps,
            "guidance_scale": args.guidance_scale,
            "wan_negative_prompt_mode": args.wan_negative_prompt_mode,
        },
        "devices": {
            "pipe": args.pipe_device,
            "vae": args.vae_device or args.pipe_device,
            "metric": args.metric_device,
            "cross_device_grad_via_cpu": args.cross_device_grad_via_cpu,
        },
        "transformer_block_checkpointing": args.transformer_block_checkpointing,
        "geometry_models": {
            "vggt_model": args.vggt_model,
            "ufm_model": args.ufm_model,
        },
        "model_identities": {
            "generator": model_identity(args.model),
            "vggt": model_identity(args.vggt_model),
            "ufm": model_identity(args.ufm_model),
        },
        "model_content_manifest": model_content_manifest,
        "runtime_devices": {
            "pipe": cuda_runtime_identity(args.pipe_device),
            "vae": cuda_runtime_identity(args.vae_device or args.pipe_device),
            "metric": cuda_runtime_identity(args.metric_device),
        },
        "geco": {
            "ufm_scale": args.ufm_scale,
            "decode_spatial_scale": args.decode_spatial_scale,
            "max_relative_delta": args.max_relative_delta,
            "loss": "residual_motion",
            "grad_through_vggt": False,
            "pair_mode": "adjacent",
            "vggt_strategy": "once",
            "time_travel": None,
        },
    }


def bootstrap_split_vae_contract_errors(args: argparse.Namespace) -> dict:
    """Return fail-closed violations for the one-time split-VAE bootstrap run."""

    expected = {
        "backbone": "wan",
        "method": "adapted_geco",
        "protocol_mode": "legacy-debug",
        "runtime_certification": True,
        "pipe_device": "cuda:0",
        "vae_device": "cuda:1",
        "metric_device": "cuda:2",
        "cross_device_grad_via_cpu": True,
        "transformer_block_checkpointing": True,
        "allow_split_vae": True,
        "adapted_geco_schedule": "conservative_late_22",
    }
    errors = {
        key: {"expected": value, "actual": getattr(args, key)}
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    for key, value in SPLIT_VAE_BOOTSTRAP_PROFILE.items():
        if getattr(args, key) != value:
            errors[key] = {"expected": value, "actual": getattr(args, key)}
    if args.split_vae_smoke_report is not None:
        errors["split_vae_smoke_report"] = {
            "expected": None,
            "actual": str(args.split_vae_smoke_report),
        }
    if len({args.pipe_device, args.vae_device, args.metric_device}) != 3:
        errors["distinct_devices"] = {
            "expected": "three distinct CUDA devices",
            "actual": [args.pipe_device, args.vae_device, args.metric_device],
        }
    return errors


def load_and_validate_split_vae_runtime_receipt(
    receipt_path: Path,
    *,
    expected_contract: dict,
) -> dict:
    """Validate a non-overwritable receipt emitted after a complete bootstrap run."""

    receipt_path = receipt_path.resolve()
    if not receipt_path.is_file():
        raise RuntimeError(f"Split-VAE runtime receipt is missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != SPLIT_VAE_RUNTIME_RECEIPT_SCHEMA or receipt.get("passed") is not True:
        raise RuntimeError("Split-VAE runtime receipt schema/status mismatch")
    if receipt.get("contract") != expected_contract:
        raise RuntimeError("Split-VAE runtime receipt contract differs from this run")

    source = receipt.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("Split-VAE runtime receipt has no source binding")
    def source_path(key: str) -> Path:
        raw = source.get(key)
        if not isinstance(raw, str) or not raw:
            raise RuntimeError(f"Split-VAE runtime receipt has no valid {key}")
        path = Path(raw)
        if not path.is_absolute():
            raise RuntimeError(f"Split-VAE runtime receipt path is not absolute: {key}")
        return path

    metadata_path = source_path("metadata_path")
    complete_path = source_path("complete_path")
    video_path = source_path("video_path")
    certificate_path = source_path("runtime_certificate_path")
    required_files = (metadata_path, complete_path, video_path, certificate_path)
    if any(not path.is_file() for path in required_files):
        raise RuntimeError("Split-VAE runtime receipt source artifact is missing")
    if source.get("metadata_sha256") != sha256_file(metadata_path):
        raise RuntimeError("Split-VAE receipt source metadata SHA mismatch")
    if source.get("complete_sha256") != sha256_file(complete_path):
        raise RuntimeError("Split-VAE receipt source COMPLETE SHA mismatch")
    if source.get("video_sha256") != sha256_file(video_path):
        raise RuntimeError("Split-VAE receipt source video SHA mismatch")
    if source.get("runtime_certificate_sha256") != sha256_file(certificate_path):
        raise RuntimeError("Split-VAE receipt source certificate SHA mismatch")

    source_metadata = json.loads(metadata_path.read_text())
    if source_metadata.get("run_id") != source.get("run_id"):
        raise RuntimeError("Split-VAE receipt source run_id mismatch")
    if complete_path.read_text().strip() != source_metadata.get("run_id"):
        raise RuntimeError("Split-VAE receipt source COMPLETE content mismatch")
    if source_metadata.get("video") != str(video_path) or source_metadata.get("video_sha256") != source.get("video_sha256"):
        raise RuntimeError("Split-VAE receipt source video metadata mismatch")
    source_certificate = source_metadata.get("runtime_certification")
    if not isinstance(source_certificate, dict):
        raise RuntimeError("Split-VAE receipt source lacks runtime certification metadata")
    if source_certificate.get("status") != "passed" or source_certificate.get("sha256") != source.get("runtime_certificate_sha256"):
        raise RuntimeError("Split-VAE receipt source runtime certification mismatch")
    if source_metadata.get("run_config") != source.get("run_config"):
        raise RuntimeError("Split-VAE receipt source configuration binding mismatch")
    if (
        source_metadata.get("adapted_geco_model_content_manifest")
        != expected_contract["model_content_manifest"]
    ):
        raise RuntimeError("Split-VAE receipt source model-content manifest mismatch")
    for key in ("model", "vggt_model", "ufm_model"):
        if source_metadata.get(key) != source.get(key):
            raise RuntimeError(f"Split-VAE receipt source {key} binding mismatch")
    source_certificate_payload = json.loads(certificate_path.read_text())
    if (
        source_certificate_payload.get("schema") != "wan_geco_runtime_certificate_v1"
        or source_certificate_payload.get("status") != "passed"
        or source_certificate_payload.get("specification_sha256")
        != source_certificate.get("specification_sha256")
    ):
        raise RuntimeError("Split-VAE receipt source certificate content mismatch")
    source_profile = source_metadata.get("generation", {})
    profile_mismatch = {
        key: {"expected": value, "actual": source_profile.get(key)}
        for key, value in expected_contract["profile"].items()
        if source_profile.get(key) != value
    }
    if profile_mismatch:
        raise RuntimeError(f"Split-VAE receipt source profile mismatch: {profile_mismatch}")
    if (
        source_metadata.get("code_identity", {}).get("commit") != expected_contract["code_commit"]
        or source_metadata.get("code_identity", {}).get("dirty") is not False
        or source_metadata.get("runner_sha256") != expected_contract["runner_sha256"]
        or source_metadata.get("pipeline_sha256") != expected_contract["pipeline_sha256"]
        or source_metadata.get("model") != expected_contract["model_identities"]["generator"]
        or source_metadata.get("vggt_model") != expected_contract["model_identities"]["vggt"]
        or source_metadata.get("ufm_model") != expected_contract["model_identities"]["ufm"]
    ):
        raise RuntimeError("Split-VAE receipt source code/model binding mismatch")
    source_devices = source_metadata.get("devices", {})
    if any(
        source_devices.get(key) != expected_contract["devices"].get(key)
        for key in ("pipe", "vae", "metric", "cross_device_grad_via_cpu")
    ):
        raise RuntimeError("Split-VAE receipt source device-request binding mismatch")
    source_geco = source_metadata.get("geco", {})
    if any(
        source_geco.get(key) != value
        for key, value in expected_contract["geco"].items()
    ):
        raise RuntimeError("Split-VAE receipt source geometry-execution binding mismatch")
    if (
        source_certificate_payload.get("specification_sha256")
        != expected_contract["runtime_certification_spec_sha256"]
        or source_certificate_payload.get("scheduler_preflight", {}).get("sha256")
        != expected_contract["scheduler_preflight_sha256"]
    ):
        raise RuntimeError("Split-VAE receipt source certificate-contract mismatch")
    source_runtime = source_metadata.get("runtime", {})
    for role, expected_device in expected_contract["runtime_devices"].items():
        device_name = expected_contract["devices"][role]
        try:
            index = int(device_name.split(":", 1)[1])
            source_device = source_runtime["devices"][index]
        except (AttributeError, IndexError, KeyError, ValueError) as error:
            raise RuntimeError(f"Split-VAE receipt source runtime lacks {role} device") from error
        observed_device = {
            "torch_version": source_runtime.get("torch"),
            "cuda_version": source_runtime.get("cuda"),
            "device_name": source_device.get("name"),
            "device_capability": source_device.get("capability"),
        }
        if observed_device != expected_device:
            raise RuntimeError(f"Split-VAE receipt source runtime mismatch for {role}")
    return receipt


def publish_split_vae_runtime_receipt(
    destination: Path,
    *,
    contract: dict,
    source_metadata: dict,
    metadata_path: Path,
    complete_path: Path,
    video_path: Path,
    certificate_path: Path,
) -> Path:
    """Publish exactly once, after the source video, metadata and COMPLETE are durable."""

    if not complete_path.is_file() or complete_path.read_text().strip() != source_metadata["run_id"]:
        raise RuntimeError("Refusing to publish split-VAE receipt before COMPLETE")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": SPLIT_VAE_RUNTIME_RECEIPT_SCHEMA,
        "passed": True,
        "contract": contract,
        "source": {
            "run_id": source_metadata["run_id"],
            "metadata_path": str(metadata_path.resolve()),
            "metadata_sha256": sha256_file(metadata_path),
            "complete_path": str(complete_path.resolve()),
            "complete_sha256": sha256_file(complete_path),
            "video_path": str(video_path.resolve()),
            "video_sha256": sha256_file(video_path),
            "runtime_certificate_path": str(certificate_path.resolve()),
            "runtime_certificate_sha256": sha256_file(certificate_path),
            "run_config": source_metadata["run_config"],
            "model": source_metadata["model"],
            "vggt_model": source_metadata["vggt_model"],
            "ufm_model": source_metadata["ufm_model"],
        },
    }
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise RuntimeError(f"Refusing to overwrite split-VAE runtime receipt: {destination}") from error
        directory_fd = os.open(destination.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=PROFILES, required=True)
    parser.add_argument(
        "--method",
        choices=("baseline", "adapted_geco", "online_geometry_selection"),
        required=True,
    )
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--protocol-mode",
        choices=("frozen", "legacy-debug"),
        required=True,
        help="Formal runs must explicitly select frozen; legacy-debug artifacts cannot enter formal pools.",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--expected-split", choices=("debug", "validation", "test"))
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--model-lock", type=Path)
    parser.add_argument("--model-verification-cache", type=Path)
    parser.add_argument("--experiment-lock", type=Path)
    parser.add_argument("--candidate-spec", type=Path)
    parser.add_argument("--online-selection-config", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case-id")
    group.add_argument("--case-index", type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vggt-model", default="facebook/VGGT-1B")
    parser.add_argument("--ufm-model", default="infinity1096/UFM-Base")
    parser.add_argument(
        "--model-content-manifest",
        type=Path,
        help=(
            "Required full content manifest for adapted-GeCo model roots; it is "
            "validated before loading Wan, VGGT-Omega, or UFM."
        ),
    )
    parser.add_argument(
        "--expected-model-content-manifest-sha256",
        help="Required SHA-256 of --model-content-manifest.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--ufm-scale", type=float, default=0.25)
    parser.add_argument("--decode-spatial-scale", type=float, default=1.0)
    parser.add_argument("--max-relative-delta", type=float, default=0.0)
    parser.add_argument(
        "--adapted-geco-schedule",
        help=(
            "Required pre-registered schedule ID for adapted-GeCo. The original "
            "DDIM time-travel is intentionally not migrated to Wan flow matching."
        ),
    )
    parser.add_argument("--guidance-start-fraction", type=float, default=0.40)
    parser.add_argument("--guidance-end-fraction", type=float, default=0.84)
    parser.add_argument("--guidance-repeats", type=int, default=1)
    parser.add_argument("--guidance-lr", type=float, default=0.1)
    parser.add_argument(
        "--debug-guidance-consistency",
        action="store_true",
        help="Emit non-mutating prediction and VAE consistency traces for runtime certification.",
    )
    parser.add_argument(
        "--runtime-certification",
        action="store_true",
        help=(
            "Require and persist a passing receipt from the actual adapted-GeCo guidance path "
            "using the committed certificate specification."
        ),
    )
    parser.add_argument(
        "--expected-runtime-certification-spec-sha256",
        help="Required SHA-256 of the committed runtime-certificate specification.",
    )
    parser.add_argument(
        "--scheduler-preflight-receipt",
        type=Path,
        help="Same-environment 50-step scheduler equivalence receipt required for certification.",
    )
    parser.add_argument(
        "--expected-scheduler-preflight-sha256",
        help="Required SHA-256 of --scheduler-preflight-receipt.",
    )
    parser.add_argument(
        "--wan-negative-prompt-mode",
        choices=("none", "frozen"),
        default="none",
        help="Use none to match the existing 70 Wan baseline videos.",
    )
    parser.add_argument("--pipe-device", default="cuda:0")
    parser.add_argument("--vae-device")
    parser.add_argument("--metric-device", default="cuda:0")
    parser.add_argument("--cross-device-grad-via-cpu", action="store_true")
    parser.add_argument("--transformer-block-checkpointing", action="store_true")
    parser.add_argument("--allow-split-vae", action="store_true")
    parser.add_argument("--split-vae-smoke-report", type=Path)
    parser.add_argument(
        "--emit-split-vae-smoke-report",
        type=Path,
        help=(
            "One-time, non-overwriting split-VAE runtime receipt. It is permitted only "
            "for the full-profile legacy-debug adapted-GeCo certification run."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for key, value in PROFILES[args.backbone].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.method == "adapted_geco" and args.decode_spatial_scale != 1.0:
        parser.error("Formal adapted-GeCo benchmark requires --decode-spatial-scale 1.0")
    if args.runtime_certification and args.method != "adapted_geco":
        parser.error("--runtime-certification requires --method adapted_geco")
    if args.runtime_certification and args.repo.resolve() != REPO_ROOT.resolve():
        parser.error("runtime-certification --repo must be the repository executing this runner")
    needs_adapted_geco_contract = args.method == "adapted_geco"
    if needs_adapted_geco_contract and not args.adapted_geco_schedule:
        parser.error("adapted-GeCo requires --adapted-geco-schedule")
    if needs_adapted_geco_contract and (
        args.guidance_start_fraction != 0.40
        or args.guidance_end_fraction != 0.84
        or args.guidance_repeats != 1
        or args.guidance_lr != 0.1
    ):
        parser.error(
            "adapted-GeCo schedule hyperparameters are registered in the certificate spec; "
            "use --adapted-geco-schedule instead of --guidance-* overrides"
        )
    if needs_adapted_geco_contract and not args.expected_runtime_certification_spec_sha256:
        parser.error("adapted-GeCo requires --expected-runtime-certification-spec-sha256")
    if needs_adapted_geco_contract and not args.expected_scheduler_preflight_sha256:
        parser.error("adapted-GeCo requires --expected-scheduler-preflight-sha256")
    if needs_adapted_geco_contract and args.scheduler_preflight_receipt is None:
        parser.error(
            "adapted-GeCo requires --scheduler-preflight-receipt and "
            "--expected-scheduler-preflight-sha256"
        )
    if needs_adapted_geco_contract and args.model_content_manifest is None:
        parser.error("adapted-GeCo requires --model-content-manifest")
    if needs_adapted_geco_contract and not args.expected_model_content_manifest_sha256:
        parser.error("adapted-GeCo requires --expected-model-content-manifest-sha256")
    is_split_vae_bootstrap = args.emit_split_vae_smoke_report is not None
    if is_split_vae_bootstrap:
        bootstrap_errors = bootstrap_split_vae_contract_errors(args)
        if bootstrap_errors:
            parser.error(
                "--emit-split-vae-smoke-report has a fail-closed contract: "
                + json.dumps(bootstrap_errors, sort_keys=True)
            )
    runtime_certification_path = None
    runtime_certification_spec = None
    runtime_certification_spec_sha256 = None
    if needs_adapted_geco_contract:
        (
            runtime_certification_path,
            runtime_certification_spec,
            runtime_certification_spec_sha256,
        ) = load_runtime_certification_spec(args.repo.resolve())
        if runtime_certification_spec_sha256 != args.expected_runtime_certification_spec_sha256:
            parser.error("runtime-certification specification SHA mismatch")
        if args.steps != runtime_certification_spec["scheduler_preflight"]["steps"]:
            parser.error("runtime-certification requires the frozen scheduler-preflight step count")
        if args.adapted_geco_schedule not in runtime_certification_spec["adapted_geco"]["schedule_candidates"]:
            parser.error("unknown pre-registered adapted-GeCo schedule")

    adapted_geco_model_manifest = None
    if needs_adapted_geco_contract:
        adapted_geco_model_manifest = load_verified_model_content_manifest(
            args.model_content_manifest,
            expected_sha256=args.expected_model_content_manifest_sha256,
            requested_models={
                "wan": args.model,
                "vggt_omega": args.vggt_model,
                "ufm": args.ufm_model,
            },
        )
        # Load only canonical roots validated above, never a caller-supplied
        # alias that could change between validation and from_pretrained().
        args.model = adapted_geco_model_manifest["verified_roots"]["wan"]
        args.vggt_model = adapted_geco_model_manifest["verified_roots"]["vggt_omega"]
        args.ufm_model = adapted_geco_model_manifest["verified_roots"]["ufm"]

    is_frozen_protocol = args.protocol_mode == "frozen"
    if is_frozen_protocol and args.method in {"adapted_geco", "online_geometry_selection"}:
        parser.error(
            "formal guided runs are disabled until the method configuration and geometry "
            "backbone are included in the frozen experiment lock"
        )
    if is_frozen_protocol and args.overwrite:
        parser.error("frozen protocol forbids --overwrite of completed generations")
    if is_frozen_protocol and args.repo.resolve() != REPO_ROOT.resolve():
        raise ValueError("formal generation --repo must be the repository executing this runner")
    online_config = None
    online_config_sha256 = None
    online_adapter = None
    online_geometry_identity = None
    if args.method == "online_geometry_selection":
        if args.online_selection_config is None:
            parser.error("online geometry selection requires --online-selection-config")
        online_config = load_online_selection_config(args.online_selection_config)
        online_config.validate(num_steps=args.steps)
        online_config_sha256 = sha256_file(args.online_selection_config)
        online_adapter = build_online_vggt_adapter(
            args, online_config, args.online_selection_config
        )
        # Content identity is computed before run_id so stale COMPLETE outputs
        # cannot be reused after a checkpoint/source update at the same path.
        online_geometry_identity = online_adapter.artifact_identity()
    elif args.online_selection_config is not None:
        parser.error("--online-selection-config requires --method online_geometry_selection")
    if is_frozen_protocol:
        if args.expected_split is None:
            parser.error("frozen protocol requires --expected-split")
        protocol = validate_formal_protocol(args.manifest)
        protocol_meta = protocol["_meta"]
        split_cases = sorted(
            (
                (key, value)
                for key, value in protocol.items()
                if not key.startswith("_") and value["split"] == args.expected_split
            ),
            key=lambda item: item[1]["split_order"],
        )
        if args.case_id is not None:
            matches = [item for item in split_cases if item[0] == args.case_id]
            if len(matches) != 1:
                parser.error(f"case {args.case_id!r} is not in split {args.expected_split}")
            case_id, case = matches[0]
        else:
            if args.case_index is None or not 0 <= args.case_index < len(split_cases):
                parser.error(
                    f"case_index must be in [0, {len(split_cases) - 1}] for "
                    f"split {args.expected_split}"
                )
            case_id, case = split_cases[args.case_index]
    else:
        case_id, case, protocol_meta = load_case(
            args.manifest, args.case_id, args.case_index
        )
    code_identity = git_identity(REPO_ROOT if is_frozen_protocol else args.repo.resolve())
    if args.runtime_certification and code_identity["dirty"]:
        raise RuntimeError("runtime-certification requires a clean repository commit")
    if args.runtime_certification:
        validate_committed_file(
            runtime_certification_path,
            REPO_ROOT,
            code_identity["commit"],
        )
    locked_model_identity = None
    model_lock_sha256 = None
    experiment_lock_sha256 = None
    implementation_sha256 = None
    candidate_spec_sha256 = None
    planned_candidate = None
    if is_frozen_protocol:
        if args.dataset_root is None:
            parser.error("frozen protocol requires --dataset-root")
        if args.expected_git_commit is None:
            parser.error("frozen protocol requires --expected-git-commit")
        if code_identity["dirty"] or code_identity["commit"] != args.expected_git_commit:
            raise RuntimeError(
                f"generation code is not the frozen clean commit: {code_identity}"
            )
        validate_committed_file(args.manifest, REPO_ROOT, code_identity["commit"])
        profile_name = {
            "wan": "Wan2.2-TI2V-5B",
            "cosmos": "Cosmos-Predict2.5-2B-post",
        }[args.backbone]
        profile = protocol_meta["generation_profiles"][profile_name]
        profile_mismatches = {
            key: (getattr(args, key), expected)
            for key, expected in profile.items()
            if getattr(args, key) != expected
        }
        if profile_mismatches:
            raise ValueError(f"generation profile differs from protocol: {profile_mismatches}")
        if args.model_lock is None:
            parser.error("frozen protocol requires --model-lock")
        if args.model_verification_cache is None:
            parser.error("frozen protocol requires --model-verification-cache")
        validate_committed_file(args.model_lock, REPO_ROOT, code_identity["commit"])
        model_lock_sha256 = protocol_file_sha256(args.model_lock)
        if model_lock_sha256 != protocol_meta["model_lock_sha256"]:
            raise ValueError("model lock digest differs from frozen protocol")
        locked_model_identity = verify_generation_model(
            load_model_lock(args.model_lock),
            profile_name,
            Path(args.model),
            verification_cache=args.model_verification_cache,
        )
        if args.experiment_lock is None:
            parser.error("frozen protocol requires --experiment-lock")
        if args.candidate_spec is None:
            parser.error("frozen protocol requires --candidate-spec")
        validate_committed_file(args.candidate_spec, REPO_ROOT, code_identity["commit"])
        candidate_spec = json.loads(args.candidate_spec.read_text(encoding="utf-8"))
        validate_candidate_spec(candidate_spec, require_candidate_videos=False)
        candidate_spec_sha256 = protocol_file_sha256(args.candidate_spec)
        if candidate_spec["split"] != args.expected_split:
            raise ValueError("candidate spec split differs from generation split")
        if candidate_spec["backbone"] != profile_name:
            raise ValueError("candidate spec backbone differs from generation profile")
        if candidate_spec["protocol_manifest_sha256"] != protocol_file_sha256(args.manifest):
            raise ValueError("candidate spec protocol digest differs from manifest")
        matching_candidates = [
            candidate
            for planned_case in candidate_spec["cases"]
            if planned_case["case_id"] == case_id
            for candidate in planned_case["candidates"]
            if candidate["seed"] == args.seed
        ]
        if len(matching_candidates) != 1:
            raise ValueError("candidate spec does not contain exactly one matching case/seed")
        planned_candidate = matching_candidates[0]
        experiment_lock = validate_experiment_lock(
            args.experiment_lock,
            REPO_ROOT,
            code_identity["commit"],
            protocol_manifest_sha256=protocol_file_sha256(args.manifest),
            model_lock_sha256=model_lock_sha256,
            split=args.expected_split,
            backbone=profile_name,
            candidate_spec_sha256=candidate_spec_sha256,
            artifact_root=args.output_root,
        )
        experiment_lock_sha256 = protocol_file_sha256(args.experiment_lock)
        implementation_sha256 = experiment_lock["implementation_sha256"]
        allowed_seeds = protocol_meta["candidate_seed_policy"]["candidate_seeds"]
        if args.seed not in allowed_seeds:
            raise ValueError(f"seed {args.seed} is outside frozen policy {allowed_seeds}")
        image_path = resolve_protocol_image(case, args.dataset_root)
        resolve_protocol_transforms(case, args.dataset_root)
    else:
        image_path = Path(case["image_prompt"])
        if not image_path.is_absolute():
            image_path = args.manifest.resolve().parent / image_path
    if not image_path.is_file():
        raise FileNotFoundError(f"Conditioning image not found: {image_path}")
    prompt = case["text_prompt"]
    image_sha256 = sha256_file(image_path)
    manifest_sha256 = sha256_file(args.manifest)
    runner_sha256 = sha256_file(Path(__file__).resolve())
    pipeline_path = (
        args.repo.resolve()
        / (
            "external/guidance_wan/pipeline_wan_i2v_full_guided.py"
            if args.backbone == "wan"
            else "external/guidance_cosmos/pipeline_cosmos2_5_predict_guided.py"
        )
    )
    pipeline_sha256 = sha256_file(pipeline_path) if args.method != "baseline" else None
    split_vae_report = None
    requested_vae_device = args.vae_device or args.pipe_device
    if (
        online_config is not None
        and torch.device(requested_vae_device) != torch.device(args.pipe_device)
    ):
        parser.error(
            "online geometry selection currently requires VAE and transformer on one device"
        )
    if torch.device(requested_vae_device) != torch.device(args.pipe_device):
        if not args.allow_split_vae or (
            args.split_vae_smoke_report is None and not is_split_vae_bootstrap
        ):
            parser.error(
                "Split VAE requires --allow-split-vae and an existing runtime receipt"
            )

    if needs_adapted_geco_contract:
        fixed_frames, guidance_step, guidance_lr, adapted_geco_schedule_meta = (
            resolve_adapted_geco_schedule(
                runtime_certification_spec,
                args.adapted_geco_schedule,
            )
        )
    else:
        fixed_frames = mapped_fixed_frames(args.frames)
        guidance_step = [0] * args.steps
        guidance_lr = [0.0] * args.steps
        adapted_geco_schedule_meta = None
    if needs_adapted_geco_contract:
        validate_runtime_certification_geco_schedule(
            args,
            runtime_certification_spec,
            schedule_id=args.adapted_geco_schedule,
            fixed_frames=fixed_frames,
            guidance_step=guidance_step,
            guidance_lr=guidance_lr,
        )
    expected_split_vae_contract = (
        split_vae_contract(
            args,
            code_identity=code_identity,
            runner_sha256=runner_sha256,
            pipeline_sha256=pipeline_sha256,
            model_content_manifest=adapted_geco_model_manifest,
        )
        if needs_adapted_geco_contract
        else None
    )
    if (
        torch.device(requested_vae_device) != torch.device(args.pipe_device)
        and not is_split_vae_bootstrap
    ):
        split_vae_report = load_and_validate_split_vae_runtime_receipt(
            args.split_vae_smoke_report,
            expected_contract=expected_split_vae_contract,
        )
    negative_prompt = (
        WAN_NEGATIVE
        if args.backbone == "wan" and args.wan_negative_prompt_mode == "frozen"
        else COSMOS_NEGATIVE
        if args.backbone == "cosmos"
        else None
    )
    config_for_id = {
        "manifest_sha256": manifest_sha256,
        "case_id": case_id,
        "image_sha256": image_sha256,
        "prompt": prompt,
        "backbone": args.backbone,
        "method": args.method,
        "code_identity": code_identity,
        "seed": args.seed,
        "model": model_identity(args.model),
        "locked_model_identity": locked_model_identity,
        "model_lock_sha256": model_lock_sha256,
        "model_content_verified": bool(is_frozen_protocol or adapted_geco_model_manifest),
        "adapted_geco_model_content_manifest": adapted_geco_model_manifest,
        "experiment_lock_sha256": experiment_lock_sha256,
        "implementation_sha256": implementation_sha256,
        "runner_sha256": runner_sha256,
        "pipeline_sha256": pipeline_sha256,
        "vggt_model": model_identity(args.vggt_model)
        if args.method == "adapted_geco"
        else None,
        "ufm_model": model_identity(args.ufm_model)
        if args.method == "adapted_geco"
        else None,
        "online_geometry_selection": (
            {
                "config_sha256": online_config_sha256,
                "config": online_config.resolved_dict(),
                "config_hash": online_config.config_hash,
                "geometry_backbone": online_geometry_identity,
            }
            if online_config is not None
            else None
        ),
        "steps": args.steps,
        "frames": args.frames,
        "height": args.height,
        "width": args.width,
        "fps": args.fps,
        "guidance_scale": args.guidance_scale,
        "negative_prompt": negative_prompt,
        "fixed_frames": fixed_frames,
        "adapted_geco_schedule": adapted_geco_schedule_meta,
        "guidance_step": guidance_step if args.method == "adapted_geco" else None,
        "guidance_lr": guidance_lr if args.method == "adapted_geco" else None,
        "ufm_scale": args.ufm_scale,
        "decode_spatial_scale": args.decode_spatial_scale,
        "max_relative_delta": args.max_relative_delta,
        "debug_guidance_consistency": (
            args.debug_guidance_consistency if args.method == "adapted_geco" else None
        ),
        "runtime_certification": args.runtime_certification if args.method == "adapted_geco" else None,
        "runtime_certification_spec_sha256": runtime_certification_spec_sha256,
        "scheduler_preflight_sha256": (
            args.expected_scheduler_preflight_sha256 if args.method == "adapted_geco" else None
        ),
        "model_content_manifest_sha256": (
            args.expected_model_content_manifest_sha256
            if args.method == "adapted_geco"
            else None
        ),
        "transformer_block_checkpointing": args.transformer_block_checkpointing,
        "split_vae_bootstrap": is_split_vae_bootstrap,
        "split_vae_smoke_report_sha256": (
            sha256_file(args.split_vae_smoke_report)
            if args.split_vae_smoke_report is not None
            else None
        ),
        "devices": {
            "pipe": args.pipe_device,
            "vae": args.vae_device or args.pipe_device,
            "metric": args.metric_device if args.method == "adapted_geco" else None,
            "cross_device_grad_via_cpu": args.cross_device_grad_via_cpu,
            "allow_split_vae": args.allow_split_vae,
            "split_vae_smoke_report": (
                str(args.split_vae_smoke_report.resolve())
                if args.split_vae_smoke_report is not None
                else None
            ),
        },
    }
    run_id = hashlib.sha256(
        json.dumps(config_for_id, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    output_dir = (
        args.output_root
        / args.backbone
        / args.method
        / case_id
        / f"seed_{args.seed}"
        / f"run_{run_id}"
    )
    video_path = output_dir / "video.mp4"
    if is_frozen_protocol and Path(planned_candidate["video"]).resolve() != video_path.resolve():
        raise ValueError(
            "candidate spec video path differs from the deterministic generation output: "
            f"planned={planned_candidate['video']} actual={video_path}"
        )
    metadata_path = output_dir / "metadata.json"
    complete_path = output_dir / "COMPLETE"
    if complete_path.exists() and not args.overwrite:
        stored = json.loads(metadata_path.read_text())
        if stored.get("run_id") != run_id:
            raise RuntimeError(f"COMPLETE metadata run_id mismatch: {output_dir}")
        if not video_path.is_file() or stored.get("video_sha256") != sha256_file(video_path):
            raise RuntimeError(f"COMPLETE video integrity mismatch: {output_dir}")
        if args.runtime_certification:
            verify_completed_runtime_certificate(
                stored,
                output_dir=output_dir,
                specification_sha256=runtime_certification_spec_sha256,
                adapted_geco_schedule=adapted_geco_schedule_meta,
                fixed_frames=fixed_frames,
                guidance_step=guidance_step,
                guidance_lr=guidance_lr,
            )
        probe_video(
            video_path,
            frames=args.frames,
            height=args.height,
            width=args.width,
            fps=args.fps,
        )
        if is_split_vae_bootstrap:
            certificate_path = output_dir / "runtime_certification.json"
            publish_split_vae_runtime_receipt(
                args.emit_split_vae_smoke_report,
                contract=expected_split_vae_contract,
                source_metadata=stored,
                metadata_path=metadata_path,
                complete_path=complete_path,
                video_path=video_path,
                certificate_path=certificate_path,
            )
        print(json.dumps({"status": "already_complete", "video": str(video_path)}, indent=2))
        return
    run_lock = GenerationRunLock(output_dir)
    atexit.register(run_lock.release)

    pipe, vae_device, transformer_block_checkpointing_actual = build_pipeline(args)
    if is_split_vae_bootstrap and not transformer_block_checkpointing_actual:
        raise RuntimeError("Split-VAE bootstrap requires active transformer checkpointing")
    if split_vae_report is not None and not transformer_block_checkpointing_actual:
        raise RuntimeError("Split-VAE receipt requires active transformer checkpointing")
    runtime_scheduler_identity = None
    runtime_scheduler_preflight = None
    if needs_adapted_geco_contract:
        runtime_scheduler_identity = validate_runtime_certification_scheduler(
            pipe.scheduler, runtime_certification_spec, repo=args.repo.resolve()
        )
        runtime_scheduler_preflight = validate_scheduler_preflight_receipt(
            args.scheduler_preflight_receipt,
            args.expected_scheduler_preflight_sha256,
            code_commit=code_identity["commit"],
            model=args.model,
            steps=args.steps,
            scheduler_identity=runtime_scheduler_identity,
            runtime_identity=cuda_runtime_identity(args.pipe_device),
            spec=runtime_certification_spec,
        )
    online_selector = None
    online_runtime = None
    if online_config is not None:
        online_selector, online_runtime = build_online_geometry_selector(
            args,
            pipe,
            online_config,
            args.online_selection_config,
            output_dir,
            online_adapter,
            online_geometry_identity,
        )
    image = Image.open(image_path).convert("RGB")
    generator = torch.Generator(device=args.pipe_device).manual_seed(args.seed)

    common = {
        "image": image,
        "prompt": prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.frames,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
    }
    if args.backbone == "wan":
        common["negative_prompt"] = negative_prompt
    else:
        common.update(video=None, negative_prompt=negative_prompt)

    if args.method == "adapted_geco":
        if (
            (
                torch.device(args.metric_device) != torch.device(args.pipe_device)
                or torch.device(requested_vae_device) != torch.device(args.pipe_device)
            )
            and not args.cross_device_grad_via_cpu
        ):
            raise ValueError(
                "Cross-device adapted-GeCo requires --cross-device-grad-via-cpu"
            )
        metric = load_metric(args)
        runtime_certification_events: list[dict] | None = (
            [] if args.runtime_certification else None
        )
        additional_inputs = {
            "residual_motion_metric": metric,
            "decode_spatial_scale": args.decode_spatial_scale,
            "cross_device_grad_via_cpu": args.cross_device_grad_via_cpu,
            "debug_guidance_consistency": (
                args.debug_guidance_consistency or args.runtime_certification
            ),
        }
        if runtime_certification_events is not None:
            additional_inputs["runtime_certification_events"] = runtime_certification_events
        if args.backbone == "wan":
            additional_inputs["max_relative_delta"] = args.max_relative_delta
        else:
            additional_inputs.update(
                transformer_activation_checkpointing=False,
                vae_activation_checkpointing=True,
                vae_checkpoint_mode="whole",
                debug_guidance_gradient=False,
                debug_guidance_stages=False,
                debug_guidance_dump_dir=None,
            )
        common.update(
            fixed_frames=fixed_frames,
            guidance_step=guidance_step,
            guidance_lr=guidance_lr,
            loss_fn="residual_motion",
            additional_inputs=additional_inputs,
        )
    if online_selector is not None:
        common["online_selector"] = online_selector

    if args.method in {"baseline", "online_geometry_selection"}:
        with torch.inference_mode():
            output = pipe(**common)
    else:
        output = pipe(**common)
    scheduler_trace = (
        scheduler_sampling_trace(pipe.scheduler) if args.method == "adapted_geco" else None
    )
    if needs_adapted_geco_contract:
        validate_scheduler_sampling_trace(scheduler_trace, runtime_scheduler_preflight)
    runtime_certificate = None
    certificate_path = None
    if args.runtime_certification:
        runtime_certificate = runtime_certification_report(
            runtime_certification_events,
            guidance_step=guidance_step,
            fixed_frames=fixed_frames,
            prediction_max_abs=float(
                runtime_certification_spec["thresholds"]["prediction_max_abs"]
            ),
            x0_max_abs=float(runtime_certification_spec["thresholds"]["x0_max_abs"]),
            vae_max_abs=float(runtime_certification_spec["thresholds"]["vae_max_abs"]),
            scheduler_config=dict(pipe.scheduler.config),
            scheduler_identity=runtime_scheduler_identity,
            scheduler_preflight=runtime_scheduler_preflight,
            specification_sha256=runtime_certification_spec_sha256,
            scheduler_sampling_trace=scheduler_trace,
            adapted_geco_schedule=adapted_geco_schedule_meta,
            guidance_lr=guidance_lr,
        )
        certificate_path = output_dir / "runtime_certification.json"
        temporary_certificate = output_dir / f".runtime_certificate.{uuid.uuid4().hex}.tmp.json"
        temporary_certificate.write_text(json.dumps(runtime_certificate, indent=2) + "\n")
        temporary_certificate.replace(certificate_path)
        if runtime_certificate["status"] != "passed":
            raise RuntimeError(
                "Wan adapted-GeCo runtime certification failed: "
                + "; ".join(runtime_certificate["failures"])
            )
    temporary_video = output_dir / f".video.{uuid.uuid4().hex}.tmp.mp4"
    export_to_video(output.frames[0], str(temporary_video), fps=args.fps)
    video_probe = probe_video(
        temporary_video,
        frames=args.frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
    )
    temporary_video.replace(video_path)
    video_sha256 = sha256_file(video_path)

    metadata = {
        "case_id": case_id,
        "case": case,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256,
        "protocol": {
            "is_frozen": is_frozen_protocol,
            "mode": args.protocol_mode,
            "split": case.get("split"),
            "expected_split": args.expected_split,
            "experiment_lock": (
                str(args.experiment_lock.resolve()) if args.experiment_lock else None
            ),
            "experiment_lock_sha256": experiment_lock_sha256,
        },
        "code_identity": code_identity,
        "image_path": str(image_path.resolve()),
        "image_sha256": image_sha256,
        "prompt": prompt,
        "backbone": args.backbone,
        "method": args.method,
        "run_id": run_id,
        "run_config": config_for_id,
        "run_config_sha256": hashlib.sha256(
            json.dumps(config_for_id, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "seed": args.seed,
        "model": model_identity(args.model),
        "locked_model_identity": locked_model_identity,
        "model_lock_sha256": model_lock_sha256,
        "model_content_verified": bool(is_frozen_protocol or adapted_geco_model_manifest),
        "adapted_geco_model_content_manifest": adapted_geco_model_manifest,
        "experiment_lock_sha256": experiment_lock_sha256,
        "implementation_sha256": implementation_sha256,
        "candidate_spec_sha256": candidate_spec_sha256,
        "vggt_model": model_identity(args.vggt_model)
        if args.method == "adapted_geco"
        else None,
        "ufm_model": model_identity(args.ufm_model)
        if args.method == "adapted_geco"
        else None,
        "online_geometry_selection": (
            {
                **online_runtime,
                "events": online_selector.events,
            }
            if online_selector is not None
            else None
        ),
        "generation": {
            "steps": args.steps,
            "frames": args.frames,
            "height": args.height,
            "width": args.width,
            "fps": args.fps,
            "guidance_scale": args.guidance_scale,
            "negative_prompt": negative_prompt,
            "wan_negative_prompt_mode": (
                args.wan_negative_prompt_mode if args.backbone == "wan" else None
            ),
        },
        "geco": {
            "enabled": args.method == "adapted_geco",
            "fixed_frames": fixed_frames,
            "guidance_step": guidance_step if args.method == "adapted_geco" else None,
            "guidance_lr": guidance_lr if args.method == "adapted_geco" else None,
            "ufm_scale": args.ufm_scale,
            "decode_spatial_scale": args.decode_spatial_scale,
            "max_relative_delta": args.max_relative_delta,
            "debug_guidance_consistency": (
                args.debug_guidance_consistency if args.method == "adapted_geco" else None
            ),
            "grad_through_vggt": False,
            "pair_mode": "adjacent",
            "time_travel": None,
            "schedule": adapted_geco_schedule_meta,
            "scheduler_sampling_trace": scheduler_trace,
            "scheduler_identity": runtime_scheduler_identity,
            "scheduler_preflight": runtime_scheduler_preflight,
            "schedule_source": (
                "Pre-registered GeCo adaptation for Wan flow matching; frozen VGGT "
                "geometry is re-estimated with stop-gradient at every update. The "
                "original DDIM time-travel/re-noising is intentionally not migrated."
            ),
            "vggt_strategy_argument": "once",
            "vggt_is_cached_across_updates": False,
            "transformer_block_checkpointing_actual": transformer_block_checkpointing_actual,
        },
        "devices": {
            "pipe": args.pipe_device,
            "vae": str(vae_device),
            "metric": args.metric_device if args.method == "adapted_geco" else None,
            "cross_device_grad_via_cpu": args.cross_device_grad_via_cpu,
            "allow_split_vae": args.allow_split_vae,
            "split_vae_smoke_report": (
                str(args.split_vae_smoke_report.resolve())
                if args.split_vae_smoke_report is not None
                else None
            ),
            "split_vae_smoke_report_sha256": (
                sha256_file(args.split_vae_smoke_report)
                if args.split_vae_smoke_report is not None
                else None
            ),
            "split_vae_bootstrap": is_split_vae_bootstrap,
            "transformer_block_checkpointing_actual": transformer_block_checkpointing_actual,
        },
        "runtime": runtime_meta(),
        "runtime_certification": (
            {
                "path": str(certificate_path.resolve()),
                "sha256": sha256_file(certificate_path),
                "status": runtime_certificate["status"],
                "schema": runtime_certificate["schema"],
                "specification_path": str(runtime_certification_path.resolve()),
                "specification_sha256": runtime_certification_spec_sha256,
                "scheduler": runtime_scheduler_identity,
                "scheduler_preflight": runtime_scheduler_preflight,
            }
            if runtime_certificate is not None
            else None
        ),
        "video": str(video_path),
        "video_sha256": video_sha256,
        "video_probe": video_probe,
        "runner_sha256": runner_sha256,
        "pipeline_sha256": pipeline_sha256,
    }
    temporary_metadata = output_dir / f".metadata.{uuid.uuid4().hex}.tmp.json"
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary_metadata.replace(metadata_path)
    temporary_complete = output_dir / f".complete.{uuid.uuid4().hex}.tmp"
    temporary_complete.write_text(f"{run_id}\n")
    temporary_complete.replace(complete_path)
    if is_split_vae_bootstrap:
        if certificate_path is None:
            raise RuntimeError("Split-VAE bootstrap cannot publish without runtime certificate")
        publish_split_vae_runtime_receipt(
            args.emit_split_vae_smoke_report,
            contract=expected_split_vae_contract,
            source_metadata=metadata,
            metadata_path=metadata_path,
            complete_path=complete_path,
            video_path=video_path,
            certificate_path=certificate_path,
        )
    run_lock.release()
    print(json.dumps({"video": str(video_path), "metadata": str(metadata_path)}, indent=2))


if __name__ == "__main__":
    main()
