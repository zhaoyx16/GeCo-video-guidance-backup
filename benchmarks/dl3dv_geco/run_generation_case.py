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
import os
import platform
import socket
import stat
import subprocess
import sys
import tempfile
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


def read_regular_file_bytes_no_follow(path: Path) -> bytes:
    """Read one regular file from one no-follow descriptor exactly once."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("this platform cannot enforce no-follow checkpoint reads")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("checkpoint contract input must be a regular file")
        blocks = []
        while True:
            block = os.read(descriptor, 8 << 20)
            if not block:
                break
            blocks.append(block)
        return b"".join(blocks)
    finally:
        os.close(descriptor)


def validate_lora_checkpoint(
    checkpoint: Path,
    expected_receipt_sha256: str,
    expected_step: int,
) -> dict:
    """Validate a model-level Diffusers LoRA checkpoint without trusting paths.

    Full-Graph DPO checkpoints are written by
    ``WanTransformer3DModel.save_lora_adapter``.  Their keys therefore do not
    have the pipeline-level ``transformer.`` prefix.  This validator binds the
    complete checkpoint directory before the loader is allowed to consume it.
    """

    if (
        len(expected_receipt_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_receipt_sha256)
        or isinstance(expected_step, bool)
        or expected_step <= 0
    ):
        raise ValueError("LoRA receipt SHA and expected step are invalid")
    if checkpoint.is_symlink():
        raise ValueError("LoRA checkpoint cannot be a symlink")
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_dir():
        raise ValueError("LoRA checkpoint must be a directory")
    receipt_path = checkpoint / "CHECKPOINT_RECEIPT.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("LoRA checkpoint receipt must be a regular file")
    receipt_payload = read_regular_file_bytes_no_follow(receipt_path)
    if hashlib.sha256(receipt_payload).hexdigest() != expected_receipt_sha256:
        raise ValueError("LoRA checkpoint receipt SHA mismatch")
    try:
        receipt = json.loads(receipt_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("LoRA checkpoint receipt is not valid UTF-8 JSON") from error
    receipt_step = receipt.get("step") if isinstance(receipt, dict) else None
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != "fullgraph-dpo-adapter-checkpoint-v1"
        or isinstance(receipt_step, bool)
        or not isinstance(receipt_step, int)
        or receipt_step != expected_step
    ):
        raise ValueError("LoRA checkpoint receipt contract mismatch")
    rows = receipt.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("LoRA checkpoint receipt contains no files")
    checked = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("LoRA checkpoint file records must be objects")
        name = row.get("name")
        digest = row.get("sha256")
        size = row.get("size")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in seen
            or name == receipt_path.name
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise ValueError("LoRA checkpoint file record is invalid")
        artifact = checkpoint / name
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError("LoRA checkpoint artifact must be a regular file")
        if artifact.stat().st_size != size or sha256_file(artifact) != digest:
            raise ValueError("LoRA checkpoint artifact content mismatch")
        seen.add(name)
        checked.append({"name": name, "sha256": digest, "size": size})
    actual = {
        path.name
        for path in checkpoint.iterdir()
        if path.name != receipt_path.name
    }
    if actual != seen:
        raise ValueError("LoRA checkpoint has unreceipted or missing artifacts")
    weight_name = "pytorch_lora_weights.safetensors"
    if weight_name not in seen:
        raise ValueError("LoRA checkpoint lacks the expected safetensors weights")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_receipt": str(receipt_path.resolve()),
        "checkpoint_receipt_sha256": expected_receipt_sha256,
        # Kept in memory only until the private snapshot is sealed.  The
        # producer receipt must never be reopened after this authenticated
        # no-follow read.
        "_checkpoint_receipt_payload": receipt_payload,
        "step": expected_step,
        "files": checked,
        "weight_name": weight_name,
    }


def snapshot_lora_checkpoint(checkpoint: Path, identity: dict) -> tuple[tempfile.TemporaryDirectory, Path]:
    """Copy already-identified artifacts through no-follow file descriptors.

    Diffusers only sees this private snapshot.  A mutation or replacement of
    the producer path after validation therefore cannot change loaded bytes.
    """

    temporary = tempfile.TemporaryDirectory(prefix="fullgraph-dpo-lora-")
    target = Path(temporary.name) / "checkpoint"
    target.mkdir(mode=0o700)
    receipt_payload = identity.get("_checkpoint_receipt_payload")
    if not isinstance(receipt_payload, bytes):
        raise ValueError("LoRA identity lacks authenticated receipt bytes")
    if hashlib.sha256(receipt_payload).hexdigest() != identity["checkpoint_receipt_sha256"]:
        raise ValueError("authenticated LoRA receipt bytes changed in memory")
    source_root = checkpoint.resolve(strict=True)
    try:
        receipt_destination = target / "CHECKPOINT_RECEIPT.json"
        with receipt_destination.open("xb") as handle:
            handle.write(receipt_payload)
            handle.flush()
            os.fsync(handle.fileno())
        receipt_destination.chmod(0o400)

        # Artifact bytes are copied and re-hashed from no-follow descriptors.
        # Unlike the receipt, they are never interpreted before this snapshot.
        for row in identity["files"]:
            source = source_root / row["name"]
            if not hasattr(os, "O_NOFOLLOW"):
                raise RuntimeError("this platform cannot enforce no-follow checkpoint reads")
            flags = os.O_RDONLY | os.O_NOFOLLOW
            descriptor = os.open(source, flags)
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError("LoRA snapshot source must be a regular file")
                digest = hashlib.sha256()
                destination = target / row["name"]
                with destination.open("xb") as handle:
                    while True:
                        block = os.read(descriptor, 8 << 20)
                        if not block:
                            break
                        digest.update(block)
                        handle.write(block)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                os.close(descriptor)
            if digest.hexdigest() != row["sha256"]:
                raise ValueError("LoRA source changed after validation")
            destination.chmod(0o400)
        target.chmod(0o500)
        return temporary, target
    except BaseException:
        temporary.cleanup()
        raise


def load_lora_transformer(
    model: str,
    checkpoint: Path,
    lora_identity: dict,
    mode: str,
    *,
    transformer_class=None,
):
    """Load the SHA-validated model-level adapter in one explicit state."""

    if mode not in {"base", "adapted"}:
        raise ValueError("LoRA mode must be base or adapted")
    if transformer_class is None:
        from diffusers import WanTransformer3DModel

        transformer_class = WanTransformer3DModel
    transformer = transformer_class.from_pretrained(
        model,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    transformer.load_lora_adapter(
        checkpoint,
        weight_name=lora_identity["weight_name"],
        use_safetensors=True,
        local_files_only=True,
        prefix=None,
        adapter_name="fullgraph_dpo",
    )
    if mode == "base":
        transformer.disable_adapters()
    else:
        transformer.set_adapters("fullgraph_dpo")
    return transformer


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


def mapped_fixed_frames(num_frames: int) -> list[int]:
    # Original GeCo uses 1-based 12,24,36,48 over 49 CogVideoX frames.
    fractions = (11 / 48, 23 / 48, 35 / 48, 47 / 48)
    return sorted({min(num_frames - 1, round(fraction * (num_frames - 1))) for fraction in fractions})


def adapted_geco_schedule(
    num_steps: int,
    start_fraction: float,
    end_fraction: float,
    repeats_per_step: int,
    learning_rate: float,
) -> tuple[list[int], list[float]]:
    """Create a one-update default schedule for the stop-gradient VGGT surrogate."""

    if not 0.0 <= start_fraction < end_fraction <= 1.0:
        raise ValueError("Guidance fractions must satisfy 0 <= start < end <= 1")
    if repeats_per_step < 1:
        raise ValueError("repeats_per_step must be positive")
    start = min(num_steps - 1, int(round(start_fraction * num_steps)))
    end = min(num_steps, max(start + 1, int(round(end_fraction * num_steps))))
    repeats = [0] * num_steps
    for index in range(start, end):
        repeats[index] = repeats_per_step
    return repeats, [learning_rate if repeat else 0.0 for repeat in repeats]


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

        if args.method in {"baseline", "lora_dpo"}:
            from diffusers import WanImageToVideoPipeline as Pipeline
        else:
            Pipeline = load_class(
                repo / "external/guidance_wan/pipeline_wan_i2v_full_guided.py",
                "WanImageToVideoPipeline",
            )
        vae = AutoencoderKLWan.from_pretrained(
            args.model, subfolder="vae", torch_dtype=torch.float32
        )
        if args.method == "lora_dpo":
            transformer = load_lora_transformer(
                args.model,
                args.lora_checkpoint,
                args.lora_identity,
                args.lora_mode,
            )
            pipe = Pipeline.from_pretrained(
                args.model,
                transformer=transformer,
                vae=vae,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            ).to(args.pipe_device)
        else:
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

    if args.method == "adapted_geco" and args.transformer_block_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
    vae_device = place_vae(
        pipe,
        args.vae_device,
        args.pipe_device,
        backbone=args.backbone,
        method=args.method,
        allow_split_vae=args.allow_split_vae,
    )
    return pipe, vae_device


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=PROFILES, required=True)
    parser.add_argument(
        "--method",
        choices=("baseline", "adapted_geco", "online_geometry_selection", "lora_dpo"),
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
    parser.add_argument("--guidance-start-fraction", type=float, default=0.40)
    parser.add_argument("--guidance-end-fraction", type=float, default=0.84)
    parser.add_argument("--guidance-repeats", type=int, default=1)
    parser.add_argument("--guidance-lr", type=float, default=0.1)
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--lora-checkpoint", type=Path)
    parser.add_argument("--expected-lora-checkpoint-receipt-sha256")
    parser.add_argument("--expected-lora-step", type=int)
    parser.add_argument("--lora-mode", choices=("base", "adapted"))
    args = parser.parse_args()

    for key, value in PROFILES[args.backbone].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.method == "adapted_geco" and args.decode_spatial_scale != 1.0:
        parser.error("Formal adapted-GeCo benchmark requires --decode-spatial-scale 1.0")
    lora_arguments = (
        args.lora_checkpoint,
        args.expected_lora_checkpoint_receipt_sha256,
        args.expected_lora_step,
        args.lora_mode,
    )
    if args.method == "lora_dpo":
        if args.backbone != "wan" or any(value is None for value in lora_arguments):
            parser.error("Wan LoRA-DPO requires all explicit LoRA checkpoint arguments")
        args.lora_identity = validate_lora_checkpoint(
            args.lora_checkpoint,
            args.expected_lora_checkpoint_receipt_sha256,
            args.expected_lora_step,
        )
        args.lora_snapshot, args.lora_checkpoint = snapshot_lora_checkpoint(
            Path(args.lora_identity["checkpoint"]), args.lora_identity
        )
        del args.lora_identity["_checkpoint_receipt_payload"]
    else:
        if any(value is not None for value in lora_arguments):
            parser.error("LoRA checkpoint arguments require --method lora_dpo")
        args.lora_identity = None

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
    pipeline_sha256 = (
        sha256_file(pipeline_path)
        if args.method not in {"baseline", "lora_dpo"}
        else None
    )
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
        if not args.allow_split_vae or args.split_vae_smoke_report is None:
            parser.error(
                "Split VAE requires --allow-split-vae and --split-vae-smoke-report"
            )
        split_vae_report = json.loads(args.split_vae_smoke_report.read_text())
        expected_report = {
            "passed": True,
            "backbone": args.backbone,
            "pipeline_sha256": pipeline_sha256,
            "pipe_device": args.pipe_device,
            "vae_device": requested_vae_device,
            "metric_device": args.metric_device,
            "cross_device_grad_via_cpu": args.cross_device_grad_via_cpu,
        }
        mismatches = {
            key: (split_vae_report.get(key), expected)
            for key, expected in expected_report.items()
            if split_vae_report.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(f"Split-VAE smoke report mismatch: {mismatches}")

    fixed_frames = mapped_fixed_frames(args.frames)
    guidance_step, guidance_lr = adapted_geco_schedule(
        args.steps,
        args.guidance_start_fraction,
        args.guidance_end_fraction,
        args.guidance_repeats,
        args.guidance_lr,
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
        "model_content_verified": bool(is_frozen_protocol),
        "experiment_lock_sha256": experiment_lock_sha256,
        "implementation_sha256": implementation_sha256,
        "runner_sha256": runner_sha256,
        "pipeline_sha256": pipeline_sha256,
        "lora_dpo": (
            {"mode": args.lora_mode, **args.lora_identity}
            if args.method == "lora_dpo"
            else None
        ),
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
        "guidance_step": guidance_step if args.method == "adapted_geco" else None,
        "guidance_lr": guidance_lr if args.method == "adapted_geco" else None,
        "ufm_scale": args.ufm_scale,
        "decode_spatial_scale": args.decode_spatial_scale,
        "max_relative_delta": args.max_relative_delta,
        "transformer_block_checkpointing": args.transformer_block_checkpointing,
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
        probe_video(
            video_path,
            frames=args.frames,
            height=args.height,
            width=args.width,
            fps=args.fps,
        )
        print(json.dumps({"status": "already_complete", "video": str(video_path)}, indent=2))
        return
    run_lock = GenerationRunLock(output_dir)
    atexit.register(run_lock.release)

    pipe, vae_device = build_pipeline(args)
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
            torch.device(args.metric_device) != torch.device(args.pipe_device)
            and not args.cross_device_grad_via_cpu
        ):
            raise ValueError(
                "Cross-device metric guidance requires --cross-device-grad-via-cpu"
            )
        metric = load_metric(args)
        additional_inputs = {
            "residual_motion_metric": metric,
            "decode_spatial_scale": args.decode_spatial_scale,
            "cross_device_grad_via_cpu": args.cross_device_grad_via_cpu,
        }
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

    if args.method in {"baseline", "online_geometry_selection", "lora_dpo"}:
        with torch.inference_mode():
            output = pipe(**common)
    else:
        output = pipe(**common)
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
        "model_content_verified": bool(is_frozen_protocol),
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
            "grad_through_vggt": False,
            "pair_mode": "adjacent",
            "time_travel": None,
            "schedule_source": (
                "Adapted single-repeat schedule for a flow-matching backbone; "
                "VGGT geometry is re-estimated with stop-gradient at every update."
            ),
            "guidance_start_fraction": args.guidance_start_fraction,
            "guidance_end_fraction": args.guidance_end_fraction,
            "guidance_repeats": args.guidance_repeats,
            "vggt_strategy_argument": "once",
            "vggt_is_cached_across_updates": False,
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
        },
        "runtime": runtime_meta(),
        "video": str(video_path),
        "video_sha256": video_sha256,
        "video_probe": video_probe,
        "runner_sha256": runner_sha256,
        "pipeline_sha256": pipeline_sha256,
        "lora_dpo": (
            {"mode": args.lora_mode, **args.lora_identity}
            if args.method == "lora_dpo"
            else None
        ),
    }
    temporary_metadata = output_dir / f".metadata.{uuid.uuid4().hex}.tmp.json"
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary_metadata.replace(metadata_path)
    temporary_complete = output_dir / f".complete.{uuid.uuid4().hex}.tmp"
    temporary_complete.write_text(f"{run_id}\n")
    temporary_complete.replace(complete_path)
    run_lock.release()
    print(json.dumps({"video": str(video_path), "metadata": str(metadata_path)}, indent=2))


if __name__ == "__main__":
    main()
