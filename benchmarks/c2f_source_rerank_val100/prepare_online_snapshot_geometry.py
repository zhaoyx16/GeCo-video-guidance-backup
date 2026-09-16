#!/usr/bin/env python3
"""Capture one frozen step-19 predicted-clean geometry bundle per Val100 case."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_selection.backbones.vggt_omega import VGGTOmegaAdapter  # noqa: E402
from geometry_selection.wan_online_geometry import WanPredictedCleanGeometry  # noqa: E402


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def load_pipeline_class(path: Path):
    spec = importlib.util.spec_from_file_location("_online_snapshot_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WanImageToVideoPipeline


def git_identity() -> dict[str, Any]:
    commit = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True).strip()
    )
    return {"commit": commit, "dirty": dirty}


def ordered_cases(manifest: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    records = [(case_id, case) for case_id, case in manifest.items() if not case_id.startswith("_")]
    records.sort(key=lambda item: (int(item[1]["split_order"]), item[0]))
    if len(records) != 100 or any(case.get("protocol_split") != "validation" for _, case in records):
        raise RuntimeError("online snapshot capture requires the frozen Val100 manifest")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--geometry-device", default="cuda:0")
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    if args.case_index is not None and (args.shard_index != 0 or args.num_shards != 1):
        raise ValueError("--case-index cannot be combined with sharding")

    config_path = args.config.resolve()
    manifest_path = args.manifest.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if config.get("schema") != "c2f-source-online-snapshot-geometry-config-v1":
        raise RuntimeError("unexpected online snapshot config schema")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != config["manifest_sha256"]:
        raise RuntimeError("manifest digest differs from online snapshot config")
    identity = git_identity()
    if args.expected_git_commit and identity["commit"] != args.expected_git_commit:
        raise RuntimeError("repository commit differs from --expected-git-commit")
    if identity["dirty"] and not args.allow_dirty:
        raise RuntimeError("refusing to run from a dirty worktree")

    all_cases = ordered_cases(manifest)
    if args.case_index is not None:
        if not 0 <= args.case_index < len(all_cases):
            raise ValueError("case index is outside Val100")
        cases = [all_cases[args.case_index]]
    else:
        cases = [record for index, record in enumerate(all_cases) if index % args.num_shards == args.shard_index]
    for case_id, case in cases:
        image_path = (args.dataset_root / case["dataset_relative_image"]).resolve()
        if not image_path.is_file() or sha256_file(image_path) != case["image_sha256"]:
            raise RuntimeError(f"conditioning image is missing or changed for {case_id}: {image_path}")
    output_root = (args.output_root or Path(config["output_root"])).resolve()
    pipeline_path = Path(config["pipeline_path"])
    if not pipeline_path.is_absolute():
        pipeline_path = REPO_ROOT / pipeline_path
    geometry_config = config["geometry"]
    geometry_source = Path(geometry_config["source_root"])
    if not geometry_source.is_absolute():
        geometry_source = REPO_ROOT / geometry_source
    checkpoint = Path(geometry_config["checkpoint"])
    if checkpoint.stat().st_size != int(geometry_config["checkpoint_size"]):
        raise RuntimeError("VGGT-Omega checkpoint size differs from config")
    if args.validate_only:
        print(f"validated {len(cases)} of {len(all_cases)} online snapshots; output={output_root}")
        return

    PipelineClass = load_pipeline_class(pipeline_path)
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    pipe = PipelineClass.from_pretrained(args.model, vae=vae, torch_dtype=torch.bfloat16).to(args.device)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    adapter = VGGTOmegaAdapter(
        source_root=geometry_source,
        checkpoint=checkpoint,
        device=args.geometry_device,
        image_resolution=int(geometry_config["image_resolution"]),
        preprocessing_mode=geometry_config["preprocessing_mode"],
        require_official_commit=True,
    )
    adapter.load()
    geometry_callback = WanPredictedCleanGeometry(
        vae=pipe.vae,
        geometry_adapter=adapter,
        frame_indices=config["frame_indices"],
        confidence_percentile=float(config["confidence_percentile"]),
        temporary_root=output_root,
    )
    generation = config["generation"]
    output_root.mkdir(parents=True, exist_ok=True)
    for ordinal, (case_id, case) in enumerate(cases, start=1):
        case_dir = output_root / case_id
        geometry_path = case_dir / "GEOMETRY.npz"
        metadata_path = case_dir / "GEOMETRY_METADATA.json"
        complete_path = case_dir / "COMPLETE.json"
        expected = {
            "config_sha256": sha256_file(config_path),
            "manifest_sha256": manifest_sha,
            "image_sha256": case["image_sha256"],
        }
        if complete_path.is_file():
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            if all(complete.get(key) == value for key, value in expected.items()) and geometry_path.is_file():
                print(f"[{ordinal}/{len(cases)}] skip complete {case_id}")
                continue
            raise RuntimeError(f"stale COMPLETE marker: {complete_path}")
        if case_dir.exists() and any(case_dir.iterdir()):
            raise RuntimeError(f"refusing to overwrite partial output: {case_dir}")
        case_dir.mkdir(parents=True, exist_ok=True)
        geometry_callback.records.clear()
        captured: dict[str, np.ndarray] = {}

        def capture(step: int, x0_latents: torch.Tensor) -> None:
            captured.update(geometry_callback(step, x0_latents))
            return None

        image_path = (args.dataset_root / case["dataset_relative_image"]).resolve()
        generator = torch.Generator(device=args.device).manual_seed(int(config["seed"]))
        started = time.perf_counter()
        with torch.inference_mode():
            output = pipe(
                prompt=case["text_prompt"],
                negative_prompt=generation["negative_prompt"],
                image=Image.open(image_path).convert("RGB"),
                height=int(generation["height"]),
                width=int(generation["width"]),
                num_frames=int(generation["frames"]),
                num_inference_steps=int(generation["steps"]),
                guidance_scale=float(generation["guidance_scale"]),
                generator=generator,
                output_type="latent",
                c2f_geometry_update_callback=capture,
                c2f_geometry_update_steps=[int(config["snapshot_step"])],
                c2f_stop_after_geometry_update=True,
            )
        wall_seconds = time.perf_counter() - started
        if not captured or len(geometry_callback.records) != 1:
            raise RuntimeError("online snapshot callback did not produce exactly one geometry bundle")
        atomic_npz(geometry_path, captured)
        metadata = {
            "schema": "c2f-source-online-snapshot-geometry-bundle-v1",
            "case_id": case_id,
            "case_index": int(case["split_order"]) - 101,
            "seed": int(config["seed"]),
            "prompt": case["text_prompt"],
            "image_path": str(image_path),
            "image_sha256": case["image_sha256"],
            "snapshot_step": int(config["snapshot_step"]),
            "generation": generation,
            "config": str(config_path),
            "config_sha256": expected["config_sha256"],
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "pipeline": str(pipeline_path),
            "pipeline_sha256": sha256_file(pipeline_path),
            "code_identity": identity,
            "geometry_model": adapter.identity(hash_checkpoint=True),
            "callback_record": geometry_callback.records[0],
            "wall_seconds": wall_seconds,
            "geometry_sha256": sha256_file(geometry_path),
        }
        atomic_json(metadata_path, metadata)
        atomic_json(
            complete_path,
            {
                "status": "complete",
                **expected,
                "geometry_sha256": metadata["geometry_sha256"],
                "metadata_sha256": sha256_file(metadata_path),
            },
        )
        print(f"[{ordinal}/{len(cases)}] saved {case_id} wall={wall_seconds:.1f}s", flush=True)
        del output, captured
        if hasattr(pipe.vae, "clear_cache"):
            pipe.vae.clear_cache()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
