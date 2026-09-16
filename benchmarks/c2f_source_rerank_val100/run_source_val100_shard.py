#!/usr/bin/env python3
"""Run one frozen source-reranking method over a deterministic Val100 shard."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
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


def load_pipeline_class(path: Path):
    spec = importlib.util.spec_from_file_location("_source_rerank_pipeline", path)
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
    orders = [int(case["split_order"]) for _, case in records]
    if len(records) != 100 or len(set(orders)) != 100:
        raise RuntimeError("frozen Val100 manifest must contain 100 unique split_order values")
    if any(case.get("protocol_split") != "validation" for _, case in records):
        raise RuntimeError("source-reranking runner only accepts validation cases")
    return records


def index_baselines(root: Path, case_ids: set[str], seed: int) -> dict[str, tuple[Path, Path, dict]]:
    indexed: dict[str, tuple[Path, Path, dict]] = {}
    for metadata_path in root.rglob("metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        case_id = metadata.get("case_id", metadata.get("run_config", {}).get("case_id"))
        record_seed = metadata.get("seed", metadata.get("run_config", {}).get("seed"))
        method = metadata.get("method", metadata.get("run_config", {}).get("method"))
        if case_id not in case_ids or record_seed != seed or method != "baseline":
            continue
        video_path = metadata_path.parent / "video.mp4"
        if not video_path.is_file():
            continue
        if case_id in indexed:
            raise RuntimeError(f"multiple seed-{seed} baseline videos for {case_id}")
        indexed[case_id] = (video_path, metadata_path, metadata)
    missing = sorted(case_ids.difference(indexed))
    if missing:
        raise RuntimeError(f"missing {len(missing)} paired baseline videos; first={missing[0]}")
    return indexed


def validate_baseline(case_id: str, case: dict, record: tuple[Path, Path, dict], config: dict) -> None:
    _, _, metadata = record
    generation = metadata.get("generation", metadata.get("run_config", {}))
    mismatches = {
        key: (generation.get(key), expected)
        for key, expected in config["generation"].items()
        if key != "negative_prompt" and generation.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"baseline generation mismatch for {case_id}: {mismatches}")
    if metadata.get("prompt") != case["text_prompt"]:
        raise RuntimeError(f"baseline prompt mismatch for {case_id}")
    if metadata.get("image_sha256") != case["image_sha256"]:
        raise RuntimeError(f"baseline conditioning image mismatch for {case_id}")


def load_completed_geometry(root: Path, case_id: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    case_dir = root / case_id
    geometry_path = case_dir / "GEOMETRY.npz"
    metadata_path = case_dir / "GEOMETRY_METADATA.json"
    complete_path = case_dir / "COMPLETE.json"
    if not geometry_path.is_file() or not metadata_path.is_file() or not complete_path.is_file():
        raise RuntimeError(f"missing completed geometry bundle for {case_id}: {case_dir}")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    geometry_sha = sha256_file(geometry_path)
    if complete.get("geometry_sha256") != geometry_sha:
        raise RuntimeError(f"geometry digest differs from COMPLETE for {case_id}")
    required = {"world_to_camera", "intrinsics", "depth", "confidence", "confidence_thresholds"}
    with np.load(geometry_path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise RuntimeError(f"geometry bundle for {case_id} is missing {sorted(missing)}")
        bundle = {key: np.asarray(archive[key]) for key in required}
    return bundle, {
        "path": str(geometry_path),
        "sha256": geometry_sha,
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "complete_path": str(complete_path),
        "complete_sha256": sha256_file(complete_path),
    }


def probe_video(path: Path) -> dict[str, Any]:
    if shutil.which("ffprobe"):
        command = [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate", "-of", "json", str(path),
        ]
        return json.loads(subprocess.check_output(command, text=True))["streams"][0]

    from imageio_ffmpeg import read_frames

    reader = read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        frame_count = sum(1 for _ in reader)
    finally:
        reader.close()
    width, height = (int(value) for value in metadata["size"])
    fps = float(metadata["fps"])
    if frame_count <= 0 or width <= 0 or height <= 0 or fps <= 0.0:
        raise RuntimeError(
            f"invalid imageio-ffmpeg video probe: frames={frame_count} size={width}x{height} fps={fps}"
        )
    rounded_fps = int(round(fps))
    if abs(fps - rounded_fps) > 1e-3:
        raise RuntimeError(f"generated video has non-integral fps={fps}")
    return {
        "nb_read_frames": str(frame_count),
        "width": width,
        "height": height,
        "r_frame_rate": f"{rounded_fps}/1",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--geometry-device", default="cuda:0")
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard-index < num-shards")
    if args.case_index is not None and (args.shard_index != 0 or args.num_shards != 1):
        raise ValueError("--case-index cannot be combined with sharding")

    config_path = args.config.resolve()
    manifest_path = args.manifest.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if config.get("schema") != "wan-c2f-source-rerank-val100-v1":
        raise RuntimeError("unexpected source-reranking config schema")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != config["manifest_sha256"]:
        raise RuntimeError("manifest digest differs from source-reranking config")
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
    case_ids = {case_id for case_id, _ in all_cases}
    baseline_index = index_baselines(Path(config["baseline_root"]), case_ids, int(config["seed"]))
    for case_id, case in all_cases:
        validate_baseline(case_id, case, baseline_index[case_id], config)
    for case_id, case in cases:
        image_path = (args.dataset_root / case["dataset_relative_image"]).resolve()
        if not image_path.is_file() or sha256_file(image_path) != case["image_sha256"]:
            raise RuntimeError(f"conditioning image is missing or changed for {case_id}: {image_path}")

    evidence = config["evidence"]
    evidence_kind = evidence["kind"]
    if evidence_kind not in {"draft", "online_snapshot", "online_refresh"}:
        raise RuntimeError(f"unsupported geometry evidence kind: {evidence_kind}")
    geometry_root = Path(evidence["geometry_root"])
    for case_id, _ in cases:
        load_completed_geometry(geometry_root, case_id)

    pipeline_path = Path(config["pipeline_path"])
    if not pipeline_path.is_absolute():
        pipeline_path = REPO_ROOT / pipeline_path
    pipeline_sha = sha256_file(pipeline_path)
    runner_sha = sha256_file(Path(__file__).resolve())
    config_sha = sha256_file(config_path)
    print("method:", config["method_id"])
    print("evidence:", evidence_kind)
    print("shard cases:", len(cases))
    print("git:", identity)
    print("pipeline_sha256:", pipeline_sha)
    print("runner_sha256:", runner_sha)
    if args.validate_only:
        print("validation-only gate passed")
        return

    PipelineClass = load_pipeline_class(pipeline_path)
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    pipe = PipelineClass.from_pretrained(args.model, vae=vae, torch_dtype=torch.bfloat16).to(args.device)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    online_callback = None
    if evidence_kind == "online_refresh":
        geometry_config = evidence["geometry_model"]
        source_root = Path(geometry_config["source_root"])
        if not source_root.is_absolute():
            source_root = REPO_ROOT / source_root
        checkpoint = Path(geometry_config["checkpoint"])
        if checkpoint.stat().st_size != int(geometry_config["checkpoint_size"]):
            raise RuntimeError("VGGT-Omega checkpoint size differs from method config")
        if sha256_file(checkpoint) != geometry_config["checkpoint_sha256"]:
            raise RuntimeError("VGGT-Omega checkpoint digest differs from method config")
        adapter = VGGTOmegaAdapter(
            source_root=source_root,
            checkpoint=checkpoint,
            device=args.geometry_device,
            image_resolution=int(geometry_config["image_resolution"]),
            preprocessing_mode=geometry_config["preprocessing_mode"],
            require_official_commit=True,
        )
        adapter.load()
        online_callback = WanPredictedCleanGeometry(
            vae=pipe.vae,
            geometry_adapter=adapter,
            frame_indices=evidence["frame_indices"],
            confidence_percentile=float(evidence["confidence_percentile"]),
            temporary_root=args.output_root,
        )

    generation = config["generation"]
    method = config["method"]
    for ordinal, (case_id, case) in enumerate(cases, start=1):
        initial_geometry, geometry_identity = load_completed_geometry(geometry_root, case_id)
        baseline_video, baseline_metadata, _ = baseline_index[case_id]
        output_dir = args.output_root.resolve() / config["method_id"] / f"{int(case['split_order']):03d}_{case_id}" / "seed_0"
        video_path = output_dir / "video.mp4"
        metadata_path = output_dir / "metadata.json"
        diagnostics_path = output_dir / "source_geometry_diagnostics.json"
        complete_path = output_dir / "COMPLETE.json"
        expected_complete = {
            "config_sha256": config_sha,
            "manifest_sha256": manifest_sha,
            "pipeline_sha256": pipeline_sha,
            "runner_sha256": runner_sha,
            "git_commit": identity["commit"],
            "initial_geometry_sha256": geometry_identity["sha256"],
        }
        if complete_path.is_file():
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            if all(complete.get(key) == value for key, value in expected_complete.items()) and video_path.is_file():
                print(f"[{ordinal}/{len(cases)}] skip complete {case_id}")
                continue
            raise RuntimeError(f"stale COMPLETE marker: {complete_path}")
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(f"refusing to overwrite partial output: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        callback_kwargs: dict[str, Any] = {}
        if online_callback is not None:
            online_callback.records.clear()
            callback_kwargs = {
                "c2f_geometry_update_callback": online_callback,
                "c2f_geometry_update_steps": evidence["refresh_steps"],
            }
        image_path = (args.dataset_root / case["dataset_relative_image"]).resolve()
        generator = torch.Generator(device=args.device).manual_seed(int(config["seed"]))
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
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
                c2f_geometry_bundle=initial_geometry,
                **method,
                **callback_kwargs,
            )
        torch.cuda.synchronize(torch.device(args.device))
        wall_seconds = time.perf_counter() - started
        peak_memory_mib = torch.cuda.max_memory_allocated(torch.device(args.device)) / 1024**2
        temporary_video = output_dir / f".video.{uuid.uuid4().hex}.tmp.mp4"
        export_to_video(output.frames[0], str(temporary_video), fps=int(generation["fps"]))
        probe = probe_video(temporary_video)
        actual_probe = {
            "nb_read_frames": probe.get("nb_read_frames"),
            "width": int(probe["width"]),
            "height": int(probe["height"]),
            "r_frame_rate": probe["r_frame_rate"],
        }
        expected_probe = {
            "nb_read_frames": str(generation["frames"]),
            "width": int(generation["width"]),
            "height": int(generation["height"]),
            "r_frame_rate": f"{int(generation['fps'])}/1",
        }
        if actual_probe != expected_probe:
            raise RuntimeError(f"generated video probe mismatch: {actual_probe} != {expected_probe}")
        temporary_video.replace(video_path)

        diagnostics = pipe._last_c2f_geometry_diagnostics
        if diagnostics is None:
            raise RuntimeError("source-reranking pipeline did not emit geometry diagnostics")
        atomic_json(diagnostics_path, diagnostics)
        metadata = {
            "schema": "wan-c2f-source-rerank-val100-output-v1",
            "method_id": config["method_id"],
            "case_id": case_id,
            "split_order": int(case["split_order"]),
            "seed": int(config["seed"]),
            "prompt": case["text_prompt"],
            "image_path": str(image_path),
            "image_sha256": case["image_sha256"],
            "generation": generation,
            "method": method,
            "evidence": evidence,
            "online_geometry_records": online_callback.records if online_callback is not None else [],
            "config": str(config_path),
            "config_sha256": config_sha,
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "pipeline": str(pipeline_path),
            "pipeline_sha256": pipeline_sha,
            "runner_sha256": runner_sha,
            "code_identity": identity,
            "baseline_video": str(baseline_video),
            "baseline_video_sha256": sha256_file(baseline_video),
            "baseline_metadata": str(baseline_metadata),
            "initial_geometry": geometry_identity,
            "diagnostics_sha256": sha256_file(diagnostics_path),
            "video_sha256": sha256_file(video_path),
            "video_probe": actual_probe,
            "wall_seconds": wall_seconds,
            "peak_memory_mib": peak_memory_mib,
            "hostname": os.uname().nodename,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
        }
        atomic_json(metadata_path, metadata)
        atomic_json(
            complete_path,
            {
                "status": "complete",
                **expected_complete,
                "video_sha256": metadata["video_sha256"],
                "metadata_sha256": sha256_file(metadata_path),
                "diagnostics_sha256": metadata["diagnostics_sha256"],
            },
        )
        print(
            f"[{ordinal}/{len(cases)}] saved {case_id} wall={wall_seconds:.1f}s peak={peak_memory_mib:.1f}MiB",
            flush=True,
        )
        del output, initial_geometry
        if hasattr(pipe.vae, "clear_cache"):
            pipe.vae.clear_cache()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
