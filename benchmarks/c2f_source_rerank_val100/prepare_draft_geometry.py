#!/usr/bin/env python3
"""Prepare one frozen 31-view VGGT-Omega draft bundle per Val100 case."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_selection.backbones.vggt_omega import VGGTOmegaAdapter  # noqa: E402


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


def git_identity() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
        ).strip()
    )
    return {"commit": commit, "dirty": dirty}


def ordered_cases(manifest: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    records = [(case_id, case) for case_id, case in manifest.items() if not case_id.startswith("_")]
    records.sort(key=lambda item: (int(item[1]["split_order"]), item[0]))
    if len(records) != 100 or any(case.get("protocol_split") != "validation" for _, case in records):
        raise RuntimeError("draft geometry requires the frozen 100-case validation manifest")
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
        raise RuntimeError(f"missing {len(missing)} baseline videos; first={missing[0]}")
    return indexed


def validate_baseline(case_id: str, case: dict, record: tuple[Path, Path, dict], config: dict) -> None:
    _, _, metadata = record
    generation = metadata.get("generation", metadata.get("run_config", {}))
    mismatches = {
        key: (generation.get(key), expected)
        for key, expected in config["generation"].items()
        if generation.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"baseline generation mismatch for {case_id}: {mismatches}")
    if metadata.get("prompt") != case["text_prompt"]:
        raise RuntimeError(f"baseline prompt mismatch for {case_id}")
    if metadata.get("image_sha256") != case["image_sha256"]:
        raise RuntimeError(f"baseline conditioning image mismatch for {case_id}")


def decode_frames(video: Path, indices: list[int], output_dir: Path) -> tuple[list[Path], tuple[int, int]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    selected = set(indices)
    paths: dict[int, Path] = {}
    size = None
    try:
        for frame_index in range(indices[-1] + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"could not decode frame {frame_index} from {video}")
            if frame_index not in selected:
                continue
            height, width = frame.shape[:2]
            size = size or (height, width)
            if size != (height, width):
                raise RuntimeError("video frame size changed")
            path = output_dir / f"frame_{frame_index:04d}.png"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"failed to write {path}")
            paths[frame_index] = path
    finally:
        capture.release()
    if set(paths) != selected or size is None:
        raise RuntimeError("not all requested frames were decoded")
    return [paths[index] for index in indices], size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
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
    if config.get("schema") != "c2f-source-draft-geometry-config-v1":
        raise RuntimeError("unexpected draft geometry config schema")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != config["manifest_sha256"]:
        raise RuntimeError("manifest digest differs from draft geometry config")
    identity = git_identity()
    if args.expected_git_commit and identity["commit"] != args.expected_git_commit:
        raise RuntimeError("repository commit differs from --expected-git-commit")
    if identity["dirty"] and not args.allow_dirty:
        raise RuntimeError("refusing to run from a dirty worktree")

    all_cases = ordered_cases(manifest)
    baseline_index = index_baselines(
        Path(config["baseline_root"]), {case_id for case_id, _ in all_cases}, int(config["seed"])
    )
    for case_id, case in all_cases:
        validate_baseline(case_id, case, baseline_index[case_id], config)
    if args.case_index is not None:
        if not 0 <= args.case_index < len(all_cases):
            raise ValueError("case index is outside Val100")
        cases = [all_cases[args.case_index]]
    else:
        cases = [record for index, record in enumerate(all_cases) if index % args.num_shards == args.shard_index]
    output_root = (args.output_root or Path(config["output_root"])).resolve()
    if args.validate_only:
        print(f"validated {len(cases)} of {len(all_cases)} frozen drafts; output={output_root}")
        return

    geometry_config = config["geometry"]
    source_root = Path(geometry_config["source_root"])
    if not source_root.is_absolute():
        source_root = REPO_ROOT / source_root
    checkpoint = Path(geometry_config["checkpoint"])
    if checkpoint.stat().st_size != int(geometry_config["checkpoint_size"]):
        raise RuntimeError("VGGT-Omega checkpoint size differs from config")
    adapter = VGGTOmegaAdapter(
        source_root=source_root,
        checkpoint=checkpoint,
        device=args.device,
        image_resolution=int(geometry_config["image_resolution"]),
        preprocessing_mode=geometry_config["preprocessing_mode"],
        require_official_commit=True,
    )
    load_started = time.perf_counter()
    adapter.load()
    torch.cuda.synchronize(torch.device(args.device))
    model_load_seconds = time.perf_counter() - load_started
    print(f"loaded VGGT-Omega in {model_load_seconds:.2f}s", flush=True)

    frame_indices = [int(value) for value in config["frame_indices"]]
    confidence_percentile = float(config["confidence_percentile"])
    output_root.mkdir(parents=True, exist_ok=True)
    for ordinal, (case_id, _) in enumerate(cases, start=1):
        video_path, baseline_metadata_path, _ = baseline_index[case_id]
        case_dir = output_root / case_id
        geometry_path = case_dir / "GEOMETRY.npz"
        metadata_path = case_dir / "GEOMETRY_METADATA.json"
        complete_path = case_dir / "COMPLETE.json"
        expected = {
            "config_sha256": sha256_file(config_path),
            "manifest_sha256": manifest_sha,
            "baseline_video_sha256": sha256_file(video_path),
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
        with tempfile.TemporaryDirectory(prefix=f"{case_id[:24]}_", dir=case_dir) as temporary:
            frame_paths, original_hw = decode_frames(video_path, frame_indices, Path(temporary))
            frame_hashes = [sha256_file(path) for path in frame_paths]
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
            started = time.perf_counter()
            geometry = adapter.predict_image_paths(
                frame_paths,
                keyframe_indices=frame_indices,
                hash_checkpoint=False,
            )
            torch.cuda.synchronize(torch.device(args.device))
            forward_seconds = time.perf_counter() - started
            peak_memory_mib = torch.cuda.max_memory_allocated(torch.device(args.device)) / 1024**2
        thresholds = np.percentile(
            geometry.confidence, confidence_percentile, axis=(1, 2)
        ).astype(np.float32)
        np.savez_compressed(
            geometry_path,
            world_to_camera=geometry.world_to_camera.astype(np.float32),
            intrinsics=geometry.intrinsics.astype(np.float32),
            depth=geometry.depth.astype(np.float32),
            confidence=geometry.confidence.astype(np.float32),
            confidence_thresholds=thresholds,
            keyframe_indices=geometry.keyframe_indices.astype(np.int64),
        )
        metadata = {
            "schema": "c2f-source-draft-geometry-bundle-v1",
            "case_id": case_id,
            "seed": int(config["seed"]),
            "baseline_video": str(video_path),
            "baseline_video_sha256": expected["baseline_video_sha256"],
            "baseline_metadata": str(baseline_metadata_path),
            "baseline_metadata_sha256": sha256_file(baseline_metadata_path),
            "config": str(config_path),
            "config_sha256": expected["config_sha256"],
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "code_identity": identity,
            "frame_indices": frame_indices,
            "frame_sha256": frame_hashes,
            "original_image_size_hw": list(original_hw),
            "processed_image_size_hw": list(geometry.image_size_hw),
            "geometry_metadata": geometry.metadata,
            "confidence_percentile": confidence_percentile,
            "model_load_seconds_shared": model_load_seconds,
            "geometry_forward_seconds": forward_seconds,
            "geometry_peak_allocated_mib": peak_memory_mib,
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
        print(
            f"[{ordinal}/{len(cases)}] {case_id} geometry={forward_seconds:.2f}s "
            f"peak={peak_memory_mib:.1f}MiB bundle={geometry_path.stat().st_size / 1024**2:.1f}MiB",
            flush=True,
        )


if __name__ == "__main__":
    main()
