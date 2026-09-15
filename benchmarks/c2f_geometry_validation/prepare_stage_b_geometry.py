#!/usr/bin/env python3
"""Prepare one pinned 31-frame geometry bundle for each locked Stage B draft."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def git_identity() -> dict:
    commit = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True).strip()
    )
    return {"commit": commit, "dirty": dirty}


def decode_frames(video: Path, indices: list[int], output_dir: Path) -> tuple[list[Path], tuple[int, int]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    selected = set(indices)
    paths = {}
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
    parser.add_argument(
        "--lock",
        type=Path,
        default=REPO_ROOT / "benchmarks/c2f_geometry_validation/STAGE_B_LOCK.json",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    if args.case_index is not None and (args.num_shards != 1 or args.shard_index != 0):
        raise ValueError("--case-index cannot be combined with sharding")
    lock_path = args.lock.resolve()
    lock = read_json(lock_path)
    if lock.get("schema") != "c2f-external-geometry-stage-b-lock-v1":
        raise RuntimeError("unexpected Stage B lock schema")
    identity = git_identity()
    if identity["dirty"] and not args.allow_dirty:
        raise RuntimeError("refusing to run from a dirty worktree")
    cases = lock["cases"]
    if args.case_index is not None:
        if not 0 <= args.case_index < len(cases):
            raise ValueError(f"case index must be in [0,{len(cases)-1}]")
        cases = [cases[args.case_index]]
    else:
        cases = [case for index, case in enumerate(cases) if index % args.num_shards == args.shard_index]
    for case in cases:
        video = Path(case["baseline_video"])
        if not video.is_file() or sha256_file(video) != case["baseline_video_sha256"]:
            raise RuntimeError(f"baseline draft is missing or changed for {case['case_id']}")
    output_root = (args.output_root or Path(lock["geometry_root"])).resolve()
    if args.validate_only:
        print(f"validated {len(cases)} Stage B drafts; output={output_root}")
        return

    geometry_config = lock["geometry"]
    checkpoint = Path(geometry_config["checkpoint"])
    if checkpoint.stat().st_size != geometry_config["checkpoint_size"]:
        raise RuntimeError("VGGT-Omega checkpoint size differs from the lock")
    adapter = VGGTOmegaAdapter(
        source_root=Path(geometry_config["source_root"]),
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

    frame_indices = [int(value) for value in lock["frame_indices"]]
    confidence_percentile = float(lock["evidence_rule"]["confidence_percentile"])
    output_root.mkdir(parents=True, exist_ok=True)
    for ordinal, case in enumerate(cases, start=1):
        case_id = case["case_id"]
        case_dir = output_root / case_id
        complete_path = case_dir / "COMPLETE.json"
        geometry_path = case_dir / "GEOMETRY.npz"
        metadata_path = case_dir / "GEOMETRY_METADATA.json"
        if complete_path.is_file():
            complete = read_json(complete_path)
            if (
                complete.get("lock_sha256") == sha256_file(lock_path)
                and geometry_path.is_file()
                and complete.get("geometry_sha256") == sha256_file(geometry_path)
            ):
                print(f"[{ordinal}/{len(cases)}] skip complete {case_id}")
                continue
            raise RuntimeError(f"stale complete output at {case_dir}")
        if case_dir.exists():
            raise RuntimeError(f"refusing to overwrite partial output at {case_dir}")
        frames_dir = case_dir / "frames"
        frames_dir.mkdir(parents=True)
        frame_paths, original_hw = decode_frames(Path(case["baseline_video"]), frame_indices, frames_dir)
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
        thresholds = np.percentile(geometry.confidence, confidence_percentile, axis=(1, 2)).astype(np.float32)
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
            "schema": "c2f-external-geometry-stage-b-bundle-v1",
            "case_id": case_id,
            "motion_stratum": case["motion_stratum"],
            "baseline_video": case["baseline_video"],
            "baseline_video_sha256": case["baseline_video_sha256"],
            "lock_path": str(lock_path),
            "lock_sha256": sha256_file(lock_path),
            "code_identity": identity,
            "frame_indices": frame_indices,
            "original_image_size_hw": list(original_hw),
            "processed_image_size_hw": list(geometry.image_size_hw),
            "geometry_metadata": geometry.metadata,
            "confidence_percentile": confidence_percentile,
            "model_load_seconds_shared": model_load_seconds,
            "geometry_forward_seconds": forward_seconds,
            "geometry_peak_allocated_mib": peak_memory_mib,
            "frame_inputs": [
                {"path": str(path), "sha256": sha256_file(path)} for path in frame_paths
            ],
            "geometry_path": str(geometry_path),
            "geometry_sha256": sha256_file(geometry_path),
        }
        atomic_json(metadata_path, metadata)
        atomic_json(
            complete_path,
            {
                "status": "complete",
                "case_id": case_id,
                "lock_sha256": sha256_file(lock_path),
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
