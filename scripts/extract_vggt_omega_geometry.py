#!/usr/bin/env python3
"""Extract a pinned VGGT-Omega geometry cache from selected video frames."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from geometry_selection.backbones.vggt_omega import VGGTOmegaAdapter, file_sha256
from geometry_selection.cache import canonical_hash, load_geometry_cache, save_geometry_cache


def decode_video_frames(video: Path, indices: list[int], output_dir: Path) -> list[Path]:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    paths: list[Path] = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"could not decode frame {index} from {video}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            path = output_dir / f"frame_{index:06d}.png"
            Image.fromarray(frame_rgb).save(path)
            paths.append(path)
    finally:
        capture.release()
    return paths


def video_frame_count(video: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if frames < 2:
        raise ValueError(f"video must contain at least 2 frames, got {frames}")
    return frames


def select_keyframes(total_frames: int, count: int) -> list[int]:
    if not 2 <= count <= total_frames:
        raise ValueError(f"keyframe count must be in [2,{total_frames}], got {count}")
    return np.linspace(0, total_frames - 1, count).round().astype(np.int64).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--num-keyframes", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--preprocessing-mode", choices=("balanced", "max_size"), default="balanced")
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    args = parser.parse_args()

    video = args.video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    total_frames = video_frame_count(video)
    indices = select_keyframes(total_frames, args.num_keyframes)
    adapter = VGGTOmegaAdapter(
        source_root=args.source_root,
        checkpoint=args.checkpoint,
        device=args.device,
        image_resolution=args.image_resolution,
        preprocessing_mode=args.preprocessing_mode,
    )
    provenance = {
        "video": {"path": str(video), "sha256": file_sha256(video)},
        "keyframe_indices": indices,
        "geometry_backbone": adapter.identity(hash_checkpoint=not args.skip_checkpoint_hash),
        "extractor_schema_version": 1,
    }
    cache_key = canonical_hash(provenance)
    try:
        cached = load_geometry_cache(args.cache_root, cache_key, expected_provenance=provenance)
    except FileNotFoundError:
        cached = None
    if cached is not None:
        print(json.dumps({"status": "cache_hit", "cache_key": cache_key}, indent=2))
        return

    with tempfile.TemporaryDirectory(prefix="vggt_omega_frames_") as directory:
        image_paths = decode_video_frames(video, indices, Path(directory))
        prediction = adapter.predict_image_paths(
            image_paths,
            keyframe_indices=indices,
            hash_checkpoint=not args.skip_checkpoint_hash,
        )
    arrays, metadata = save_geometry_cache(
        args.cache_root,
        cache_key,
        prediction,
        provenance,
    )
    print(
        json.dumps(
            {
                "status": "computed",
                "cache_key": cache_key,
                "arrays": str(arrays),
                "metadata": str(metadata),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
