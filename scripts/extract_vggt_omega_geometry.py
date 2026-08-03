#!/usr/bin/env python3
"""Extract a pinned VGGT-Omega geometry cache from selected video frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_selection.backbones.vggt_omega import VGGTOmegaAdapter, file_sha256
from geometry_selection.cache import canonical_hash, load_geometry_cache, save_geometry_cache


def decode_video_frames(
    video: Path,
    indices: list[int],
    output_dir: Path,
) -> tuple[list[Path], dict]:
    import cv2

    if indices != sorted(set(indices)) or not indices or indices[0] < 0:
        raise ValueError("frame indices must be non-negative, sorted, and unique")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    backend = capture.getBackendName() if hasattr(capture, "getBackendName") else "unknown"
    selected = set(indices)
    paths_by_index: dict[int, Path] = {}
    frame_records = []
    try:
        for index in range(indices[-1] + 1):
            ok, frame_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"could not decode frame {index} from {video}")
            if index not in selected:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            path = output_dir / f"frame_{index:06d}.png"
            Image.fromarray(frame_rgb).save(path)
            paths_by_index[index] = path
            frame_records.append(
                {
                    "index": index,
                    "shape": list(frame_rgb.shape),
                    "dtype": str(frame_rgb.dtype),
                    "pixels_sha256": hashlib.sha256(frame_rgb.tobytes()).hexdigest(),
                }
            )
    finally:
        capture.release()
    paths = [paths_by_index[index] for index in indices]
    return paths, {
        "decoder": "opencv-sequential-v1",
        "opencv_version": cv2.__version__,
        "backend": backend,
        "frames": frame_records,
    }


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
    with tempfile.TemporaryDirectory(prefix="vggt_omega_frames_") as directory:
        image_paths, decode_identity = decode_video_frames(video, indices, Path(directory))
        provenance = {
            "video_sha256": file_sha256(video),
            "keyframe_indices": indices,
            "decoded_frames": decode_identity,
            "geometry_backbone": (
                adapter.cache_identity()
                if not args.skip_checkpoint_hash
                else adapter.identity(hash_checkpoint=False)
            ),
            "extractor_schema_version": 2,
        }
        cache_key = canonical_hash(provenance)
        try:
            cached = load_geometry_cache(
                args.cache_root,
                cache_key,
                expected_provenance=provenance,
            )
        except FileNotFoundError:
            cached = None
        if cached is not None:
            print(json.dumps({"status": "cache_hit", "cache_key": cache_key}, indent=2))
            return
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
