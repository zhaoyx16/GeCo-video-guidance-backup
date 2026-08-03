#!/usr/bin/env python3
"""Extract VGGT-Omega geometry and freeze a candidate-pool manifest."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_selection.backbones.vggt_omega import VGGTOmegaAdapter, file_sha256
from geometry_selection.cache import canonical_hash, load_geometry_cache, save_geometry_cache
from geometry_selection.selection import (
    load_and_validate_candidate_pool,
    materialize_candidate_pool,
    validate_candidate_spec,
)
from scripts.extract_vggt_omega_geometry import (
    decode_video_frames,
    select_keyframes,
    video_frame_count,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT / "external/vggt_omega",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-keyframes", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument(
        "--preprocessing-mode",
        choices=("balanced", "max_size"),
        default="balanced",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen candidate pool: {args.output}")

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    validate_candidate_spec(spec)
    adapter = VGGTOmegaAdapter(
        source_root=args.source_root,
        checkpoint=args.checkpoint,
        device=args.device,
        image_resolution=args.image_resolution,
        preprocessing_mode=args.preprocessing_mode,
    )
    backbone_identity = adapter.cache_identity()
    geometry_keys: dict[tuple[str, str], str] = {}
    for case in spec.get("cases", []):
        for candidate in case.get("candidates", []):
            video = Path(candidate["video"]).resolve()
            video_hash = file_sha256(video)
            indices = select_keyframes(video_frame_count(video), args.num_keyframes)
            with tempfile.TemporaryDirectory(prefix="vggt_omega_frames_") as directory:
                image_paths, decode_identity = decode_video_frames(
                    video,
                    indices,
                    Path(directory),
                )
                provenance = {
                    "video_sha256": video_hash,
                    "keyframe_indices": indices,
                    "decoded_frames": decode_identity,
                    "geometry_backbone": backbone_identity,
                    "extractor_schema_version": 2,
                }
                cache_key = canonical_hash(provenance)
                try:
                    load_geometry_cache(
                        args.cache_root,
                        cache_key,
                        expected_provenance=provenance,
                    )
                    status = "cache_hit"
                except FileNotFoundError:
                    prediction = adapter.predict_image_paths(
                        image_paths,
                        keyframe_indices=indices,
                        hash_checkpoint=True,
                    )
                    save_geometry_cache(
                        args.cache_root,
                        cache_key,
                        prediction,
                        provenance,
                    )
                    status = "computed"
            key = (case["case_id"], candidate["candidate_id"])
            geometry_keys[key] = cache_key
            print(json.dumps({"case": key[0], "candidate": key[1], "status": status}))

    pool = materialize_candidate_pool(spec, geometry_keys)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(pool, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    load_and_validate_candidate_pool(
        temporary,
        expected_split=spec["split"],
        verify_video_hashes=True,
    )
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "cases": len(pool["cases"])}, indent=2))


if __name__ == "__main__":
    main()
