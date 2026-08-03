#!/usr/bin/env python3
"""Extract VGGT-Omega geometry and freeze a candidate-pool manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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
from geometry_selection.protocol import (
    FORMAL_SPLIT_COUNTS,
    file_sha256 as protocol_file_sha256,
    validate_candidate_spec_against_protocol,
    validate_committed_file,
    validate_committed_test_release,
    validate_formal_protocol,
)
from geometry_selection.model_lock import load_model_lock
from scripts.extract_vggt_omega_geometry import (
    decode_video_frames,
    select_keyframes,
    video_frame_count,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT / "external/vggt_omega",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--artifact-mode",
        choices=("formal", "legacy-debug"),
        required=True,
    )
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--model-lock", type=Path)
    parser.add_argument("--test-release", type=Path)
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
    code_commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    formal = args.artifact_mode == "formal"
    model_lock = None
    if formal:
        if args.expected_git_commit is None:
            parser.error("formal mode requires --expected-git-commit")
        if dirty or code_commit != args.expected_git_commit:
            raise RuntimeError(
                f"geometry producer is not the expected clean commit: "
                f"commit={code_commit}, dirty={dirty}"
            )
        validate_committed_file(args.protocol_manifest, REPO_ROOT, code_commit)
        if args.model_lock is None:
            parser.error("formal mode requires --model-lock")
        validate_committed_file(args.model_lock, REPO_ROOT, code_commit)
        protocol = validate_formal_protocol(args.protocol_manifest)
        if protocol_file_sha256(args.model_lock) != protocol["_meta"]["model_lock_sha256"]:
            raise ValueError("model lock digest differs from frozen protocol")
        model_lock = load_model_lock(args.model_lock)
    validate_candidate_spec_against_protocol(
        spec,
        args.protocol_manifest,
        args.dataset_root,
        formal=formal,
        expected_git_commit=code_commit if formal else None,
        model_lock=model_lock,
    )
    if formal and spec["split"] == "test":
        if args.test_release is None:
            parser.error("formal test preparation requires --test-release")
        validate_committed_test_release(
            args.test_release,
            REPO_ROOT,
            code_commit,
            {
                "schema": "geometry-test-release-v1",
                "protocol_manifest_sha256": protocol_file_sha256(args.protocol_manifest),
            },
        )
    adapter = VGGTOmegaAdapter(
        source_root=args.source_root,
        checkpoint=args.checkpoint,
        device=args.device,
        image_resolution=args.image_resolution,
        preprocessing_mode=args.preprocessing_mode,
    )
    backbone_identity = adapter.cache_identity()
    if formal:
        expected_geometry = model_lock["geometry_backbone"]
        geometry_mismatches = {
            key: (backbone_identity.get(key), expected)
            for key, expected in expected_geometry.items()
            if backbone_identity.get(key) != expected
        }
        if geometry_mismatches:
            raise ValueError(f"geometry backbone identity mismatch: {geometry_mismatches}")
    producer_identity = {
        "commit": code_commit,
        "dirty": dirty,
        "adapter_sha256": file_sha256(REPO_ROOT / "geometry_selection/backbones/vggt_omega.py"),
        "extractor_sha256": file_sha256(REPO_ROOT / "scripts/extract_vggt_omega_geometry.py"),
        "preparer_sha256": file_sha256(Path(__file__).resolve()),
    }
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
                    "producer": producer_identity,
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

    pool = materialize_candidate_pool(
        spec,
        geometry_keys,
        artifact_mode=args.artifact_mode,
        producer_identity=producer_identity,
        candidate_spec_sha256=file_sha256(args.spec.resolve()),
    )
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
        expected_case_count=FORMAL_SPLIT_COUNTS[spec["split"]] if formal else None,
    )
    try:
        os.link(temporary, args.output)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite frozen candidate pool: {args.output}") from error
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output), "cases": len(pool["cases"])}, indent=2))


if __name__ == "__main__":
    main()
