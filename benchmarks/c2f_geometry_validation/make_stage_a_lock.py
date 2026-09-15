#!/usr/bin/env python3
"""Freeze the outcome-independent five-scene Stage A geometry protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_STRATA = [
    "forward",
    "forward_left",
    "forward_right",
    "lateral_left",
    "lateral_right",
]
EXPECTED_ZERO_OVERLAP = {"test": 0, "validation": 0, "debug": 0}
OMEGA_SHA256 = "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934"
OMEGA_SIZE = 4_576_706_117
OMEGA_COMMIT = "39a0cb8af88554f15ddcb5354cd52bde588fa014"


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
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "benchmarks/c2f_validation/dev25_manifest.json",
    )
    parser.add_argument(
        "--p0-lock",
        type=Path,
        default=REPO_ROOT / "benchmarks/c2f_forensics/P0_LOCK.json",
    )
    parser.add_argument(
        "--c2f-root",
        type=Path,
        default=Path("/vol/dissolve/yz10325/outputs/c2f_validation_0914/dev25/c2f_k3_a0025"),
    )
    parser.add_argument(
        "--p0-replay-root",
        type=Path,
        default=Path(
            "/vol/dissolve/yz10325/outputs/c2f_p0_forensics_0915/"
            "replay_v1/baseline_observe/diagnostic/through_step_29"
        ),
    )
    parser.add_argument(
        "--dense-replay-root",
        type=Path,
        default=Path(
            "/vol/dissolve/yz10325/outputs/c2f_geometry_validation_0915/"
            "dense_replay_v1/baseline_observe/diagnostic/through_step_29"
        ),
    )
    parser.add_argument(
        "--omega-source-root",
        type=Path,
        default=REPO_ROOT / "external/vggt_omega",
    )
    parser.add_argument(
        "--omega-checkpoint",
        type=Path,
        default=Path("/vol/dissolve/yz10325/checkpoints/VGGT-Omega/vggt_omega_1b_512.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmarks/c2f_geometry_validation/STAGE_A_LOCK.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    p0_lock_path = args.p0_lock.resolve()
    manifest = read_json(manifest_path)
    p0_lock = read_json(p0_lock_path)

    if manifest.get("_meta", {}).get("reserved_overlap_counts") != EXPECTED_ZERO_OVERLAP:
        raise RuntimeError("Dev25 is not disjoint from all reserved splits")
    if p0_lock.get("reserved_overlap_counts") != EXPECTED_ZERO_OVERLAP:
        raise RuntimeError("P0 lock is not disjoint from all reserved splits")
    expected_manifest_sha = p0_lock["frozen_inputs"]["manifest"]["sha256"]
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        raise RuntimeError("Dev25 manifest differs from the frozen P0 manifest")

    checkpoint = args.omega_checkpoint.resolve()
    source_root = args.omega_source_root.resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size != OMEGA_SIZE:
        raise RuntimeError("VGGT-Omega checkpoint is missing or has the wrong size")
    upstream = (source_root / "UPSTREAM_COMMIT").read_text(encoding="utf-8").strip()
    if upstream != OMEGA_COMMIT:
        raise RuntimeError(f"VGGT-Omega source commit mismatch: {upstream}")

    selected = []
    for stratum in EXPECTED_STRATA:
        matches = [
            (case_id, case)
            for case_id, case in manifest.items()
            if not case_id.startswith("_")
            and case["c2f_dev_selection"]["stratum"] == stratum
            and case["c2f_dev_selection"]["stratum_quantile_rank"] == 2
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one middle-quantile case for {stratum}, got {len(matches)}")
        case_id, case = matches[0]
        c2f_dir = args.c2f_root.resolve() / case_id / "seed_0"
        c2f_metadata_path = c2f_dir / "metadata.json"
        c2f_video_path = c2f_dir / "video.mp4"
        sparse_replay_path = args.p0_replay_root.resolve() / case_id / "seed_0/replay.json"
        dense_replay_path = args.dense_replay_root.resolve() / case_id / "seed_0/replay.json"
        if not c2f_metadata_path.is_file() or not c2f_video_path.is_file() or not sparse_replay_path.is_file():
            raise FileNotFoundError(f"missing frozen artifact for {case_id}")
        metadata = read_json(c2f_metadata_path)
        baseline_video = Path(metadata["baseline_video"]).resolve()
        if not baseline_video.is_file():
            raise FileNotFoundError(baseline_video)
        selected.append(
            {
                "case_id": case_id,
                "motion_stratum": stratum,
                "selection_order": case["c2f_dev_selection"]["selection_order"],
                "selection_rule": "stratum_quantile_rank == 2",
                "conditioning_image": case["image_prompt"],
                "text_prompt": case["text_prompt"],
                "baseline_video": str(baseline_video),
                "baseline_video_sha256": metadata["baseline_video_sha256"],
                "c2f_video": str(c2f_video_path.resolve()),
                "c2f_video_sha256": metadata["video_sha256"],
                "c2f_metadata": str(c2f_metadata_path.resolve()),
                "c2f_metadata_sha256": sha256_file(c2f_metadata_path),
                "sparse_p0_replay": str(sparse_replay_path),
                "sparse_p0_replay_sha256": sha256_file(sparse_replay_path),
                "dense_replay": str(dense_replay_path),
            }
        )

    payload = {
        "schema": "c2f-external-geometry-stage-a-lock-v1",
        "purpose": "coordinate_and_signal_sanity_only",
        "selection_uses_generated_outputs": False,
        "selection_uses_metrics": False,
        "selection_rule": "one middle pose-quantile case per frozen Dev25 motion stratum",
        "manifest": {"path": str(manifest_path), "sha256": actual_manifest_sha},
        "p0_lock": {"path": str(p0_lock_path), "sha256": sha256_file(p0_lock_path)},
        "reserved_overlap_counts": EXPECTED_ZERO_OVERLAP,
        "cases": selected,
        "c2f_observation": {
            "replay_mode": "baseline_observe",
            "step": 20,
            "layer": 10,
            "expected_token_grid_thw": [31, 22, 40],
            "target_token_times": [10, 25],
            "memory_lags": [1, 2, 3],
            "temporal_pixel_frames_per_token": 4,
            "dense_sample_size_per_layer_step": 8192,
            "uses_actual_per_token_selected_source": True,
        },
        "geometry": {
            "name": "VGGT-Omega-1B-512",
            "source_root": str(source_root),
            "source_commit": OMEGA_COMMIT,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": OMEGA_SHA256,
            "checkpoint_size": OMEGA_SIZE,
            "image_resolution": 512,
            "preprocessing_mode": "balanced",
            "camera_convention": "opencv_world_to_camera",
            "depth_definition": "camera_z_depth",
        },
        "evidence_rule": {
            "confidence_percentile": 20.0,
            "depth_relative_tolerance": 0.15,
            "max_reprojection_error_tokens": 1.5,
            "behind_target_policy": "abstain_occluded",
            "in_front_of_target_policy": "reject_conflict",
            "low_confidence_or_out_of_bounds_policy": "abstain",
        },
        "frame_indices": [28, 32, 36, 40, 88, 92, 96, 100],
    }
    atomic_json(args.output.resolve(), payload)
    print(f"locked {len(selected)} cases: {args.output.resolve()}")
    for case in selected:
        print(case["motion_stratum"], case["case_id"])


if __name__ == "__main__":
    main()
