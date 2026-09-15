#!/usr/bin/env python3
"""Freeze the outcome-independent ten-scene Stage B geometry-gating experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_STRATA = ["forward", "forward_left", "forward_right", "lateral_left", "lateral_right"]
EXPECTED_ZERO_OVERLAP = {"test": 0, "validation": 0, "debug": 0}
SELECTED_RANKS = (1, 3)
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
        "--frozen-c2f-config",
        type=Path,
        default=REPO_ROOT / "benchmarks/c2f_validation/frozen_c2f_k3_a0025.json",
    )
    parser.add_argument(
        "--c2f-root",
        type=Path,
        default=Path("/vol/dissolve/yz10325/outputs/c2f_validation_0914/dev25/c2f_k3_a0025"),
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=Path("/vol/dissolve/yz10325/outputs/c2f_geometry_validation_0915/stage_b_geometry_v1"),
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
        "--pipeline",
        type=Path,
        default=REPO_ROOT / "external/guidance_wan/pipeline_wan_i2v_c2f_geometry_gate.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks/c2f_geometry_validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    p0_lock_path = args.p0_lock.resolve()
    c2f_config_path = args.frozen_c2f_config.resolve()
    manifest = read_json(manifest_path)
    p0_lock = read_json(p0_lock_path)
    c2f_config = read_json(c2f_config_path)
    if manifest.get("_meta", {}).get("reserved_overlap_counts") != EXPECTED_ZERO_OVERLAP:
        raise RuntimeError("Dev25 is not disjoint from all reserved splits")
    if p0_lock.get("reserved_overlap_counts") != EXPECTED_ZERO_OVERLAP:
        raise RuntimeError("P0 lock is not disjoint from all reserved splits")
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != p0_lock["frozen_inputs"]["manifest"]["sha256"]:
        raise RuntimeError("Dev25 manifest differs from the frozen P0 manifest")

    checkpoint = args.omega_checkpoint.resolve()
    source_root = args.omega_source_root.resolve()
    pipeline = args.pipeline.resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size != OMEGA_SIZE:
        raise RuntimeError("VGGT-Omega checkpoint is missing or has the wrong size")
    if (source_root / "UPSTREAM_COMMIT").read_text(encoding="utf-8").strip() != OMEGA_COMMIT:
        raise RuntimeError("VGGT-Omega source commit differs from the lock")
    if not pipeline.is_file():
        raise FileNotFoundError(pipeline)

    selected = []
    selected_ids = set()
    for stratum in EXPECTED_STRATA:
        for rank in SELECTED_RANKS:
            matches = [
                (case_id, case)
                for case_id, case in manifest.items()
                if not case_id.startswith("_")
                and case["c2f_dev_selection"]["stratum"] == stratum
                and case["c2f_dev_selection"]["stratum_quantile_rank"] == rank
            ]
            if len(matches) != 1:
                raise RuntimeError(f"expected one case for stratum={stratum}, rank={rank}; got {len(matches)}")
            case_id, case = matches[0]
            selected_ids.add(case_id)
            c2f_dir = args.c2f_root.resolve() / case_id / "seed_0"
            c2f_metadata_path = c2f_dir / "metadata.json"
            c2f_video_path = c2f_dir / "video.mp4"
            if not c2f_metadata_path.is_file() or not c2f_video_path.is_file():
                raise FileNotFoundError(f"missing frozen C2F artifact for {case_id}")
            metadata = read_json(c2f_metadata_path)
            baseline_video = Path(metadata["baseline_video"]).resolve()
            if not baseline_video.is_file():
                raise FileNotFoundError(baseline_video)
            selected.append(
                {
                    "case_id": case_id,
                    "motion_stratum": stratum,
                    "stratum_quantile_rank": rank,
                    "selection_order": case["c2f_dev_selection"]["selection_order"],
                    "selection_rule": "stratum_quantile_rank in {1,3}",
                    "conditioning_image": case["image_prompt"],
                    "text_prompt": case["text_prompt"],
                    "baseline_video": str(baseline_video),
                    "baseline_video_sha256": metadata["baseline_video_sha256"],
                    "c2f_video": str(c2f_video_path.resolve()),
                    "c2f_video_sha256": metadata["video_sha256"],
                    "c2f_metadata": str(c2f_metadata_path.resolve()),
                    "c2f_metadata_sha256": sha256_file(c2f_metadata_path),
                }
            )
    selected.sort(key=lambda case: case["selection_order"])
    if len(selected_ids) != 10:
        raise RuntimeError("Stage B selection contains duplicate cases")

    stage_a_path = args.output_dir.resolve() / "STAGE_A_LOCK.json"
    stage_a_ids = {case["case_id"] for case in read_json(stage_a_path)["cases"]}
    if selected_ids.intersection(stage_a_ids):
        raise RuntimeError("Stage B selection unexpectedly overlaps Stage A")

    subset_manifest = {"_meta": dict(manifest["_meta"])}
    subset_manifest["_meta"].update(
        {
            "purpose": "c2f_external_geometry_stage_b_dev10",
            "selection_uses_generated_outputs": False,
            "selection_uses_metrics": False,
            "selection_rule": "ranks 1 and 3 in each frozen Dev25 motion stratum",
            "stage_a_overlap_count": 0,
        }
    )
    for case in selected:
        subset_manifest[case["case_id"]] = manifest[case["case_id"]]

    output_dir = args.output_dir.resolve()
    subset_path = output_dir / "stage_b10_manifest.json"
    lock_path = output_dir / "STAGE_B_LOCK.json"
    atomic_json(subset_path, subset_manifest)
    evidence_rule = {
        "confidence_percentile": 20.0,
        "depth_relative_tolerance": 0.15,
        "max_reprojection_error_tokens": 1.5,
        "accepted_policy": "permit_c2f_value_residual",
        "all_other_statuses": "abstain_no_residual",
    }
    lock = {
        "schema": "c2f-external-geometry-stage-b-lock-v1",
        "purpose": "paired_geometry_gating_vs_uniform_energy_control",
        "selection_uses_generated_outputs": False,
        "selection_uses_metrics": False,
        "selection_rule": "two non-Stage-A pose quantiles per motion stratum: ranks 1 and 3",
        "manifest": {"path": str(manifest_path), "sha256": actual_manifest_sha},
        "subset_manifest": {"path": str(subset_path), "sha256": sha256_file(subset_path)},
        "p0_lock": {"path": str(p0_lock_path), "sha256": sha256_file(p0_lock_path)},
        "frozen_c2f_config": {"path": str(c2f_config_path), "sha256": sha256_file(c2f_config_path)},
        "stage_a_lock": {"path": str(stage_a_path), "sha256": sha256_file(stage_a_path)},
        "reserved_overlap_counts": EXPECTED_ZERO_OVERLAP,
        "stage_a_overlap_count": 0,
        "cases": selected,
        "generation": c2f_config["generation"],
        "frozen_c2f_method": c2f_config["method"],
        "pipeline": {"path": str(pipeline), "sha256": sha256_file(pipeline)},
        "geometry_root": str(args.geometry_root.resolve()),
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
        "evidence_rule": evidence_rule,
        "frame_indices": list(range(0, int(c2f_config["generation"]["frames"]), 4)),
        "groups": {
            "B": "existing official Wan baseline",
            "C": "existing frozen C2F K=3 alpha=0.025",
            "G": "hard positive external-geometry gate on each selected C2F correspondence",
            "U": "uniform residual scaling with per-layer-step L2 norm matched to G",
        },
        "primary_comparisons": ["G-C", "G-U", "G-B"],
    }
    atomic_json(lock_path, lock)

    common = {
        "schema": "wan-c2f-geometry-stage-b-v1",
        "source_manifest": c2f_config["source_manifest"],
        "source_manifest_sha256": c2f_config["source_manifest_sha256"],
        "selection_manifest": str(subset_path),
        "reserved_split_csv": c2f_config["reserved_split_csv"],
        "reserved_split_sha256": c2f_config["reserved_split_sha256"],
        "required_zero_overlap_splits": c2f_config["required_zero_overlap_splits"],
        "baseline_root": c2f_config["baseline_root"],
        "model_path": c2f_config["model_path"],
        "pipeline_path": str(pipeline),
        "geometry_root": str(args.geometry_root.resolve()),
        "stage_b_lock": str(lock_path),
        "stage_b_lock_sha256": sha256_file(lock_path),
        "seed": c2f_config["seed"],
        "generation": c2f_config["generation"],
    }
    geometry_method = dict(c2f_config["method"])
    geometry_method.update(
        {
            "c2f_geometry_confidence_percentile": evidence_rule["confidence_percentile"],
            "c2f_geometry_depth_tolerance": evidence_rule["depth_relative_tolerance"],
            "c2f_geometry_max_error_tokens": evidence_rule["max_reprojection_error_tokens"],
        }
    )
    for method_id, mode in (
        ("c2f_geometry_hard_gate", "hard_gate"),
        ("c2f_geometry_uniform_norm_control", "uniform_norm_control"),
    ):
        config = dict(common)
        config["method_id"] = method_id
        config["method"] = dict(geometry_method, c2f_geometry_mode=mode)
        atomic_json(output_dir / f"{method_id}.json", config)

    print(f"locked {len(selected)} Stage B cases: {lock_path}")
    for case in selected:
        print(case["motion_stratum"], case["stratum_quantile_rank"], case["case_id"])


if __name__ == "__main__":
    main()
