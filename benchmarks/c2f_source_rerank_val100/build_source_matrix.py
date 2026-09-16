#!/usr/bin/env python3
"""Build the preregistered 36-method plus 24-control Val100 matrix."""

from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"
MANIFEST_SHA = "ef59a10d21d7d30729a9ac24cfba7cb37c153f5ae05debaa18387d3720afb862"
BASELINE_ROOT = Path(
    "/vol/dissolve/yz10325/outputs/geometry-selection/evaluator_inputs/wan_unguided_v2_4seed_metric_v1"
)
DRAFT_ROOT = Path(
    "/vol/dissolve/yz10325/outputs/geometry-selection/c2f_source_val100_draft_geometry_v1"
)
SNAPSHOT_ROOT = Path(
    "/vol/dissolve/yz10325/outputs/geometry-selection/c2f_source_val100_online_snapshot_geometry_v1"
)
FRAME_INDICES = list(range(0, 121, 4))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def alpha_id(alpha: float) -> str:
    return {0.0125: "a00125", 0.025: "a0025"}[alpha]


def evidence_config(kind: str) -> dict:
    geometry_root = DRAFT_ROOT if kind == "draft" else SNAPSHOT_ROOT
    evidence = {
        "kind": kind,
        "geometry_root": str(geometry_root),
    }
    if kind == "online_refresh":
        evidence.update(
            {
                "refresh_steps": [22, 25, 28],
                "frame_indices": FRAME_INDICES,
                "confidence_percentile": 20.0,
                "geometry_model": {
                    "name": "VGGT-Omega-1B-512",
                    "source_root": "external/vggt_omega",
                    "source_commit": "39a0cb8af88554f15ddcb5354cd52bde588fa014",
                    "checkpoint": "/vol/dissolve/yz10325/checkpoints/VGGT-Omega/vggt_omega_1b_512.pt",
                    "checkpoint_sha256": "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934",
                    "checkpoint_size": 4576706117,
                    "image_resolution": 512,
                    "preprocessing_mode": "balanced",
                },
            }
        )
    return evidence


def method_config(alpha: float, pair: str, policy: str, mode: str) -> dict:
    return {
        "attn_avg_alpha": alpha,
        "attn_avg_layers": [10, 15, 20],
        "attn_avg_mode": "c2f_value_residual_memory",
        "attn_avg_start": 20,
        "attn_avg_end": 29,
        "attn_avg_temporal_radius": 1,
        "attn_avg_match_radius": 2,
        "attn_avg_match_confidence": 0.4,
        "attn_avg_match_mutual": True,
        "attn_avg_descriptor_dim": 64,
        "attn_avg_coarse_factor": 2,
        "attn_avg_memory_lookback": 3,
        "attn_avg_cond_only": True,
        "attn_avg_preserve_first_frame": True,
        "attn_avg_debug": False,
        "c2f_geometry_mode": mode,
        "c2f_retrieval_policy": policy,
        "c2f_pair_weighting": pair,
        "c2f_source_support_history": 3,
        "c2f_force_source_score_one": False,
        "c2f_geometry_confidence_percentile": 20.0,
        "c2f_geometry_depth_tolerance": 0.15,
        "c2f_geometry_max_error_tokens": 1.5,
    }


def make_config(method_id: str, alpha: float, pair: str, policy: str, evidence: str, mode: str) -> dict:
    return {
        "schema": "wan-c2f-source-rerank-val100-v1",
        "method_id": method_id,
        "manifest_sha256": MANIFEST_SHA,
        "pipeline_path": "external/guidance_wan/pipeline_wan_i2v_c2f_source_rerank.py",
        "baseline_root": str(BASELINE_ROOT),
        "seed": 0,
        "generation": {
            "steps": 50,
            "frames": 121,
            "height": 704,
            "width": 1280,
            "fps": 24,
            "guidance_scale": 5.0,
            "negative_prompt": None,
        },
        "factors": {
            "alpha": alpha,
            "pair_weighting": pair,
            "retrieval_policy": policy,
            "geometry_evidence": evidence,
            "intervention": "main" if mode == "rerank" else "uniform_norm_control",
        },
        "evidence": evidence_config(evidence),
        "method": method_config(alpha, pair, policy, mode),
    }


def main() -> None:
    records = []
    for alpha, pair, policy, evidence in product(
        (0.0125, 0.025), ("hard", "soft"), ("V", "P", "S"),
        ("draft", "online_snapshot", "online_refresh"),
    ):
        method_id = f"c2f_src_{evidence}_{pair}_{policy.lower()}_{alpha_id(alpha)}"
        path = CONFIG_DIR / f"{method_id}.json"
        write_json(path, make_config(method_id, alpha, pair, policy, evidence, "rerank"))
        records.append({"kind": "main", "method_id": method_id, "config": str(path.relative_to(HERE)), "sha256": sha256_file(path)})

    for alpha, pair, policy, evidence in product(
        (0.0125, 0.025), ("hard", "soft"), ("V", "P"),
        ("draft", "online_snapshot", "online_refresh"),
    ):
        method_id = f"c2f_u_{evidence}_{pair}_{policy.lower()}_{alpha_id(alpha)}"
        path = CONFIG_DIR / f"{method_id}.json"
        write_json(path, make_config(method_id, alpha, pair, policy, evidence, "uniform_norm_control"))
        records.append({"kind": "uniform_control", "method_id": method_id, "config": str(path.relative_to(HERE)), "sha256": sha256_file(path)})

    if len(records) != 60 or len({record["method_id"] for record in records}) != 60:
        raise RuntimeError("matrix must contain exactly 36 main methods and 24 controls")
    write_json(
        HERE / "SOURCE_MATRIX.json",
        {
            "schema": "wan-c2f-source-rerank-matrix-v1",
            "manifest_sha256": MANIFEST_SHA,
            "main_count": sum(record["kind"] == "main" for record in records),
            "uniform_control_count": sum(record["kind"] == "uniform_control" for record in records),
            "public_reference_ids": ["wan_unguided", "c2f_k3_a0025", "c2f_k3_a00125"],
            "adapted_geco": "reuse-existing-not-regenerated",
            "records": records,
        },
    )
    print(f"wrote {len(records)} configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
