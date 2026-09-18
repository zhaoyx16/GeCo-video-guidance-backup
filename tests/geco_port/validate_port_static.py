#!/usr/bin/env python3
"""Static source lint for the Wan/Cosmos GeCo ports.

This does not prove numerical equivalence. It catches known migration mistakes
before expensive GPU tests and reports configuration differences explicitly.
Passing this script is never sufficient to certify a migration.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Check:
    name: str
    status: str
    detail: str


def contains(path: Path, pattern: str, flags: int = 0) -> bool:
    return re.search(pattern, path.read_text(), flags) is not None


def add(checks: list[Check], name: str, ok: bool, detail: str) -> None:
    checks.append(Check(name, "PASS" if ok else "FAIL", detail))


def warn(checks: list[Check], name: str, condition: bool, detail: str) -> None:
    checks.append(Check(name, "WARN" if condition else "PASS", detail))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()

    cog = repo / "external/guidance_cogvideox/cogvideox.py"
    demo = repo / "demo_guidance.py"
    wan = repo / "external/guidance_wan/pipeline_wan_i2v_full_guided.py"
    wan_runner = repo / "run_wan_geco_case_full.py"
    cosmos = repo / "external/guidance_cosmos/pipeline_cosmos2_5_predict_guided.py"
    cosmos_runner = repo / "run_cosmos_geco_case.py"
    required = [cog, demo, wan, wan_runner, cosmos, cosmos_runner]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    checks: list[Check] = []
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--quiet", "origin/main", "--", str(cog), str(demo)],
        check=False,
    )
    add(
        checks,
        "official_geco_reference_unchanged",
        diff.returncode == 0,
        "CogVideoX guidance and demo match origin/main.",
    )

    wan_text = wan.read_text()
    x0_match = re.search(
        r"x0_pred\s*=\s*self\.scheduler\.convert_model_output\((.*?)sample=latents\.float\(\)",
        wan_text,
        re.S,
    )
    add(
        checks,
        "wan_transformer_jacobian",
        bool(x0_match and ".detach()" not in x0_match.group(1)),
        "Wan x0 conversion must use differentiable noise_pred_g.",
    )
    add(
        checks,
        "wan_fresh_prediction_each_repeat",
        contains(wan, r"for rep in range\(guidance_step\[i\]\).*?current_model\(", re.S),
        "A fresh transformer prediction is made inside each guidance repeat.",
    )
    add(
        checks,
        "wan_scheduler_x0",
        "self.scheduler.convert_model_output" in wan_text,
        "Wan uses the scheduler's configured flow-prediction conversion.",
    )
    add(
        checks,
        "wan_full_spatial_decode",
        "Full Wan GeCo guidance requires decode_spatial_scale=1.0" in wan_text,
        "Formal Wan port rejects reduced-resolution differentiable decode.",
    )
    add(
        checks,
        "wan_condition_preserved",
        contains(wan, r"\(1 - first_frame_mask\.float\(\)\).*?condition\.float\(\)", re.S),
        "Conditioned frame/region is restored before metric decode.",
    )
    add(
        checks,
        "wan_normalized_update",
        contains(wan, r"update\s*=\s*guidance_lr\[i\]\s*\*\s*grad\s*/\s*grad_norm"),
        "Wan update is normalized as in GeCo.",
    )
    add(
        checks,
        "wan_checkpointed_vae_decode",
        "checkpoint(" in wan_text and "self.vae.decode" in wan_text,
        "Full-resolution VAE decode supports activation recomputation.",
    )
    warn(
        checks,
        "wan_no_time_travel",
        "travel_time" not in wan_text,
        "Original GeCo DDIM time-travel is absent; it must not be copied to flow matching without derivation.",
    )
    warn(
        checks,
        "wan_runner_ufm_default",
        contains(wan_runner, r'--ufm_scale".*?default=0\.125', re.S),
        "Runner default is 0.125; formal GeCo-faithful jobs should pass --ufm_scale 0.25 explicitly.",
    )
    warn(
        checks,
        "wan_runner_model_defaults",
        all(
            token in wan_runner.read_text()
            for token in ["default=10", "default=21", "default=480", "default=832"]
        ),
        "Runner defaults are smoke settings; formal jobs must pass 50/121/704/1280/24 explicitly.",
    )

    cosmos_text = cosmos.read_text()
    add(
        checks,
        "cosmos_fresh_prediction_each_repeat",
        contains(cosmos, r"for rep in range\(guidance_step\[i\]\).*?_predict_noise_for_latents", re.S),
        "Cosmos recomputes the transformer prediction after every latent update.",
    )
    add(
        checks,
        "cosmos_transformer_jacobian",
        contains(
            cosmos,
            r"x0_pred\s*=\s*latents_req\s*-\s*sigma_for_guidance\s*\*\s*noise_pred_for_guidance",
        ),
        "Cosmos x0 prediction retains the transformer Jacobian.",
    )
    add(
        checks,
        "cosmos_generated_region_gradient",
        contains(cosmos, r"grad\s*=\s*grad\s*\*\s*\(1\s*-\s*cond_mask\)"),
        "Cosmos guidance does not update conditioned latent regions.",
    )
    add(
        checks,
        "cosmos_normalized_update",
        contains(cosmos, r"guidance_lr\[i\]\)\s*\*\s*grad\s*/\s*grad_norm"),
        "Cosmos update is normalized as in GeCo.",
    )
    add(
        checks,
        "cosmos_prediction_recomputed_after_guidance",
        contains(cosmos, r"if do_metric_guidance:.*?noise_pred\s*=\s*_predict_noise_for_latents\(latents\)", re.S),
        "The scheduler receives a prediction evaluated at the updated latent.",
    )
    warn(
        checks,
        "cosmos_no_time_travel",
        "travel_time" not in cosmos_text,
        "Original GeCo DDIM time-travel is absent; this remains a controlled method difference.",
    )
    warn(
        checks,
        "cosmos_runner_negative_prompt",
        "distorted geometry, flickering, object deformation" in cosmos_runner.read_text(),
        "Runner overrides the official model negative prompt; use one frozen prompt for every compared method.",
    )

    payload = {
        "kind": "static_source_lint_only",
        "certifies_runtime_correctness": False,
        "repo": str(repo),
        "summary": {
            "pass": sum(item.status == "PASS" for item in checks),
            "warn": sum(item.status == "WARN" for item in checks),
            "fail": sum(item.status == "FAIL" for item in checks),
        },
        "checks": [asdict(item) for item in checks],
    }
    print(json.dumps(payload, indent=2))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
    raise SystemExit(1 if payload["summary"]["fail"] else 0)


if __name__ == "__main__":
    main()
