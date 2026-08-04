#!/usr/bin/env python3
"""Static source lint for the Wan/Cosmos GeCo ports.

This does not prove numerical equivalence. It catches known migration mistakes
before expensive GPU tests and reports configuration differences explicitly.
Passing this script is never sufficient to certify a migration.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Check:
    name: str
    status: str
    detail: str


def contains(path: Path, pattern: str, flags: int = 0) -> bool:
    return re.search(pattern, path.read_text(), flags) is not None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_enable_grad_context(node: ast.withitem) -> bool:
    context = node.context_expr
    return (
        isinstance(context, ast.Call)
        and isinstance(context.func, ast.Attribute)
        and isinstance(context.func.value, ast.Name)
        and context.func.value.id == "torch"
        and context.func.attr == "enable_grad"
    )


def live_x0_lines_in_guidance_repeat(path: Path) -> set[int]:
    """Locate the x0 calls used by the differentiable guidance repeat path."""

    tree = ast.parse(path.read_text(), filename=str(path))
    lines: set[int] = set()

    class Visitor(ast.NodeVisitor):
        grad_depth = 0
        repeat_depth = 0

        @staticmethod
        def is_guidance_repeat(node: ast.For) -> bool:
            return (
                isinstance(node.target, ast.Name)
                and node.target.id == "rep"
                and isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
                and len(node.iter.args) == 1
                and isinstance(node.iter.args[0], ast.Subscript)
                and isinstance(node.iter.args[0].value, ast.Name)
                and node.iter.args[0].value.id == "guidance_step"
                and isinstance(node.iter.args[0].slice, ast.Name)
                and node.iter.args[0].slice.id == "i"
            )

        def visit_For(self, node: ast.For) -> None:
            is_repeat = self.is_guidance_repeat(node)
            self.repeat_depth += int(is_repeat)
            self.generic_visit(node)
            self.repeat_depth -= int(is_repeat)

        def visit_With(self, node: ast.With) -> None:
            enabled = any(is_enable_grad_context(item) for item in node.items)
            self.grad_depth += int(enabled)
            self.generic_visit(node)
            self.grad_depth -= int(enabled)

        def visit_Assign(self, node: ast.Assign) -> None:
            is_x0_target = any(
                isinstance(target, ast.Name) and target.id == "x0_pred"
                for target in node.targets
            )
            call = node.value
            if (
                is_x0_target
                and isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "flow_match_predicted_x0"
                and len(call.args) >= 3
                and isinstance(call.args[1], ast.Name)
                and call.args[1].id == "noise_pred_g"
                and isinstance(call.args[2], ast.Name)
                and call.args[2].id == "latents"
                and any(
                    keyword.arg == "step_index"
                    and isinstance(keyword.value, ast.Name)
                    and keyword.value.id == "i"
                    for keyword in call.keywords
                )
                and self.grad_depth > 0
                and self.repeat_depth > 0
            ):
                lines.add(node.lineno)
            self.generic_visit(node)

    Visitor().visit(tree)
    return lines


def has_live_x0_call_in_grad_context(path: Path) -> bool:
    return bool(live_x0_lines_in_guidance_repeat(path))


def detaches_noise_prediction_before_live_x0(path: Path) -> bool:
    """Reject direct or aliased detach operations before the guidance x0 call."""

    tree = ast.parse(path.read_text(), filename=str(path))
    x0_lines = live_x0_lines_in_guidance_repeat(path)
    if not x0_lines:
        return True
    live_x0_line = min(x0_lines)
    aliases = {"noise_pred_g"}
    for node in sorted(ast.walk(tree), key=lambda item: getattr(item, "lineno", -1)):
        if getattr(node, "lineno", live_x0_line) >= live_x0_line:
            break
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            aliases.update(target.id for target in node.targets if isinstance(target, ast.Name))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "detach"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in aliases
        ):
            return True
    return False


def flow_helper_is_pure_x0(path: Path) -> bool:
    """Reject scheduler mutation and require x_t - sigma_t * model_output."""

    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "flow_match_predicted_x0"
        ),
        None,
    )
    if function is None:
        return False
    has_sigma_index = False
    has_return_formula = False
    mutates_scheduler = False
    scheduler_aliases = {"scheduler"}
    nodes = sorted(ast.walk(function), key=lambda item: getattr(item, "lineno", -1))
    for node in nodes:
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Name) and node.value.id in scheduler_aliases:
                scheduler_aliases.update(
                    target.id for target in node.targets if isinstance(target, ast.Name)
                )
            has_sigma_target = any(
                isinstance(target, ast.Name) and target.id == "sigma"
                for target in node.targets
            )
            sigma_source = (
                node.value.func.value
                if isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                else None
            )
            has_sigma_index |= (
                has_sigma_target
                and isinstance(sigma_source, ast.Subscript)
                and isinstance(sigma_source.value, ast.Name)
                and sigma_source.value.id == "sigmas"
                and isinstance(sigma_source.slice, ast.Name)
                and sigma_source.slice.id == "step_index"
            )
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            mutates_scheduler |= any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id in scheduler_aliases
                for target in targets
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            mutates_scheduler |= (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id in scheduler_aliases
                and node.func.attr not in {"__class__"}
            )
        if isinstance(node, ast.Return):
            value = node.value
            has_return_formula |= (
                isinstance(value, ast.BinOp)
                and isinstance(value.op, ast.Sub)
                and isinstance(value.left, ast.Call)
                and isinstance(value.left.func, ast.Attribute)
                and isinstance(value.left.func.value, ast.Name)
                and value.left.func.value.id == "sample"
                and value.left.func.attr == "float"
                and isinstance(value.right, ast.BinOp)
                and isinstance(value.right.op, ast.Mult)
                and isinstance(value.right.left, ast.Name)
                and value.right.left.id == "sigma"
                and isinstance(value.right.right, ast.Call)
                and isinstance(value.right.right.func, ast.Attribute)
                and isinstance(value.right.right.func.value, ast.Name)
                and value.right.right.func.value.id == "model_output"
                and value.right.right.func.attr == "float"
            )
    return has_sigma_index and has_return_formula and not mutates_scheduler


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
    wan_x0 = repo / "geometry_selection/online.py"
    wan_runner = repo / "run_wan_geco_case_full.py"
    cosmos = repo / "external/guidance_cosmos/pipeline_cosmos2_5_predict_guided.py"
    cosmos_runner = repo / "run_cosmos_geco_case.py"
    required = [cog, demo, wan, wan_x0, wan_runner, cosmos, cosmos_runner]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    checks: list[Check] = []
    add(
        checks,
        "official_geco_reference_pinned",
        sha256(cog) == "f19ffee43f55354a6e154ab5ae7bcf9331f6ba436927bd944ddf7a51dfa3992c"
        and sha256(demo) == "7a1d376179feae23eaf23a426d097778c9cb2682018271f1e7c6bc3b6a7dce54",
        "CogVideoX guidance and demo match the frozen official reference hashes.",
    )

    wan_text = wan.read_text()
    add(
        checks,
        "wan_flowmatch_helper_import",
        contains(
            wan,
            r"from\s+geometry_selection\.online\s+import\s*\(.*?flow_match_predicted_x0",
            re.S,
        ),
        "The Wan pipeline imports the checked pure flow-matching helper from geometry_selection.online.",
    )
    add(
        checks,
        "wan_flowmatch_scheduler_contract",
        contains(wan, r"scheduler:\s*FlowMatchEulerDiscreteScheduler"),
        "The Wan pipeline declares a FlowMatchEulerDiscreteScheduler.",
    )
    add(
        checks,
        "wan_transformer_jacobian_source_path",
        has_live_x0_call_in_grad_context(wan)
        and not detaches_noise_prediction_before_live_x0(wan),
        "Wan clean prediction is called with the live transformer output; runtime autograd is checked separately.",
    )
    add(
        checks,
        "wan_transformer_autograd_context",
        contains(
            wan,
            r"with\s+torch\.enable_grad\(\):.*?noise_pred_g\s*=\s*current_model\(",
            re.S,
        ),
        "The guidance transformer prediction is computed in an enabled autograd context.",
    )
    add(
        checks,
        "wan_fresh_prediction_each_repeat",
        contains(wan, r"for rep in range\(guidance_step\[i\]\).*?current_model\(", re.S),
        "A fresh transformer prediction is made inside each guidance repeat.",
    )
    add(
        checks,
        "wan_prediction_recomputed_after_guidance",
        contains(
            wan,
            r"if guidance_step\[i\] > 0.*?# Recompute noise_pred after latent update.*?"
            r"noise_pred\s*=\s*current_model\(.*?self\.scheduler\.step\(noise_pred,\s*t,\s*latents",
            re.S,
        ),
        "The Wan scheduler receives a transformer prediction recomputed at the updated latent.",
    )
    add(
        checks,
        "wan_scheduler_x0",
        flow_helper_is_pure_x0(wan_x0),
        "Wan uses the FlowMatchEuler clean prediction x_t - sigma_t * model_output without mutating scheduler state.",
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
