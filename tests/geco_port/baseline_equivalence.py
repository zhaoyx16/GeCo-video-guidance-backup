#!/usr/bin/env python3
"""Numerical baseline-equivalence test for custom Wan/Cosmos pipelines.

Run ``official`` and ``custom`` as separate GPU jobs to avoid holding two
pipelines in memory, then run ``compare`` on CPU. Both generation modes save
the final latent rather than decoded RGB, making the comparison stricter and
cheaper than comparing MP4 files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import io
import json
import platform
import re
import socket
import uuid
from contextlib import redirect_stdout
from pathlib import Path

import torch
from PIL import Image


WAN_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

COSMOS_NEGATIVE = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
    "pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color "
    "balance, washed out colors, choppy sequences, jerky movements, low frame rate, "
    "artifacting, color banding, unnatural transitions, outdated special effects, fake "
    "elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and "
    "flickering. Overall, the video is of poor quality."
)


def load_class(path: Path, class_name: str):
    spec = importlib.util.spec_from_file_location(f"_geco_test_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def extract_tensor(output) -> torch.Tensor:
    value = output.frames
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"Unexpected latent output type: {type(value)}")


def model_identity(model: str, revision: str | None) -> dict:
    path = Path(model).expanduser()
    resolved = str(path.resolve()) if path.exists() else model
    parts = Path(resolved).parts
    snapshot = None
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            snapshot = parts[index + 1]
    return {
        "requested": model,
        "resolved": resolved,
        "requested_revision": revision,
        "snapshot_commit": snapshot,
    }


def runtime_identity() -> dict:
    import diffusers
    import transformers

    cuda = None
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        cuda = {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "cuda_version": torch.version.cuda,
        }
    return {
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "cuda": cuda,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scheduler_identity(scheduler) -> dict:
    config = json.loads(json.dumps(dict(scheduler.config), default=str))
    if isinstance(config.get("_use_default_values"), list):
        config["_use_default_values"] = sorted(config["_use_default_values"])
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return {
        "class": scheduler.__class__.__name__,
        "config": config,
        "config_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def make_noop_cosmos_safety_checker(pipeline_class):
    """Construct a type-compatible benign checker for latent-only testing."""

    expected_type = inspect.signature(pipeline_class.__init__).parameters[
        "safety_checker"
    ].annotation

    class NoOpCosmosSafetyChecker(expected_type):
        def __init__(self):
            pass

        def to(self, *args, **kwargs):
            return self

        def check_text_safety(self, prompt: str) -> bool:
            return True

        def check_video_safety(self, video):
            return video

    return NoOpCosmosSafetyChecker()


def save_result(
    path: Path,
    tensor: torch.Tensor,
    args: argparse.Namespace,
    *,
    pipe,
    negative_prompt: str | None,
    guidance_scale: float,
    implementation_source: Path,
    diagnostics: dict | None = None,
) -> None:
    tensor = tensor.detach().float().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "latent": tensor,
        "meta": {
            "backbone": args.backbone,
            "implementation": args.action,
            "run_label": args.run_label,
            "seed": args.seed,
            "steps": args.steps,
            "frames": args.frames,
            "height": args.height,
            "width": args.width,
            "prompt": args.prompt,
            "negative_prompt_mode": args.negative_prompt_mode,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "guidance_smoke": diagnostics,
            "implementation_source": {
                "path": str(implementation_source),
                "sha256": file_sha256(implementation_source),
            },
            "image": {
                "path": str(args.image.resolve()),
                "sha256": file_sha256(args.image),
            },
            "scheduler": scheduler_identity(pipe.scheduler),
            "determinism": {
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
            },
            "model": model_identity(args.model, args.revision),
            "runtime": runtime_identity(),
        },
    }
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(
        payload,
        temporary,
    )
    temporary.replace(path)
    print(
        json.dumps(
            {
                "saved": str(path),
                "shape": list(tensor.shape),
                "mean": tensor.mean().item(),
                "std": tensor.std().item(),
                "min": tensor.min().item(),
                "max": tensor.max().item(),
                "finite": torch.isfinite(tensor).float().mean().item(),
            },
            indent=2,
        )
    )


def generate(args: argparse.Namespace) -> None:
    from diffusers import AutoencoderKLWan

    repo = args.repo.resolve()
    if not args.image.is_file():
        raise FileNotFoundError(f"Correctness image does not exist: {args.image}")
    if args.backbone == "wan":
        if args.action == "official":
            from diffusers import WanImageToVideoPipeline as Pipeline
            implementation_source = Path(inspect.getfile(Pipeline)).resolve()
        else:
            implementation_source = (
                repo / "external/guidance_wan/pipeline_wan_i2v_full_guided.py"
            ).resolve()
            Pipeline = load_class(
                implementation_source,
                "WanImageToVideoPipeline",
            )
        load_kwargs = {}
        if args.revision and not Path(args.model).expanduser().exists():
            load_kwargs["revision"] = args.revision
        vae = AutoencoderKLWan.from_pretrained(
            args.model,
            subfolder="vae",
            torch_dtype=torch.float32,
            **load_kwargs,
        )
        pipe = Pipeline.from_pretrained(
            args.model,
            vae=vae,
            torch_dtype=torch.bfloat16,
            **load_kwargs,
        ).to(args.device)
    else:
        if args.action == "official":
            from diffusers import Cosmos2_5_PredictBasePipeline as Pipeline
            implementation_source = Path(inspect.getfile(Pipeline)).resolve()
        else:
            implementation_source = (
                repo
                / "external/guidance_cosmos/pipeline_cosmos2_5_predict_guided.py"
            ).resolve()
            Pipeline = load_class(
                implementation_source,
                "Cosmos2_5_PredictBasePipeline",
            )
        load_kwargs = {"torch_dtype": torch.bfloat16}
        if args.revision and not Path(args.model).expanduser().exists():
            load_kwargs["revision"] = args.revision
        pipe = Pipeline.from_pretrained(
            args.model,
            safety_checker=make_noop_cosmos_safety_checker(Pipeline),
            **load_kwargs,
        ).to(args.device)
    image = Image.open(args.image).convert("RGB")
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    common = dict(
        image=image,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        generator=generator,
        output_type="latent",
    )
    frozen_negative = WAN_NEGATIVE if args.backbone == "wan" else COSMOS_NEGATIVE
    negative_prompt = frozen_negative if args.negative_prompt_mode == "frozen" else None
    guidance_scale = 5.0 if args.backbone == "wan" else 7.0
    if args.backbone == "wan":
        common.update(negative_prompt=negative_prompt, guidance_scale=guidance_scale)
    else:
        common.update(video=None, negative_prompt=negative_prompt, guidance_scale=guidance_scale)

    if args.action == "custom":
        guidance_step = [0] * args.steps
        guidance_lr = [0.0] * args.steps
        loss_fn = None
        additional_inputs = None
        if args.guidance_smoke:
            if not 0 <= args.guidance_step_index < args.steps:
                raise ValueError("--guidance-step-index must be inside the denoising schedule")
            guidance_step[args.guidance_step_index] = args.guidance_repeats
            guidance_lr[args.guidance_step_index] = args.guidance_lr
            loss_fn = "latent_l2"
            additional_inputs = {
                "debug_guidance_consistency": True,
                "debug_guidance_gradient": True,
            }
        common.update(
            fixed_frames=[0, max(0, args.frames // 2), args.frames - 1],
            guidance_step=guidance_step,
            guidance_lr=guidance_lr,
            loss_fn=loss_fn,
            additional_inputs=additional_inputs,
        )

    diagnostics = None
    if args.guidance_smoke:
        transformer_trace = []

        def trace_transformer_input(module, positional, keyword):
            hidden = keyword.get("hidden_states")
            if hidden is None and positional:
                hidden = positional[0]
            transformer_trace.append(
                {
                    "shape": list(hidden.shape) if isinstance(hidden, torch.Tensor) else None,
                    "mean": hidden.detach().float().mean().item()
                    if isinstance(hidden, torch.Tensor)
                    else None,
                }
            )

        trace_handle = pipe.transformer.register_forward_pre_hook(
            trace_transformer_input, with_kwargs=True
        )
        captured = io.StringIO()
        try:
            with redirect_stdout(captured):
                output = pipe(**common)
        finally:
            trace_handle.remove()
        log_text = captured.getvalue()
        print(log_text, end="")
        prefix = "wan" if args.backbone == "wan" else "cosmos"
        losses = [
            float(value)
            for value in re.findall(
                rf"{prefix}_guidance_loss\(\d+/\d+\): ([+-]?[0-9.eE-]+)",
                log_text,
            )
        ]
        deltas = [
            float(value)
            for value in re.findall(r"latent_delta=([+-]?[0-9.eE-]+)", log_text)
        ]
        cfg_branches = 2 if guidance_scale > 1.0 else 1
        minimum_forward_count = (
            args.steps * cfg_branches + args.guidance_repeats * cfg_branches
        )
        diagnostics = {
            "step_index": args.guidance_step_index,
            "repeats": args.guidance_repeats,
            "lr": args.guidance_lr,
            "losses": losses,
            "latent_deltas": deltas,
            "transformer_forward_count": len(transformer_trace),
            "minimum_transformer_forward_count": minimum_forward_count,
            "fresh_transformer_for_each_repeat": (
                len(transformer_trace) >= minimum_forward_count
            ),
            "fresh_loss_decreased": (
                len(losses) == args.guidance_repeats
                and all(torch.isfinite(torch.tensor(losses)))
                and losses[-1] < losses[0]
            ),
            "nonzero_updates": (
                len(deltas) == args.guidance_repeats and all(value > 0.0 for value in deltas)
            ),
        }
        if (
            not diagnostics["fresh_loss_decreased"]
            or not diagnostics["nonzero_updates"]
            or not diagnostics["fresh_transformer_for_each_repeat"]
        ):
            raise SystemExit(f"Guidance smoke failed: {json.dumps(diagnostics, indent=2)}")
    else:
        with torch.inference_mode():
            output = pipe(**common)
    save_result(
        args.output,
        extract_tensor(output),
        args,
        pipe=pipe,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        implementation_source=implementation_source,
        diagnostics=diagnostics,
    )


def difference_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    delta = b - a
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "relative_l2": (delta.norm() / a.norm().clamp_min(1e-12)).item(),
        "cosine": torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0
        ).item(),
    }


def validate_comparable_meta(reference: dict, candidate: dict) -> None:
    keys = (
        "backbone",
        "seed",
        "steps",
        "frames",
        "height",
        "width",
        "prompt",
        "negative_prompt_mode",
        "negative_prompt",
        "guidance_scale",
        "image",
        "scheduler",
        "determinism",
        "model",
    )
    def comparable_value(key: str, value):
        if key != "scheduler" or not isinstance(value, dict):
            return value
        normalized = dict(value)
        config = dict(normalized.get("config", {}))
        if isinstance(config.get("_use_default_values"), list):
            config["_use_default_values"] = sorted(config["_use_default_values"])
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
        normalized["config"] = config
        normalized["config_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        return normalized

    mismatches = {}
    for key in keys:
        left = comparable_value(key, reference.get(key))
        right = comparable_value(key, candidate.get(key))
        if left != right:
            mismatches[key] = (left, right)
    if mismatches:
        raise SystemExit(f"Metadata mismatch: {json.dumps(mismatches, indent=2)}")


def compare(args: argparse.Namespace) -> None:
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    custom = torch.load(args.custom, map_location="cpu", weights_only=False)
    a = reference["latent"].float()
    b = custom["latent"].float()
    validate_comparable_meta(reference["meta"], custom["meta"])
    if a.shape != b.shape:
        raise SystemExit(f"Shape mismatch: official={tuple(a.shape)} custom={tuple(b.shape)}")
    candidate_metrics = difference_metrics(a, b)
    control_metrics = None
    control_pass = None
    if args.control:
        control = torch.load(args.control, map_location="cpu", weights_only=False)
        validate_comparable_meta(reference["meta"], control["meta"])
        c = control["latent"].float()
        if a.shape != c.shape:
            raise SystemExit(
                f"Control shape mismatch: official_a={tuple(a.shape)} official_b={tuple(c.shape)}"
            )
        control_metrics = difference_metrics(a, c)
        control_pass = torch.equal(a, c)

    exact = torch.equal(a, b)
    allclose = torch.allclose(a, b, atol=args.atol, rtol=args.rtol)
    within_control_floor = False
    if control_metrics is not None:
        within_control_floor = (
            candidate_metrics["max_abs"]
            <= max(args.atol, args.control_factor * control_metrics["max_abs"])
            and candidate_metrics["rmse"]
            <= max(args.atol, args.control_factor * control_metrics["rmse"])
        )
    control_valid = control_pass is True
    passed = control_valid and exact
    metrics = {
        "shape": list(a.shape),
        "candidate": candidate_metrics,
        "official_repeat_control": control_metrics,
        "official_repeat_allclose": control_pass,
        "official_repeat_valid": control_valid,
        "candidate_allclose": allclose,
        "candidate_exact": exact,
        "within_control_floor": within_control_floor,
        "passed": passed,
        "atol": args.atol,
        "rtol": args.rtol,
        "control_factor": args.control_factor,
        "reference_meta": reference["meta"],
        "candidate_meta": custom["meta"],
    }
    print(json.dumps(metrics, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(metrics, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


def compare_guidance(args: argparse.Namespace) -> None:
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    guided = torch.load(args.custom, map_location="cpu", weights_only=False)
    validate_comparable_meta(reference["meta"], guided["meta"])
    a = reference["latent"].float()
    b = guided["latent"].float()
    if a.shape != b.shape:
        raise SystemExit(f"Shape mismatch: baseline={tuple(a.shape)} guided={tuple(b.shape)}")
    cond_tokens = args.condition_latent_tokens
    if not 0 < cond_tokens < a.shape[2]:
        raise SystemExit(
            f"Invalid --condition-latent-tokens={cond_tokens} for latent T={a.shape[2]}"
        )
    cond_delta = difference_metrics(a[:, :, :cond_tokens], b[:, :, :cond_tokens])
    generated_delta = difference_metrics(a[:, :, cond_tokens:], b[:, :, cond_tokens:])
    smoke = guided["meta"].get("guidance_smoke") or {}
    passed = (
        smoke.get("fresh_loss_decreased") is True
        and smoke.get("nonzero_updates") is True
        and smoke.get("fresh_transformer_for_each_repeat") is True
        and cond_delta["max_abs"] <= args.atol
        and generated_delta["max_abs"] > args.min_generated_delta
    )
    report = {
        "condition_latent_tokens": cond_tokens,
        "condition_delta": cond_delta,
        "generated_delta": generated_delta,
        "guidance_smoke": smoke,
        "passed": passed,
        "atol": args.atol,
        "min_generated_delta": args.min_generated_delta,
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["official", "custom", "compare", "guidance-compare"]
    )
    parser.add_argument("--backbone", choices=["wan", "cosmos"])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt", default="A continuous first-person camera shot moving smoothly forward.")
    parser.add_argument(
        "--negative-prompt-mode",
        choices=["none", "frozen"],
        default="none",
        help="Use no negative prompt (matching existing Wan baselines) or the frozen benchmark negative prompt.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-label")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--custom", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--control-factor", type=float, default=2.0)
    parser.add_argument("--guidance-smoke", action="store_true")
    parser.add_argument("--guidance-step-index", type=int, default=2)
    parser.add_argument("--guidance-repeats", type=int, default=2)
    parser.add_argument("--guidance-lr", type=float, default=0.1)
    parser.add_argument("--condition-latent-tokens", type=int, default=1)
    parser.add_argument("--min-generated-delta", type=float, default=1e-8)
    args = parser.parse_args()

    if args.action == "compare":
        if not args.reference or not args.custom or not args.control:
            parser.error("compare requires --reference, --control, and --custom")
        compare(args)
    elif args.action == "guidance-compare":
        if not args.reference or not args.custom:
            parser.error("guidance-compare requires --reference and --custom")
        compare_guidance(args)
    else:
        missing = [
            name
            for name in ("backbone", "model", "image", "output")
            if getattr(args, name) in (None, "")
        ]
        if missing:
            parser.error(f"{args.action} missing: {', '.join(missing)}")
        generate(args)


if __name__ == "__main__":
    main()
