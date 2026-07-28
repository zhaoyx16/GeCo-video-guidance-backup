#!/usr/bin/env python3
"""Train and evaluate constant, linear, and 3D-Conv probes on real caches."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from latent_geometry.data import CachedLatentDataset, LinearFlowNoiseSchedule
from latent_geometry.models import ConstantPoseBaseline, LinearLatentProbe, Small3DConvCritic
from latent_geometry.training import evaluate_probe, train_probe_steps


def _loader(dataset, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        pin_memory=False,
    )


def _evaluate_at_timesteps(model, manifest, split, timesteps, batch_size, device):
    output = {}
    for timestep in timesteps:
        dataset = CachedLatentDataset(
            manifest,
            split,
            noise_schedule=LinearFlowNoiseSchedule(timestep, timestep),
            base_seed=1000 + round(timestep * 1000),
        )
        metrics = evaluate_probe(model, _loader(dataset, batch_size, False), device=device, input_key="zt")
        output[f"{timestep:.3f}"] = asdict(metrics)
    return output


def _evaluate_by_scene(model, dataset, batch_size, device, input_key):
    scene_indices: dict[str, list[int]] = {}
    for index, record in enumerate(dataset.records):
        scene_indices.setdefault(record.scene_id, []).append(index)
    return {
        scene_id: asdict(
            evaluate_probe(
                model,
                _loader(Subset(dataset, indices), batch_size, False),
                device=device,
                input_key=input_key,
            )
        )
        for scene_id, indices in sorted(scene_indices.items())
    }


def _scene_macro(per_scene):
    if not per_scene:
        raise ValueError("Cannot aggregate an empty per-scene result")
    translation = [
        metrics["translation_direction_deg"]
        for metrics in per_scene.values()
        if metrics["translation_direction_deg"] is not None
    ]
    return {
        "rotation_deg": sum(metrics["rotation_deg"] for metrics in per_scene.values()) / len(per_scene),
        "translation_direction_deg": sum(translation) / len(translation) if translation else None,
        "scenes": len(per_scene),
    }


def _git_state() -> dict[str, object]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--short"))}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--constant-learning-rate", type=float, default=3e-3)
    parser.add_argument(
        "--train-domain",
        choices=("z0", "diffusion_z0", "zt"),
        default="diffusion_z0",
    )
    parser.add_argument("--train-min-t", type=float, default=0.0)
    parser.add_argument("--train-max-t", type=float, default=0.7)
    parser.add_argument("--eval-timesteps", default="0.0,0.3,0.5,0.7")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--timestep-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        raise SystemExit("--steps and --batch-size must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    schedule = LinearFlowNoiseSchedule(args.train_min_t, args.train_max_t)
    val_dataset = CachedLatentDataset(args.manifest, "val", noise_schedule=schedule, base_seed=args.seed + 1)
    test_dataset = CachedLatentDataset(args.manifest, "test", noise_schedule=schedule, base_seed=args.seed + 2)
    val_loader = _loader(val_dataset, args.batch_size, False)
    test_loader = _loader(test_dataset, args.batch_size, False)
    probe_dataset = CachedLatentDataset(args.manifest, "train", noise_schedule=schedule, base_seed=args.seed)
    in_channels = int(probe_dataset[0]["z0"].shape[0])
    model_factories = {
        "constant": lambda: ConstantPoseBaseline(),
        "linear": lambda: LinearLatentProbe(in_channels),
        "small_3dconv": lambda: Small3DConvCritic(
            in_channels, width=args.width, timestep_dim=args.timestep_dim, hidden_dim=args.hidden_dim
        ),
    }
    eval_timesteps = [float(value) for value in args.eval_timesteps.split(",") if value.strip()]
    results = {
        "config": {
            **vars(args),
            "manifest": str(args.manifest),
            "manifest_sha256": _file_sha256(args.manifest),
            "output_root": str(args.output_root),
            "git": _git_state(),
        },
        "dataset_sizes": {
            "train": len(probe_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "models": {},
    }
    started = time.perf_counter()
    for name, model_factory in model_factories.items():
        # Recreate the data stream so every candidate gets identical shuffles
        # and deterministic online-noise draws.
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        train_dataset = CachedLatentDataset(
            args.manifest, "train", noise_schedule=schedule, base_seed=args.seed
        )
        train_loader = _loader(train_dataset, args.batch_size, True, seed=args.seed)
        model = model_factory()
        learning_rate = args.constant_learning_rate if name == "constant" else args.learning_rate
        before = evaluate_probe(model, val_loader, device=args.device, input_key=args.train_domain)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        train_started = time.perf_counter()
        history = train_probe_steps(
            model,
            train_loader,
            optimizer,
            steps=args.steps,
            device=args.device,
            input_key=args.train_domain,
        )
        elapsed = time.perf_counter() - train_started
        val_metrics = evaluate_probe(model, val_loader, device=args.device, input_key=args.train_domain)
        test_metrics = evaluate_probe(model, test_loader, device=args.device, input_key=args.train_domain)
        timestep_curve = _evaluate_at_timesteps(
            model, args.manifest, "test", eval_timesteps, args.batch_size, args.device
        )
        checkpoint = args.output_root / f"{name}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "model_name": name,
                "in_channels": in_channels,
                "config": results["config"],
            },
            checkpoint,
        )
        val_by_scene = _evaluate_by_scene(
            model, val_dataset, args.batch_size, args.device, args.train_domain
        )
        test_by_scene = _evaluate_by_scene(
            model, test_dataset, args.batch_size, args.device, args.train_domain
        )
        results["models"][name] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "learning_rate": learning_rate,
            "val_before": asdict(before),
            "val_after": asdict(val_metrics),
            "test": asdict(test_metrics),
            "val_by_scene": val_by_scene,
            "test_by_scene": test_by_scene,
            "val_scene_macro": _scene_macro(val_by_scene),
            "test_scene_macro": _scene_macro(test_by_scene),
            "test_timestep_curve": timestep_curve,
            "train_seconds": elapsed,
            "loss_first_20_mean": sum(history[:20]) / min(20, len(history)),
            "loss_last_20_mean": sum(history[-20:]) / min(20, len(history)),
            "checkpoint": str(checkpoint),
        }
        print(
            f"{name}: val rot={val_metrics.rotation_deg:.3f} trans={val_metrics.translation_direction_deg} "
            f"test rot={test_metrics.rotation_deg:.3f} trans={test_metrics.translation_direction_deg} "
            f"time={elapsed:.1f}s",
            flush=True,
        )
        (args.output_root / "results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    results["total_seconds"] = time.perf_counter() - started
    (args.output_root / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"results: {args.output_root / 'results.json'}")


if __name__ == "__main__":
    main()
