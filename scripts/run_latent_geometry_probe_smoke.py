#!/usr/bin/env python3
"""Run a tiny deterministic CPU smoke experiment for latent_geometry.

This script creates only synthetic tensors in a temporary directory.  It does
not load Wan, a VAE, VGGT, Any4D, or any downloaded checkpoint.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from latent_geometry.data import CachedLatentDataset, LinearFlowNoiseSchedule
from latent_geometry.models import ConstantPoseBaseline, LinearLatentProbe, Small3DConvCritic
from latent_geometry.synthetic import create_synthetic_probe_manifest
from latent_geometry.training import evaluate_probe, train_probe_steps


def _format_metrics(label: str, metrics) -> str:
    direction = "n/a" if metrics.translation_direction_deg is None else f"{metrics.translation_direction_deg:.2f} deg"
    return (
        f"{label}: loss={metrics.loss:.6f}, rotation={metrics.rotation_deg:.2f} deg, "
        f"translation_direction={direction}, examples={metrics.examples}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=24, help="Training steps per learned probe")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--keep-dir", type=Path, default=None, help="Optional location for synthetic test data")
    arguments = parser.parse_args()
    if arguments.steps <= 0 or arguments.batch_size <= 0:
        raise SystemExit("--steps and --batch-size must be positive")

    torch.manual_seed(0)
    temporary = None
    if arguments.keep_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="latent_geometry_probe_smoke_")
        root = Path(temporary.name)
    else:
        root = arguments.keep_dir
        root.mkdir(parents=True, exist_ok=True)

    try:
        manifest = create_synthetic_probe_manifest(root, train_scenes=2, val_scenes=1, records_per_scene=4)
        schedule = LinearFlowNoiseSchedule(0.0, 0.0)
        train_dataset = CachedLatentDataset(manifest, "train", noise_schedule=schedule, base_seed=11)
        val_dataset = CachedLatentDataset(manifest, "val", noise_schedule=schedule, base_seed=13)
        train_loader = DataLoader(train_dataset, batch_size=arguments.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=arguments.batch_size, shuffle=False)
        in_channels = int(train_dataset[0]["z0"].shape[0])

        models = {
            "constant": ConstantPoseBaseline(),
            "linear": LinearLatentProbe(in_channels),
            "small_3dconv": Small3DConvCritic(in_channels, width=8, timestep_dim=8, hidden_dim=16),
        }
        for name, model in models.items():
            before = evaluate_probe(model, val_loader, input_key="z0")
            train_probe_steps(
                model,
                train_loader,
                torch.optim.Adam(model.parameters(), lr=5e-2),
                steps=arguments.steps,
                input_key="z0",
            )
            after = evaluate_probe(model, val_loader, input_key="z0")
            print(_format_metrics(f"{name} before", before))
            print(_format_metrics(f"{name} after", after))
        print(f"synthetic manifest: {manifest}")
        print("CPU smoke completed; this is a contract test, not a real geometry result.")
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
