"""Deterministic unit tests for the latent-geometry probe foundation."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from latent_geometry.data import (
    CachedLatentDataset,
    LinearFlowNoiseSchedule,
    ManifestError,
    load_manifest_records,
)
from latent_geometry.geometry import make_relative_pose_target, rotation_6d_to_matrix
from latent_geometry.models import ConstantPoseBaseline, LinearLatentProbe, Small3DConvCritic
from latent_geometry.synthetic import create_synthetic_probe_manifest
from latent_geometry.training import evaluate_probe, train_probe_steps


class LatentGeometryProbeTests(unittest.TestCase):
    def test_relative_pose_target_is_scale_free_and_ordered(self) -> None:
        pose_source = torch.eye(4)
        pose_target = torch.eye(4)
        theta = math.pi / 2.0
        pose_target[:3, :3] = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        pose_target[:3, 3] = torch.tensor([4.0, 0.0, 0.0])
        poses = torch.stack((pose_source, pose_target))

        target = make_relative_pose_target(poses, 0, 1)
        rotation = rotation_6d_to_matrix(target["rotation_6d"])
        self.assertTrue(torch.allclose(rotation, pose_target[:3, :3], atol=1e-5))
        self.assertTrue(torch.allclose(target["translation_direction"], torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(bool(target["translation_valid"]))

        pose_target_scaled = pose_target.clone()
        pose_target_scaled[:3, 3] *= 123.0
        scaled = make_relative_pose_target(torch.stack((pose_source, pose_target_scaled)), 0, 1)
        self.assertTrue(torch.allclose(target["translation_direction"], scaled["translation_direction"]))

    def test_scene_split_leakage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "leaky.jsonl"
            records = [
                {
                    "format_version": 1,
                    "record_id": "scene_a_train",
                    "scene_id": "scene_a",
                    "split": "train",
                    "cache_path": "ignored.pt",
                    "source_pose_index": 0,
                    "target_pose_index": 1,
                    "source_latent_index": 0,
                    "target_latent_index": 1,
                    "static_scene": True,
                },
                {
                    "format_version": 1,
                    "record_id": "scene_a_val",
                    "scene_id": "scene_a",
                    "split": "val",
                    "cache_path": "ignored.pt",
                    "source_pose_index": 0,
                    "target_pose_index": 1,
                    "source_latent_index": 0,
                    "target_latent_index": 1,
                    "static_scene": True,
                },
            ]
            manifest.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "Scene-disjoint split violation"):
                load_manifest_records(manifest)

    def test_online_zt_is_deterministic_per_record_and_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = create_synthetic_probe_manifest(temporary, train_scenes=1, val_scenes=1, records_per_scene=1)
            dataset = CachedLatentDataset(
                manifest,
                "train",
                noise_schedule=LinearFlowNoiseSchedule(0.1, 0.8),
                base_seed=77,
            )
            first = dataset[0]
            repeated = dataset[0]
            self.assertTrue(torch.equal(first["zt"], repeated["zt"]))
            self.assertEqual(float(first["timestep"]), float(repeated["timestep"]))
            dataset.set_epoch(1)
            next_epoch = dataset[0]
            self.assertFalse(torch.equal(first["zt"], next_epoch["zt"]))

    def test_models_and_tiny_cpu_smoke_training(self) -> None:
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = create_synthetic_probe_manifest(temporary, train_scenes=2, val_scenes=1, records_per_scene=4)
            dataset = CachedLatentDataset(
                manifest,
                "train",
                noise_schedule=LinearFlowNoiseSchedule(0.0, 0.0),
                base_seed=5,
            )
            loader = DataLoader(dataset, batch_size=4, shuffle=False)
            batch = next(iter(loader))
            in_channels = int(batch["z0"].shape[1])
            models = [
                ConstantPoseBaseline(),
                LinearLatentProbe(in_channels),
                Small3DConvCritic(in_channels, width=8, timestep_dim=8, hidden_dim=16),
            ]
            for model in models:
                output = model(batch["z0"], batch["timestep"])
                self.assertEqual(tuple(output["rotation_6d"].shape), (4, 6))
                self.assertEqual(tuple(output["translation_direction"].shape), (4, 3))
                self.assertTrue(torch.isfinite(output["rotation_6d"]).all())

            linear = LinearLatentProbe(in_channels)
            initial = evaluate_probe(linear, loader, input_key="z0")
            history = train_probe_steps(
                linear,
                loader,
                torch.optim.Adam(linear.parameters(), lr=5e-2),
                steps=24,
                input_key="z0",
            )
            final = evaluate_probe(linear, loader, input_key="z0")
            self.assertEqual(len(history), 24)
            self.assertTrue(all(math.isfinite(value) for value in history))
            self.assertLess(final.loss, initial.loss)
            self.assertTrue(math.isfinite(final.rotation_deg))
            self.assertIsNotNone(final.translation_direction_deg)


if __name__ == "__main__":
    unittest.main()
