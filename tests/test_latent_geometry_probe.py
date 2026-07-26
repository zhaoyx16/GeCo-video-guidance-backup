"""Deterministic contract tests for the latent-geometry probe foundation."""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from latent_geometry.data import (
    CachedLatentDataset,
    LATENT_DOMAIN_X0_PRED,
    LinearFlowNoiseSchedule,
    ManifestError,
    load_manifest_records,
)
from latent_geometry.geometry import make_relative_pose_target, rotation_6d_to_matrix
from latent_geometry.models import ConstantPoseBaseline, LinearLatentProbe, Small3DConvCritic
from latent_geometry.synthetic import create_synthetic_probe_manifest
from latent_geometry.training import evaluate_probe, probe_loss_for_batch, train_probe_steps


class _RecordingPoseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.tensor(1.0))
        self.last_timestep: torch.Tensor | None = None

    def forward(self, latent: torch.Tensor, timestep: torch.Tensor) -> dict[str, torch.Tensor]:
        self.last_timestep = timestep.detach().clone()
        batch_size = latent.shape[0]
        rotation = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=latent.device).expand(batch_size, -1)
        direction = torch.tensor([1.0, 0.0, 0.0], device=latent.device).expand(batch_size, -1)
        return {
            "rotation_6d": rotation + self.parameter * 0.0,
            "translation_direction": direction + self.parameter * 0.0,
        }


def _read_records(manifest: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")


class LatentGeometryProbeTests(unittest.TestCase):
    def test_relative_pose_target_is_scale_free_and_ordered(self) -> None:
        pose_source = torch.eye(4)
        pose_target = torch.eye(4)
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

    def test_manifest_rejects_same_source_uid_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = create_synthetic_probe_manifest(temporary, train_scenes=1, val_scenes=1, records_per_scene=1)
            records = _read_records(manifest)
            records[1]["source_scene_uid"] = records[0]["source_scene_uid"]
            leaky = Path(temporary) / "same_source_uid.jsonl"
            _write_records(leaky, records)
            with self.assertRaisesRegex(ManifestError, "source_scene_uid"):
                load_manifest_records(leaky)

    def test_cache_provenance_rejects_cross_split_reuse_under_renamed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = create_synthetic_probe_manifest(temporary, train_scenes=1, val_scenes=1, records_per_scene=1)
            records = _read_records(manifest)
            train_record, val_record = records
            malicious = dict(val_record)
            malicious.update(
                {
                    "record_id": "renamed_val_record",
                    "scene_id": "totally_different_human_label",
                    "source_dataset": "different_dataset_label",
                    "source_scene_uid": "different_scene_uid",
                    "source_clip_uid": "different_clip_uid",
                    "cache_sha256": "0" * 64,
                    "cache_path": train_record["cache_path"],
                }
            )
            leaky = Path(temporary) / "renamed_reuse.jsonl"
            _write_records(leaky, [train_record, malicious])
            with self.assertRaisesRegex(ManifestError, "Manifest/cache provenance mismatch"):
                CachedLatentDataset(leaky, "train")

    def test_cache_rejects_invalid_se3_pose(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = create_synthetic_probe_manifest(temporary, train_scenes=1, val_scenes=1, records_per_scene=1)
            record = _read_records(manifest)[0]
            cache_path = Path(temporary) / str(record["cache_path"])
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            payload["camera_poses_w2c"][1, 3, 3] = 0.0
            torch.save(payload, cache_path)
            with self.assertRaisesRegex(ValueError, "bottom row"):
                CachedLatentDataset(manifest, "train")

    def test_cache_sha_detects_tampered_latent_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = create_synthetic_probe_manifest(temporary, train_scenes=1, val_scenes=1, records_per_scene=1)
            record = _read_records(manifest)[0]
            cache_path = Path(temporary) / str(record["cache_path"])
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            payload["z0"][0, 0, 0, 0] += 1.0
            torch.save(payload, cache_path)
            with self.assertRaisesRegex(ManifestError, "Cache SHA256 mismatch"):
                CachedLatentDataset(manifest, "train")

    def test_cache_rejects_pose_to_latent_mapping_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = create_synthetic_probe_manifest(temporary, train_scenes=1, val_scenes=1, records_per_scene=1)
            records = _read_records(manifest)
            records[0]["source_pose_index"] = 1
            mismatch = Path(temporary) / "mapping_mismatch.jsonl"
            _write_records(mismatch, records)
            with self.assertRaisesRegex(ManifestError, "RGB-pose to latent mapping mismatch"):
                CachedLatentDataset(mismatch, "train")

    def test_online_zt_is_deterministic_and_z0_control_uses_zero_timestep(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = create_synthetic_probe_manifest(temporary, train_scenes=1, val_scenes=1, records_per_scene=1)
            dataset = CachedLatentDataset(
                manifest,
                "train",
                noise_schedule=LinearFlowNoiseSchedule(0.4, 0.8),
                base_seed=77,
            )
            first = dataset[0]
            repeated = dataset[0]
            self.assertTrue(torch.equal(first["zt"], repeated["zt"]))
            self.assertGreater(float(first["timestep"]), 0.0)
            self.assertEqual(float(first["z0_timestep"]), 0.0)
            dataset.set_epoch(1)
            self.assertFalse(torch.equal(first["zt"], dataset[0]["zt"]))

            model = _RecordingPoseModel()
            one_item_batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
            probe_loss_for_batch(model, one_item_batch, input_key="z0")
            self.assertIsNotNone(model.last_timestep)
            self.assertTrue(torch.equal(model.last_timestep, torch.zeros_like(model.last_timestep)))

    def test_models_constant_training_and_cpu_smoke_training(self) -> None:
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
                output = model(batch["z0"], batch["z0_timestep"])
                self.assertEqual(tuple(output["rotation_6d"].shape), (4, 6))
                self.assertEqual(tuple(output["translation_direction"].shape), (4, 3))
                self.assertTrue(torch.isfinite(output["rotation_6d"]).all())

            constant = ConstantPoseBaseline()
            initial_constant = constant.pose.detach().clone()
            train_probe_steps(
                constant,
                loader,
                torch.optim.Adam(constant.parameters(), lr=5e-2),
                steps=4,
                input_key="z0",
            )
            self.assertFalse(torch.equal(initial_constant, constant.pose.detach()))

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

    def test_incompatible_x0_pred_domain_is_rejected_by_probe_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = create_synthetic_probe_manifest(temporary, train_scenes=1, val_scenes=1, records_per_scene=1)
            with self.assertRaisesRegex(ValueError, "only trains z0/online-zt"):
                CachedLatentDataset(manifest, "train", expected_latent_domain=LATENT_DOMAIN_X0_PRED)

    def test_x0_pred_cache_requires_scheduler_and_condition_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = create_synthetic_probe_manifest(temporary, train_scenes=1, val_scenes=1, records_per_scene=1)
            record = _read_records(manifest)[0]
            cache_path = Path(temporary) / str(record["cache_path"])
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            payload["latent_spec"]["domain"] = "x0_pred"
            torch.save(payload, cache_path)
            with self.assertRaisesRegex(ManifestError, "scheduler must be a mapping"):
                CachedLatentDataset(manifest, "train")

    @unittest.skipUnless(
        torch.cuda.is_available() and os.environ.get("RUN_CUDA_PROBE_TESTS") == "1",
        "Set RUN_CUDA_PROBE_TESTS=1 with an explicitly assigned CUDA device to run this test",
    )
    def test_evaluation_moves_model_to_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = create_synthetic_probe_manifest(temporary, train_scenes=1, val_scenes=1, records_per_scene=2)
            dataset = CachedLatentDataset(manifest, "train", noise_schedule=LinearFlowNoiseSchedule(0.0, 0.0))
            loader = DataLoader(dataset, batch_size=2, shuffle=False)
            model = LinearLatentProbe(int(dataset[0]["z0"].shape[0]))
            result = evaluate_probe(model, loader, device="cuda", input_key="z0")
            self.assertTrue(next(model.parameters()).is_cuda)
            self.assertTrue(math.isfinite(result.loss))


if __name__ == "__main__":
    unittest.main()
