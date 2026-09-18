from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "external" / "guidance_wan"))

from draft_geometry_map import (
    _project_token_centres,
    _sample_conservative_source_depth,
    _source_free_space_violation_map,
    build_anchor_transport_map,
    frame_to_latent_index,
)
from geometry_provenance import (
    build_generation_contract,
    build_source_artifact,
    validate_generation_contract,
)
from geometry_free_space import (
    FREE_SPACE_VALUE_MODE,
    apply_observed_background_value_residual,
    apply_observed_background_value_residual_to_attention,
    prepare_observed_background_transport,
    validate_frozen_free_space_map,
    validate_frozen_free_space_transport_settings,
)
from geometry_suppression import (
    apply_hidden_input_suppression,
    validate_confidence,
    validate_frozen_offline_suppression_contract,
)


def identity_geometry(frame_count: int, depth_values: list[float]):
    intrinsic = torch.eye(3).repeat(frame_count, 1, 1)
    extrinsic = torch.cat(
        [torch.eye(3), torch.zeros(3, 1)],
        dim=1,
    ).repeat(frame_count, 1, 1)
    depth = torch.stack(
        [torch.full((8, 8, 1), value) for value in depth_values],
        dim=0,
    )
    confidence = torch.ones_like(depth)
    return intrinsic, extrinsic, depth, confidence


class GeometryMathTests(unittest.TestCase):
    def test_causal_frame_mapping(self):
        expected = {
            0: 0,
            1: 1,
            4: 1,
            5: 2,
            8: 2,
            9: 3,
            12: 3,
            13: 4,
            16: 4,
        }
        for frame, latent in expected.items():
            self.assertEqual(frame_to_latent_index(frame, 4, 31), latent)

    def test_identity_projection_uses_pixel_centres(self):
        depth = torch.ones(8, 8)
        source_index, _, _, u, v, target_z, inside = _project_token_centres(
            depth,
            torch.eye(3),
            torch.cat([torch.eye(3), torch.zeros(3, 1)], dim=1),
            torch.eye(3),
            torch.cat([torch.eye(3), torch.zeros(3, 1)], dim=1),
            2,
            2,
        )
        self.assertTrue(torch.equal(source_index, torch.arange(4)))
        self.assertTrue(torch.allclose(u, torch.tensor([1.5, 5.5, 1.5, 5.5])))
        self.assertTrue(torch.allclose(v, torch.tensor([1.5, 1.5, 5.5, 5.5])))
        self.assertTrue(torch.allclose(target_z, torch.ones(4)))
        self.assertTrue(inside.all())

    def _free_space(self, source_depth, target_depth, target_confidence=None):
        if target_confidence is None:
            target_confidence = torch.ones_like(target_depth)
        return _source_free_space_violation_map(
            source_depth=source_depth,
            target_depth=target_depth,
            source_confidence=torch.ones_like(source_depth),
            target_confidence=target_confidence,
            source_intrinsic=torch.eye(3),
            target_intrinsic=torch.eye(3),
            source_extrinsic=torch.cat(
                [torch.eye(3), torch.zeros(3, 1)],
                dim=1,
            ),
            target_extrinsic=torch.cat(
                [torch.eye(3), torch.zeros(3, 1)],
                dim=1,
            ),
            token_height=2,
            token_width=2,
            confidence_percentile=0.0,
            confidence_floor=0.0,
            source_depth_percentile=100.0,
            behind_threshold=0.05,
            front_threshold=0.40,
            spatial_samples_per_axis=1,
            min_spatial_support=1,
            source_depth_edge_radius=0,
            max_source_depth_spread=0.10,
        )

    def test_free_space_sign_and_low_confidence_abstention(self):
        source = torch.full((8, 8), 5.0)
        _, same_confidence, _ = self._free_space(
            source,
            torch.full((8, 8), 5.0),
        )
        _, closer_confidence, _ = self._free_space(
            source,
            torch.full((8, 8), 3.0),
        )
        _, farther_confidence, _ = self._free_space(
            source,
            torch.full((8, 8), 7.0),
        )
        _, low_confidence, _ = self._free_space(
            source,
            torch.full((8, 8), 3.0),
            target_confidence=torch.zeros(8, 8),
        )
        self.assertEqual(torch.count_nonzero(same_confidence).item(), 0)
        self.assertGreater(torch.count_nonzero(closer_confidence).item(), 0)
        self.assertEqual(torch.count_nonzero(farther_confidence).item(), 0)
        self.assertEqual(torch.count_nonzero(low_confidence).item(), 0)

    def test_depth_discontinuity_abstains(self):
        depth = torch.full((8, 8), 3.0)
        depth[:, 4:] = 9.0
        sampled, valid, spread = _sample_conservative_source_depth(
            depth,
            torch.tensor([1.5, 3.5]),
            torch.tensor([3.5, 3.5]),
            radius=1,
            max_relative_spread=0.10,
        )
        self.assertTrue(valid[0])
        self.assertFalse(valid[1])
        self.assertEqual(sampled[1].item(), 3.0)
        self.assertGreater(spread[1].item(), 0.10)

    def test_temporal_aggregation_is_order_independent(self):
        selected = [0, 13, 14, 15, 16]
        intrinsic, extrinsic, depth, confidence = identity_geometry(
            5,
            [5.0, 3.0, 3.0, 5.0, 3.0],
        )

        def build(target_frames):
            return build_anchor_transport_map(
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                depth_map=depth,
                confidence_map=confidence,
                selected_video_frames=selected,
                anchor_video_frames=[0],
                target_video_frames=target_frames,
                token_grid=(5, 2, 2),
                temporal_scale=4,
                memory_slots=1,
                min_temporal_support=2,
                min_temporal_observations=4,
                confidence_percentile=0.0,
                confidence_floor=0.0,
                visibility_mode="source_free_space_violation",
                behind_threshold=0.05,
                front_threshold=0.40,
                source_depth_edge_radius=0,
            )

        forward = build([13, 14, 15, 16])
        reverse = build([16, 15, 14, 13])
        self.assertTrue(
            torch.equal(
                forward.conflict_confidence,
                reverse.conflict_confidence,
            )
        )
        self.assertTrue(
            torch.equal(
                forward.temporal_support_count,
                reverse.temporal_support_count,
            )
        )
        self.assertEqual(forward.temporal_observation_count[-1].item(), 4)
        self.assertEqual(torch.count_nonzero(forward.confidence).item(), 0)
        self.assertGreater(
            torch.count_nonzero(
                forward.observed_background_confidence
            ).item(),
            0,
        )
        self.assertTrue(
            torch.equal(
                forward.observed_background_confidence.amax(dim=-1) > 0,
                forward.conflict_confidence > 0,
            )
        )
        self.assertEqual(
            forward.metadata["evidence_semantics"],
            {
                "confidence": "same_surface_transport",
                "conflict_confidence": (
                    "source_observed_free_space_conflict"
                ),
                "observed_background_confidence": (
                    "source_ray_background_behind_free_space_conflict"
                ),
            },
        )


class SuppressionTests(unittest.TestCase):
    def setUp(self):
        self.hidden = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
        self.grid = (3, 1, 2)

    def test_alpha_zero_and_zero_mask_are_exact_noops(self):
        confidence = torch.ones(2, 1, 2)
        alpha_zero = apply_hidden_input_suppression(
            self.hidden,
            confidence,
            token_grid=self.grid,
            alpha=0.0,
        )
        zero_mask = apply_hidden_input_suppression(
            self.hidden,
            torch.zeros_like(confidence),
            token_grid=self.grid,
            alpha=0.5,
        )
        self.assertTrue(torch.equal(alpha_zero, self.hidden))
        self.assertTrue(torch.equal(zero_mask, self.hidden))

    def test_single_and_all_token_masks(self):
        single = torch.zeros(2, 1, 2)
        single[0, 0, 1] = 1.0
        result = apply_hidden_input_suppression(
            self.hidden,
            single,
            token_grid=self.grid,
            alpha=0.25,
        ).reshape(1, 3, 2, 4)
        original = self.hidden.reshape(1, 3, 2, 4)
        self.assertTrue(torch.equal(result[:, 0], original[:, 0]))
        self.assertTrue(torch.equal(result[:, 1, 0], original[:, 1, 0]))
        self.assertTrue(
            torch.equal(result[:, 1, 1], original[:, 1, 1] * 0.75)
        )
        self.assertTrue(torch.equal(result[:, 2], original[:, 2]))

        all_one = apply_hidden_input_suppression(
            self.hidden,
            torch.ones(2, 1, 2),
            token_grid=self.grid,
            alpha=0.25,
        ).reshape(1, 3, 2, 4)
        self.assertTrue(torch.equal(all_one[:, 0], original[:, 0]))
        self.assertTrue(torch.equal(all_one[:, 1:], original[:, 1:] * 0.75))

    def test_invalid_confidence_rejected(self):
        for invalid in (
            torch.tensor([float("nan")]),
            torch.tensor([float("inf")]),
            torch.tensor([-0.1]),
            torch.tensor([1.1]),
        ):
            with self.assertRaises(ValueError):
                validate_confidence(invalid, name="test")

    def test_frozen_contract_rejects_unreviewed_paths(self):
        valid = {
            "enabled": True,
            "alpha": 0.05,
            "layers": [0],
            "mode": "hidden_input_suppress",
            "map_path": "map.pt",
            "frame_indices": None,
            "cond_only": True,
            "attn_avg_alpha": 0.0,
            "attn_avg_layers": None,
            "expected_provenance": {"contract_version": 1},
        }
        validate_frozen_offline_suppression_contract(**valid)
        for field, value in (
            ("layers", [1]),
            ("mode", "value_residual"),
            ("map_path", None),
            ("frame_indices", [0, 4]),
            ("cond_only", False),
            ("attn_avg_alpha", 0.1),
            ("expected_provenance", None),
        ):
            invalid = dict(valid)
            invalid[field] = value
            with self.assertRaises(ValueError, msg=field):
                validate_frozen_offline_suppression_contract(**invalid)


class FreeSpaceValueTransportTests(unittest.TestCase):
    def test_exact_value_blend_and_noops(self):
        attended = torch.arange(16, dtype=torch.float32).reshape(
            1, 2, 2, 2, 2
        )
        source = attended + 10.0
        confidence = torch.tensor([[[0.0, 0.5], [1.0, 0.0]]])
        result = apply_observed_background_value_residual(
            attended,
            source,
            confidence,
            alpha=0.2,
        )
        expected = attended.clone()
        expected[:, 0, 1] = (
            attended[:, 0, 1] * 0.9 + source[:, 0, 1] * 0.1
        )
        expected[:, 1, 0] = (
            attended[:, 1, 0] * 0.8 + source[:, 1, 0] * 0.2
        )
        self.assertTrue(torch.allclose(result, expected))
        self.assertTrue(
            torch.equal(
                apply_observed_background_value_residual(
                    attended,
                    source,
                    confidence,
                    alpha=0.0,
                ),
                attended,
            )
        )
        self.assertTrue(
            torch.equal(
                apply_observed_background_value_residual(
                    attended,
                    source,
                    torch.zeros_like(confidence),
                    alpha=0.2,
                ),
                attended,
            )
        )

    def test_frozen_free_space_settings_reject_unreviewed_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            map_path = Path(directory) / "map.pt"
            map_path.write_bytes(b"map")
            valid = {
                "enabled": True,
                "alpha": 0.1,
                "layers": [0],
                "mode": FREE_SPACE_VALUE_MODE,
                "map_path": str(map_path),
                "frame_indices": None,
                "cond_only": True,
                "attn_avg_alpha": 0.0,
                "attn_avg_layers": None,
                "expected_provenance": {"contract_version": 1},
            }
            validate_frozen_free_space_transport_settings(**valid)
            for field, value in (
                ("layers", [1]),
                ("mode", "value_residual"),
                ("map_path", None),
                ("frame_indices", [0, 4]),
                ("cond_only", False),
                ("attn_avg_alpha", 0.1),
                ("expected_provenance", None),
            ):
                invalid = dict(valid)
                invalid[field] = value
                with self.assertRaises(ValueError, msg=field):
                    validate_frozen_free_space_transport_settings(**invalid)

    def test_conflict_confidence_controls_blend_and_indices_are_causal(self):
        source_time = torch.tensor([[[0], [-1]], [[0], [0]]])
        source_index = torch.tensor([[[1], [-1]], [[0], [1]]])
        source_confidence = torch.tensor(
            [[[[1.0], [0.0]]], [[[0.9], [0.2]]]]
        )
        conflict_confidence = torch.tensor(
            [[[0.6, 0.0]], [[0.4, 0.8]]]
        )
        safe_time, safe_index, blend_confidence = (
            prepare_observed_background_transport(
                source_time,
                source_index,
                source_confidence,
                conflict_confidence,
                spatial_tokens=2,
            )
        )
        self.assertEqual(safe_time[0, 1, 0].item(), 0)
        self.assertEqual(safe_index[0, 1, 0].item(), 0)
        self.assertTrue(
            torch.equal(
                blend_confidence,
                conflict_confidence.unsqueeze(-1),
            )
        )

        attended = torch.zeros(1, 2, 2, 1, 1)
        source = torch.ones_like(attended)
        result = apply_observed_background_value_residual(
            attended,
            source,
            blend_confidence.reshape(1, 2, 2),
            alpha=0.5,
        )
        self.assertAlmostEqual(result[0, 0, 0].item(), 0.3)
        self.assertAlmostEqual(result[0, 1, 0].item(), 0.2)
        self.assertAlmostEqual(result[0, 1, 1].item(), 0.4)

        invalid_time = source_time.clone()
        invalid_time[1, 0, 0] = 2
        with self.assertRaises(ValueError):
            prepare_observed_background_transport(
                invalid_time,
                source_index,
                source_confidence,
                conflict_confidence,
                spatial_tokens=2,
            )
        invalid_index = source_index.clone()
        invalid_index[1, 0, 0] = 2
        with self.assertRaises(ValueError):
            prepare_observed_background_transport(
                source_time,
                invalid_index,
                source_confidence,
                conflict_confidence,
                spatial_tokens=2,
            )

        slot_one_only_confidence = torch.cat(
            [
                torch.zeros_like(source_confidence),
                source_confidence,
            ],
            dim=-1,
        )
        slot_one_only_time = torch.cat([source_time, source_time], dim=-1)
        slot_one_only_index = torch.cat([source_index, source_index], dim=-1)
        with self.assertRaises(ValueError):
            prepare_observed_background_transport(
                slot_one_only_time,
                slot_one_only_index,
                slot_one_only_confidence,
                conflict_confidence,
                spatial_tokens=2,
            )

        with self.assertRaisesRegex(ValueError, "integer tensor dtype"):
            prepare_observed_background_transport(
                source_time.float() - 0.1,
                source_index,
                source_confidence,
                conflict_confidence,
                spatial_tokens=2,
            )

    def test_flat_attention_blend_uses_runtime_batch_size(self):
        attended = torch.zeros(2, 6, 1, 1)
        source = torch.stack(
            [
                torch.ones(2, 2, 1, 1),
                torch.full((2, 2, 1, 1), 2.0),
            ],
            dim=0,
        )
        confidence = torch.tensor(
            [
                [[1.0, 0.0], [0.5, 1.0]],
                [[0.25, 1.0], [0.0, 0.5]],
            ]
        )
        result = apply_observed_background_value_residual_to_attention(
            attended,
            source,
            confidence,
            token_grid=(3, 1, 2),
            alpha=0.2,
        ).reshape(2, 3, 2, 1, 1)
        self.assertTrue(torch.equal(result[:, :1], torch.zeros_like(result[:, :1])))
        self.assertAlmostEqual(result[0, 1, 0].item(), 0.2)
        self.assertAlmostEqual(result[0, 2, 0].item(), 0.1)
        self.assertAlmostEqual(result[1, 1, 0].item(), 0.1)
        self.assertAlmostEqual(result[1, 1, 1].item(), 0.4)


class ProvenanceAndCleanupTests(unittest.TestCase):
    def test_provenance_accepts_exact_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            image.write_bytes(b"image")
            video = root / "draft.mp4"
            video.write_bytes(b"video")
            model = root / "model"
            (model / "scheduler").mkdir(parents=True)
            (model / "model_index.json").write_text("{}")
            (model / "scheduler" / "scheduler_config.json").write_text("{}")
            contract = build_generation_contract(
                prompt="same corridor",
                negative_prompt=None,
                seed=1,
                steps=50,
                frames=121,
                height=704,
                width=1280,
                fps=24,
                guidance_scale=5.0,
                image_path=image,
                model_path=model,
            )
            metadata = {
                "format_version": 3,
                "visibility_mode": "source_free_space_violation",
                "evidence_semantics": {
                    "confidence": "same_surface_transport",
                    "conflict_confidence": (
                        "source_observed_free_space_conflict"
                    ),
                    "observed_background_confidence": (
                        "source_ray_background_behind_free_space_conflict"
                    ),
                },
                "generation_contract": contract,
                "source_artifact": build_source_artifact(video),
            }
            validate_generation_contract(metadata, contract)
            map_payload = {
                "source_time": torch.zeros(1, 1, 1, dtype=torch.long),
                "source_index": torch.zeros(1, 1, 1, dtype=torch.long),
                "confidence": torch.zeros(1, 1, 1, 1),
                "observed_background_time": torch.zeros(1, 1, 1, dtype=torch.long),
                "observed_background_index": torch.zeros(1, 1, 1, dtype=torch.long),
                "observed_background_confidence": torch.ones(1, 1, 1, 1),
                "conflict_confidence": torch.ones(1, 1, 1),
                "metadata": metadata,
            }
            validate_frozen_free_space_map(map_payload, contract)
            invalid_map = dict(map_payload)
            invalid_map["conflict_confidence"] = torch.zeros(1, 1, 1)
            with self.assertRaises(ValueError):
                validate_frozen_free_space_map(invalid_map, contract)
            mismatch = dict(contract)
            mismatch["seed"] = 2
            with self.assertRaises(ValueError):
                validate_generation_contract(metadata, mismatch)
            video.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                validate_generation_contract(metadata, contract)

    def test_installation_is_exception_safe(self):
        pipeline_path = (
            REPO
            / "external"
            / "guidance_wan"
            / "pipeline_wan_i2v_geometry_transport.py"
        )
        tree = ast.parse(pipeline_path.read_text())
        guarded_install = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try) or not node.handlers:
                continue
            body_text = ast.unparse(node.body)
            handler_text = ast.unparse(node.handlers)
            if (
                (
                    "register_forward" in body_text
                    or "set_processor" in body_text
                )
                and "_restore_attention_modifications" in handler_text
            ):
                guarded_install.append(node)
        self.assertTrue(
            guarded_install,
            "Hook/processor installation must restore partial changes on failure",
        )

    def test_pipeline_has_exception_safe_hook_cleanup(self):
        pipeline_path = (
            REPO
            / "external"
            / "guidance_wan"
            / "pipeline_wan_i2v_geometry_transport.py"
        )
        tree = ast.parse(pipeline_path.read_text())
        cleanup_try = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try) or not node.finalbody:
                continue
            final_text = ast.unparse(node.finalbody)
            if "_restore_attention_modifications" in final_text:
                cleanup_try.append(node)
        self.assertTrue(
            cleanup_try,
            "Denoising must restore hooks/processors from a finally block",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
