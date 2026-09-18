from __future__ import annotations

import unittest

import torch

from external.guidance_wan.pipeline_wan_i2v_full_guided import (
    _frame_guidance_update_base,
    _prepare_frame_guidance_targets,
    _wan_causal_frame_decode_plan,
)


class FakeVideoProcessor:
    def __init__(self) -> None:
        self.calls: list[tuple[object, int, int]] = []

    def preprocess(self, value: object, *, height: int, width: int) -> torch.Tensor:
        self.calls.append((value, height, width))
        base = 1.0 if value == "middle" else 0.0
        return torch.full((1, 3, height, width), base)


class FrameGuidancePipelineHelperTest(unittest.TestCase):
    def test_targets_use_prediction_layout_and_do_not_require_grad(self) -> None:
        processor = FakeVideoProcessor()
        targets = _prepare_frame_guidance_targets(
            processor,
            {0: "first", "8": "middle"},
            fixed_frames=[0, 8],
            num_frames=9,
            height=2,
            width=3,
            vae_device=torch.device("cpu"),
        )
        self.assertEqual(processor.calls, [("first", 2, 3), ("middle", 2, 3)])
        self.assertEqual(tuple(targets[8].shape), (1, 2, 3, 3))
        self.assertFalse(targets[8].requires_grad)
        self.assertTrue(torch.all(targets[8] == 1.0))

    def test_targets_match_official_latent_downscale_resolution(self) -> None:
        processor = FakeVideoProcessor()
        targets = _prepare_frame_guidance_targets(
            processor,
            {0: "first", 8: "middle"},
            fixed_frames=[0, 8],
            num_frames=9,
            height=704,
            width=1280,
            vae_device=torch.device("cpu"),
            latent_downscale_factor=4,
        )
        self.assertEqual(processor.calls, [("first", 176, 320), ("middle", 176, 320)])
        self.assertEqual(tuple(targets[8].shape), (1, 176, 320, 3))

    def test_rejects_unsupported_latent_downscale_factor(self) -> None:
        with self.assertRaisesRegex(ValueError, "one of"):
            _prepare_frame_guidance_targets(
                FakeVideoProcessor(),
                {0: "first", 8: "middle"},
                fixed_frames=[0, 8],
                num_frames=9,
                height=704,
                width=1280,
                vae_device=torch.device("cpu"),
                latent_downscale_factor=3,
            )

    def test_rejects_target_not_selected_for_decoding(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing from fixed_frames"):
            _prepare_frame_guidance_targets(
                FakeVideoProcessor(),
                {0: "first", 4: "middle"},
                fixed_frames=[0, 8],
                num_frames=9,
                height=2,
                width=3,
                vae_device=torch.device("cpu"),
            )


class WanCausalFrameDecodePlanTest(unittest.TestCase):
    def assert_plan(
        self,
        frame_index: int,
        expected: tuple[int, int, int],
        *,
        context_latents: int = 2,
    ) -> None:
        plan = _wan_causal_frame_decode_plan(
            frame_index,
            num_frames=121,
            latent_frames=31,
            temporal_scale=4,
            context_latents=context_latents,
        )
        self.assertEqual(plan, expected)

    def test_first_middle_last_anchor_mapping(self) -> None:
        self.assert_plan(0, (0, 1, 0))
        self.assert_plan(60, (14, 16, 4))
        self.assert_plan(120, (29, 31, 4))

    def test_three_token_context_preserves_target_mapping(self) -> None:
        self.assert_plan(0, (0, 1, 0), context_latents=3)
        self.assert_plan(5, (0, 3, 5), context_latents=3)
        self.assert_plan(60, (13, 16, 8), context_latents=3)
        self.assert_plan(120, (28, 31, 8), context_latents=3)

    def test_noninitial_frames_select_the_target_token_not_a_future_slot(self) -> None:
        for frame_index in [1, 4, 5, 8, 60, 119, 120]:
            start, end, local_index = _wan_causal_frame_decode_plan(
                frame_index,
                num_frames=121,
                latent_frames=31,
                temporal_scale=4,
            )
            target_latent = (frame_index - 1) // 4 + 1
            self.assertEqual((start, end), (target_latent - 1, target_latent + 1))
            self.assertEqual(local_index, (frame_index - 1) % 4 + 1)
            reconstructed_frame = 1 + (target_latent - 1) * 4 + (local_index - 1)
            self.assertEqual(reconstructed_frame, frame_index)

    def test_rejects_frame_not_representable_by_latent_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be represented"):
            _wan_causal_frame_decode_plan(
                9,
                num_frames=10,
                latent_frames=3,
                temporal_scale=4,
            )

    def test_rejects_context_shorter_than_predecessor_target_pair(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            _wan_causal_frame_decode_plan(
                8,
                num_frames=121,
                latent_frames=31,
                temporal_scale=4,
                context_latents=1,
            )


class FrameGuidanceUpdateBaseTest(unittest.TestCase):
    def test_direct_mode_preserves_current_latent(self) -> None:
        latents = torch.randn(1, 2, 3)
        x0_pred = torch.randn_like(latents)
        base = _frame_guidance_update_base(
            latents,
            x0_pred,
            sigma=torch.tensor(0.6),
            generator=torch.Generator().manual_seed(7),
            use_vlo_renoising=False,
        )
        self.assertIs(base, latents)

    def test_vlo_mode_matches_flow_matching_renoising_equation(self) -> None:
        latents = torch.zeros(1, 2, 3)
        x0_pred = torch.full_like(latents, 2.0)
        sigma = torch.tensor(0.25)
        expected_generator = torch.Generator().manual_seed(11)
        actual_generator = torch.Generator().manual_seed(11)
        expected_noise = torch.randn(
            x0_pred.shape,
            generator=expected_generator,
            dtype=x0_pred.dtype,
        )
        expected = sigma * expected_noise + (1.0 - sigma) * x0_pred
        actual = _frame_guidance_update_base(
            latents,
            x0_pred,
            sigma,
            actual_generator,
            use_vlo_renoising=True,
        )
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
