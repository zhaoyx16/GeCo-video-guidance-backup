from __future__ import annotations

import unittest

import torch

from external.guidance_wan.pipeline_wan_i2v_full_guided import (
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
    def assert_plan(self, frame_index: int, expected: tuple[int, int, int]) -> None:
        plan = _wan_causal_frame_decode_plan(
            frame_index,
            num_frames=121,
            latent_frames=31,
            temporal_scale=4,
        )
        self.assertEqual(plan, expected)

    def test_first_middle_last_anchor_mapping(self) -> None:
        self.assert_plan(0, (0, 1, 0))
        self.assert_plan(60, (14, 16, 4))
        self.assert_plan(120, (29, 31, 4))

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


if __name__ == "__main__":
    unittest.main()
