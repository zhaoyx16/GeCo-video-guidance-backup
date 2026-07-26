from __future__ import annotations

import unittest

import torch

from external.guidance_wan.pipeline_wan_i2v_full_guided import _prepare_frame_guidance_targets


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


if __name__ == "__main__":
    unittest.main()
