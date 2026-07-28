from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch
from PIL import Image

from latent_geometry.dl3dv import (
    choose_clip_starts,
    latent_anchor_rgb_indices,
    load_dl3dv_scene,
    resize_cover_crop_geometry,
    transform_intrinsics_for_cover_crop,
)


class DL3DVProbeHelperTests(unittest.TestCase):
    def test_load_scene_inverts_c2w_and_sorts_frame_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "scene" / "images_4"
            images.mkdir(parents=True)
            for frame_id in (2, 1):
                Image.new("RGB", (8, 4), color=(frame_id, 0, 0)).save(images / f"frame_{frame_id:05d}.png")
            c2w_1 = torch.eye(4)
            c2w_1[0, 3] = 3.0
            c2w_2 = torch.eye(4)
            c2w_2[1, 3] = 4.0
            payload = {
                "w": 8,
                "h": 4,
                "fl_x": 4.0,
                "fl_y": 4.0,
                "cx": 4.0,
                "cy": 2.0,
                "frames": [
                    {"file_path": "images/frame_00002.png", "transform_matrix": c2w_2.tolist()},
                    {"file_path": "images/frame_00001.png", "transform_matrix": c2w_1.tolist()},
                ],
            }
            transforms = root / "scene" / "transforms.json"
            transforms.write_text(json.dumps(payload), encoding="utf-8")
            scene = load_dl3dv_scene(transforms, root)
            self.assertEqual(scene.frame_ids.tolist(), [1, 2])
            self.assertAlmostEqual(float(scene.camera_poses_w2c[0, 0, 3]), -3.0)
            self.assertAlmostEqual(float(scene.camera_poses_w2c[1, 1, 3]), 4.0)
            self.assertEqual(scene.camera_model, "UNKNOWN")

    def test_cover_crop_intrinsics_matches_resize_and_crop(self) -> None:
        geometry = resize_cover_crop_geometry(960, 540, 1280, 704)
        self.assertEqual(geometry, (1280, 720, 0, 8))
        intrinsics = torch.tensor([[[2000.0, 0.0, 1920.0], [0.0, 2000.0, 1080.0], [0.0, 0.0, 1.0]]])
        transformed = transform_intrinsics_for_cover_crop(
            intrinsics,
            calibration_width=3840,
            calibration_height=2160,
            image_width=960,
            image_height=540,
            target_width=1280,
            target_height=704,
        )
        self.assertTrue(torch.allclose(transformed[0, :2, :3], torch.tensor([[666.6667, 0.0, 640.0], [0.0, 666.6667, 352.0]]), atol=1e-3))

    def test_clip_starts_and_wan_causal_anchors(self) -> None:
        self.assertEqual(latent_anchor_rgb_indices(17, 4), [0, 4, 8, 12, 16])
        self.assertEqual(choose_clip_starts(40, clip_frames=17, frame_step=1, clip_stride=8, max_clips=None), [0, 8, 16])
        with self.assertRaises(ValueError):
            latent_anchor_rgb_indices(16, 4)


if __name__ == "__main__":
    unittest.main()
