from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from geometry_selection.wan_online_geometry import WanPredictedCleanGeometry


class _FakeVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(
            z_dim=3,
            latents_mean=[0.0, 0.0, 0.0],
            latents_std=[1.0, 1.0, 1.0],
        )

    def decode(self, latents: torch.Tensor, return_dict: bool = False):
        return (latents,)


class _FakePrediction:
    def __init__(self, frames: int, height: int, width: int, indices: tuple[int, ...]) -> None:
        self.world_to_camera = np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0)
        self.intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], frames, axis=0)
        self.depth = np.ones((frames, height, width), dtype=np.float32)
        self.confidence = np.full((frames, height, width), 2.0, dtype=np.float32)
        self.keyframe_indices = np.asarray(indices, dtype=np.int64)


class _FakeAdapter:
    def predict_image_paths(self, paths, *, keyframe_indices, hash_checkpoint):
        assert hash_checkpoint is False
        assert all(path.is_file() for path in paths)
        from PIL import Image

        with Image.open(paths[0]) as image:
            width, height = image.size
        return _FakePrediction(len(paths), height, width, tuple(keyframe_indices))


def test_predicted_clean_callback_decodes_and_returns_geometry(tmp_path) -> None:
    callback = WanPredictedCleanGeometry(
        vae=_FakeVAE(),
        geometry_adapter=_FakeAdapter(),
        frame_indices=[0, 1],
        confidence_percentile=20.0,
        temporary_root=tmp_path,
    )
    x0 = torch.zeros((1, 3, 2, 4, 5), dtype=torch.float32)
    x0[:, :, 1] = 1.0
    bundle = callback(19, x0)

    assert bundle["world_to_camera"].shape == (2, 4, 4)
    assert bundle["depth"].shape == (2, 4, 5)
    assert np.array_equal(bundle["keyframe_indices"], np.array([0, 1]))
    assert np.allclose(bundle["confidence_thresholds"], np.array([2.0, 2.0]))
    assert len(callback.records) == 1
    assert callback.records[0]["step"] == 19
    assert len(callback.records[0]["frame_sha256"]) == 2
