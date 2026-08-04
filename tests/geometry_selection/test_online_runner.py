from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.dl3dv_geco.run_generation_case import make_wan_provisional_keyframe_decoder


class _FakeVAE:
    dtype = torch.float32

    class config:
        latents_mean = [0.0]
        latents_std = [1.0]
        z_dim = 1

    def __init__(self) -> None:
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.cache_cleared = False

    def parameters(self):
        return iter((self.parameter,))

    def decode(self, latent, return_dict=False):
        return (latent,)

    def clear_cache(self):
        self.cache_cleared = True


class _FakeProcessor:
    def postprocess_video(self, decoded, output_type):
        assert output_type == "np"
        # The runner only needs the standard [B,T,H,W,C] result.
        return np.array(
            [[[[[0, 1, 2], [3, 4, 5]]], [[[6, 7, 8], [9, 10, 11]]]]],
            dtype=np.uint8,
        )


class _FakePipe:
    def __init__(self) -> None:
        self.vae = _FakeVAE()
        self.video_processor = _FakeProcessor()


def test_wan_provisional_decoder_writes_exact_requested_frames(tmp_path: Path) -> None:
    pipe = _FakePipe()
    decoder = make_wan_provisional_keyframe_decoder(pipe, expected_frames=2)
    destination = tmp_path / "frames"
    paths = decoder(torch.zeros((1, 1, 1, 1, 1)), [0, 1], destination)

    assert sorted(paths) == [0, 1]
    assert all(path.is_file() for path in paths.values())
    assert pipe.vae.cache_cleared
    with pytest.raises(FileExistsError, match="not empty"):
        decoder(torch.zeros((1, 1, 1, 1, 1)), [0], destination)
