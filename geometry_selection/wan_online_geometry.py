"""Predicted-clean Wan video decoding for online geometry evidence."""

from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class WanPredictedCleanGeometry:
    """Convert a scheduler x0 estimate into one VGGT-Omega geometry bundle.

    The callback is deliberately outside the pipeline's attention logic. It
    receives standardized Wan latents, uses the pipeline VAE's exact
    normalization, and returns NumPy geometry arrays without gradients.
    """

    def __init__(
        self,
        *,
        vae: torch.nn.Module,
        geometry_adapter: Any,
        frame_indices: Sequence[int],
        confidence_percentile: float = 20.0,
        temporary_root: Path | None = None,
    ) -> None:
        indices = tuple(int(index) for index in frame_indices)
        if not indices or indices != tuple(sorted(set(indices))) or indices[0] < 0:
            raise ValueError("frame_indices must be non-empty, sorted, unique, and non-negative")
        if not 0.0 <= confidence_percentile <= 100.0:
            raise ValueError("confidence_percentile must lie in [0, 100]")
        self.vae = vae
        self.geometry_adapter = geometry_adapter
        self.frame_indices = indices
        self.confidence_percentile = float(confidence_percentile)
        self.temporary_root = Path(temporary_root).resolve() if temporary_root is not None else None
        self.records: list[dict[str, Any]] = []

    def _vae_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        parameter = next(self.vae.parameters())
        return parameter.device, parameter.dtype

    def _decode_selected_frames(self, x0_latents: torch.Tensor) -> list[np.ndarray]:
        if x0_latents.ndim != 5 or x0_latents.shape[0] != 1:
            raise ValueError("x0_latents must have shape (1,C,T,H,W)")
        vae_device, vae_dtype = self._vae_device_dtype()
        latent_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(vae_device, vae_dtype)
        )
        latent_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(vae_device, vae_dtype)
        normalized = x0_latents.detach().to(vae_device, vae_dtype)
        normalized = normalized / latent_std + latent_mean
        with torch.inference_mode():
            decoded = self.vae.decode(normalized, return_dict=False)[0]
        if self.frame_indices[-1] >= decoded.shape[2]:
            raise ValueError(
                f"requested RGB frame {self.frame_indices[-1]} but VAE decoded {decoded.shape[2]} frames"
            )
        selected = []
        for frame_index in self.frame_indices:
            frame = ((decoded[0, :, frame_index].float().permute(1, 2, 0) + 1.0) / 2.0).clamp(0, 1)
            selected.append((frame.mul(255).round().to(torch.uint8).cpu().numpy()))
        del normalized, decoded, latent_mean, latent_std
        if hasattr(self.vae, "clear_cache"):
            self.vae.clear_cache()
        if vae_device.type == "cuda":
            torch.cuda.empty_cache()
        return selected

    def __call__(self, step_index: int, x0_latents: torch.Tensor) -> dict[str, np.ndarray]:
        started = time.perf_counter()
        frames = self._decode_selected_frames(x0_latents)
        decode_seconds = time.perf_counter() - started
        temporary_parent = str(self.temporary_root) if self.temporary_root is not None else None
        with tempfile.TemporaryDirectory(prefix=f"wan_x0_step_{int(step_index):03d}_", dir=temporary_parent) as tmp:
            paths = []
            for frame_index, frame in zip(self.frame_indices, frames):
                path = Path(tmp) / f"frame_{frame_index:04d}.png"
                Image.fromarray(frame).save(path)
                paths.append(path)
            frame_hashes = [_sha256_file(path) for path in paths]
            geometry_started = time.perf_counter()
            prediction = self.geometry_adapter.predict_image_paths(
                paths,
                keyframe_indices=self.frame_indices,
                hash_checkpoint=False,
            )
            geometry_seconds = time.perf_counter() - geometry_started
        thresholds = np.percentile(
            prediction.confidence,
            self.confidence_percentile,
            axis=(1, 2),
        ).astype(np.float32)
        bundle = {
            "world_to_camera": prediction.world_to_camera.astype(np.float32),
            "intrinsics": prediction.intrinsics.astype(np.float32),
            "depth": prediction.depth.astype(np.float32),
            "confidence": prediction.confidence.astype(np.float32),
            "confidence_thresholds": thresholds,
            "keyframe_indices": prediction.keyframe_indices.astype(np.int64),
        }
        self.records.append(
            {
                "step": int(step_index),
                "frame_indices": list(self.frame_indices),
                "frame_sha256": frame_hashes,
                "decode_seconds": decode_seconds,
                "geometry_seconds": geometry_seconds,
                "geometry_image_size_hw": list(prediction.depth.shape[1:]),
            }
        )
        return bundle
