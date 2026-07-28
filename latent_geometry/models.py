"""Small latent geometry probe models.

All models consume a latent pair shaped ``[B, C, T_pair, H, W]`` and a scalar
normalized timestep per example.  They deliberately do not load a VAE, VDM, or
geometry teacher; this keeps the probe cheap enough to diagnose latent signal.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F


def _expand_timestep(timestep: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    timestep = timestep.to(device=device, dtype=torch.float32)
    if timestep.ndim == 0:
        return timestep.expand(batch_size)
    if timestep.ndim == 1 and timestep.shape[0] == batch_size:
        return timestep
    if timestep.ndim == 2 and timestep.shape == (batch_size, 1):
        return timestep[:, 0]
    raise ValueError(f"timestep must be scalar or [B], got {tuple(timestep.shape)} for batch {batch_size}")


def _pose_prediction(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    if raw.shape[-1] != 9:
        raise ValueError(f"Expected pose head output [..., 9], got {tuple(raw.shape)}")
    return {
        "rotation_6d": raw[..., :6],
        "translation_direction": F.normalize(raw[..., 6:], dim=-1, eps=1e-6),
    }


class ConstantPoseBaseline(nn.Module):
    """Learned constant pose baseline that ignores latent and timestep."""

    def __init__(self) -> None:
        super().__init__()
        initial = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        self.pose = nn.Parameter(initial)

    def forward(self, latent: torch.Tensor, timestep: torch.Tensor) -> dict[str, torch.Tensor]:
        if latent.ndim != 5:
            raise ValueError(f"latent must be [B, C, T, H, W], got {tuple(latent.shape)}")
        del timestep
        return _pose_prediction(self.pose.unsqueeze(0).expand(latent.shape[0], -1))


class LinearLatentProbe(nn.Module):
    """Ordered linear readout from source, target, and target-source features."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.head = nn.Linear(3 * self.in_channels + 1, 9)
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[:6].copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, latent: torch.Tensor, timestep: torch.Tensor) -> dict[str, torch.Tensor]:
        if latent.ndim != 5:
            raise ValueError(f"latent must be [B, C, T, H, W], got {tuple(latent.shape)}")
        if latent.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} latent channels, got {latent.shape[1]}")
        time = _expand_timestep(timestep, latent.shape[0], latent.device).unsqueeze(-1)
        if latent.shape[2] != 2:
            raise ValueError(f"LinearLatentProbe expects T=2, got T={latent.shape[2]}")
        source = latent[:, :, 0].mean(dim=(2, 3))
        target = latent[:, :, 1].mean(dim=(2, 3))
        features = torch.cat((source, target, target - source, time), dim=-1)
        return _pose_prediction(self.head(features))


class SinusoidalTimestepEmbedding(nn.Module):
    """Fixed sinusoidal embedding for normalized timesteps in [0, 1]."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 4 or embedding_dim % 2:
            raise ValueError("embedding_dim must be even and at least 4")
        self.embedding_dim = embedding_dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.embedding_dim // 2
        exponent = torch.arange(half, device=timestep.device, dtype=timestep.dtype)
        frequencies = torch.exp(-math.log(10000.0) * exponent / max(half - 1, 1))
        angles = timestep.unsqueeze(-1) * frequencies.unsqueeze(0)
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _Residual3DBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class Small3DConvCritic(nn.Module):
    """A deliberately small nonlinear 3D-convolutional latent geometry critic.

    It sees the ordered latent pair jointly along its temporal axis.  The model
    is a candidate critic, not an encoder replacement or a video-model adapter.
    """

    def __init__(
        self,
        in_channels: int,
        width: int = 32,
        timestep_dim: int = 16,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if width < 4:
            raise ValueError("width must be at least 4")
        self.in_channels = int(in_channels)
        self.timestep_embedding = SinusoidalTimestepEmbedding(timestep_dim)
        self.stem = nn.Conv3d(self.in_channels, width, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(_Residual3DBlock(width), _Residual3DBlock(width))
        self.time_mlp = nn.Sequential(nn.Linear(timestep_dim, timestep_dim), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(width + timestep_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 9),
        )
        with torch.no_grad():
            self.head[-1].bias.zero_()
            self.head[-1].bias[:6].copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, latent: torch.Tensor, timestep: torch.Tensor) -> dict[str, torch.Tensor]:
        if latent.ndim != 5:
            raise ValueError(f"latent must be [B, C, T, H, W], got {tuple(latent.shape)}")
        if latent.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} latent channels, got {latent.shape[1]}")
        time = _expand_timestep(timestep, latent.shape[0], latent.device)
        features = self.blocks(self.stem(latent))
        pooled = features.mean(dim=(2, 3, 4))
        conditioned = torch.cat((pooled, self.time_mlp(self.timestep_embedding(time))), dim=-1)
        return _pose_prediction(self.head(conditioned))
