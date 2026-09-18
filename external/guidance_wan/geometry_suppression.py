"""Small, testable primitives for Wan hidden-input suppression."""

from __future__ import annotations

import torch


def validate_frozen_offline_suppression_contract(
    *,
    enabled: bool,
    alpha: float,
    layers: list[int] | None,
    mode: str,
    map_path: str | None,
    frame_indices: list[int] | None,
    cond_only: bool,
    attn_avg_alpha: float,
    attn_avg_layers: list[int] | None,
    expected_provenance: dict | None,
) -> None:
    if not enabled or alpha <= 0.0:
        return
    if layers != [0]:
        raise ValueError("Frozen offline suppression requires geometry layer [0]")
    if mode != "hidden_input_suppress":
        raise ValueError(
            "Frozen offline suppression requires hidden_input_suppress"
        )
    if map_path is None:
        raise ValueError(
            "Frozen offline suppression requires a precomputed geometry map"
        )
    if frame_indices is not None:
        raise ValueError(
            "Frozen offline suppression forbids online geometry_frame_indices"
        )
    if not cond_only:
        raise ValueError(
            "Frozen offline suppression must modify only the conditional CFG branch"
        )
    if attn_avg_alpha != 0.0 or attn_avg_layers:
        raise ValueError(
            "Frozen offline suppression cannot be combined with attention averaging"
        )
    if expected_provenance is None:
        raise ValueError(
            "Frozen offline suppression requires expected generation provenance"
        )


def validate_confidence(
    confidence: torch.Tensor,
    *,
    name: str,
) -> None:
    if not torch.isfinite(confidence).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if (confidence < 0).any() or (confidence > 1).any():
        raise ValueError(f"{name} must lie in [0, 1]")


def apply_hidden_input_suppression(
    hidden_states: torch.Tensor,
    confidence: torch.Tensor,
    *,
    token_grid: tuple[int, int, int],
    alpha: float,
) -> torch.Tensor:
    """Suppress target tokens while preserving conditioning token t0 exactly."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    validate_confidence(confidence, name="suppression confidence")

    batch, sequence_length, channels = hidden_states.shape
    temporal, height, width = token_grid
    spatial = height * width
    expected_shape = (temporal - 1, height, width)
    if confidence.shape != expected_shape:
        raise ValueError(
            f"Expected confidence {expected_shape}, got {tuple(confidence.shape)}"
        )
    if sequence_length != temporal * spatial:
        raise ValueError(
            "Hidden sequence length does not match the declared Wan token grid"
        )

    hidden_grid = hidden_states.reshape(
        batch,
        temporal,
        spatial,
        channels,
    )
    blend = alpha * confidence.reshape(
        1,
        temporal - 1,
        spatial,
        1,
    ).to(device=hidden_states.device, dtype=torch.float32)
    result = hidden_grid.clone()
    result[:, 1:] = (
        hidden_grid[:, 1:].float() * (1.0 - blend)
    ).to(hidden_states.dtype)
    return result.reshape_as(hidden_states)
