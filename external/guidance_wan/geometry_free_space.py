"""Reviewed intervention helpers for observed free-space violations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from geometry_provenance import validate_generation_contract
from geometry_suppression import validate_confidence


FREE_SPACE_VALUE_MODE = "free_space_value_residual"
INTEGER_INDEX_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _require_integer_indices(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.dtype not in INTEGER_INDEX_DTYPES:
        raise ValueError(f"{name} must use an integer tensor dtype")


def validate_frozen_free_space_transport_settings(
    *,
    enabled: bool,
    alpha: float,
    layers: Sequence[int] | None,
    mode: str,
    map_path: str | None,
    frame_indices: Sequence[int] | None,
    cond_only: bool,
    attn_avg_alpha: float,
    attn_avg_layers: Sequence[int] | None,
    expected_provenance: dict[str, Any] | None,
) -> None:
    """Reject settings outside the reviewed free-space transport protocol."""
    if not enabled or alpha == 0.0:
        return
    if list(layers or []) != [0]:
        raise ValueError("Frozen free-space transport requires transformer block 0")
    if mode != FREE_SPACE_VALUE_MODE:
        raise ValueError(
            f"Frozen free-space transport requires mode={FREE_SPACE_VALUE_MODE}"
        )
    if not map_path or not Path(map_path).is_file():
        raise ValueError("Frozen free-space transport requires a precomputed map")
    if frame_indices is not None:
        raise ValueError("Frozen free-space transport forbids online geometry frames")
    if not cond_only:
        raise ValueError("Frozen free-space transport must modify only the CFG conditional branch")
    if attn_avg_alpha != 0.0 or attn_avg_layers:
        raise ValueError("Frozen free-space transport cannot be combined with feature averaging")
    if expected_provenance is None:
        raise ValueError("Frozen free-space transport requires generation provenance")


def validate_frozen_free_space_map(
    payload: dict[str, Any],
    expected_provenance: dict[str, Any] | None,
) -> None:
    """Validate semantics, tensor ranges, and provenance of a free-space map."""
    metadata = payload.get("metadata", {})
    if int(metadata.get("format_version", 0)) < 3:
        raise ValueError("Frozen free-space transport requires a format-v3 map")
    if metadata.get("visibility_mode") != "source_free_space_violation":
        raise ValueError(
            "Frozen free-space transport requires source_free_space_violation evidence"
        )
    required = {
        "source_time",
        "source_index",
        "confidence",
        "observed_background_time",
        "observed_background_index",
        "observed_background_confidence",
        "conflict_confidence",
    }
    expected_semantics = {
        "confidence": "same_surface_transport",
        "conflict_confidence": "source_observed_free_space_conflict",
        "observed_background_confidence": (
            "source_ray_background_behind_free_space_conflict"
        ),
    }
    if metadata.get("evidence_semantics") != expected_semantics:
        raise ValueError(
            "Frozen free-space map must explicitly separate same-surface, "
            "free-space-conflict, and observed-background evidence"
        )
    missing = required.difference(payload)
    if missing:
        raise ValueError(
            f"Frozen free-space map is missing fields: {sorted(missing)}"
        )
    validate_confidence(
        payload["confidence"],
        name="same-surface transport confidence",
    )
    validate_confidence(
        payload["observed_background_confidence"],
        name="observed-background confidence",
    )
    validate_confidence(
        payload["conflict_confidence"],
        name="free-space conflict confidence",
    )
    source_time = payload["observed_background_time"]
    source_index = payload["observed_background_index"]
    source_confidence = payload["observed_background_confidence"]
    conflict_confidence = payload["conflict_confidence"]
    if source_time.ndim != 3 or source_index.shape != source_time.shape:
        raise ValueError(
            "Observed-background time/index must have shape [T,P,S]"
        )
    if (
        source_confidence.ndim != 4
        or source_confidence.shape[0] != source_time.shape[0]
        or source_confidence.shape[1] * source_confidence.shape[2]
        != source_time.shape[1]
        or source_confidence.shape[3] != source_time.shape[2]
    ):
        raise ValueError(
            "Observed-background confidence must match time/index [T,H,W,S]"
        )
    if conflict_confidence.shape != source_confidence.shape[:3]:
        raise ValueError(
            "Free-space conflict confidence must have shape [T,H,W]"
        )
    if not torch.equal(
        payload["observed_background_confidence"].amax(dim=-1) > 0,
        payload["conflict_confidence"] > 0,
    ):
        raise ValueError(
            "Observed-background evidence and free-space conflict masks must agree"
        )
    validate_generation_contract(metadata, expected_provenance)


def prepare_observed_background_transport(
    source_time: torch.Tensor,
    source_index: torch.Tensor,
    source_confidence: torch.Tensor,
    conflict_confidence: torch.Tensor,
    *,
    spatial_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate source evidence and return safe indices plus target confidence."""
    if source_time.ndim != 3 or source_index.shape != source_time.shape:
        raise ValueError("source_time/source_index must have shape [T,P,S]")
    _require_integer_indices(source_time, name="source_time")
    _require_integer_indices(source_index, name="source_index")
    source_time = source_time.long()
    source_index = source_index.long()
    if source_time.shape[1] != spatial_tokens:
        raise ValueError(
            "Observed-background source indices do not match the token grid"
        )
    if source_confidence.numel() != source_time.numel():
        raise ValueError(
            "Observed-background confidence does not match source slots"
        )
    source_confidence_flat = source_confidence.reshape_as(source_time)
    if conflict_confidence.shape[0] != source_time.shape[0] or (
        conflict_confidence[0].numel()
        if conflict_confidence.shape[0]
        else 0
    ) != spatial_tokens:
        raise ValueError(
            "Free-space conflict confidence does not match the token grid"
        )

    validate_confidence(
        source_confidence,
        name="observed-background confidence",
    )
    validate_confidence(
        conflict_confidence,
        name="free-space conflict confidence",
    )
    source_active = source_confidence_flat > 0
    target_active = conflict_confidence.reshape(
        source_time.shape[0],
        spatial_tokens,
    ) > 0
    if not torch.equal(source_active[..., 0], target_active):
        raise ValueError(
            "Frozen free-space transport requires one active source in slot 0 "
            "for every active conflict token"
        )

    target_time = torch.arange(
        1,
        source_time.shape[0] + 1,
        device=source_time.device,
        dtype=source_time.dtype,
    ).reshape(-1, 1, 1)
    invalid_active = source_active & (
        (source_time < 0)
        | (source_time >= target_time)
        | (source_index < 0)
        | (source_index >= spatial_tokens)
    )
    if torch.any(invalid_active):
        raise ValueError(
            "Active observed-background correspondence must be causal and in range"
        )

    safe_time = torch.where(
        source_active,
        source_time,
        torch.zeros_like(source_time),
    )
    safe_index = torch.where(
        source_active,
        source_index,
        torch.zeros_like(source_index),
    )
    return safe_time, safe_index, conflict_confidence.unsqueeze(-1)


def apply_observed_background_value_residual(
    attended: torch.Tensor,
    source_value: torch.Tensor,
    confidence: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    """Blend ray-aligned observed background values into conflict tokens."""
    if attended.shape != source_value.shape:
        raise ValueError(
            "attended and source_value must have identical [B,T,P,H,D] shapes"
        )
    if confidence.shape != attended.shape[:3]:
        raise ValueError("confidence must have shape [B,T,P]")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    validate_confidence(confidence, name="observed-background confidence")
    if alpha == 0.0 or not torch.count_nonzero(confidence):
        return attended
    blend = (alpha * confidence).unsqueeze(-1).unsqueeze(-1)
    return (
        attended.float() * (1.0 - blend)
        + source_value.float() * blend
    ).to(attended.dtype)


def apply_observed_background_value_residual_to_attention(
    attended: torch.Tensor,
    source_value: torch.Tensor,
    confidence: torch.Tensor,
    *,
    token_grid: tuple[int, int, int],
    alpha: float,
) -> torch.Tensor:
    """Apply the free-space blend to flat Wan attention using its runtime batch."""
    if attended.ndim != 4:
        raise ValueError("attended must have shape [B,N,H,D]")
    batch_size, sequence_length, heads, head_dim = attended.shape
    num_tokens_t, height_tokens, width_tokens = token_grid
    spatial_tokens = height_tokens * width_tokens
    if sequence_length != num_tokens_t * spatial_tokens:
        raise ValueError("attended sequence length does not match token_grid")
    expected_source_shape = (
        batch_size,
        num_tokens_t - 1,
        spatial_tokens,
        heads,
        head_dim,
    )
    if source_value.shape != expected_source_shape:
        raise ValueError(
            "source_value must have shape "
            f"{expected_source_shape}, got {tuple(source_value.shape)}"
        )
    expected_confidence_shape = (
        batch_size,
        num_tokens_t - 1,
        spatial_tokens,
    )
    if confidence.shape != expected_confidence_shape:
        raise ValueError(
            "confidence must have shape "
            f"{expected_confidence_shape}, got {tuple(confidence.shape)}"
        )

    attended_grid = attended.reshape(
        batch_size,
        num_tokens_t,
        spatial_tokens,
        heads,
        head_dim,
    )
    mixed_grid = attended_grid.clone()
    mixed_grid[:, 1:] = apply_observed_background_value_residual(
        attended_grid[:, 1:],
        source_value,
        confidence,
        alpha=alpha,
    )
    return mixed_grid.reshape_as(attended)
