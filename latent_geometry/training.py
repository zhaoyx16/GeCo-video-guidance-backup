"""Minimal CPU/GPU-agnostic training utilities for latent geometry probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from .geometry import pose_losses, pose_metrics


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _target_from_batch(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "rotation_6d": batch["rotation_6d"],
        "translation_direction": batch["translation_direction"],
        "translation_valid": batch["translation_valid"],
    }


def _select_model_input(batch: Mapping[str, Any], input_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an input and the only valid conditioning timestep for its domain."""
    if input_key == "z0":
        if "z0_timestep" not in batch:
            raise KeyError("z0 control batches must provide z0_timestep=0")
        return batch["z0"], batch["z0_timestep"]
    if input_key == "diffusion_z0":
        if "diffusion_z0_timestep" not in batch:
            raise KeyError("diffusion-z0 batches must provide diffusion_z0_timestep=0")
        return batch["diffusion_z0"], batch["diffusion_z0_timestep"]
    if input_key == "zt":
        return batch["zt"], batch["timestep"]
    raise ValueError("input_key must be 'z0', 'diffusion_z0', or 'zt'")


def probe_loss_for_batch(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    input_key: str = "zt",
    rotation_weight: float = 1.0,
    translation_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    model_input, model_timestep = _select_model_input(batch, input_key)
    prediction = model(model_input, model_timestep)
    return pose_losses(
        prediction,
        _target_from_batch(batch),
        rotation_weight=rotation_weight,
        translation_weight=translation_weight,
    )


@dataclass(frozen=True)
class ProbeEvaluation:
    loss: float
    rotation_deg: float
    translation_direction_deg: float | None
    examples: int
    translation_examples: int


@torch.no_grad()
def evaluate_probe(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    *,
    device: str | torch.device = "cpu",
    input_key: str = "zt",
) -> ProbeEvaluation:
    device = torch.device(device)
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_rotation = 0.0
    total_examples = 0
    total_direction = 0.0
    total_direction_examples = 0
    for raw_batch in dataloader:
        batch = _move_batch(raw_batch, device)
        losses = probe_loss_for_batch(model, batch, input_key=input_key)
        model_input, model_timestep = _select_model_input(batch, input_key)
        prediction = model(model_input, model_timestep)
        metrics = pose_metrics(prediction, _target_from_batch(batch))
        batch_size = int(model_input.shape[0])
        total_examples += batch_size
        total_loss += float(losses["loss"].item()) * batch_size
        total_rotation += float(metrics["rotation_deg"].sum().item())
        valid = metrics["translation_valid"]
        if bool(valid.any()):
            total_direction += float(metrics["translation_direction_deg"][valid].sum().item())
            total_direction_examples += int(valid.sum().item())
    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty dataloader")
    return ProbeEvaluation(
        loss=total_loss / total_examples,
        rotation_deg=total_rotation / total_examples,
        translation_direction_deg=(
            total_direction / total_direction_examples if total_direction_examples else None
        ),
        examples=total_examples,
        translation_examples=total_direction_examples,
    )


def train_probe_steps(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
    device: str | torch.device = "cpu",
    input_key: str = "zt",
    rotation_weight: float = 1.0,
    translation_weight: float = 1.0,
) -> list[float]:
    """Run a small deterministic training loop and return per-step total losses."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    device = torch.device(device)
    model.to(device)
    model.train()
    losses: list[float] = []
    epoch = 0
    iterator = iter(dataloader)
    for _ in range(steps):
        try:
            raw_batch = next(iterator)
        except StopIteration:
            epoch += 1
            dataset = getattr(dataloader, "dataset", None)
            if dataset is not None and hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
            iterator = iter(dataloader)
            raw_batch = next(iterator)
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss_terms = probe_loss_for_batch(
            model,
            batch,
            input_key=input_key,
            rotation_weight=rotation_weight,
            translation_weight=translation_weight,
        )
        loss_terms["loss"].backward()
        optimizer.step()
        losses.append(float(loss_terms["loss"].detach().cpu().item()))
    return losses
