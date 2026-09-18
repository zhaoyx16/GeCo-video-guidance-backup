#!/usr/bin/env python3
"""Augment a frozen Wan draft-delta cache with multi-anchor agreement statistics."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta_cache", type=Path, required=True)
    parser.add_argument("--kv_cache", type=Path, required=True)
    parser.add_argument("--geometry_map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(name: str, left, right) -> None:
    if left != right:
        raise ValueError(f"{name} differs: delta={left!r} kv={right!r}")


def main() -> None:
    args = parse_args()
    delta_payload = torch.load(
        args.delta_cache,
        map_location="cpu",
        weights_only=False,
    )
    kv_payload = torch.load(
        args.kv_cache,
        map_location="cpu",
        weights_only=False,
    )
    geometry = torch.load(
        args.geometry_map,
        map_location="cpu",
        weights_only=False,
    )

    delta_meta = dict(delta_payload["metadata"])
    kv_meta = kv_payload["metadata"]
    if delta_meta.get("cache_kind") != "attention_value_delta":
        raise ValueError("delta_cache is not an attention_value_delta cache")
    if kv_meta.get("cache_kind") != "attention_kv":
        raise ValueError("kv_cache is not an attention_kv cache")
    for name in (
        "token_grid",
        "anchor_token_times",
        "layers",
        "num_inference_steps",
        "run_fingerprint",
        "geometry_transport_interval",
    ):
        require_equal(name, delta_meta.get(name), kv_meta.get(name))

    geometry_sha256 = sha256_file(args.geometry_map)
    if delta_meta.get("geometry_map_sha256") != geometry_sha256:
        raise ValueError(
            "geometry_map does not match the delta cache: "
            f"cache={delta_meta.get('geometry_map_sha256')} file={geometry_sha256}"
        )

    token_grid = tuple(delta_meta["token_grid"])
    num_tokens_t, height_tokens, width_tokens = token_grid
    spatial_tokens = height_tokens * width_tokens
    source_time = geometry["source_time"]
    source_index = geometry["source_index"]
    confidence = geometry["confidence"]
    if source_time.shape != source_index.shape:
        raise ValueError("geometry source_time/source_index shapes differ")
    if confidence.shape[:3] != (
        num_tokens_t - 1,
        height_tokens,
        width_tokens,
    ):
        raise ValueError("geometry confidence shape is incompatible with token_grid")
    memory_slots = confidence.shape[-1]
    source_time = source_time.reshape(
        num_tokens_t - 1,
        spatial_tokens,
        memory_slots,
    )
    source_index = source_index.reshape_as(source_time)
    confidence = confidence.reshape_as(source_time).float()

    features = {}
    max_delta_error = 0.0
    agreement_values = []
    effective_anchor_values = []
    unique_anchor_values = []
    for cache_key, delta_entry in delta_payload["features"].items():
        if cache_key not in kv_payload["features"]:
            raise KeyError(f"kv_cache has no matching entry for {cache_key}")
        kv_entry = kv_payload["features"][cache_key]
        anchor_values = kv_entry["value"].float()
        if anchor_values.ndim != 4:
            raise ValueError(
                f"{cache_key}: expected cached anchor V as [B,A,S,C], "
                f"got {tuple(anchor_values.shape)}"
            )
        target_flat = delta_entry["target_flat"].long()
        delta = delta_entry["delta"].float()
        target_flat_parts = []
        candidate_value_parts = []
        candidate_confidence_parts = []

        for target_time in range(1, num_tokens_t):
            confidence_t = confidence[target_time - 1]
            support = confidence_t.sum(dim=-1) > 0
            if not support.any():
                continue
            target_spatial = support.nonzero(as_tuple=False).squeeze(-1)
            expected_target_flat = (
                target_time * spatial_tokens + target_spatial
            )
            target_flat_parts.append(expected_target_flat)
            source_time_t = source_time[target_time - 1, target_spatial]
            source_index_t = source_index[target_time - 1, target_spatial]
            if (
                source_time_t.min().item() < 0
                or source_time_t.max().item() >= anchor_values.shape[1]
            ):
                raise ValueError(
                    f"{cache_key}: geometry source_time is outside cached anchors"
                )
            candidate_value_parts.append(
                anchor_values[
                    :,
                    source_time_t.reshape(-1),
                    source_index_t.reshape(-1),
                ].reshape(
                    anchor_values.shape[0],
                    target_spatial.numel(),
                    memory_slots,
                    anchor_values.shape[-1],
                )
            )
            candidate_confidence_parts.append(confidence_t[target_spatial])

        rebuilt_target_flat = torch.cat(target_flat_parts)
        if not torch.equal(target_flat, rebuilt_target_flat):
            raise ValueError(f"{cache_key}: target order does not match geometry map")
        candidate_values = torch.cat(candidate_value_parts, dim=1)
        candidate_confidence = torch.cat(
            candidate_confidence_parts,
            dim=0,
        )
        candidate_weights = (
            candidate_confidence
            / candidate_confidence.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        )
        matched_source = (
            candidate_weights[None, :, :, None] * candidate_values
        ).sum(dim=2)

        # The aggregate cache stores mu = matched_source - target. Recovering
        # target this way is exact up to the original BF16 cache quantization.
        target_value = matched_source - delta
        candidate_delta = candidate_values - target_value[:, :, None, :]
        rebuilt_delta = (
            candidate_weights[None, :, :, None] * candidate_delta
        ).sum(dim=2)
        delta_error = (rebuilt_delta - delta).abs().max().item()
        max_delta_error = max(max_delta_error, delta_error)

        mean_energy = delta.float().square().mean(dim=-1)
        candidate_energy = (
            candidate_weights[None]
            * candidate_delta.float().square().mean(dim=-1)
        ).sum(dim=-1)
        agreement = (
            mean_energy / candidate_energy.clamp_min(1e-12)
        ).clamp(0, 1)
        if agreement.shape[0] != 1:
            raise ValueError(
                "Consensus cache currently expects one video per pipeline call"
            )
        agreement = agreement[0]
        unique_anchor_count = (candidate_confidence > 0).sum(dim=-1)
        effective_anchor_count = (
            1.0
            / candidate_weights.square().sum(dim=-1).clamp_min(1e-8)
        )

        features[cache_key] = {
            "target_flat": delta_entry["target_flat"],
            "delta": delta_entry["delta"],
            "confidence": delta_entry["confidence"],
            "agreement": agreement.to(torch.float32).contiguous(),
            "unique_anchor_count": unique_anchor_count.to(
                torch.uint8
            ).contiguous(),
            "effective_anchor_count": effective_anchor_count.to(
                torch.float32
            ).contiguous(),
        }
        agreement_values.append(agreement)
        effective_anchor_values.append(effective_anchor_count)
        unique_anchor_values.append(unique_anchor_count)

    delta_meta["format_version"] = 4
    delta_meta["cache_kind"] = "attention_value_delta_consensus"
    delta_meta["consensus_source_delta_cache"] = str(args.delta_cache)
    delta_meta["consensus_source_kv_cache"] = str(args.kv_cache)
    delta_meta["consensus_geometry_map"] = str(args.geometry_map)
    payload = {
        "metadata": delta_meta,
        "features": features,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    agreement_all = torch.cat(agreement_values)
    effective_all = torch.cat(effective_anchor_values)
    unique_all = torch.cat(unique_anchor_values)
    print(f"saved: {args.output}")
    print(f"entries: {len(features)}")
    print(f"max_rebuilt_delta_error: {max_delta_error:.8g}")
    print(
        "agreement quantiles:",
        torch.quantile(
            agreement_all,
            torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]),
        ).tolist(),
    )
    print(
        "effective anchors quantiles:",
        torch.quantile(
            effective_all,
            torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]),
        ).tolist(),
    )
    print(
        "tokens with >=2 anchors:",
        f"{(unique_all >= 2).float().mean().item():.4f}",
    )


if __name__ == "__main__":
    main()
