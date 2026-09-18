"""Reproducible generation contracts for offline Wan geometry maps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


GENERATION_FIELDS = (
    "prompt",
    "negative_prompt",
    "seed",
    "steps",
    "frames",
    "height",
    "width",
    "fps",
    "guidance_scale",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_config_fingerprints(model_path: str | Path) -> dict[str, str]:
    root = Path(model_path).resolve()
    relative_paths = (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "transformer/config.json",
        "transformer_2/config.json",
        "vae/config.json",
        "image_encoder/config.json",
        "text_encoder/config.json",
    )
    fingerprints = {}
    for relative_path in relative_paths:
        candidate = root / relative_path
        if candidate.is_file():
            fingerprints[relative_path] = sha256_file(candidate)
    if not fingerprints:
        raise ValueError(f"No model configuration files found under {root}")
    return fingerprints


def build_generation_contract(
    *,
    prompt: str,
    negative_prompt: str | None,
    seed: int,
    steps: int,
    frames: int,
    height: int,
    width: int,
    fps: int,
    guidance_scale: float,
    image_path: str | Path,
    model_path: str | Path,
) -> dict[str, Any]:
    image_path = Path(image_path).resolve()
    model_path = Path(model_path).resolve()
    return {
        "contract_version": 1,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": int(seed),
        "steps": int(steps),
        "frames": int(frames),
        "height": int(height),
        "width": int(width),
        "fps": int(fps),
        "guidance_scale": float(guidance_scale),
        "conditioning_image": str(image_path),
        "conditioning_image_sha256": sha256_file(image_path),
        "model_path": str(model_path),
        "model_config_sha256": model_config_fingerprints(model_path),
    }


def build_source_artifact(video_path: str | Path) -> dict[str, str]:
    video_path = Path(video_path).resolve()
    return {
        "draft_video": str(video_path),
        "draft_video_sha256": sha256_file(video_path),
    }


def validate_generation_contract(
    metadata: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    actual = metadata.get("generation_contract")
    if not isinstance(actual, dict):
        raise ValueError(
            "Geometry map has no generation_contract; rebuild it with strict provenance."
        )
    actual_json = json.dumps(actual, sort_keys=True, separators=(",", ":"))
    expected_json = json.dumps(expected, sort_keys=True, separators=(",", ":"))
    if actual_json != expected_json:
        mismatches = []
        for key in sorted(set(actual) | set(expected)):
            if actual.get(key) != expected.get(key):
                mismatches.append(
                    f"{key}: map={actual.get(key)!r}, run={expected.get(key)!r}"
                )
        raise ValueError(
            "Geometry map provenance does not match the guided run: "
            + "; ".join(mismatches)
        )

    source = metadata.get("source_artifact")
    if not isinstance(source, dict):
        raise ValueError("Geometry map has no source_artifact provenance.")
    draft_path = Path(source.get("draft_video", ""))
    expected_digest = source.get("draft_video_sha256")
    if not draft_path.is_file() or not expected_digest:
        raise ValueError("Geometry map draft-video provenance is incomplete.")
    actual_digest = sha256_file(draft_path)
    if actual_digest != expected_digest:
        raise ValueError(
            "Geometry map draft video changed after map construction: "
            f"{draft_path}"
        )
