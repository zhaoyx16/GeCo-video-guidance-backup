"""Validation helpers binding generated artifacts to the frozen DL3DV protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DL3DV_PROTOCOL_SCHEMA = "dl3dv-geometry-selection-v1"


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("_meta", {}).get("schema") != DL3DV_PROTOCOL_SCHEMA:
        raise ValueError(f"protocol schema must be {DL3DV_PROTOCOL_SCHEMA}")
    return payload


def resolve_protocol_image(case: dict[str, Any], dataset_root: Path) -> Path:
    root = dataset_root.resolve()
    relative = Path(case["dataset_relative_image"])
    if relative.is_absolute():
        raise ValueError("dataset_relative_image must be relative")
    image = (root / relative).resolve()
    try:
        image.relative_to(root)
    except ValueError as error:
        raise ValueError("dataset_relative_image escapes dataset_root") from error
    if not image.is_file():
        raise FileNotFoundError(image)
    if file_sha256(image) != case["image_sha256"]:
        raise ValueError(f"frozen conditioning image hash mismatch: {image}")
    return image


def validate_candidate_spec_against_protocol(
    spec: dict[str, Any],
    protocol_path: Path,
    dataset_root: Path,
) -> None:
    protocol_path = protocol_path.resolve()
    if file_sha256(protocol_path) != spec["protocol_manifest_sha256"]:
        raise ValueError("candidate spec does not match the frozen protocol digest")
    protocol = load_protocol(protocol_path)
    for source_case in spec["cases"]:
        case_id = source_case["case_id"]
        if case_id not in protocol or case_id.startswith("_"):
            raise ValueError(f"candidate case is absent from frozen protocol: {case_id}")
        frozen = protocol[case_id]
        expected = {
            "scene_uid": frozen["scene_uid"],
            "split": frozen["split"],
            "prompt": frozen["text_prompt"],
        }
        actual = {
            "scene_uid": source_case["scene_uid"],
            "split": spec["split"],
            "prompt": source_case["prompt"],
        }
        if actual != expected:
            raise ValueError(f"candidate case differs from frozen protocol: {case_id}")
        frozen_image = resolve_protocol_image(frozen, dataset_root)
        candidate_image = Path(source_case["conditioning_image"]).resolve()
        if file_sha256(candidate_image) != file_sha256(frozen_image):
            raise ValueError(f"candidate conditioning image differs from protocol: {case_id}")
