"""Manifest validation and provenance helpers for controlled Frame Guidance runs.

The manifest is intentionally small and JSON-only so it can be frozen alongside a
paired experiment.  A case must name the conditioning frame and the sparse target
anchors used by Frame Guidance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable


class FrameGuidanceManifestError(ValueError):
    """Raised when a Frame Guidance manifest is incomplete or ambiguous."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _case_table(document: dict[str, Any]) -> dict[str, Any]:
    cases = document.get("cases", document)
    if not isinstance(cases, dict):
        raise FrameGuidanceManifestError("Manifest must be a case mapping or contain a 'cases' mapping.")
    return cases


def _resolve_path(
    raw_path: str,
    manifest_path: Path,
    path_mapper: Callable[[str], str] | None,
) -> Path:
    mapped_path = path_mapper(raw_path) if path_mapper is not None else raw_path
    resolved = Path(mapped_path)
    if not resolved.is_absolute():
        resolved = manifest_path.parent / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FrameGuidanceManifestError(f"Anchor image does not exist: {resolved}")
    return resolved


def load_frame_guidance_case(
    manifest_file: str | Path,
    case_id: str,
    *,
    path_mapper: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Load one case and return normalized absolute anchor paths.

    Accepted entry shape::

        {
          "text_prompt": "...",
          "image_prompt": "first.png",
          "frame_guidance": {
            "anchors": [
              {"frame_index": 0, "image_path": "first.png"},
              {"frame_index": 60, "image_path": "middle.png"},
              {"frame_index": 120, "image_path": "last.png"}
            ]
          }
        }

    ``anchors`` may also be top-level for convenience.  We require a frame-0
    anchor because it is both the Wan I2V condition and an immutable provenance
    record; only anchors after frame zero enter the frame MSE objective.
    """
    manifest_path = Path(manifest_file).resolve()
    if not manifest_path.is_file():
        raise FrameGuidanceManifestError(f"Manifest does not exist: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise FrameGuidanceManifestError("Manifest root must be a JSON object.")

    cases = _case_table(document)
    if case_id not in cases:
        raise FrameGuidanceManifestError(f"Case '{case_id}' is not present in {manifest_path}.")
    entry = cases[case_id]
    if not isinstance(entry, dict):
        raise FrameGuidanceManifestError(f"Case '{case_id}' must be a JSON object.")

    prompt = entry.get("text_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise FrameGuidanceManifestError(f"Case '{case_id}' requires a non-empty 'text_prompt'.")

    frame_guidance = entry.get("frame_guidance", entry)
    if not isinstance(frame_guidance, dict):
        raise FrameGuidanceManifestError(f"Case '{case_id}'.frame_guidance must be an object.")
    raw_anchors = frame_guidance.get("anchors")
    if not isinstance(raw_anchors, list) or len(raw_anchors) < 3:
        raise FrameGuidanceManifestError(
            f"Case '{case_id}' requires at least first/middle/last anchors."
        )

    anchors: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for anchor in raw_anchors:
        if not isinstance(anchor, dict):
            raise FrameGuidanceManifestError("Every anchor must be a JSON object.")
        frame_index = anchor.get("frame_index", anchor.get("index"))
        image_path = anchor.get("image_path", anchor.get("path", anchor.get("frame_path")))
        if not isinstance(frame_index, int) or frame_index < 0:
            raise FrameGuidanceManifestError("Every anchor requires a non-negative integer 'frame_index'.")
        if frame_index in seen_indices:
            raise FrameGuidanceManifestError(f"Duplicate anchor index {frame_index} in case '{case_id}'.")
        if not isinstance(image_path, str) or not image_path:
            raise FrameGuidanceManifestError(
                f"Anchor {frame_index} in case '{case_id}' requires 'image_path'."
            )
        seen_indices.add(frame_index)
        normalized = dict(anchor)
        normalized["frame_index"] = frame_index
        normalized["image_path"] = str(_resolve_path(image_path, manifest_path, path_mapper))
        normalized["sha256"] = sha256_file(Path(normalized["image_path"]))
        anchors.append(normalized)

    anchors.sort(key=lambda anchor: anchor["frame_index"])
    if anchors[0]["frame_index"] != 0:
        raise FrameGuidanceManifestError(
            f"Case '{case_id}' must include a frame_index=0 conditioning anchor."
        )

    condition_raw = entry.get("image_prompt", entry.get("condition_image_path", anchors[0]["image_path"]))
    if not isinstance(condition_raw, str) or not condition_raw:
        raise FrameGuidanceManifestError(f"Case '{case_id}' requires 'image_prompt' or a frame-0 anchor.")
    condition_path = _resolve_path(condition_raw, manifest_path, path_mapper)
    condition_sha = sha256_file(condition_path)
    if condition_sha != anchors[0]["sha256"]:
        raise FrameGuidanceManifestError(
            "The conditioning image and frame-0 anchor differ. Use the same first GT frame for both."
        )

    return {
        "case_id": case_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "text_prompt": prompt,
        "condition_image_path": str(condition_path),
        "condition_image_sha256": condition_sha,
        "anchors": anchors,
        "frame_guidance_metadata": {
            key: value for key, value in frame_guidance.items() if key != "anchors"
        },
        "case_metadata": {
            key: value
            for key, value in entry.items()
            if key not in {"text_prompt", "image_prompt", "condition_image_path", "frame_guidance", "anchors"}
        },
    }

