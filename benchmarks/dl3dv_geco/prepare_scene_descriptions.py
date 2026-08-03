#!/usr/bin/env python3
"""Prepare motion-free scene descriptions from released DL3DV captions."""

from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path

from build_manifest import (
    FORMAL_SPLIT_COUNTS,
    find_forbidden_description_terms,
    load_frozen_assignments,
    load_scene_descriptions,
    sha256_file,
)


PREFIXES = (
    r"^the video takes place in\s+",
    r"^the video (?:features|explores|showcases|shows|captures|depicts|presents)\s+",
    r"^the video takes (?:us|viewers) (?:through|to)\s+",
)


def caption_index(payload: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, caption in payload.items():
        if not isinstance(key, str) or not isinstance(caption, str):
            raise ValueError("caption file must map strings to strings")
        parts = key.split("/")
        scene_ids = [part for part in parts if len(part) == 64]
        if len(scene_ids) != 1:
            raise ValueError(f"cannot resolve scene id from caption key {key!r}")
        scene_id = scene_ids[0]
        if scene_id in result:
            raise ValueError(f"duplicate caption for scene {scene_id}")
        result[scene_id] = caption
    return result


def _same_scene_phrase(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(?:a|an|the)\s+", "", text, flags=re.IGNORECASE)
    if not text:
        raise ValueError("caption became empty after prefix removal")
    return f"the same {text[0].lower() + text[1:]}"


def clean_caption(caption: str, max_words: int = 45) -> str:
    first_sentence = re.split(r"(?<=[.!?])\s+", " ".join(caption.split()), maxsplit=1)[0]
    content = first_sentence.rstrip(".!? ")
    for pattern in PREFIXES:
        stripped, count = re.subn(pattern, "", content, count=1, flags=re.IGNORECASE)
        if count:
            content = stripped
            break
    content = re.sub(r"\bstarting with a view of\b", "with", content, flags=re.IGNORECASE)
    content = re.sub(r"\b(?:a series of )?still scenes? (?:at|in|of)\b", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\bvarious scenes of\b", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\bis showcased\b", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\bshowcasing\b", "with", content, flags=re.IGNORECASE)
    content = " ".join(content.split()).strip(" ,")
    description = _same_scene_phrase(content)
    words = description.split()
    if len(words) > max_words:
        candidate = " ".join(words[:max_words])
        comma = candidate.rfind(",")
        if comma >= 20:
            candidate = candidate[:comma]
        description = candidate.rstrip(",;: ")
    forbidden = find_forbidden_description_terms(description)
    if forbidden:
        raise ValueError(f"motion/temporal terms remain: {forbidden}")
    return description


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--expected-captions-sha256", required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["debug", "validation", "test"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overrides", type=Path)
    parser.add_argument(
        "--review-status",
        choices=["requires_manual_review", "manually_reviewed"],
        default="requires_manual_review",
    )
    args = parser.parse_args()

    captions_sha256 = sha256_file(args.captions)
    if captions_sha256 != args.expected_captions_sha256:
        raise ValueError("caption source SHA256 mismatch")
    assignments = load_frozen_assignments(
        args.split_csv,
        set(args.splits),
        FORMAL_SPLIT_COUNTS,
    )
    captions = caption_index(json.loads(args.captions.read_text(encoding="utf-8")))
    overrides = load_scene_descriptions(args.overrides) if args.overrides else {}
    descriptions: dict[str, str] = {}
    failures = []
    for assignment in assignments:
        if assignment.scene_id in overrides:
            descriptions[assignment.scene_id] = overrides[assignment.scene_id]
            continue
        caption = captions.get(assignment.scene_id)
        if caption is None:
            failures.append({"scene_id": assignment.scene_id, "error": "missing caption"})
            continue
        try:
            descriptions[assignment.scene_id] = clean_caption(caption)
        except ValueError as error:
            failures.append({"scene_id": assignment.scene_id, "error": str(error)})

    atomic_json(args.output, descriptions)
    # Reuse the manifest-side policy as a final independent parse of the output.
    load_scene_descriptions(args.output)
    report = {
        "schema": "dl3dv-scene-description-preparation-v1",
        "source_captions": str(args.captions.resolve()),
        "source_captions_sha256": captions_sha256,
        "source_url": (
            "https://github.com/Hongyang-Du/VideoGPA/blob/main/"
            "dl3dv_video_captions/captions_1K.json"
        ),
        "split_csv": str(args.split_csv.resolve()),
        "split_csv_sha256": sha256_file(args.split_csv),
        "selected_splits": sorted(set(args.splits)),
        "overrides": str(args.overrides.resolve()) if args.overrides else None,
        "overrides_sha256": sha256_file(args.overrides) if args.overrides else None,
        "expected_scenes": len(assignments),
        "prepared_scenes": len(descriptions),
        "failed_scenes": len(failures),
        "failures": failures,
        "output_sha256": sha256_file(args.output),
        "review_status": args.review_status,
    }
    atomic_json(args.report, report)
    if failures:
        raise SystemExit(f"{len(failures)} descriptions require manual resolution")


if __name__ == "__main__":
    main()
