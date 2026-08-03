#!/usr/bin/env python3
"""Fail-fast checks for the fixed development-validation candidate array."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_selection.model_lock import (
    load_model_lock,
    model_directory_identity,
    runtime_identity_matches_lock,
    validate_frozen_model_snapshot,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--expected-split", required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--model-lock-sha256", required=True)
    parser.add_argument("--model-profile", required=True)
    args = parser.parse_args()

    if args.expected_cases <= 0:
        parser.error("expected-cases must be positive")
    actual_manifest_sha256 = file_sha256(args.manifest)
    if actual_manifest_sha256 != args.manifest_sha256:
        raise ValueError("validation manifest SHA-256 differs from the submitted contract")
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = [(key, value) for key, value in payload.items() if not key.startswith("_")]
    if len(cases) != args.expected_cases:
        raise ValueError(f"expected {args.expected_cases} cases, found {len(cases)}")
    if not 0 <= args.case_index < len(cases):
        raise ValueError(f"case index outside fixed manifest: {args.case_index}")
    if {case.get("split") for _, case in cases} != {args.expected_split}:
        raise ValueError("manifest cases do not all belong to the expected split")
    if sorted(case.get("split_order") for _, case in cases) != list(range(args.expected_cases)):
        raise ValueError("manifest split_order is not exactly 0..expected-cases-1")
    if len({case.get("scene_id") for _, case in cases}) != args.expected_cases:
        raise ValueError("manifest scene IDs are not unique")
    for case_id, case in cases:
        if "case_id" in case and case["case_id"] != case_id:
            raise ValueError(f"manifest key/case_id mismatch: {case_id}")
        if not Path(case["image_prompt"]).is_file():
            raise FileNotFoundError(case["image_prompt"])
        if not Path(case["transforms_path"]).is_file():
            raise FileNotFoundError(case["transforms_path"])

    selected_id, selected_case = cases[args.case_index]
    provenance = selected_case.get("conditioning_image_provenance", {})
    image_path = Path(selected_case["image_prompt"])
    transforms_path = Path(selected_case["transforms_path"])
    if file_sha256(image_path) != provenance.get("image_sha256"):
        raise ValueError(f"conditioning-image SHA-256 mismatch: {selected_id}")
    if file_sha256(transforms_path) != provenance.get("transforms_sha256"):
        raise ValueError(f"transforms SHA-256 mismatch: {selected_id}")
    with Image.open(image_path) as image:
        actual_size = image.size
    expected_size = (provenance.get("width"), provenance.get("height"))
    if actual_size != expected_size:
        raise ValueError(
            f"conditioning-image size mismatch for {selected_id}: "
            f"{actual_size} != {expected_size}"
        )

    actual_model_lock_sha256 = file_sha256(args.model_lock)
    if actual_model_lock_sha256 != args.model_lock_sha256:
        raise ValueError("model-lock SHA-256 differs from the submitted contract")
    model_lock = load_model_lock(args.model_lock)
    if args.model_profile not in model_lock["generation_models"]:
        raise ValueError(f"model profile is absent from lock: {args.model_profile}")
    validate_frozen_model_snapshot(args.model)
    runtime = model_directory_identity(args.model, hash_weights=False)
    if not runtime_identity_matches_lock(
        runtime,
        model_lock["generation_models"][args.model_profile],
    ):
        raise ValueError("read-only generation model identity differs from the frozen lock")
    validate_frozen_model_snapshot(args.model)

    print(
        json.dumps(
            {
                "case_id": selected_id,
                "case_index": args.case_index,
                "manifest_sha256": actual_manifest_sha256,
                "model_lock_sha256": actual_model_lock_sha256,
                "model_profile": args.model_profile,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
