#!/usr/bin/env python3
"""Create the pre-generation lock that authorizes ranking configurations."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_selection.config import load_offline_ranking_config
from geometry_selection.protocol import (
    file_sha256,
    implementation_tree_sha256,
    validate_formal_protocol,
)
from geometry_selection.selection import validate_candidate_spec


def parse_authorization(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SPLIT=/path/to/ranking.yaml")
    split, raw_path = value.split("=", 1)
    if split not in {"debug", "validation", "test"}:
        raise argparse.ArgumentTypeError(f"invalid split: {split}")
    return split, Path(raw_path).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--candidate-spec", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--authorize", action="append", type=parse_authorization, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = validate_formal_protocol(args.protocol)
    spec = json.loads(args.candidate_spec.read_text(encoding="utf-8"))
    validate_candidate_spec(spec, require_candidate_videos=False)
    protocol_sha256 = file_sha256(args.protocol)
    candidate_spec_sha256 = file_sha256(args.candidate_spec)
    if spec["protocol_manifest_sha256"] != protocol_sha256:
        raise ValueError("candidate spec protocol digest differs from protocol")
    artifact_root = args.artifact_root.expanduser().resolve()
    for case in spec["cases"]:
        for candidate in case["candidates"]:
            try:
                Path(candidate["video"]).expanduser().resolve().relative_to(artifact_root)
            except ValueError as error:
                raise ValueError("candidate video path escapes artifact root") from error
    model_lock_sha256 = file_sha256(args.model_lock)
    if model_lock_sha256 != protocol["_meta"]["model_lock_sha256"]:
        raise ValueError("model lock digest differs from protocol")
    authorized: list[str] = []
    for split, config_path in args.authorize:
        if split != spec["split"]:
            raise ValueError("all ranking configs must match the candidate-spec split")
        config = load_offline_ranking_config(config_path)
        if config.expected_split != split:
            raise ValueError(f"ranking config split mismatch: {config_path}")
        authorized.append(config.config_hash)
    authorized = sorted(set(authorized))
    if spec["split"] == "test" and len(authorized) != 1:
        raise ValueError("test split must authorize exactly one final ranking config")
    payload = {
        "schema": "geometry-experiment-lock-v1",
        "protocol_manifest_sha256": protocol_sha256,
        "model_lock_sha256": model_lock_sha256,
        "implementation_sha256": implementation_tree_sha256(REPO_ROOT),
        "split": spec["split"],
        "backbone": spec["backbone"],
        "candidate_spec_sha256": candidate_spec_sha256,
        "artifact_root": str(artifact_root),
        "authorized_ranking_config_hashes": authorized,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, output)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite experiment lock: {output}") from error
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
