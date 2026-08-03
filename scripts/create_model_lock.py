#!/usr/bin/env python3
"""Create an immutable model lock with one-time full weight hashing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_selection.backbones.vggt_omega import VGGTOmegaAdapter
from geometry_selection.model_lock import MODEL_LOCK_SCHEMA, model_directory_identity


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected PROFILE=/absolute/model/path")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not name or not path.is_absolute():
        raise argparse.ArgumentTypeError("expected PROFILE=/absolute/model/path")
    return name, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-model",
        action="append",
        type=parse_named_path,
        required=True,
    )
    parser.add_argument("--geometry-source", type=Path, required=True)
    parser.add_argument("--geometry-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = [name for name, _ in args.generation_model]
    if len(set(names)) != len(names):
        parser.error("generation model profile names must be unique")

    geometry = VGGTOmegaAdapter(
        source_root=args.geometry_source,
        checkpoint=args.geometry_checkpoint,
        device="cpu",
    ).cache_identity()
    payload = {
        "schema": MODEL_LOCK_SCHEMA,
        "generation_models": {
            name: model_directory_identity(path, hash_weights=True)
            for name, path in args.generation_model
        },
        "geometry_backbone": geometry,
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
        raise FileExistsError(f"refusing to overwrite model lock: {output}") from error
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "generation_models": names,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
