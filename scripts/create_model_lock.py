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
    expanded = Path(raw_path).expanduser()
    if not name or not expanded.is_absolute():
        raise argparse.ArgumentTypeError("expected PROFILE=/absolute/model/path")
    path = expanded.resolve()
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
    parser.add_argument(
        "--reuse-weight-hashes-from",
        type=Path,
        help="Reuse previously computed hashes only when every weight path and size matches.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = [name for name, _ in args.generation_model]
    if len(set(names)) != len(names):
        parser.error("generation model profile names must be unique")

    previous = None
    if args.reuse_weight_hashes_from is not None:
        previous = json.loads(args.reuse_weight_hashes_from.read_text(encoding="utf-8"))

    geometry_adapter = VGGTOmegaAdapter(
        source_root=args.geometry_source,
        checkpoint=args.geometry_checkpoint,
        device="cpu",
    )
    if previous is None:
        geometry = geometry_adapter.artifact_identity()
    else:
        geometry = geometry_adapter.identity(hash_checkpoint=False)
        geometry.pop("source_root")
        geometry.pop("checkpoint")
        previous_geometry = previous.get("geometry_backbone")
        if not isinstance(previous_geometry, dict):
            raise ValueError("previous model lock has no geometry backbone identity")
        comparable = dict(geometry)
        comparable.pop("checkpoint_sha256")
        previous_comparable = dict(previous_geometry)
        previous_digest = previous_comparable.pop("checkpoint_sha256", None)
        if comparable != previous_comparable or not previous_digest:
            raise ValueError(
                "cannot reuse geometry checkpoint hash after geometry identity change"
            )
        geometry["checkpoint_sha256"] = previous_digest

    generation_models = {}
    for name, path in args.generation_model:
        if previous is None:
            identity = model_directory_identity(path, hash_weights=True)
        else:
            identity = model_directory_identity(path, hash_weights=False)
            previous_identity = previous.get("generation_models", {}).get(name)
            if previous_identity is None:
                raise ValueError(f"previous model lock has no profile: {name}")
            previous_weights = {
                (record["path"], record["size"]): record.get("sha256")
                for record in previous_identity.get("weight_files", [])
            }
            for record in identity["weight_files"]:
                digest = previous_weights.get((record["path"], record["size"]))
                if not digest:
                    raise ValueError(
                        f"cannot reuse weight hash after path/size change: {name}/{record['path']}"
                    )
                record["sha256"] = digest
        generation_models[name] = identity

    payload = {
        "schema": MODEL_LOCK_SCHEMA,
        "generation_models": generation_models,
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
