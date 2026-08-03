#!/usr/bin/env python3
"""Publish a model directory as a closed, read-only formal snapshot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_selection.model_lock import FROZEN_MODEL_MARKER


def freeze_snapshot(source: Path, target: Path) -> Path:
    source = source.expanduser().resolve()
    raw_target = target.expanduser().absolute()
    if not source.is_dir():
        raise FileNotFoundError(f"model source is not a directory: {source}")
    target = raw_target.parent.resolve() / raw_target.name
    try:
        target.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("frozen snapshot target must not be inside its source tree")
    directory_links = [
        path for path in source.rglob("*") if path.is_symlink() and path.resolve().is_dir()
    ]
    if directory_links:
        raise ValueError(f"model source contains directory symlink: {directory_links[0]}")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to replace existing target: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, staging, symlinks=False)
        symlinks = [path for path in staging.rglob("*") if path.is_symlink()]
        if symlinks:
            raise ValueError(f"frozen staging tree contains symlink: {symlinks[0]}")
        marker = {
            "schema": "geometry-frozen-model-snapshot-v1",
            "publication": "closed-staging-read-only-atomic-rename",
            "source": str(source),
        }
        (staging / FROZEN_MODEL_MARKER).write_text(
            json.dumps(marker, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        entries = sorted(staging.rglob("*"), key=lambda path: len(path.parts), reverse=True)
        for path in entries:
            mode = path.stat().st_mode
            if stat.S_ISREG(mode):
                path.chmod(0o444)
            elif stat.S_ISDIR(mode):
                path.chmod(0o555)
        staging.chmod(0o555)
        os.replace(staging, target)
    except Exception:
        if staging.exists():
            for path in staging.rglob("*"):
                if path.is_dir():
                    path.chmod(0o755)
                elif path.is_file():
                    path.chmod(0o644)
            staging.chmod(0o755)
            shutil.rmtree(staging)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    print(freeze_snapshot(args.source, args.target))


if __name__ == "__main__":
    main()
