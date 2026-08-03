#!/usr/bin/env python3
"""Download pinned generation models and publish frozen local snapshots."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from huggingface_hub import get_token, snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_selection.model_lock import validate_frozen_model_snapshot
from scripts.freeze_model_snapshot import freeze_snapshot


MODEL_SPECS = {
    "wan": {
        "repo_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "revision": "b8fff7315c768468a5333511427288870b2e9635",
        "requires_token": False,
    },
    "cosmos": {
        "repo_id": "nvidia/Cosmos-Predict2.5-2B",
        "revision": "0d37c7498f54cee3c599d438d895a0a4a8608064",
        "requires_token": True,
    },
}


def parse_model_keys(value: str) -> list[str]:
    keys = [key.strip() for key in value.split(",") if key.strip()]
    if not keys or len(keys) != len(set(keys)) or any(key not in MODEL_SPECS for key in keys):
        raise ValueError(f"models must be a unique comma-separated subset of {sorted(MODEL_SPECS)}")
    return keys


@contextmanager
def model_publication_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def prepare_models(
    keys: list[str],
    *,
    cache_dir: Path,
    model_root: Path,
    download: Callable[..., str] = snapshot_download,
) -> list[dict[str, str]]:
    receipts = []
    (model_root / "frozen").mkdir(parents=True, exist_ok=True)
    (model_root / "refs").mkdir(parents=True, exist_ok=True)
    for key in keys:
        spec = MODEL_SPECS[key]
        if spec["requires_token"] and not get_token():
            raise RuntimeError(
                f"{spec['repo_id']} is gated; authenticate in the configured HF_HOME first"
            )
        lock_path = model_root / ".locks" / f"{key}.lock"
        with model_publication_lock(lock_path):
            print(
                "downloading {}: {}@{}".format(key, spec["repo_id"], spec["revision"]),
                flush=True,
            )
            snapshot = Path(
                download(
                    repo_id=spec["repo_id"],
                    revision=spec["revision"],
                    cache_dir=cache_dir,
                    max_workers=4,
                )
            ).resolve()
            frozen = model_root / "frozen" / "{}-{}".format(key, spec["revision"])
            if frozen.exists():
                validate_frozen_model_snapshot(frozen)
            else:
                freeze_snapshot(snapshot, frozen)
                validate_frozen_model_snapshot(frozen)

            receipt = {
                "schema": "geometry-prefetched-model-v1",
                "key": key,
                "repo_id": spec["repo_id"],
                "revision": spec["revision"],
                "snapshot_path": str(snapshot),
                "frozen_path": str(frozen),
            }
            receipt_path = model_root / "refs" / f"{key}.json"
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=receipt_path.parent,
                prefix=f".{receipt_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            temporary.replace(receipt_path)
            receipts.append(receipt)
            print(json.dumps(receipt, indent=2), flush=True)
    return receipts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="wan")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()
    prepare_models(
        parse_model_keys(args.models),
        cache_dir=args.cache_dir,
        model_root=args.model_root,
    )


if __name__ == "__main__":
    main()
