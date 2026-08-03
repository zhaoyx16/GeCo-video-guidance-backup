#!/usr/bin/env python3
"""Freeze an extracted candidate pool and all referenced geometry caches."""

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

from geometry_selection.pool_lock import make_candidate_pool_lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = make_candidate_pool_lock(args.candidate_pool, args.cache_root)
    if payload["artifact_mode"] != "formal":
        raise ValueError("only a formal candidate pool can be frozen for formal ranking")
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
        raise FileExistsError(f"refusing to overwrite candidate pool lock: {output}") from error
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
