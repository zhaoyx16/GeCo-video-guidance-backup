#!/usr/bin/env python3
"""Verify that a diagnostic replay leaves the frozen step-29 latent unchanged."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = read_json(args.reference.resolve())
    diagnostic = read_json(args.diagnostic.resolve())
    checks = {
        "case_id": reference.get("case_id") == diagnostic.get("case_id"),
        "replay_mode": reference.get("replay_mode") == diagnostic.get("replay_mode"),
        "last_computed_step": reference.get("last_computed_step") == diagnostic.get("last_computed_step"),
        "generation": reference.get("generation") == diagnostic.get("generation"),
        "method": reference.get("method") == diagnostic.get("method"),
        "latent_sha256": reference.get("latent_sha256") == diagnostic.get("latent_sha256"),
        "reference_has_no_diagnostics": reference.get("diagnostics") is None,
        "diagnostic_has_records": bool((diagnostic.get("diagnostics") or {}).get("records")),
    }
    payload = {
        "schema": "c2f-p0-replay-equivalence-v1",
        "reference_record": str(args.reference.resolve()),
        "diagnostic_record": str(args.diagnostic.resolve()),
        "case_id": reference.get("case_id"),
        "replay_mode": reference.get("replay_mode"),
        "reference_pipeline_sha256": reference.get("pipeline_sha256"),
        "diagnostic_pipeline_sha256": diagnostic.get("pipeline_sha256"),
        "reference_latent_sha256": reference.get("latent_sha256"),
        "diagnostic_latent_sha256": diagnostic.get("latent_sha256"),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2))
    if not payload["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
