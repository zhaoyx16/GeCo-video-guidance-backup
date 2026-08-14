#!/usr/bin/env python3
"""Mandatory immediately-before-worker integrity check for a locked Hippasus mirror."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
import types
from pathlib import Path
from typing import Any


FORMAL_INPUT_SCHEMA = "geometry-selection-five-metric-hippasus-input-lock-v2"
TRAINDEV_INPUT_SCHEMA = "geometry-selection-five-metric-traindev-input-lock-v1"
TRAINDEV_METHOD_IDS = {
    "wan_lora_dpo_step64_base_traindev",
    "wan_lora_dpo_step64_adapted_traindev",
}


def load_exact_sibling(path: Path, expected_sha256: str, module_name: str) -> types.ModuleType:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino):
            raise RuntimeError(f"{module_name} helper is not a stable regular file")
        digest, chunks = hashlib.sha256(), []
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
        named_after = os.stat(path, follow_symlinks=False)
        if (
            digest.hexdigest() != expected_sha256
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or (after.st_dev, after.st_ino) != (named_after.st_dev, named_after.st_ino)
        ):
            raise RuntimeError(f"{module_name} helper differs from its reviewed SHA")
    finally:
        os.close(descriptor)
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    exec(compile(b"".join(chunks), str(path), "exec"), module.__dict__)
    return module


_provenance_module = load_exact_sibling(
    Path(__file__).resolve(strict=True).with_name("traindev_provenance_v1.py"),
    "89283249fba8e4176574b7bf0bf1cbe0cdfe94792a8433e51edb2d331b7fa7a1",
    "locked_traindev_provenance_v1",
)
verify_train_dev_provenance = _provenance_module.verify_train_dev_provenance


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)}


def safe_output(path: Path, derived_root: Path, inputs: list[Path]) -> None:
    if derived_root.is_symlink() or not derived_root.is_dir() or tuple(derived_root.resolve(strict=True).parts[-3:]) != ("outputs", "geometry-selection", "hippasus_evaluation"):
        raise ValueError("derived root must be the approved Hippasus evaluation root")
    derived = derived_root.resolve(strict=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("output parent must be an existing regular directory")
    target = path.parent.resolve(strict=True) / path.name
    if target == derived or derived not in target.parents or target.relative_to(derived).parts[0] != "receipts":
        raise ValueError("preflight receipt must publish below approved receipts root")
    for source in inputs:
        resolved = source.resolve(strict=True)
        if target == resolved or target in resolved.parents or resolved in target.parents:
            raise ValueError("output must not overlap an input")


def publish(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing output: {path}")
    fd, name = tempfile.mkstemp(prefix=".metric-input-preflight-", suffix=".tmp", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_bytes(payload) + b"\n")
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        os.link(temporary, path)
        dir_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
        raise ValueError("input lock must be a read-only regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("input lock must be a JSON object")
    return value


def readonly_path(root: Path, raw: Any, expected_sha: Any, label: str) -> None:
    if not isinstance(raw, str) or not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError(f"{label} path/SHA is malformed")
    int(expected_sha, 16)
    path = Path(raw)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    resolved_root, resolved = root.resolve(strict=True), path.resolve(strict=True)
    if resolved_root not in resolved.parents:
        raise ValueError(f"{label} escapes its mirror root")
    for item in (resolved_root, *resolved.parents):
        if item == resolved_root or resolved_root in item.parents:
            if item.stat().st_mode & 0o222:
                raise ValueError(f"{label} has writable ancestor: {item}")
    if resolved.stat().st_mode & 0o222:
        raise ValueError(f"{label} is writable")
    if sha256_file(resolved) != expected_sha:
        raise ValueError(f"{label} SHA changed after input lock publication")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-lock", type=Path, required=True)
    parser.add_argument("--evaluator-lock", type=Path)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = read_json(args.input_lock)
    schema = lock.get("schema")
    train_dev = schema == TRAINDEV_INPUT_SCHEMA
    if schema not in {FORMAL_INPUT_SCHEMA, TRAINDEV_INPUT_SCHEMA} or lock.get("evaluation_site") != "Hippasus":
        raise ValueError("unexpected input-lock identity")
    if train_dev and (
        lock.get("scope") != "train_dev_evaluation"
        or lock.get("dataset_split") != "dev"
        or "formal_validation" in lock
        or lock.get("reserved_ids_disclosed") is not False
        or lock.get("method_id") not in TRAINDEV_METHOD_IDS
        or not isinstance(lock.get("traindev_reference_isolation_receipt_sha256"), str)
        or len(lock["traindev_reference_isolation_receipt_sha256"]) != 64
    ):
        raise ValueError("unexpected train-dev input-lock identity")
    evaluator_binding = None
    traindev_provenance = None
    if train_dev:
        if args.evaluator_lock is None:
            raise ValueError("train-dev preflight requires its exact evaluator lock")
        evaluator = read_json(args.evaluator_lock)
        evaluator_binding = binding(args.evaluator_lock)
        if (
            evaluator.get("schema") != "geometry-selection-evaluator-lock-v2"
            or evaluator.get("scope") != "train_dev"
            or evaluator.get("site") != "Hippasus"
            or evaluator.get("input_manifest") != binding(args.input_lock)
            or not isinstance(evaluator.get("traindev_provenance"), dict)
        ):
            raise ValueError("train-dev evaluator lock does not bind this exact input")
        locked_provenance = evaluator["traindev_provenance"]
        traindev_provenance = verify_train_dev_provenance(
            input_payload=lock,
            input_binding=binding(args.input_lock),
            source_bundle_ready=locked_provenance.get("source_bundle_ready"),
            expected_generation_receipt_sha256=locked_provenance.get(
                "expected_generation_receipt_sha256"
            ),
            expected_baseline_eligibility_sha256=locked_provenance.get(
                "expected_baseline_eligibility_sha256"
            ),
        )
        if traindev_provenance != locked_provenance:
            raise ValueError("train-dev provenance differs from the evaluator-lock closure")
    raw_root = lock.get("mirror_root")
    if not isinstance(raw_root, str) or not raw_root:
        raise ValueError("input lock lacks mirror root")
    root = Path(raw_root)
    if root.is_symlink() or not root.is_dir() or root.stat().st_mode & 0o222:
        raise ValueError("mirror root must be a read-only regular directory")
    entries = lock.get("entries")
    if not isinstance(entries, list) or not entries or (train_dev and len(entries) != 100):
        raise ValueError("input lock has no entries")
    seen: set[str] = set()
    verified_entries = []
    for order, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("case_id"), str) or entry["case_id"] in seen:
            raise ValueError("input lock case entries are malformed")
        if train_dev and entry.get("split_order") != order:
            raise ValueError("train-dev input lock case order changed")
        seen.add(entry["case_id"])
        readonly_path(root, entry.get("metric_video_path"), entry.get("video_sha256"), f"{entry['case_id']}.video")
        readonly_path(root, entry.get("metric_metadata_path"), entry.get("metadata_sha256"), f"{entry['case_id']}.metadata")
        readonly_path(root, entry.get("metric_complete_path"), entry.get("complete_sha256"), f"{entry['case_id']}.COMPLETE")
        verified = {
            "case_id": entry["case_id"],
            "video_sha256": entry["video_sha256"],
            "metadata_sha256": entry["metadata_sha256"],
            "complete_sha256": entry["complete_sha256"],
        }
        if train_dev:
            readonly_path(
                root,
                entry.get("metric_generation_lock_path"),
                entry.get("generation_lock_sha256"),
                f"{entry['case_id']}.generation_lock",
            )
            verified["split_order"] = order
            verified["generation_lock_sha256"] = entry["generation_lock_sha256"]
        verified_entries.append(verified)
    payload = {
        "schema": (
            "geometry-selection-metric-traindev-input-preflight-receipt-v1"
            if train_dev
            else "geometry-selection-metric-input-preflight-receipt-v1"
        ),
        "site": "Hippasus",
        "scope": lock.get("scope"),
        "input_lock": binding(args.input_lock),
        "case_artifacts": verified_entries if train_dev else sorted(verified_entries, key=lambda item: item["case_id"]),
    }
    if train_dev:
        payload["evaluator_lock"] = evaluator_binding
        payload["traindev_provenance"] = traindev_provenance
    protected_inputs = [args.input_lock, root]
    if args.evaluator_lock is not None:
        protected_inputs.append(args.evaluator_lock)
    safe_output(args.output, args.derived_root, protected_inputs)
    publish(args.output, payload)
    print(json.dumps({"case_count": len(seen), "input_lock_sha256": payload["input_lock"]["sha256"], "output": str(args.output), "verified": True}, sort_keys=True))


if __name__ == "__main__":
    main()
