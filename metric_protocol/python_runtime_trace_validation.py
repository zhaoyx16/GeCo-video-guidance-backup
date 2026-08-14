#!/usr/bin/env python3
"""Re-hash adapter-emitted PyTorch/TorchVision module and native load traces."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _verify_loaded_files(
    values: Any, expected: dict[str, str], label: str, required_path: str
) -> list[dict[str, str]]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} is empty")
    paths: list[str] = []
    normalized: list[dict[str, str]] = []
    for item in values:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256"}
            or not isinstance(item["path"], str)
            or not isinstance(item["sha256"], str)
        ):
            raise ValueError(f"{label} entry is malformed")
        raw = Path(item["path"])
        info = raw.stat(follow_symlinks=False)
        path = raw.resolve(strict=True)
        if (
            raw.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or info.st_mode & 0o222
            or str(path) in paths
            or expected.get(str(path)) != item["sha256"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"{label} file differs from its sealed manifest")
        paths.append(str(path))
        normalized.append({"path": str(path), "sha256": item["sha256"]})
    if paths != sorted(paths) or required_path not in paths:
        raise ValueError(f"{label} is non-canonical or omits its required native/runtime file")
    return normalized


def verify_package_runtime_trace(
    trace: Any, expected_identity: Any, distribution: str
) -> dict[str, Any]:
    base_keys = {
        "runtime_identity", "package_file_count", "loaded_module_files",
        "loaded_module_files_sha256", "loaded_environment_native_files",
        "loaded_environment_native_files_sha256", "native_extension_loaded",
    }
    expected_keys = set(base_keys)
    if distribution == "torchvision":
        expected_keys |= {"native_ops_available", "torch_runtime_load_trace"}
    if (
        not isinstance(trace, dict)
        or set(trace) != expected_keys
        or not isinstance(expected_identity, dict)
        or trace.get("runtime_identity") != expected_identity
        or trace.get("native_extension_loaded") is not True
        or (distribution == "torchvision" and trace.get("native_ops_available") is not True)
    ):
        raise ValueError(f"{distribution} runtime trace schema/identity differs")
    package_manifest = expected_identity.get("package_manifest")
    environment = expected_identity.get("environment_closure")
    if not isinstance(package_manifest, dict) or not isinstance(environment, dict):
        raise ValueError(f"{distribution} locked runtime closure is malformed")
    package_payload = read_json(Path(package_manifest["path"]))
    package_root = Path(expected_identity["package_root"]).resolve(strict=True)
    package_files = package_payload.get("files")
    if (
        not isinstance(package_files, list)
        or trace.get("package_file_count") != len(package_files)
    ):
        raise ValueError(f"{distribution} trace package count differs")
    package_expected = {
        str((package_root / item["path"]).resolve(strict=True)): item["sha256"]
        for item in package_files
    }
    modules = _verify_loaded_files(
        trace.get("loaded_module_files"),
        package_expected,
        f"{distribution} loaded modules",
        str(Path(expected_identity["init"]["path"]).resolve(strict=True)),
    )
    if trace.get("loaded_module_files_sha256") != hashlib.sha256(canonical_bytes(modules)).hexdigest():
        raise ValueError(f"{distribution} loaded-module trace SHA differs")
    environment_root = Path(environment["root"]).resolve(strict=True)
    environment_payload = read_json(Path(environment["content_manifest"]["path"]))
    environment_expected = {
        str((environment_root / item["path"]).resolve(strict=True)): item["sha256"]
        for item in environment_payload["files"]
    }
    native = _verify_loaded_files(
        trace.get("loaded_environment_native_files"),
        environment_expected,
        f"{distribution} loaded native files",
        str(Path(expected_identity["native_extension"]["path"]).resolve(strict=True)),
    )
    if trace.get("loaded_environment_native_files_sha256") != hashlib.sha256(canonical_bytes(native)).hexdigest():
        raise ValueError(f"{distribution} loaded-native trace SHA differs")
    return trace
