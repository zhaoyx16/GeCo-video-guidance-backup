#!/usr/bin/env python3
"""Content-lock a Python wheel, its installed package, and native closure.

This module is deliberately shared by evaluator-lock construction, preflight,
the guarded worker, and the adapter runtime guard.  A symbolic package version
or ``pip freeze`` line is not sufficient evidence: every caller re-hashes the
sealed wheelhouse, sealed environment, installation/RECORD receipt, package
tree, native extension, and its complete ``ldd`` closure.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any


PACKAGE_MANIFEST_SCHEMA = "geometry-selection-python-package-content-manifest-v2"
TREE_MANIFEST_SCHEMA = "geometry-selection-sealed-tree-content-manifest-v1"
TREE_READY_SCHEMA = "geometry-selection-sealed-tree-ready-receipt-v1"
INSTALL_RECEIPT_SCHEMA = "geometry-selection-python-wheel-install-receipt-v1"
INSTALL_RECEIPT_STATUS = "wheel_record_verified"
ENVIRONMENT_FINGERPRINT_SCHEMA = "geometry-selection-evaluator-environment-fingerprint-v2"
PARITY_CERTIFICATE_SCHEMA = "geometry-selection-torchvision-preprocessing-parity-certificate-v1"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    return value


def readonly_regular(path: Path, label: str) -> Path:
    info = path.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222:
        raise ValueError(f"{label} must be an immutable regular file: {path}")
    return path.resolve(strict=True)


def readonly_directory(path: Path, label: str) -> Path:
    info = path.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o222:
        raise ValueError(f"{label} must be an immutable real directory: {path}")
    return path.resolve(strict=True)


def verify_file_binding(value: Any, label: str, *, sized: bool = False) -> dict[str, Any]:
    expected = {"path", "sha256", "size"} if sized else {"path", "sha256"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} binding is malformed")
    if not isinstance(value["path"], str) or not value["path"]:
        raise ValueError(f"{label} path is malformed")
    digest = require_sha256(value["sha256"], f"{label}.sha256")
    path = readonly_regular(Path(value["path"]), label)
    if sized and (
        not isinstance(value["size"], int)
        or isinstance(value["size"], bool)
        or value["size"] < 0
        or path.stat().st_size != value["size"]
    ):
        raise ValueError(f"{label} size differs")
    if sha256_file(path) != digest:
        raise ValueError(f"{label} SHA256 differs")
    result: dict[str, Any] = {"path": str(path), "sha256": digest}
    if sized:
        result["size"] = value["size"]
    return result


def read_bound_json(value: Any, label: str, schema: str) -> tuple[dict[str, str], dict[str, Any]]:
    binding = verify_file_binding(value, label)
    payload = json.loads(Path(binding["path"]).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"{label} schema differs")
    return binding, payload


def verify_relative_files(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "size", "sha256"}
            or not isinstance(item["path"], str)
            or not item["path"]
            or Path(item["path"]).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(item["path"]).parts)
            or not isinstance(item["size"], int)
            or isinstance(item["size"], bool)
            or item["size"] < 0
            or item["path"] in seen
        ):
            raise ValueError(f"{label}[{index}] is malformed")
        require_sha256(item["sha256"], f"{label}[{index}].sha256")
        seen.add(item["path"])
        files.append({"path": item["path"], "size": item["size"], "sha256": item["sha256"]})
    if files != sorted(files, key=lambda item: item["path"]):
        raise ValueError(f"{label} is not canonically sorted")
    return files


def rehash_readonly_tree(root: Path, label: str) -> list[dict[str, Any]]:
    root = readonly_directory(root, label)
    files: list[dict[str, Any]] = []
    for raw in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        info = raw.stat(follow_symlinks=False)
        if raw.is_symlink() or (not stat.S_ISDIR(info.st_mode) and not stat.S_ISREG(info.st_mode)):
            raise ValueError(f"{label} contains a symlink or special file: {raw}")
        if info.st_mode & 0o222:
            raise ValueError(f"{label} contains a writable descendant: {raw}")
        if stat.S_ISREG(info.st_mode):
            files.append({
                "path": raw.relative_to(root).as_posix(),
                "size": info.st_size,
                "sha256": sha256_file(raw),
            })
    if not files:
        raise ValueError(f"{label} tree is empty")
    return files


def verify_sealed_tree_closure(value: Any, label: str, expected_parent: Path) -> dict[str, Any]:
    keys = {"root", "content_manifest", "ready", "file_count", "tree_sha256"}
    if not isinstance(value, dict) or set(value) != keys or not isinstance(value["root"], str):
        raise ValueError(f"{label} closure is malformed")
    parent = expected_parent.resolve(strict=True)
    root = readonly_directory(Path(value["root"]), f"{label} root")
    if root.parent != parent:
        raise ValueError(f"{label} root is not a direct child of its approved parent")
    manifest_binding, manifest = read_bound_json(
        value["content_manifest"], f"{label} content manifest", TREE_MANIFEST_SCHEMA
    )
    ready_binding, ready = read_bound_json(value["ready"], f"{label} READY", TREE_READY_SCHEMA)
    declared_files = verify_relative_files(manifest.get("files"), f"{label}.content_manifest.files")
    actual_files = rehash_readonly_tree(root, label)
    tree_sha = hashlib.sha256(canonical_bytes(declared_files) + b"\n").hexdigest()
    if manifest.get("root") != str(root) or actual_files != declared_files:
        raise ValueError(f"{label} content manifest differs from the sealed tree")
    if (
        not isinstance(value["file_count"], int)
        or isinstance(value["file_count"], bool)
        or value["file_count"] != len(declared_files)
        or require_sha256(value["tree_sha256"], f"{label}.tree_sha256") != tree_sha
    ):
        raise ValueError(f"{label} closure count/tree SHA differs")
    if (
        set(ready) != {"schema", "status", "root", "content_manifest", "file_count", "tree_sha256"}
        or ready.get("status") != "full_rehash_verified"
        or ready.get("root") != str(root)
        or ready.get("content_manifest") != manifest_binding
        or ready.get("file_count") != len(declared_files)
        or ready.get("tree_sha256") != tree_sha
    ):
        raise ValueError(f"{label} READY does not attest the exact sealed tree")
    return {
        "root": str(root),
        "content_manifest": manifest_binding,
        "ready": ready_binding,
        "file_count": len(declared_files),
        "tree_sha256": tree_sha,
    }


def native_dependency_inspector_identity(
    inspector: dict[str, str], interpreter: dict[str, str], label: str
) -> dict[str, str]:
    inspector_binding = verify_file_binding(inspector, f"{label} dependency inspector")
    interpreter_binding = verify_file_binding(interpreter, f"{label} inspector interpreter")
    first_line = Path(inspector_binding["path"]).open("rb").readline().decode("utf-8", errors="strict").strip()
    if not first_line.startswith("#!") or Path(first_line[2:].split(maxsplit=1)[0]).resolve(strict=True) != Path(interpreter_binding["path"]):
        raise ValueError(f"{label} dependency inspector does not use its sealed interpreter")
    return {
        "path": inspector_binding["path"],
        "sha256": inspector_binding["sha256"],
        "interpreter_path": interpreter_binding["path"],
        "interpreter_sha256": interpreter_binding["sha256"],
    }


def native_library_closure(native: Path, inspector: dict[str, str], label: str) -> list[dict[str, str]]:
    injected = sorted(name for name, raw in os.environ.items() if name.startswith("LD_") and raw)
    if injected:
        raise ValueError(f"dynamic-loader environment must be empty for {label}: {injected}")
    result = subprocess.run(
        [inspector["path"], str(native)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    libraries: dict[str, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text or "linux-vdso" in text:
            continue
        if "not found" in text:
            raise ValueError(f"{label} native dependency is unresolved: {text}")
        tokens = text.replace("=>", " ").split()
        candidate = next((token for token in tokens if token.startswith("/")), None)
        if candidate is None:
            raise ValueError(f"could not resolve {label} native dependency: {text}")
        path = readonly_regular(Path(candidate), f"{label} native dependency")
        libraries[str(path)] = {"path": str(path), "sha256": sha256_file(path)}
    if not libraries:
        raise ValueError(f"ldd returned no sealed {label} native dependencies")
    return [libraries[path] for path in sorted(libraries)]


def _verify_install_receipt(
    binding: Any,
    *,
    distribution: str,
    version: str,
    package_root: Path,
    wheel: dict[str, Any],
    environment_closure: dict[str, Any],
    package_files: list[dict[str, Any]],
) -> dict[str, str]:
    receipt_binding, receipt = read_bound_json(binding, f"{distribution} install receipt", INSTALL_RECEIPT_SCHEMA)
    expected_keys = {
        "schema", "status", "distribution", "version", "package_root", "environment_root",
        "wheel", "record", "files", "record_mappings",
    }
    if (
        set(receipt) != expected_keys
        or receipt.get("status") != INSTALL_RECEIPT_STATUS
        or receipt.get("distribution") != distribution
        or receipt.get("version") != version
        or receipt.get("package_root") != str(package_root)
        or receipt.get("environment_root") != environment_closure["root"]
        or receipt.get("wheel") != wheel
        or verify_relative_files(receipt.get("files"), f"{distribution}.install_receipt.files") != package_files
    ):
        raise ValueError(f"{distribution} install receipt identity differs")
    record = verify_file_binding(receipt.get("record"), f"{distribution} RECORD", sized=True)
    environment_root = Path(environment_closure["root"])
    record_path = Path(record["path"])
    if environment_root not in record_path.parents:
        raise ValueError(f"{distribution} RECORD escapes the sealed environment")
    mappings = receipt.get("record_mappings")
    if not isinstance(mappings, list) or len(mappings) != len(package_files):
        raise ValueError(f"{distribution} RECORD mapping count differs")
    by_path = {item["path"]: item for item in package_files}
    seen: set[str] = set()
    for index, item in enumerate(mappings):
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "wheel_member_path", "record_sha256"}
            or not isinstance(item["path"], str)
            or item["path"] in seen
            or item["path"] not in by_path
            or not isinstance(item["wheel_member_path"], str)
            or not item["wheel_member_path"]
            or Path(item["wheel_member_path"]).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(item["wheel_member_path"]).parts)
            or require_sha256(item["record_sha256"], f"{distribution}.record_mappings[{index}]")
            != by_path[item["path"]]["sha256"]
        ):
            raise ValueError(f"{distribution} RECORD mapping is malformed")
        seen.add(item["path"])
    if seen != set(by_path) or mappings != sorted(mappings, key=lambda item: item["path"]):
        raise ValueError(f"{distribution} RECORD mappings are incomplete or non-canonical")
    return receipt_binding


def verify_declared_runtime_identity(
    identity: Any, expected_distribution: str, expected_version: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    expected_keys = {
        "distribution", "version", "environment_closure", "wheelhouse_closure",
        "package_root", "package_manifest", "wheel", "install_receipt", "init",
        "native_extension", "native_library_closure", "native_dependency_inspector",
        "runtime_helper",
    }
    if not isinstance(identity, dict) or set(identity) != expected_keys:
        raise ValueError(f"{expected_distribution} runtime identity is malformed")
    if identity["distribution"] != expected_distribution or identity["version"] != expected_version:
        raise ValueError(f"{expected_distribution} version differs from the frozen platform adaptation")
    helper = verify_file_binding(identity["runtime_helper"], f"{expected_distribution} runtime helper")
    if Path(helper["path"]) != Path(__file__).resolve(strict=True) or helper["sha256"] != sha256_file(Path(__file__)):
        raise ValueError(f"{expected_distribution} runtime helper differs from the executing verifier")
    environment = verify_sealed_tree_closure(
        identity["environment_closure"],
        f"{expected_distribution} environment",
        Path("/vol/dissolve/yz10325/evaluator_envs"),
    )
    wheelhouse = verify_sealed_tree_closure(
        identity["wheelhouse_closure"],
        f"{expected_distribution} wheelhouse",
        Path("/vol/dissolve/yz10325/evaluator_wheelhouses"),
    )
    manifest_binding, manifest = read_bound_json(
        identity["package_manifest"], f"{expected_distribution} package manifest", PACKAGE_MANIFEST_SCHEMA
    )
    manifest_keys = {
        "schema", "distribution", "version", "environment_closure", "wheelhouse_closure",
        "package_root", "wheel", "install_receipt", "files",
    }
    if (
        set(manifest) != manifest_keys
        or manifest.get("distribution") != expected_distribution
        or manifest.get("version") != expected_version
        or manifest.get("environment_closure") != environment
        or manifest.get("wheelhouse_closure") != wheelhouse
        or manifest.get("wheel") != identity["wheel"]
        or manifest.get("install_receipt") != identity["install_receipt"]
    ):
        raise ValueError(f"{expected_distribution} package manifest identity differs")
    package_root = readonly_directory(Path(manifest.get("package_root", "")), f"{expected_distribution} package root")
    environment_root = Path(environment["root"])
    if environment_root not in package_root.parents or str(package_root) != identity["package_root"]:
        raise ValueError(f"{expected_distribution} package root escapes the sealed environment")
    # Every component from the sealed environment root to the import package
    # must be a real, non-writable directory; writable/symlink ancestors would
    # permit package replacement between preflight and import.
    relative_parts = package_root.relative_to(environment_root).parts
    current = environment_root
    readonly_directory(current, f"{expected_distribution} environment root")
    for part in relative_parts:
        current = readonly_directory(current / part, f"{expected_distribution} package path component")
    package_files = verify_relative_files(manifest.get("files"), f"{expected_distribution}.package_manifest.files")
    actual_package_files = rehash_readonly_tree(package_root, f"{expected_distribution} package")
    if actual_package_files != package_files:
        raise ValueError(f"{expected_distribution} package tree differs from its manifest")
    by_relative = {item["path"]: item for item in package_files}
    wheel = verify_file_binding(identity["wheel"], f"{expected_distribution} wheel", sized=True)
    wheel_path = Path(wheel["path"])
    wheelhouse_root = Path(wheelhouse["root"])
    if wheelhouse_root not in wheel_path.parents:
        raise ValueError(f"{expected_distribution} wheel escapes the sealed wheelhouse")
    wheel_relative = wheel_path.relative_to(wheelhouse_root).as_posix()
    wheelhouse_manifest = json.loads(Path(wheelhouse["content_manifest"]["path"]).read_text(encoding="utf-8"))
    wheelhouse_files = {item["path"]: item for item in wheelhouse_manifest["files"]}
    if wheelhouse_files.get(wheel_relative) != {"path": wheel_relative, "size": wheel["size"], "sha256": wheel["sha256"]}:
        raise ValueError(f"{expected_distribution} wheel is not bound by the wheelhouse closure")
    if manifest["wheel"] != wheel:
        raise ValueError(f"{expected_distribution} wheel binding is non-canonical")
    install_receipt = _verify_install_receipt(
        identity["install_receipt"],
        distribution=expected_distribution,
        version=expected_version,
        package_root=package_root,
        wheel=wheel,
        environment_closure=environment,
        package_files=package_files,
    )
    if manifest["install_receipt"] != install_receipt:
        raise ValueError(f"{expected_distribution} install receipt binding differs")
    for label in ("init", "native_extension"):
        binding = verify_file_binding(identity[label], f"{expected_distribution} {label}")
        path = Path(binding["path"])
        if package_root not in path.parents:
            raise ValueError(f"{expected_distribution} {label} escapes package root")
        relative = path.relative_to(package_root).as_posix()
        if by_relative.get(relative) != {"path": relative, "size": path.stat().st_size, "sha256": binding["sha256"]}:
            raise ValueError(f"{expected_distribution} {label} is not package-manifest bound")
    inspector = identity["native_dependency_inspector"]
    if not isinstance(inspector, dict) or set(inspector) != {"path", "sha256", "interpreter_path", "interpreter_sha256"}:
        raise ValueError(f"{expected_distribution} native inspector identity is malformed")
    checked_inspector = native_dependency_inspector_identity(
        {"path": inspector["path"], "sha256": inspector["sha256"]},
        {"path": inspector["interpreter_path"], "sha256": inspector["interpreter_sha256"]},
        expected_distribution,
    )
    if checked_inspector != inspector:
        raise ValueError(f"{expected_distribution} native inspector identity differs")
    native = Path(identity["native_extension"]["path"])
    libraries = native_library_closure(native, inspector, expected_distribution)
    if identity["native_library_closure"] != libraries:
        raise ValueError(f"{expected_distribution} native-library closure differs")
    return identity, manifest, by_relative


def package_runtime_identity(
    roles: dict[str, dict[str, str]], prefix: str, distribution: str, version: str
) -> dict[str, Any]:
    helper = verify_file_binding(roles["python_package_runtime_lock"], f"{distribution} runtime helper")
    if Path(helper["path"]) != Path(__file__).resolve(strict=True) or helper["sha256"] != sha256_file(Path(__file__)):
        raise ValueError(f"{distribution} runtime helper role differs from executing implementation")
    manifest_binding, manifest = read_bound_json(
        roles[f"{prefix}_package_content_manifest"], f"{distribution} package manifest", PACKAGE_MANIFEST_SCHEMA
    )
    inspector = native_dependency_inspector_identity(
        roles[f"{prefix}_native_dependency_inspector"],
        roles[f"{prefix}_native_dependency_inspector_interpreter"],
        distribution,
    )
    native = verify_file_binding(roles[f"{prefix}_native_extension"], f"{distribution} native extension")
    libraries = native_library_closure(Path(native["path"]), inspector, distribution)
    declared_libraries = sorted(
        (
            verify_file_binding(binding, f"{distribution} native library role")
            for role, binding in roles.items()
            if role.startswith(f"{prefix}_native_library_")
        ),
        key=lambda item: item["path"],
    )
    if libraries != declared_libraries:
        raise ValueError(f"{distribution} native-library closure differs from code manifest")
    identity = {
        "distribution": distribution,
        "version": version,
        "environment_closure": manifest.get("environment_closure"),
        "wheelhouse_closure": manifest.get("wheelhouse_closure"),
        "package_root": manifest.get("package_root"),
        "package_manifest": manifest_binding,
        "wheel": manifest.get("wheel"),
        "install_receipt": manifest.get("install_receipt"),
        "init": verify_file_binding(roles[f"{prefix}_runtime"], f"{distribution} runtime"),
        "native_extension": native,
        "native_library_closure": libraries,
        "native_dependency_inspector": inspector,
        "runtime_helper": helper,
    }
    verified, _manifest, _files = verify_declared_runtime_identity(identity, distribution, version)
    return verified


def verify_preprocessing_parity_certificate(
    value: Any, environment_closure: dict[str, Any]
) -> dict[str, str]:
    binding, payload = read_bound_json(
        value, "TorchVision preprocessing parity certificate", PARITY_CERTIFICATE_SCHEMA
    )
    expected_keys = {
        "schema", "status", "site", "environment_root", "torch_version",
        "torchvision_version", "device", "runner", "input_image", "tests",
    }
    if (
        set(payload) != expected_keys
        or payload.get("status") != "passed"
        or payload.get("site") != "Hippasus"
        or payload.get("environment_root") != environment_closure["root"]
        or payload.get("torch_version") != "2.8.0+cu128"
        or payload.get("torchvision_version") != "0.23.0+cu128"
        or payload.get("device") != "cpu"
    ):
        raise ValueError("TorchVision preprocessing parity certificate identity differs")
    runner = verify_file_binding(payload.get("runner"), "parity certificate runner")
    image = verify_file_binding(payload.get("input_image"), "parity certificate input image", sized=True)
    environment_root = Path(environment_closure["root"])
    if environment_root not in Path(image["path"]).parents:
        raise ValueError("parity certificate image is outside the sealed evaluator environment")
    tests = payload.get("tests")
    required_names = {"raft_large_C_T_SKHT_V2"}
    if not isinstance(tests, list) or len(tests) != len(required_names):
        raise ValueError("parity certificate must contain exactly the frozen RAFT preprocessing case")
    names: set[str] = set()
    for index, item in enumerate(tests):
        if (
            not isinstance(item, dict)
            or set(item) != {
                "name", "shape", "dtype", "locked_output_sha256",
                "reference_output_sha256", "max_abs_error", "mean_abs_error",
            }
            or item.get("name") in names
            or item.get("name") not in required_names
            or not isinstance(item.get("shape"), list)
            or not item["shape"]
            or not all(isinstance(size, int) and not isinstance(size, bool) and size > 0 for size in item["shape"])
            or item.get("dtype") != "float32"
            or require_sha256(item.get("locked_output_sha256"), f"parity.tests[{index}].locked")
            != require_sha256(item.get("reference_output_sha256"), f"parity.tests[{index}].reference")
            or item.get("max_abs_error") != 0.0
            or item.get("mean_abs_error") != 0.0
        ):
            raise ValueError("TorchVision preprocessing parity result differs from the fixed reference")
        names.add(item["name"])
    if names != required_names or tests != sorted(tests, key=lambda item: item["name"]):
        raise ValueError("TorchVision preprocessing parity cases are incomplete or non-canonical")
    # The runner is part of the content-addressed source closure rather than
    # the environment tree, while the fixed image is part of the environment.
    if Path(runner["path"]).stat().st_mode & 0o222:
        raise ValueError("parity runner is writable")
    return binding


def verify_environment_fingerprint_payload(payload: Any, site: str) -> dict[str, Any]:
    expected_keys = {
        "schema", "site", "offline_runtime", "network_access", "offline_guards",
        "python_executable", "python_version", "pip_freeze_sha256",
        "container_or_environment_identity", "python_packages", "environment_closure",
        "wheelhouse_closure", "torchvision_preprocessing_parity_certificate",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("environment fingerprint schema/fields differ")
    if (
        payload.get("schema") != ENVIRONMENT_FINGERPRINT_SCHEMA
        or payload.get("site") != site
        or payload.get("offline_runtime") is not True
        or payload.get("network_access") != "disabled"
        or payload.get("python_version") != "3.10.20"
        or not isinstance(payload.get("container_or_environment_identity"), str)
        or not payload["container_or_environment_identity"]
    ):
        raise ValueError("environment fingerprint site/runtime identity differs")
    guards = payload.get("offline_guards")
    if guards != {
        "GEOMETRY_EVAL_NETWORK": "disabled",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }:
        raise ValueError("environment fingerprint offline guards differ")
    environment = verify_sealed_tree_closure(
        payload["environment_closure"], "evaluator environment", Path("/vol/dissolve/yz10325/evaluator_envs")
    )
    wheelhouse = verify_sealed_tree_closure(
        payload["wheelhouse_closure"], "evaluator wheelhouse", Path("/vol/dissolve/yz10325/evaluator_wheelhouses")
    )
    python = verify_file_binding(payload.get("python_executable"), "environment Python")
    if Path(environment["root"]) not in Path(python["path"]).parents:
        raise ValueError("locked Python executable escapes the sealed environment")
    require_sha256(payload.get("pip_freeze_sha256"), "environment pip_freeze_sha256")
    packages = payload.get("python_packages")
    if (
        not isinstance(packages, dict)
        or packages.get("torch") != "2.8.0+cu128"
        or packages.get("torchvision") != "0.23.0+cu128"
        or not all(isinstance(name, str) and name and isinstance(version, str) and version for name, version in packages.items())
    ):
        raise ValueError("environment package identity lacks the exact Torch/TorchVision pair")
    parity = verify_preprocessing_parity_certificate(
        payload.get("torchvision_preprocessing_parity_certificate"), environment
    )
    normalized = dict(payload)
    normalized["python_executable"] = python
    normalized["environment_closure"] = environment
    normalized["wheelhouse_closure"] = wheelhouse
    normalized["torchvision_preprocessing_parity_certificate"] = parity
    return normalized
