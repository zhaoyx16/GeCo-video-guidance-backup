#!/usr/bin/env python3
"""Mandatory pre-worker revalidation of a locked Hippasus evaluator environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

def load_exact_sibling(path: Path, expected_sha256: str, module_name: str) -> types.ModuleType:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
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
        if digest.hexdigest() != expected_sha256 or (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns
        ) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ) or (after.st_dev, after.st_ino) != (named_after.st_dev, named_after.st_ino):
            raise RuntimeError(f"{module_name} helper differs from its reviewed SHA")
    finally:
        os.close(descriptor)
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    exec(compile(b"".join(chunks), str(path), "exec"), module.__dict__)
    return module


_PACKAGE_RUNTIME_HELPER = Path(__file__).resolve(strict=True).with_name("python_package_runtime_lock.py")
_package_runtime_module = load_exact_sibling(
    _PACKAGE_RUNTIME_HELPER,
    "0a344fff0b0daf534ec46d60acb57cc041ba3527e7aea40041261b201efd0211",
    "locked_python_package_runtime_lock",
)
package_runtime_identity = _package_runtime_module.package_runtime_identity
verify_environment_fingerprint_payload = _package_runtime_module.verify_environment_fingerprint_payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_clean_dynamic_loader_environment() -> None:
    injected = sorted(name for name, value in os.environ.items() if name.startswith("LD_") and value)
    if injected:
        raise ValueError(f"dynamic-loader environment must be empty for sealed OpenCV execution: {injected}")


def native_dependency_inspector_identity(inspector: dict[str, str], interpreter: dict[str, str]) -> dict[str, str]:
    inspector_path = Path(inspector["path"]).resolve(strict=True)
    interpreter_path = Path(interpreter["path"]).resolve(strict=True)
    if sha256_file(inspector_path) != inspector["sha256"] or sha256_file(interpreter_path) != interpreter["sha256"]:
        raise ValueError("OpenCV dependency inspector or interpreter changed after sealing")
    first_line = inspector_path.open("rb").readline().decode("utf-8", errors="strict").strip()
    if not first_line.startswith("#!") or Path(first_line[2:].split(maxsplit=1)[0]).resolve(strict=True) != interpreter_path:
        raise ValueError("OpenCV dependency inspector does not use its sealed interpreter")
    return {"path": str(inspector_path), "sha256": inspector["sha256"], "interpreter_path": str(interpreter_path), "interpreter_sha256": interpreter["sha256"]}


def native_library_closure(native: Path, inspector: dict[str, str]) -> list[dict[str, str]]:
    require_clean_dynamic_loader_environment()
    result = subprocess.run(
        [inspector["path"], str(native)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env={"PATH": "/usr/bin:/bin"},
    )
    libraries: dict[str, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text or "linux-vdso" in text:
            continue
        if "not found" in text:
            raise ValueError(f"OpenCV native dependency is unresolved: {text}")
        tokens = text.replace("=>", " ").split()
        candidate = next((token for token in tokens if token.startswith("/")), None)
        if candidate is None:
            raise ValueError(f"could not resolve OpenCV native dependency: {text}")
        path = Path(candidate).resolve(strict=True)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"OpenCV native dependency is not a sealed regular file: {path}")
        libraries[str(path)] = {"path": str(path), "sha256": sha256_file(path)}
    if not libraries:
        raise ValueError("ldd returned no sealed OpenCV native dependencies")
    return [libraries[path] for path in sorted(libraries)]


def opencv_runtime_identity(inspector: dict[str, str], interpreter: dict[str, str]) -> dict[str, Any]:
    require_clean_dynamic_loader_environment()
    import cv2

    package_root = Path(cv2.__file__).resolve(strict=True).parent
    candidates = sorted(
        {path.resolve(strict=True) for pattern in ("*.so", "*.pyd", "*.dylib") for path in package_root.glob(pattern) if path.is_file()}
    )
    if len(candidates) != 1:
        raise ValueError(f"could not identify one native OpenCV implementation below {package_root}: {candidates}")
    native = candidates[0]
    inspector_identity = native_dependency_inspector_identity(inspector, interpreter)
    return {
        "module_version": str(cv2.__version__),
        "native_extension_path": str(native),
        "native_extension_sha256": sha256_file(native),
        "build_information_sha256": hashlib.sha256(cv2.getBuildInformation().encode("utf-8")).hexdigest(),
        "native_library_closure": native_library_closure(native, inspector_identity),
        "native_dependency_inspector": inspector_identity,
    }


def runtime_identities_for_metric(name: str, roles: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    identities: dict[str, dict[str, Any]] = {
        "torch": package_runtime_identity(roles, "torch", "torch", "2.8.0+cu128")
    }
    if name == "relative_total_motion_percent":
        identities["opencv"] = opencv_runtime_identity(
            roles["opencv_native_dependency_inspector"],
            roles["opencv_native_dependency_inspector_interpreter"],
        )
    if name in {"relative_total_motion_percent", "vbench_quality"}:
        identities["torchvision"] = package_runtime_identity(
            roles, "torchvision", "torchvision", "0.23.0+cu128"
        )
    return identities


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
    fd, name = tempfile.mkstemp(prefix=".evaluator-preflight-", suffix=".tmp", dir=str(path.parent))
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
        raise ValueError(f"read-only regular JSON required: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def verify_binding(binding: Any, label: str) -> Path:
    if not isinstance(binding, dict) or not isinstance(binding.get("path"), str) or not isinstance(binding.get("sha256"), str) or len(binding["sha256"]) != 64:
        raise ValueError(f"{label} binding malformed")
    int(binding["sha256"], 16)
    path = Path(binding["path"])
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222 or sha256_file(path) != binding["sha256"]:
        raise ValueError(f"{label} changed after evaluator lock publication")
    return path


def verify_manifest(binding: Any, label: str, expected_schema: str) -> None:
    path = verify_binding(binding, label)
    payload = read_json(path)
    if payload.get("schema") != expected_schema or payload.get("site") != "Hippasus" or payload.get("offline_ready") is not True or payload.get("load_closure_complete") is not True or payload.get("seal_mode") != "file_level_readonly_with_preworker_rehash":
        raise ValueError(f"{label} identity/closure mismatch")
    trace_path = verify_binding(payload.get("runtime_load_trace"), f"{label}.runtime_load_trace")
    trace = read_json(trace_path)
    if trace.get("schema") != "geometry-selection-offline-load-trace-v1" or trace.get("site") != "Hippasus" or trace.get("offline_runtime") is not True or trace.get("network_access") != "disabled":
        raise ValueError(f"{label} load trace identity mismatch")
    raw_root = payload.get("trusted_root")
    if not isinstance(raw_root, str) or not raw_root:
        raise ValueError(f"{label} lacks a trusted root")
    trusted_root = Path(raw_root)
    if trusted_root.is_symlink() or not trusted_root.is_dir() or trusted_root.stat().st_mode & 0o222:
        raise ValueError(f"{label} trusted root is writable or invalid")
    trusted_root = trusted_root.resolve(strict=True)
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{label} has no content files")
    roles: set[str] = set()
    for index, item in enumerate(files):
        if not isinstance(item, dict) or not isinstance(item.get("role"), str) or not item["role"] or item["role"] in roles:
            raise ValueError(f"{label} file roles malformed")
        roles.add(item["role"])
        content_path = verify_binding(item, f"{label}.files[{index}]")
        if trusted_root not in content_path.parents:
            raise ValueError(f"{label} content file escapes its trusted root")
        for parent in (trusted_root, *content_path.parents):
            if parent == trusted_root or trusted_root in parent.parents:
                if parent.stat().st_mode & 0o222:
                    raise ValueError(f"{label} content has writable trusted-root ancestor")
    loaded = trace.get("loaded_files")
    normalized_loaded = sorted(
        [{"role": item.get("role"), "path": item.get("path"), "sha256": item.get("sha256")} for item in loaded if isinstance(item, dict)] if isinstance(loaded, list) else [],
        key=lambda item: str(item.get("role")),
    )
    expected = sorted([{"role": item["role"], "path": item["path"], "sha256": item["sha256"]} for item in files], key=lambda item: item["role"])
    if normalized_loaded != expected:
        raise ValueError(f"{label} load trace does not match its content closure")


def rehash_sealed_source_snapshot(bindings: list[dict[str, str]], locked: Any) -> dict[str, Any]:
    """Run the source verifier in this worker's isolated environment.

    Source code for all five metrics lives below one sealed snapshot.  Merely
    hashing the individual code-manifest entries is insufficient: the worker
    must also prove that SOURCE_READY still binds a full immutable-tree rehash
    immediately before loading any adapter.
    """
    if not bindings or any(set(item) != {"path", "sha256"} for item in bindings):
        raise ValueError("source snapshot verifier bindings are malformed")
    if not isinstance(locked, dict) or set(locked) != {"root", "snapshot", "ready", "full_rehash_receipt", "file_count"}:
        raise ValueError("evaluator lock lacks its exact source snapshot closure")
    unique = {(item["path"], item["sha256"]) for item in bindings}
    if len(unique) != 1:
        raise ValueError("all metrics must bind one exact source snapshot verifier")
    verifier_path_text, verifier_sha256 = next(iter(unique))
    verifier = Path(verifier_path_text)
    if verifier.is_symlink() or not verifier.is_file() or verifier.stat().st_mode & 0o222 or sha256_file(verifier) != verifier_sha256:
        raise ValueError("source snapshot verifier differs from its code-manifest binding")
    source_root = verifier.resolve(strict=True).parents[2]
    expected_verifier = source_root / "protocol" / "metric_protocol" / "verify_evaluator_source_snapshot.py"
    if verifier.resolve(strict=True) != expected_verifier or source_root.is_symlink() or not source_root.is_dir() or source_root.stat().st_mode & 0o222:
        raise ValueError("source snapshot verifier is not located below one sealed evaluator root")
    if str(source_root) != locked["root"]:
        raise ValueError("runtime source root differs from the evaluator-lock snapshot root")
    for label, name in (("snapshot", "SOURCE_SNAPSHOT.json"), ("ready", "SOURCE_READY.json")):
        value = locked.get(label)
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise ValueError(f"locked source {label} binding is malformed")
        path = Path(value["path"])
        if path.resolve(strict=True) != source_root / name or sha256_file(path) != value["sha256"]:
            raise ValueError(f"locked source {label} changed")
    receipt_binding = locked.get("full_rehash_receipt")
    if not isinstance(receipt_binding, dict) or set(receipt_binding) != {"path", "sha256"}:
        raise ValueError("locked source full-rehash receipt binding is malformed")
    receipt_path = Path(receipt_binding["path"])
    if receipt_path.is_symlink() or not receipt_path.is_file() or receipt_path.stat().st_mode & 0o222 or sha256_file(receipt_path) != receipt_binding["sha256"]:
        raise ValueError("locked source full-rehash receipt changed")
    receipt_payload = read_json(receipt_path)
    if receipt_payload.get("schema") != "geometry-selection-hippasus-evaluator-source-snapshot-rehash-receipt-v1" or receipt_payload.get("status") != "full_rehash_verified" or receipt_payload.get("source_snapshot_root") != str(source_root) or receipt_payload.get("source_snapshot_sha256") != locked["snapshot"]["sha256"] or receipt_payload.get("source_ready_sha256") != locked["ready"]["sha256"] or receipt_payload.get("file_count") != locked["file_count"]:
        raise ValueError("locked source full-rehash receipt identity differs")
    environment = {
        "PATH": "/usr/bin:/bin",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "GEOMETRY_EVAL_NETWORK": "disabled",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-I", str(verifier), "--root", str(source_root), "--expected-self-sha256", verifier_sha256],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        proof = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("source snapshot verifier did not emit its structured rehash proof") from error
    snapshot = source_root / "SOURCE_SNAPSHOT.json"
    ready = source_root / "SOURCE_READY.json"
    if (
        not isinstance(proof, dict)
        or proof.get("root") != str(source_root)
        or proof.get("verified") is not True
        or not isinstance(proof.get("file_count"), int)
        or proof["file_count"] <= 0
        or proof.get("snapshot_sha256") != locked["snapshot"]["sha256"]
        or proof["file_count"] != locked["file_count"]
    ):
        raise ValueError("source snapshot rehash proof differs from the sealed tree")
    return {
        "source_root": str(source_root),
        "verifier": {"path": str(verifier.resolve(strict=True)), "sha256": verifier_sha256},
        "source_snapshot": {"path": str(snapshot), "sha256": sha256_file(snapshot)},
        "source_ready": {"path": str(ready), "sha256": sha256_file(ready)},
        "full_rehash_receipt": receipt_binding,
        "file_count": proof["file_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator-lock", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = read_json(args.evaluator_lock)
    if lock.get("schema") != "geometry-selection-evaluator-lock-v2" or lock.get("site") != "Hippasus" or lock.get("offline_runtime") is not True:
        raise ValueError("unexpected evaluator-lock identity")
    environment_path = verify_binding(lock.get("environment_fingerprint"), "environment")
    environment = verify_environment_fingerprint_payload(read_json(environment_path), "Hippasus")
    guards = environment.get("offline_guards")
    if environment.get("site") != "Hippasus" or environment.get("offline_runtime") is not True or environment.get("network_access") != "disabled" or not isinstance(guards, dict) or guards.get("HF_HUB_OFFLINE") != "1" or guards.get("TRANSFORMERS_OFFLINE") != "1" or guards.get("GEOMETRY_EVAL_NETWORK") != "disabled":
        raise ValueError("evaluator environment no longer guarantees offline network-disabled execution")
    python_path = verify_binding(environment.get("python_executable"), "environment.python_executable")
    if Path(sys.executable).resolve(strict=True) != python_path.resolve(strict=True):
        raise ValueError("verifier interpreter differs from the locked Python executable")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1" or os.environ.get("GEOMETRY_EVAL_NETWORK") != "disabled":
        raise ValueError("current worker environment does not enforce the locked offline/network guards")
    self_namespace = os.readlink("/proc/self/ns/net")
    init_namespace = os.readlink("/proc/1/ns/net")
    interfaces = sorted(name for _, name in __import__("socket").if_nameindex())
    if self_namespace == init_namespace or interfaces != ["lo"]:
        raise ValueError("worker is not running in an isolated loopback-only network namespace")
    expected_freeze = environment.get("pip_freeze_sha256")
    if not isinstance(expected_freeze, str) or len(expected_freeze) != 64:
        raise ValueError("environment pip-freeze SHA malformed")
    int(expected_freeze, 16)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False, env={"PATH": os.environ.get("PATH", ""), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "GEOMETRY_EVAL_NETWORK": "disabled", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INDEX": "1"})
    if hashlib.sha256(freeze.stdout).hexdigest() != expected_freeze:
        raise ValueError("runtime Python package environment differs from the locked fingerprint")
    verify_binding(lock.get("metric_protocol"), "protocol")
    verify_binding(lock.get("schedule"), "schedule")
    decoder_path = verify_binding(lock.get("decoder_binary"), "decoder")
    raw_decoder_root = lock.get("decoder_trusted_root")
    if not isinstance(raw_decoder_root, str) or not raw_decoder_root:
        raise ValueError("decoder trusted root missing")
    decoder_root = Path(raw_decoder_root)
    if decoder_root.is_symlink() or not decoder_root.is_dir() or decoder_root.stat().st_mode & 0o222:
        raise ValueError("decoder trusted root is writable or invalid")
    decoder_root = decoder_root.resolve(strict=True)
    if decoder_root not in decoder_path.parents:
        raise ValueError("decoder escapes its trusted root")
    for parent in (decoder_root, *decoder_path.parents):
        if parent == decoder_root or decoder_root in parent.parents:
            if parent.stat().st_mode & 0o222:
                raise ValueError("decoder has writable trusted-root ancestor")
    metrics = lock.get("metrics")
    if not isinstance(metrics, dict) or len(metrics) != 5:
        raise ValueError("evaluator lock lacks five metrics")
    parity_certificate = read_json(Path(environment["torchvision_preprocessing_parity_certificate"]["path"]))
    parity_runner = parity_certificate.get("runner")
    if not isinstance(parity_runner, dict):
        raise ValueError("environment parity certificate lacks its runner binding")
    manifest_bindings = []
    launcher_shas: set[str] = set()
    input_verifier_shas: set[str] = set()
    evaluator_verifier_shas: set[str] = set()
    source_snapshot_verifier_bindings: list[dict[str, str]] = []
    for name, metric in metrics.items():
        if not isinstance(metric, dict):
            raise ValueError(f"metric binding malformed: {name}")
        verify_binding(metric.get("adapter_entrypoint"), f"{name}.adapter_entrypoint")
        verify_binding(metric.get("adapter_common"), f"{name}.adapter_common")
        verify_binding(metric.get("core_evaluator"), f"{name}.core_evaluator")
        verify_manifest(metric.get("code_manifest"), f"{name}.code_manifest", "geometry-selection-code-content-manifest-v1")
        verify_manifest(metric.get("weight_manifest"), f"{name}.weight_manifest", "geometry-selection-weight-content-manifest-v1")
        code_manifest = read_json(Path(metric["code_manifest"]["path"]))
        if not any(item.get("role") == "metric_adapter_common" and {"path": item.get("path"), "sha256": item.get("sha256")} == metric["adapter_common"] for item in code_manifest["files"]):
            raise ValueError(f"{name} adapter_common is not bound by its code manifest")
        code_manifest = read_json(Path(metric["code_manifest"]["path"]))
        role_to_file = {
            item["role"]: {"path": item["path"], "sha256": item["sha256"]}
            for item in code_manifest["files"]
        }
        if role_to_file.get("torchvision_preprocessing_parity_runner") != parity_runner:
            raise ValueError(f"{name} parity runner differs from the environment certificate")
        if metric.get("runtime_identities") != runtime_identities_for_metric(name, role_to_file):
            raise ValueError(f"{name} runtime implementation differs from evaluator lock")
        for distribution, identity in metric["runtime_identities"].items():
            if distribution in {"torch", "torchvision"} and (
                identity.get("environment_closure") != environment["environment_closure"]
                or identity.get("wheelhouse_closure") != environment["wheelhouse_closure"]
            ):
                raise ValueError(f"{name} {distribution} runtime differs from the frozen evaluator environment")
        launcher_shas.update(item["sha256"] for item in code_manifest["files"] if item.get("role") == "guarded_worker_launcher")
        input_verifier_shas.update(item["sha256"] for item in code_manifest["files"] if item.get("role") == "input_preflight_verifier")
        evaluator_verifier_shas.update(item["sha256"] for item in code_manifest["files"] if item.get("role") == "evaluator_preflight_verifier")
        source_snapshot_verifier_bindings.extend(
            {"path": item["path"], "sha256": item["sha256"]}
            for item in code_manifest["files"]
            if item.get("role") == "evaluator_source_snapshot_verifier"
        )
        manifest_bindings.extend([metric["code_manifest"], metric["weight_manifest"]])
    if len(launcher_shas) != 1 or len(input_verifier_shas) != 1 or len(evaluator_verifier_shas) != 1:
        raise ValueError("evaluator content manifests do not bind one exact guarded/preflight verifier set")
    source_snapshot_rehash = rehash_sealed_source_snapshot(source_snapshot_verifier_bindings, lock.get("source_snapshot_closure"))
    shared = lock.get("shared_evaluator_identity_sha256")
    if not isinstance(shared, str) or len(shared) != 64:
        raise ValueError("shared evaluator identity malformed")
    int(shared, 16)
    payload = {"schema": "geometry-selection-evaluator-preflight-receipt-v1", "site": "Hippasus", "integrity_guarantee": "file_level_readonly_with_immediately_before_worker_full_rehash", "evaluator_lock": binding(args.evaluator_lock), "shared_evaluator_identity_sha256": shared, "verifier_python_executable": str(Path(sys.executable).resolve(strict=True)), "network_namespace": self_namespace, "network_interfaces": interfaces, "guarded_launcher_sha256": next(iter(launcher_shas)), "input_preflight_verifier_sha256": next(iter(input_verifier_shas)), "evaluator_preflight_verifier_sha256": next(iter(evaluator_verifier_shas)), "source_snapshot_rehash": source_snapshot_rehash, "content_manifests": sorted(manifest_bindings, key=lambda item: (item["path"], item["sha256"]))}
    safe_output(args.output, args.derived_root, [args.evaluator_lock])
    publish(args.output, payload)
    print(json.dumps({"evaluator_lock_sha256": payload["evaluator_lock"]["sha256"], "shared_evaluator_identity_sha256": shared, "output": str(args.output), "verified": True}, sort_keys=True))


if __name__ == "__main__":
    main()
