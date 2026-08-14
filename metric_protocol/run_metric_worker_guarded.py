#!/usr/bin/env python3
"""Run one locked metric in an isolated Python process and seal its receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
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

_TRACE_VALIDATION_HELPER = Path(__file__).resolve(strict=True).with_name("python_runtime_trace_validation.py")
_trace_validation_module = load_exact_sibling(
    _TRACE_VALIDATION_HELPER,
    "9e2a973b477ad8052868b0389ae312acfb4a2c3d3287b3b5c97278d58689ccbc",
    "locked_python_runtime_trace_validation",
)
verify_package_runtime_trace = _trace_validation_module.verify_package_runtime_trace


FORBIDDEN_ENV = {
    "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE",
    "PYTHONBREAKPOINT", "PYTHONINSPECT", "LD_PRELOAD", "LD_AUDIT",
}
PASSTHROUGH_ENV = {
    "HOME", "TMPDIR", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
    "CUDA_HOME", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_HOME",
}
INDEPENDENT_LRE_CODE_ROLES = {
    "metric_adapter_common",
    "independent_lre_reprojection",
    "independent_lre_megasam_runner",
    "independent_lre_runtime_trace",
    "megasam_camera_tracking_runtime",
    "megasam_cvd_runtime",
    "ufm_runtime",
    "lre_synthetic_geometry_certificate",
}
LOCKED_WORKER_CODE_ROLES = {
    "metric_adapter_entrypoint",
    "guarded_worker_launcher",
    "input_preflight_verifier",
    "evaluator_preflight_verifier",
    "evaluator_source_snapshot_verifier",
}
INDEPENDENT_LRE_FORBIDDEN_TOKENS = ("vgg" + "t", "vgg" + "t" + "_" + "omega", "vgg" + "t" + "-" + "omega")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


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


def binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)}


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
        raise ValueError(f"read-only regular JSON required: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def validate_fresh_output(path: Path, derived_root: Path, top_level: str) -> Path:
    if derived_root.is_symlink() or not derived_root.is_dir():
        raise ValueError("derived root must be an existing regular directory")
    derived = derived_root.resolve(strict=True)
    if tuple(derived.parts[-3:]) != ("outputs", "geometry-selection", "hippasus_evaluation") or "validation_wan_candidates" in derived.parts:
        raise ValueError("derived root is not the approved Hippasus evaluator root")
    if path.exists() or path.is_symlink() or path.parent.is_symlink() or not path.parent.is_dir():
        raise FileExistsError(f"fresh regular output parent required: {path}")
    target = path.parent.resolve(strict=True) / path.name
    if target == derived or derived not in target.parents or target.relative_to(derived).parts[0] != top_level:
        raise ValueError(f"output must be fresh below {top_level}/ in the derived root")
    return target


def publish(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing output: {path}")
    fd, name = tempfile.mkstemp(prefix=".metric-worker-receipt-", suffix=".tmp", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_bytes(payload) + b"\n")
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def open_component_once(path: Path) -> tuple[int, os.stat_result, dict[str, Any], dict[str, str]]:
    """Hold one immutable component inode through worker-receipt publication."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222:
            raise ValueError("metric component inode must be a read-only regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("metric component must contain a JSON object")
        path_info = os.stat(path, follow_symlinks=False)
        if (path_info.st_dev, path_info.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError("metric component pathname changed while it was opened")
        return descriptor, info, value, {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    except Exception:
        os.close(descriptor)
        raise


def require_held_path_identity(path: Path, descriptor: int, original: os.stat_result) -> None:
    current_fd = os.fstat(descriptor)
    current_path = os.stat(path, follow_symlinks=False)
    expected = (original.st_dev, original.st_ino, original.st_size)
    if (
        (current_fd.st_dev, current_fd.st_ino, current_fd.st_size) != expected
        or (current_path.st_dev, current_path.st_ino, current_path.st_size) != expected
        or not stat.S_ISREG(current_path.st_mode)
        or current_path.st_mode & 0o222
        or path.is_symlink()
    ):
        raise ValueError("metric component pathname/inode changed before receipt publication")


def isolated_environment() -> dict[str, str]:
    inherited = dict(os.environ)
    dangerous = sorted(name for name in FORBIDDEN_ENV if inherited.get(name))
    if dangerous:
        raise ValueError(f"forbidden runtime environment injection: {dangerous}")
    environment = {name: inherited[name] for name in PASSTHROUGH_ENV if inherited.get(name)}
    environment.update({
        "PATH": "/usr/bin:/bin",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "GEOMETRY_EVAL_NETWORK": "disabled",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return environment


def require_exact_adapter_arguments(tokens: list[str], expected: dict[str, Path]) -> list[str]:
    """Allow only the guard-bound adapter arguments plus a single device choice.

    The worker receipt must attest the exact input lock that reaches the adapter;
    accepting an arbitrary trailing command would otherwise permit a valid
    preflight to be paired with unrelated score inputs.
    """
    seen: dict[str, str] = {}
    canonical: list[str] = []
    index = 0
    while index < len(tokens):
        option = tokens[index]
        if option not in {*expected, "--device"} or index + 1 >= len(tokens):
            raise ValueError(f"unrecognised or incomplete guarded adapter argument: {option!r}")
        if option in seen:
            raise ValueError(f"duplicate guarded adapter argument: {option}")
        value = tokens[index + 1]
        if option in expected:
            # --output must be a new file by contract, so it cannot be
            # strict-resolved yet.  Every other bound input must already
            # exist and resolve without traversing a symlink.
            strict = option != "--output"
            candidate = Path(value).resolve(strict=strict)
            expected_path = expected[option].resolve(strict=strict)
            if candidate != expected_path:
                raise ValueError(f"guarded adapter argument {option} differs from its locked binding")
            canonical.extend([option, str(expected_path)])
        elif not value or value.startswith("-"):
            raise ValueError("guarded adapter --device must be a nonempty device selector")
        else:
            canonical.extend([option, value])
        seen[option] = value
        index += 2
    if set(seen) - {"--device"} != set(expected):
        missing = sorted(set(expected) - set(seen))
        raise ValueError(f"guarded adapter command misses required binding(s): {missing}")
    return canonical


def require_independent_lre_runtime_closure(
    metric_id: str,
    role_to_file: dict[str, dict[str, str]],
    adapter_entrypoint: Path,
    adapter_common: Path,
    core_evaluator: Path,
) -> None:
    """Reject an accidental selector-model dependency before the LRE worker loads.

    The formal source snapshot contains GeCo's VGGT code for the separate
    GeCo-Fused metric.  LRE therefore needs an executable per-process guard,
    not merely a statement in the protocol: its locked code manifest must be
    exactly the independent MegaSaM/UFM closure and none of its executable
    files may mention a VGGT/Omega import or path.
    """
    if metric_id != "long_range_reprojection_error":
        return
    if set(role_to_file) != INDEPENDENT_LRE_CODE_ROLES | LOCKED_WORKER_CODE_ROLES:
        raise ValueError("Independent LRE code roles differ from the MegaSaM/UFM-only contract")
    candidates = {
        "adapter_entrypoint": (adapter_entrypoint, os.environ["GEOMETRY_EVAL_LOCKED_ADAPTER_ENTRYPOINT_SHA256"]),
        "adapter_common": (adapter_common, os.environ["GEOMETRY_EVAL_LOCKED_ADAPTER_COMMON_SHA256"]),
        "core_evaluator": (core_evaluator, os.environ["GEOMETRY_EVAL_LOCKED_CORE_EVALUATOR_SHA256"]),
        **{
            role: (Path(binding["path"]).resolve(strict=True), binding["sha256"])
            for role, binding in role_to_file.items()
            if role in INDEPENDENT_LRE_CODE_ROLES
        },
    }
    for label, (path, expected_sha256) in candidates.items():
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
            raise ValueError(f"Independent LRE executable path is unsafe: {label}")
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"Independent LRE executable SHA differs from the evaluator lock: {label}")
        if any(token in str(path).lower() for token in INDEPENDENT_LRE_FORBIDDEN_TOKENS):
            raise ValueError(f"Independent LRE executable path contains forbidden selector token: {label}")
        try:
            text = path.read_text(encoding="utf-8").lower()
        except UnicodeDecodeError as error:
            raise ValueError(f"Independent LRE executable is not UTF-8 source: {label}") from error
        if any(token in text for token in INDEPENDENT_LRE_FORBIDDEN_TOKENS):
            raise ValueError(f"Independent LRE executable imports or references a forbidden selector token: {label}")


def require_worker_source_rehash(evaluator_receipt: dict[str, Any], role_to_file: dict[str, dict[str, str]], evaluator_lock: dict[str, Any]) -> dict[str, Any]:
    binding = role_to_file.get("evaluator_source_snapshot_verifier")
    if not isinstance(binding, dict):
        raise ValueError("metric code manifest lacks the sealed source snapshot verifier")
    rehash = evaluator_receipt.get("source_snapshot_rehash")
    if not isinstance(rehash, dict):
        raise ValueError("evaluator preflight lacks an immediate source snapshot rehash")
    verifier = rehash.get("verifier")
    snapshot = rehash.get("source_snapshot")
    ready = rehash.get("source_ready")
    if (
        verifier != {"path": str(Path(binding["path"]).resolve(strict=True)), "sha256": binding["sha256"]}
        or not isinstance(rehash.get("source_root"), str)
        or not isinstance(rehash.get("file_count"), int)
        or rehash["file_count"] <= 0
        or not isinstance(snapshot, dict)
        or not isinstance(ready, dict)
        or not isinstance(snapshot.get("path"), str)
        or not isinstance(snapshot.get("sha256"), str)
        or not isinstance(ready.get("path"), str)
        or not isinstance(ready.get("sha256"), str)
    ):
        raise ValueError("evaluator source snapshot rehash proof differs from the locked verifier")
    source_root = Path(rehash["source_root"]).resolve(strict=True)
    if Path(snapshot["path"]).resolve(strict=True) != source_root / "SOURCE_SNAPSHOT.json" or Path(ready["path"]).resolve(strict=True) != source_root / "SOURCE_READY.json":
        raise ValueError("evaluator source snapshot rehash proof paths differ")
    if sha256_file(Path(snapshot["path"])) != snapshot["sha256"] or sha256_file(Path(ready["path"])) != ready["sha256"]:
        raise ValueError("evaluator source snapshot changed after its worker rehash")
    locked = evaluator_lock.get("source_snapshot_closure")
    if (
        not isinstance(locked, dict)
        or rehash.get("source_root") != locked.get("root")
        or snapshot != locked.get("snapshot")
        or ready != locked.get("ready")
        or rehash.get("full_rehash_receipt") != locked.get("full_rehash_receipt")
        or rehash.get("file_count") != locked.get("file_count")
    ):
        raise ValueError("worker source rehash differs from the evaluator-lock closure")
    return rehash


def verify_independent_lre_runtime_trace(component: dict[str, Any], source_root: Path) -> dict[str, Any]:
    details = component.get("details")
    trace = details.get("independent_runtime_load_trace") if isinstance(details, dict) else None
    if not isinstance(trace, dict):
        raise ValueError("Independent LRE component lacks its runtime load trace")
    expected = {"schema", "files", "forbidden_selector_tokens", "sha256"}
    if set(trace) != expected or trace.get("schema") != "geometry-selection-independent-lre-runtime-load-trace-v1":
        raise ValueError("Independent LRE runtime trace schema differs")
    if trace.get("forbidden_selector_tokens") != list(INDEPENDENT_LRE_FORBIDDEN_TOKENS) or not isinstance(trace.get("files"), list):
        raise ValueError("Independent LRE runtime trace selector-exclusion contract differs")
    unsigned = {key: trace[key] for key in expected - {"sha256"}}
    expected_sha = hashlib.sha256(canonical_bytes(unsigned) + b"\n").hexdigest()
    if trace.get("sha256") != expected_sha:
        raise ValueError("Independent LRE runtime trace SHA differs")
    files: set[str] = set()
    root = source_root.resolve(strict=True)
    for item in trace["files"]:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"} or not isinstance(item["path"], str) or not isinstance(item["sha256"], str):
            raise ValueError("Independent LRE runtime trace file entry is malformed")
        path = Path(item["path"])
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
            raise ValueError("Independent LRE runtime trace references an unsafe file")
        resolved = path.resolve(strict=True)
        if root not in resolved.parents or resolved.suffix != ".py" or str(resolved) in files or sha256_file(resolved) != item["sha256"]:
            raise ValueError("Independent LRE runtime trace file binding differs")
        files.add(str(resolved))
        text = resolved.read_text(encoding="utf-8").lower()
        if any(token in str(resolved).lower() or token in text for token in INDEPENDENT_LRE_FORBIDDEN_TOKENS):
            raise ValueError("Independent LRE runtime trace contains a selector dependency")
    if not files:
        raise ValueError("Independent LRE runtime trace is empty")
    return {"sha256": trace["sha256"], "file_count": len(files), "files": trace["files"]}


def verify_torchvision_component_trace(
    metric_id: str, component: dict[str, Any], expected_identity: Any
) -> dict[str, Any] | None:
    if metric_id not in {"relative_total_motion_percent", "vbench_quality"}:
        return None
    details = component.get("details")
    if not isinstance(details, dict) or not isinstance(expected_identity, dict):
        raise ValueError("TorchVision metric lacks locked component/runtime details")
    if metric_id == "relative_total_motion_percent":
        trace = details.get("torchvision_runtime")
    else:
        runtime = details.get("runtime_load_trace")
        sealed = runtime.get("sealed_imports") if isinstance(runtime, dict) else None
        torchvision = sealed.get("torchvision") if isinstance(sealed, dict) else None
        trace = torchvision.get("runtime_trace") if isinstance(torchvision, dict) else None
    return verify_package_runtime_trace(trace, expected_identity, "torchvision")


def verify_vbench_source_trace(component: dict[str, Any], source_root: Path) -> dict[str, Any]:
    details = component.get("details")
    runtime = details.get("runtime_load_trace") if isinstance(details, dict) else None
    trace = runtime.get("sealed_source_runtime_trace") if isinstance(runtime, dict) else None
    if not isinstance(trace, dict) or set(trace) != {"module_names", "files", "sha256"}:
        raise ValueError("VBench component lacks its final sealed-source runtime trace")
    module_names, files = trace.get("module_names"), trace.get("files")
    required = {"vbench", "clip", "pyiqa", "vision_transformer", "constant", "locked_vbench_quality_score"}
    if (
        not isinstance(module_names, list)
        or module_names != sorted(module_names)
        or len(module_names) != len(set(module_names))
        or not required.issubset(set(module_names))
        or not isinstance(files, list)
        or not files
        or trace.get("sha256") != hashlib.sha256(
            canonical_bytes({"module_names": module_names, "files": files})
        ).hexdigest()
    ):
        raise ValueError("VBench final sealed-source trace identity differs")
    root = source_root.resolve(strict=True)
    paths: list[str] = []
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("VBench final sealed-source trace entry is malformed")
        raw = Path(item["path"])
        info = raw.stat(follow_symlinks=False)
        resolved = raw.resolve(strict=True)
        if (
            raw.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or info.st_mode & 0o222
            or root not in resolved.parents
            or str(resolved) in paths
            or sha256_file(resolved) != item["sha256"]
        ):
            raise ValueError("VBench final sealed-source trace file differs")
        paths.append(str(resolved))
    if paths != sorted(paths):
        raise ValueError("VBench final sealed-source trace is non-canonical")
    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-lock", type=Path, required=True)
    parser.add_argument("--evaluator-lock", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--input-preflight-output", type=Path, required=True)
    parser.add_argument("--evaluator-preflight-output", type=Path, required=True)
    parser.add_argument("--metric-id", required=True)
    parser.add_argument("--metric-output", type=Path, required=True)
    parser.add_argument("--worker-receipt-output", type=Path, required=True)
    parser.add_argument("metric_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    require_clean_dynamic_loader_environment()
    if not args.metric_command or args.metric_command[0] != "--" or len(args.metric_command) < 4:
        raise ValueError("metric command must be '-- <locked-python> -I <locked-entrypoint> ...'")
    python_path = Path(args.metric_command[1]).resolve(strict=True)
    if python_path != Path(sys.executable).resolve(strict=True) or args.metric_command[2] != "-I":
        raise ValueError("guarded metric command must use the locked Python in isolated (-I) mode")
    metric_output = validate_fresh_output(args.metric_output, args.derived_root, "raw")
    worker_receipt_output = validate_fresh_output(args.worker_receipt_output, args.derived_root, "receipts")
    environment = isolated_environment()
    script_dir = Path(__file__).resolve().parent
    subprocess.run([sys.executable, "-I", str(script_dir / "verify_metric_input_lock.py"), "--input-lock", str(args.input_lock), "--evaluator-lock", str(args.evaluator_lock), "--derived-root", str(args.derived_root), "--output", str(args.input_preflight_output)], check=True, env=environment)
    subprocess.run([sys.executable, "-I", str(script_dir / "verify_evaluator_lock.py"), "--evaluator-lock", str(args.evaluator_lock), "--derived-root", str(args.derived_root), "--output", str(args.evaluator_preflight_output)], check=True, env=environment)
    input_receipt, evaluator_receipt = read_json(args.input_preflight_output), read_json(args.evaluator_preflight_output)
    evaluator_lock = read_json(args.evaluator_lock)
    metrics = evaluator_lock.get("metrics")
    metric = metrics.get(args.metric_id) if isinstance(metrics, dict) else None
    adapter_entrypoint = metric.get("adapter_entrypoint") if isinstance(metric, dict) else None
    core_evaluator = metric.get("core_evaluator") if isinstance(metric, dict) else None
    if not isinstance(adapter_entrypoint, dict) or not isinstance(adapter_entrypoint.get("path"), str) or not isinstance(adapter_entrypoint.get("sha256"), str):
        raise ValueError("metric id lacks a locked adapter entrypoint")
    if not isinstance(core_evaluator, dict) or not isinstance(core_evaluator.get("path"), str) or not isinstance(core_evaluator.get("sha256"), str):
        raise ValueError("metric id lacks a locked core evaluator")
    adapter_entrypoint_path = Path(adapter_entrypoint["path"]).resolve(strict=True)
    core_evaluator_path = Path(core_evaluator["path"]).resolve(strict=True)
    if Path(args.metric_command[3]).resolve(strict=True) != adapter_entrypoint_path or sha256_file(adapter_entrypoint_path) != adapter_entrypoint["sha256"]:
        raise ValueError("guarded metric command does not use the exact locked adapter entrypoint")
    if sha256_file(core_evaluator_path) != core_evaluator["sha256"]:
        raise ValueError("locked core evaluator changed after evaluator preflight")
    code_manifest = read_json(Path(metric["code_manifest"]["path"])) if isinstance(metric, dict) and isinstance(metric.get("code_manifest"), dict) and isinstance(metric["code_manifest"].get("path"), str) else None
    role_to_file = {
        item["role"]: {"path": item["path"], "sha256": item["sha256"]}
        for item in code_manifest.get("files", [])
        if isinstance(item, dict) and isinstance(item.get("role"), str) and isinstance(item.get("path"), str) and isinstance(item.get("sha256"), str)
    } if isinstance(code_manifest, dict) else {}
    runtime_identities = metric.get("runtime_identities") if isinstance(metric, dict) else None
    if not isinstance(runtime_identities, dict) or runtime_identities != runtime_identities_for_metric(args.metric_id, role_to_file):
        raise ValueError("runtime implementation differs from the locked evaluator identity")
    environment_binding = evaluator_lock.get("environment_fingerprint")
    environment_payload = read_json(Path(environment_binding["path"])) if isinstance(environment_binding, dict) and isinstance(environment_binding.get("path"), str) else None
    if not isinstance(environment_payload, dict):
        raise ValueError("evaluator lock lacks its environment fingerprint")
    for distribution, identity in runtime_identities.items():
        if distribution in {"torch", "torchvision"} and (
            identity.get("environment_closure") != environment_payload.get("environment_closure")
            or identity.get("wheelhouse_closure") != environment_payload.get("wheelhouse_closure")
        ):
            raise ValueError(f"{distribution} runtime differs from the evaluator environment fingerprint")
    source_snapshot_rehash = require_worker_source_rehash(evaluator_receipt, role_to_file, evaluator_lock)
    adapter_common = metric.get("adapter_common") if isinstance(metric, dict) else None
    if not isinstance(adapter_common, dict) or not isinstance(adapter_common.get("path"), str) or not isinstance(adapter_common.get("sha256"), str):
        raise ValueError("metric id lacks a locked shared adapter runtime")
    adapter_common_path = Path(adapter_common["path"]).resolve(strict=True)
    if sha256_file(adapter_common_path) != adapter_common["sha256"]:
        raise ValueError("locked shared adapter runtime changed after evaluator preflight")
    require_independent_lre_runtime_closure(
        args.metric_id,
        role_to_file,
        adapter_entrypoint_path,
        adapter_common_path,
        core_evaluator_path,
    )
    weight_manifest = metric.get("weight_manifest") if isinstance(metric, dict) else None
    schedule = evaluator_lock.get("schedule")
    decoder = evaluator_lock.get("decoder_binary")
    for label, value in (("weight manifest", weight_manifest), ("schedule", schedule), ("decoder", decoder)):
        if not isinstance(value, dict) or not isinstance(value.get("path"), str) or not isinstance(value.get("sha256"), str):
            raise ValueError(f"metric id lacks a locked {label}")
    weight_manifest_path = Path(weight_manifest["path"]).resolve(strict=True)
    schedule_path = Path(schedule["path"]).resolve(strict=True)
    decoder_path = Path(decoder["path"]).resolve(strict=True)
    if sha256_file(weight_manifest_path) != weight_manifest["sha256"] or sha256_file(schedule_path) != schedule["sha256"] or sha256_file(decoder_path) != decoder["sha256"]:
        raise ValueError("locked metric dependency changed after evaluator preflight")
    adapter_arguments = require_exact_adapter_arguments(
        args.metric_command[4:],
        {
            "--input-lock": args.input_lock,
            "--core-evaluator": core_evaluator_path,
            "--weight-manifest": weight_manifest_path,
            "--schedule": schedule_path,
            "--decoder": decoder_path,
            "--output": metric_output,
        },
    )
    canonical_adapter_command = [str(python_path), "-I", str(adapter_entrypoint_path), *adapter_arguments]
    launcher_sha = sha256_file(Path(__file__))
    input_verifier_sha = sha256_file(script_dir / "verify_metric_input_lock.py")
    evaluator_verifier_sha = sha256_file(script_dir / "verify_evaluator_lock.py")
    if evaluator_receipt.get("verifier_python_executable") != str(Path(sys.executable).resolve(strict=True)) or evaluator_receipt.get("network_namespace") != os.readlink("/proc/self/ns/net") or evaluator_receipt.get("guarded_launcher_sha256") != launcher_sha or evaluator_receipt.get("input_preflight_verifier_sha256") != input_verifier_sha or evaluator_receipt.get("evaluator_preflight_verifier_sha256") != evaluator_verifier_sha:
        raise ValueError("guard runtime differs from the verified evaluator runtime")
    environment.update({
        "GEOMETRY_EVAL_INPUT_PREFLIGHT_RECEIPT_SHA256": sha256_file(args.input_preflight_output),
        "GEOMETRY_EVAL_EVALUATOR_PREFLIGHT_RECEIPT_SHA256": sha256_file(args.evaluator_preflight_output),
        "GEOMETRY_EVAL_VERIFIED_PYTHON": evaluator_receipt["verifier_python_executable"],
        "GEOMETRY_EVAL_VERIFIED_NETWORK_NAMESPACE": evaluator_receipt["network_namespace"],
        "GEOMETRY_EVAL_GUARDED_LAUNCHER_SHA256": launcher_sha,
        "GEOMETRY_EVAL_METRIC_ID": args.metric_id,
        "GEOMETRY_EVAL_LOCKED_RUNTIME_IDENTITIES_JSON": json.dumps(runtime_identities, sort_keys=True, separators=(",", ":")),
        "GEOMETRY_EVAL_LOCKED_ADAPTER_ENTRYPOINT_PATH": str(adapter_entrypoint_path),
        "GEOMETRY_EVAL_LOCKED_ADAPTER_ENTRYPOINT_SHA256": adapter_entrypoint["sha256"],
        "GEOMETRY_EVAL_LOCKED_ADAPTER_COMMON_PATH": str(adapter_common_path),
        "GEOMETRY_EVAL_LOCKED_ADAPTER_COMMON_SHA256": adapter_common["sha256"],
        "GEOMETRY_EVAL_LOCKED_CORE_EVALUATOR_PATH": str(core_evaluator_path),
        "GEOMETRY_EVAL_LOCKED_CORE_EVALUATOR_SHA256": core_evaluator["sha256"],
        "GEOMETRY_EVAL_LOCKED_WEIGHT_MANIFEST_PATH": str(weight_manifest_path),
        "GEOMETRY_EVAL_LOCKED_WEIGHT_MANIFEST_SHA256": weight_manifest["sha256"],
        "GEOMETRY_EVAL_LOCKED_SCHEDULE_PATH": str(schedule_path),
        "GEOMETRY_EVAL_LOCKED_SCHEDULE_SHA256": schedule["sha256"],
        "GEOMETRY_EVAL_LOCKED_DECODER_BINARY": str(decoder_path),
        "GEOMETRY_EVAL_LOCKED_DECODER_SHA256": decoder["sha256"],
        "GEOMETRY_EVAL_INPUT_PREFLIGHT_RECEIPT_PATH": str(args.input_preflight_output.resolve(strict=True)),
        "GEOMETRY_EVAL_EVALUATOR_PREFLIGHT_RECEIPT_PATH": str(args.evaluator_preflight_output.resolve(strict=True)),
        "GEOMETRY_EVAL_LOCKED_INPUT_LOCK_PATH": str(args.input_lock.resolve(strict=True)),
        "GEOMETRY_EVAL_LOCKED_INPUT_LOCK_SHA256": sha256_file(args.input_lock),
        "GEOMETRY_EVAL_LOCKED_EVALUATOR_LOCK_PATH": str(args.evaluator_lock.resolve(strict=True)),
        "GEOMETRY_EVAL_LOCKED_EVALUATOR_LOCK_SHA256": sha256_file(args.evaluator_lock),
        "GEOMETRY_EVAL_METRIC_OUTPUT_PATH": str(metric_output),
    })
    # -I deliberately omits the script directory from sys.path.  Add only the
    # already hash-locked adapter directory inside the isolated interpreter so
    # its separately locked shared helper can be imported; no inherited path is
    # permitted.
    launcher = (
        "import runpy,sys;"
        f"sys.path.insert(0,{str(adapter_entrypoint_path.parent)!r});"
        f"sys.argv={[str(adapter_entrypoint_path), *adapter_arguments]!r};"
        f"runpy.run_path({str(adapter_entrypoint_path)!r},run_name='__main__')"
    )
    wrapper_sha256 = hashlib.sha256(launcher.encode("utf-8")).hexdigest()
    subprocess.run([str(python_path), "-I", "-c", launcher], check=True, env=environment)
    component_fd, component_info, component, component_binding = open_component_once(metric_output)
    if component.get("schema") != "geometry-selection-metric-component-v1" or component.get("metric_id") != args.metric_id or not isinstance(component.get("records"), list):
        raise ValueError("metric component schema, ID, or records are invalid")
    lre_runtime_trace = None
    if args.metric_id == "long_range_reprojection_error":
        lre_runtime_trace = verify_independent_lre_runtime_trace(component, core_evaluator_path.parents[2])
    details = component.get("details")
    torch_runtime_trace = verify_package_runtime_trace(
        details.get("torch_runtime_load_trace") if isinstance(details, dict) else None,
        runtime_identities.get("torch"),
        "torch",
    )
    torchvision_runtime_trace = verify_torchvision_component_trace(
        args.metric_id, component, runtime_identities.get("torchvision")
    )
    if torchvision_runtime_trace is not None and torchvision_runtime_trace.get("torch_runtime_load_trace") != torch_runtime_trace:
        raise ValueError("TorchVision nested PyTorch trace differs from the component's final PyTorch trace")
    vbench_source_trace = (
        verify_vbench_source_trace(component, Path(source_snapshot_rehash["source_root"]))
        if args.metric_id == "vbench_quality" else None
    )
    runtime = {
        "input_preflight_receipt_sha256": sha256_file(args.input_preflight_output),
        "evaluator_preflight_receipt_sha256": sha256_file(args.evaluator_preflight_output),
        "python_executable": evaluator_receipt["verifier_python_executable"],
        "network_namespace": evaluator_receipt["network_namespace"],
        "launcher_sha256": launcher_sha,
        "input_preflight_verifier_sha256": input_verifier_sha,
        "evaluator_preflight_verifier_sha256": evaluator_verifier_sha,
        "metric_id": args.metric_id,
        "adapter_entrypoint": {"path": str(adapter_entrypoint_path), "sha256": adapter_entrypoint["sha256"]},
        "adapter_common": {"path": str(adapter_common_path), "sha256": adapter_common["sha256"]},
        "core_evaluator": {"path": str(core_evaluator_path), "sha256": core_evaluator["sha256"]},
        "runtime_identities": runtime_identities,
        "source_snapshot_rehash": source_snapshot_rehash,
        "independent_lre_runtime_load_trace": lre_runtime_trace,
        "torch_runtime_load_trace": torch_runtime_trace,
        "torchvision_runtime_load_trace": torchvision_runtime_trace,
        "vbench_sealed_source_runtime_load_trace": vbench_source_trace,
        "execution_wrapper_sha256": wrapper_sha256,
    }
    receipt = {
        "schema": "geometry-selection-metric-worker-receipt-v1",
        "site": "Hippasus",
        "metric_id": args.metric_id,
        "input_lock": binding(args.input_lock),
        "evaluator_lock": binding(args.evaluator_lock),
        "input_preflight_receipt": binding(args.input_preflight_output),
        "evaluator_preflight_receipt": binding(args.evaluator_preflight_output),
        "runtime": runtime,
        "adapter_command": canonical_adapter_command,
        "guarded_adapter_arguments": adapter_arguments,
        "executed_wrapper": {
            "argv_prefix": [str(python_path), "-I", "-c"],
            "program_sha256": wrapper_sha256,
            "adapter_directory": str(adapter_entrypoint_path.parent),
        },
        "metric_component": component_binding,
    }
    require_held_path_identity(metric_output, component_fd, component_info)
    publish(worker_receipt_output, receipt)
    require_held_path_identity(metric_output, component_fd, component_info)
    os.close(component_fd)
    print(json.dumps({"metric_id": args.metric_id, "metric_component_sha256": receipt["metric_component"]["sha256"], "worker_receipt": str(worker_receipt_output), "verified": True}, sort_keys=True))


if __name__ == "__main__":
    main()
