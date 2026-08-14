#!/usr/bin/env python3
"""Validate a complete Hippasus five-metric result against the frozen denominator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import statistics
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any


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


_TRACE_VALIDATION_HELPER = Path(__file__).resolve(strict=True).with_name("python_runtime_trace_validation.py")
_trace_validation_module = load_exact_sibling(
    _TRACE_VALIDATION_HELPER,
    "9e2a973b477ad8052868b0389ae312acfb4a2c3d3287b3b5c97278d58689ccbc",
    "locked_python_runtime_trace_validation",
)
verify_package_runtime_trace = _trace_validation_module.verify_package_runtime_trace


METRIC_NAMES = {
    "geco_fused",
    "met3r",
    "long_range_reprojection_error",
    "relative_total_motion_percent",
    "vbench_quality",
}
TRAINDEV_METHOD_LABELS = {
    "wan_lora_dpo_step64_base_traindev": "Wan LoRA-DPO step-64 paired base train-dev",
    "wan_lora_dpo_step64_adapted_traindev": "Full-Graph LoRA-DPO step-64 paired adapted train-dev",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"regular JSON file required: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)}


def verify_bound_receipt(value: Any, schema: str, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str) or not value["path"] or not isinstance(value.get("sha256"), str) or len(value["sha256"]) != 64:
        raise ValueError(f"{label} binding malformed")
    int(value["sha256"], 16)
    path = Path(value["path"])
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222 or sha256_file(path) != value["sha256"]:
        raise ValueError(f"{label} changed after worker preflight")
    payload = read_json(path)
    if payload.get("schema") != schema or payload.get("site") != "Hippasus":
        raise ValueError(f"{label} schema/site mismatch")
    return payload, {"path": str(path.resolve()), "sha256": value["sha256"]}


def verify_complete_adapter_attestation(
    receipt: dict[str, Any], evaluator_lock: dict[str, Any], input_binding: dict[str, str],
    expected_python: Any, expected_adapter: Any, expected_core: Any, component_binding: Any,
) -> None:
    """Verify every adapter argv token and re-hash the deterministic -I wrapper."""
    adapter_command = receipt.get("adapter_command")
    guarded_arguments = receipt.get("guarded_adapter_arguments")
    executed_wrapper = receipt.get("executed_wrapper")
    if not isinstance(adapter_command, list) or not all(isinstance(token, str) for token in adapter_command) or not isinstance(guarded_arguments, list) or not all(isinstance(token, str) for token in guarded_arguments) or not isinstance(executed_wrapper, dict) or not isinstance(expected_adapter, dict) or not isinstance(expected_core, dict) or not isinstance(component_binding, dict):
        raise ValueError("worker receipt adapter attestation is malformed")
    required = {
        "--input-lock": input_binding.get("path"),
        "--core-evaluator": expected_core.get("path"),
        "--weight-manifest": evaluator_lock.get("metrics", {}).get(receipt.get("metric_id"), {}).get("weight_manifest", {}).get("path"),
        "--schedule": evaluator_lock.get("schedule", {}).get("path"),
        "--decoder": evaluator_lock.get("decoder_binary", {}).get("path"),
        "--output": component_binding.get("path"),
    }
    if not all(isinstance(value, str) and value for value in required.values()):
        raise ValueError("worker receipt lacks a complete immutable adapter binding")
    if adapter_command[:3] != [expected_python, "-I", expected_adapter.get("path")] or adapter_command[3:] != guarded_arguments:
        raise ValueError("worker receipt adapter command and guarded arguments disagree")
    seen: dict[str, str] = {}
    index = 0
    while index < len(guarded_arguments):
        option = guarded_arguments[index]
        if option not in {*required, "--device"} or index + 1 >= len(guarded_arguments) or option in seen:
            raise ValueError("worker receipt contains malformed or duplicate adapter arguments")
        value = guarded_arguments[index + 1]
        if option == "--device":
            if not value or value.startswith("-"):
                raise ValueError("worker receipt has an invalid adapter device selector")
        elif str(Path(value).resolve(strict=True)) != str(Path(required[option]).resolve(strict=True)):
            raise ValueError(f"worker receipt adapter argument {option} differs from its immutable binding")
        seen[option] = value
        index += 2
    if set(seen) - {"--device"} != set(required):
        raise ValueError("worker receipt omits an immutable adapter argument")
    adapter_path = str(Path(expected_adapter["path"]).resolve(strict=True))
    wrapper = (
        "import runpy,sys;"
        f"sys.path.insert(0,{str(Path(adapter_path).parent)!r});"
        f"sys.argv={[adapter_path, *guarded_arguments]!r};"
        f"runpy.run_path({adapter_path!r},run_name='__main__')"
    )
    wrapper_sha256 = hashlib.sha256(wrapper.encode("utf-8")).hexdigest()
    if executed_wrapper.get("argv_prefix") != [expected_python, "-I", "-c"] or executed_wrapper.get("adapter_directory") != str(Path(adapter_path).parent) or executed_wrapper.get("program_sha256") != wrapper_sha256 or receipt.get("runtime", {}).get("execution_wrapper_sha256") != wrapper_sha256:
        raise ValueError("worker receipt execution wrapper cannot be reconstructed from locked argv")


def code_role_bindings(metric_name: str, metric: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Load and re-hash the exact code-role map locked for one metric."""
    manifest_binding = metric.get("code_manifest")
    if not isinstance(manifest_binding, dict) or set(manifest_binding) != {"path", "sha256"}:
        raise ValueError(f"{metric_name} lacks a bound code manifest")
    manifest_path = Path(manifest_binding["path"])
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.stat().st_mode & 0o222
        or binding(manifest_path) != manifest_binding
    ):
        raise ValueError(f"{metric_name} code manifest changed after evaluator locking")
    manifest = read_json(manifest_path)
    files = manifest.get("files")
    if manifest.get("schema") != "geometry-selection-code-content-manifest-v1" or not isinstance(files, list):
        raise ValueError(f"{metric_name} code manifest schema differs")
    roles: dict[str, dict[str, str]] = {}
    for item in files:
        if (
            not isinstance(item, dict)
            or set(item) != {"role", "path", "sha256"}
            or not isinstance(item["role"], str)
            or not item["role"]
            or not isinstance(item["path"], str)
            or not isinstance(item["sha256"], str)
            or item["role"] in roles
        ):
            raise ValueError(f"{metric_name} code-role binding is malformed")
        roles[item["role"]] = {"path": item["path"], "sha256": item["sha256"]}
    return roles


def revalidate_source_snapshot_rehash(
    metric_name: str,
    runtime: dict[str, Any],
    evaluator_preflight: dict[str, Any],
    role_to_file: dict[str, dict[str, str]],
    locked_closure: Any,
) -> dict[str, Any]:
    """Re-run the sealed-tree verifier; do not trust a worker's copied proof."""
    proof = evaluator_preflight.get("source_snapshot_rehash")
    if runtime.get("source_snapshot_rehash") != proof or not isinstance(proof, dict):
        raise ValueError(f"{metric_name} worker source-snapshot proof differs from evaluator preflight")
    if set(proof) != {"source_root", "verifier", "source_snapshot", "source_ready", "full_rehash_receipt", "file_count"}:
        raise ValueError(f"{metric_name} source-snapshot proof schema differs")
    if (
        not isinstance(locked_closure, dict)
        or proof.get("source_root") != locked_closure.get("root")
        or proof.get("source_snapshot") != locked_closure.get("snapshot")
        or proof.get("source_ready") != locked_closure.get("ready")
        or proof.get("full_rehash_receipt") != locked_closure.get("full_rehash_receipt")
        or proof.get("file_count") != locked_closure.get("file_count")
    ):
        raise ValueError(f"{metric_name} source-snapshot proof differs from the evaluator lock")
    verifier_binding = role_to_file.get("evaluator_source_snapshot_verifier")
    if not isinstance(verifier_binding, dict) or proof.get("verifier") != verifier_binding:
        raise ValueError(f"{metric_name} source-snapshot verifier differs from the code manifest")
    source_root_value = proof.get("source_root")
    if not isinstance(source_root_value, str):
        raise ValueError(f"{metric_name} source-snapshot root is malformed")
    source_root = Path(source_root_value)
    if source_root.is_symlink() or not source_root.is_dir() or source_root.stat().st_mode & 0o222:
        raise ValueError(f"{metric_name} source-snapshot root is not sealed")
    source_root = source_root.resolve(strict=True)
    snapshot = source_root / "SOURCE_SNAPSHOT.json"
    ready = source_root / "SOURCE_READY.json"
    expected_snapshot = {"path": str(snapshot), "sha256": sha256_file(snapshot)}
    expected_ready = {"path": str(ready), "sha256": sha256_file(ready)}
    if proof.get("source_snapshot") != expected_snapshot or proof.get("source_ready") != expected_ready:
        raise ValueError(f"{metric_name} source-snapshot metadata changed after worker execution")
    if not isinstance(proof.get("file_count"), int) or proof["file_count"] <= 0:
        raise ValueError(f"{metric_name} source-snapshot file count is malformed")
    verifier_path = Path(verifier_binding["path"])
    if (
        verifier_path.is_symlink()
        or not verifier_path.is_file()
        or verifier_path.stat().st_mode & 0o222
        or sha256_file(verifier_path) != verifier_binding["sha256"]
    ):
        raise ValueError(f"{metric_name} source-snapshot verifier changed after locking")
    python_value = evaluator_preflight.get("verifier_python_executable")
    if not isinstance(python_value, str) or Path(python_value).resolve(strict=True) != Path(sys.executable).resolve(strict=True):
        raise ValueError(f"{metric_name} final receipt interpreter differs from the evaluator preflight")
    result = subprocess.run(
        [python_value, "-I", str(verifier_path), "--root", str(source_root), "--expected-self-sha256", verifier_binding["sha256"]],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    try:
        rehash = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"{metric_name} source-snapshot rehash did not return JSON") from error
    if (
        not isinstance(rehash, dict)
        or rehash != {
            "root": str(source_root),
            "snapshot_sha256": expected_snapshot["sha256"],
            "file_count": proof["file_count"],
            "verified": True,
        }
    ):
        raise ValueError(f"{metric_name} final source-snapshot rehash differs from worker proof")
    return proof


def verify_independent_lre_runtime_trace(
    runtime: dict[str, Any], component: dict[str, Any], source_proof: dict[str, Any]
) -> dict[str, Any]:
    """Bind the component trace to the worker trace and re-hash every loaded file."""
    trace = runtime.get("independent_lre_runtime_load_trace")
    details = component.get("details")
    if trace != (details.get("independent_runtime_load_trace") if isinstance(details, dict) else None):
        raise ValueError("Independent LRE component trace differs from its guarded worker receipt")
    expected_keys = {"schema", "files", "forbidden_selector_tokens", "sha256"}
    forbidden = ("vgg" + "t", "vgg" + "t" + "_" + "omega", "vgg" + "t" + "-" + "omega")
    if (
        not isinstance(trace, dict)
        or set(trace) != expected_keys
        or trace.get("schema") != "geometry-selection-independent-lre-runtime-load-trace-v1"
        or trace.get("forbidden_selector_tokens") != list(forbidden)
        or not isinstance(trace.get("files"), list)
    ):
        raise ValueError("Independent LRE runtime trace schema differs")
    unsigned = {key: trace[key] for key in expected_keys - {"sha256"}}
    if trace["sha256"] != hashlib.sha256(canonical_bytes(unsigned) + b"\n").hexdigest():
        raise ValueError("Independent LRE runtime trace SHA differs")
    source_root = Path(source_proof["source_root"]).resolve(strict=True)
    paths: list[str] = []
    for item in trace["files"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256"}
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
        ):
            raise ValueError("Independent LRE runtime trace entry is malformed")
        path = Path(item["path"])
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
            raise ValueError("Independent LRE runtime trace references an unsafe file")
        resolved = path.resolve(strict=True)
        if (
            source_root not in resolved.parents
            or resolved.suffix != ".py"
            or str(resolved) in paths
            or sha256_file(resolved) != item["sha256"]
        ):
            raise ValueError("Independent LRE runtime trace file binding differs")
        text = resolved.read_text(encoding="utf-8").lower()
        if any(token in str(resolved).lower() or token in text for token in forbidden):
            raise ValueError("Independent LRE runtime trace contains a selector dependency")
        paths.append(str(resolved))
    if not paths or paths != sorted(paths):
        raise ValueError("Independent LRE runtime trace is empty or non-canonical")
    return trace


def verify_torch_runtime_trace(
    metric_name: str, runtime: dict[str, Any], component: dict[str, Any], expected_identity: Any
) -> dict[str, Any]:
    details = component.get("details")
    trace = details.get("torch_runtime_load_trace") if isinstance(details, dict) else None
    verified = verify_package_runtime_trace(trace, expected_identity, "torch")
    if runtime.get("torch_runtime_load_trace") != verified:
        raise ValueError(f"{metric_name} worker receipt does not bind its final PyTorch load trace")
    return verified


def verify_torchvision_runtime_trace(
    metric_name: str, component: dict[str, Any], expected_identity: Any
) -> dict[str, Any]:
    """Bind loaded TorchVision modules to the exact locked wheel package."""
    if not isinstance(expected_identity, dict):
        raise ValueError(f"{metric_name} has no locked TorchVision runtime identity")
    details = component.get("details")
    if not isinstance(details, dict):
        raise ValueError(f"{metric_name} component details are missing")
    if metric_name == "relative_total_motion_percent":
        trace = details.get("torchvision_runtime")
    elif metric_name == "vbench_quality":
        runtime = details.get("runtime_load_trace")
        sealed = runtime.get("sealed_imports") if isinstance(runtime, dict) else None
        torchvision = sealed.get("torchvision") if isinstance(sealed, dict) else None
        trace = torchvision.get("runtime_trace") if isinstance(torchvision, dict) else None
    else:
        raise ValueError(f"unexpected TorchVision metric: {metric_name}")
    return verify_package_runtime_trace(trace, expected_identity, "torchvision")
def verify_vbench_source_runtime_trace(
    runtime: dict[str, Any], component: dict[str, Any], source_proof: dict[str, Any]
) -> dict[str, Any]:
    details = component.get("details")
    component_runtime = details.get("runtime_load_trace") if isinstance(details, dict) else None
    trace = component_runtime.get("sealed_source_runtime_trace") if isinstance(component_runtime, dict) else None
    if trace != runtime.get("vbench_sealed_source_runtime_load_trace"):
        raise ValueError("VBench component and worker sealed-source traces differ")
    if not isinstance(trace, dict) or set(trace) != {"module_names", "files", "sha256"}:
        raise ValueError("VBench sealed-source trace schema differs")
    names, files = trace.get("module_names"), trace.get("files")
    required = {"vbench", "clip", "pyiqa", "vision_transformer", "constant", "locked_vbench_quality_score"}
    if (
        not isinstance(names, list) or names != sorted(names) or len(names) != len(set(names))
        or not required.issubset(set(names)) or not isinstance(files, list) or not files
        or trace.get("sha256") != hashlib.sha256(canonical_bytes({"module_names": names, "files": files})).hexdigest()
    ):
        raise ValueError("VBench sealed-source trace identity differs")
    root = Path(source_proof["source_root"]).resolve(strict=True)
    paths: list[str] = []
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("VBench sealed-source trace entry is malformed")
        raw = Path(item["path"])
        info = raw.stat(follow_symlinks=False)
        resolved = raw.resolve(strict=True)
        if raw.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222 or root not in resolved.parents or str(resolved) in paths or sha256_file(resolved) != item["sha256"]:
            raise ValueError("VBench sealed-source trace file differs")
        paths.append(str(resolved))
    if paths != sorted(paths):
        raise ValueError("VBench sealed-source trace paths are non-canonical")
    return trace


def verify_runtime_attestations(
    raw_units: dict[str, Any],
    evaluator_lock: dict[str, Any],
    evaluator_binding: dict[str, str],
    input_binding: dict[str, str],
    *,
    train_dev: bool = False,
) -> dict[str, dict[str, dict[str, str]]]:
    attestations = raw_units.get("metric_worker_receipts")
    components = raw_units.get("metric_components")
    units = raw_units.get("units")
    metrics = evaluator_lock.get("metrics")
    if not isinstance(attestations, dict) or set(attestations) != METRIC_NAMES or not isinstance(components, dict) or set(components) != METRIC_NAMES or not isinstance(units, dict) or set(units) != METRIC_NAMES or not isinstance(metrics, dict) or set(metrics) != METRIC_NAMES:
        raise ValueError("metric units require one guarded receipt and component per locked metric")
    verified: dict[str, dict[str, dict[str, str]]] = {}
    for metric_name in sorted(METRIC_NAMES):
        receipt, receipt_binding = verify_bound_receipt(attestations[metric_name], "geometry-selection-metric-worker-receipt-v1", f"{metric_name} guarded worker receipt")
        input_preflight_schema = (
            "geometry-selection-metric-traindev-input-preflight-receipt-v1"
            if train_dev
            else "geometry-selection-metric-input-preflight-receipt-v1"
        )
        input_preflight, input_preflight_binding = verify_bound_receipt(
            receipt.get("input_preflight_receipt"),
            input_preflight_schema,
            f"{metric_name} input preflight receipt",
        )
        evaluator_preflight, evaluator_preflight_binding = verify_bound_receipt(receipt.get("evaluator_preflight_receipt"), "geometry-selection-evaluator-preflight-receipt-v1", f"{metric_name} evaluator preflight receipt")
        expected_adapter = metrics[metric_name].get("adapter_entrypoint") if isinstance(metrics[metric_name], dict) else None
        expected_common = metrics[metric_name].get("adapter_common") if isinstance(metrics[metric_name], dict) else None
        expected_core = metrics[metric_name].get("core_evaluator") if isinstance(metrics[metric_name], dict) else None
        expected_runtime_identities = metrics[metric_name].get("runtime_identities") if isinstance(metrics[metric_name], dict) else None
        expected = {
            "input_preflight_receipt_sha256": input_preflight_binding["sha256"],
            "evaluator_preflight_receipt_sha256": evaluator_preflight_binding["sha256"],
            "python_executable": evaluator_preflight.get("verifier_python_executable"),
            "network_namespace": evaluator_preflight.get("network_namespace"),
            "launcher_sha256": evaluator_preflight.get("guarded_launcher_sha256"),
            "input_preflight_verifier_sha256": evaluator_preflight.get("input_preflight_verifier_sha256"),
            "evaluator_preflight_verifier_sha256": evaluator_preflight.get("evaluator_preflight_verifier_sha256"),
            "metric_id": metric_name,
            "adapter_entrypoint": expected_adapter,
            "adapter_common": expected_common,
            "core_evaluator": expected_core,
            "runtime_identities": expected_runtime_identities,
        }
        component_binding = receipt.get("metric_component")
        if input_preflight.get("input_lock") != input_binding or evaluator_preflight.get("evaluator_lock") != evaluator_binding or evaluator_preflight.get("shared_evaluator_identity_sha256") != evaluator_lock.get("shared_evaluator_identity_sha256") or receipt.get("input_lock") != input_binding or receipt.get("evaluator_lock") != evaluator_binding or receipt.get("metric_id") != metric_name or not isinstance(receipt.get("runtime"), dict) or any(receipt["runtime"].get(key) != value for key, value in expected.items()) or not isinstance(expected_adapter, dict) or not isinstance(expected_common, dict) or not isinstance(expected_core, dict) or components[metric_name] != component_binding:
            raise ValueError(f"{metric_name} worker receipt does not attest its locked adapter and core evaluator")
        verify_complete_adapter_attestation(receipt, evaluator_lock, input_binding, expected["python_executable"], expected_adapter, expected_core, component_binding)
        component_path = Path(component_binding.get("path", "")) if isinstance(component_binding, dict) else Path()
        if component_path.is_symlink() or not component_path.is_file() or component_path.stat().st_mode & 0o222 or binding(component_path) != component_binding:
            raise ValueError(f"{metric_name} component changed after guarded execution")
        component = read_json(component_path)
        if component.get("schema") != "geometry-selection-metric-component-v1" or component.get("metric_id") != metric_name or component.get("records") != units[metric_name]:
            raise ValueError(f"{metric_name} raw units differ from the guarded metric component")
        if not isinstance(expected_runtime_identities, dict):
            raise ValueError(f"{metric_name} locked runtime identities are malformed")
        if metric_name == "relative_total_motion_percent" and component.get("details", {}).get("opencv_runtime") != expected_runtime_identities.get("opencv"):
            raise ValueError("motion component does not attest the locked OpenCV native runtime")
        torch_trace = verify_torch_runtime_trace(
            metric_name, receipt["runtime"], component, expected_runtime_identities.get("torch")
        )
        torchvision_trace = (
            verify_torchvision_runtime_trace(metric_name, component, expected_runtime_identities.get("torchvision"))
            if metric_name in {"relative_total_motion_percent", "vbench_quality"}
            else None
        )
        if receipt["runtime"].get("torchvision_runtime_load_trace") != torchvision_trace:
            raise ValueError(f"{metric_name} worker receipt does not bind its TorchVision load trace")
        if torchvision_trace is not None and torchvision_trace.get("torch_runtime_load_trace") != torch_trace:
            raise ValueError(f"{metric_name} TorchVision/PyTorch runtime traces disagree")
        source_proof = revalidate_source_snapshot_rehash(
            metric_name, receipt["runtime"], evaluator_preflight, code_role_bindings(metric_name, metrics[metric_name]), evaluator_lock.get("source_snapshot_closure")
        )
        vbench_source_trace = (
            verify_vbench_source_runtime_trace(receipt["runtime"], component, source_proof)
            if metric_name == "vbench_quality" else None
        )
        lre_trace = (
            verify_independent_lre_runtime_trace(receipt["runtime"], component, source_proof)
            if metric_name == "long_range_reprojection_error"
            else None
        )
        verified[metric_name] = {"worker_receipt": receipt_binding, "input_preflight_receipt": input_preflight_binding, "evaluator_preflight_receipt": evaluator_preflight_binding, "adapter_entrypoint": expected_adapter, "adapter_common": expected_common, "core_evaluator": expected_core, "metric_component": component_binding, "source_snapshot_rehash": source_proof, "independent_lre_runtime_load_trace": lre_trace, "torch_runtime_load_trace": torch_trace, "torchvision_runtime_load_trace": torchvision_trace, "vbench_sealed_source_runtime_load_trace": vbench_source_trace}
    return verified


def finite(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("metric value must be finite")
    return float(value)


def record_map(
    records: Any,
    expected: set[tuple[str, str]],
    metric: str,
    *,
    allowed_extras: set[tuple[str, str]] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError(f"{metric} records must be a list")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{metric} records must be objects")
        case_id, unit_id = record.get("case_id"), record.get("unit_id")
        key = (case_id, unit_id)
        if not isinstance(case_id, str) or not case_id or not isinstance(unit_id, str) or not unit_id or key in result:
            raise ValueError(f"{metric} record identifiers are invalid")
        finite(record.get("value"))
        result[key] = record
    allowed = allowed_extras or set()
    if expected & allowed or set(result) != expected | allowed:
        raise ValueError(f"{metric} units do not exactly match the locked denominator")
    return {key: result[key] for key in expected}


def safe_output(path: Path, derived_root: Path, inputs: list[Path]) -> None:
    if derived_root.is_symlink() or not derived_root.is_dir():
        raise ValueError("derived root must be an existing regular directory")
    derived = derived_root.resolve(strict=True)
    if "validation_wan_candidates" in derived.parts:
        raise ValueError("derived root overlaps frozen candidate root")
    if tuple(derived.parts[-3:]) != ("outputs", "geometry-selection", "hippasus_evaluation"):
        raise ValueError("derived root must be the approved Hippasus evaluation root")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("output parent must be an existing regular directory")
    target = path.parent.resolve(strict=True) / path.name
    if target == derived or derived not in target.parents:
        raise ValueError("output must be strictly below derived root")
    if target.relative_to(derived).parts[0] != "receipts":
        raise ValueError("metric receipts may only publish below the approved receipts root")
    for source in inputs:
        resolved = source.resolve(strict=True)
        if target == resolved or target in resolved.parents or resolved in target.parents:
            raise ValueError("output must not overlap an input")


def publish(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing output: {path}")
    fd, name = tempfile.mkstemp(prefix=".metric-receipt-", suffix=".tmp", dir=str(path.parent))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-lock", type=Path, required=True)
    parser.add_argument("--evaluator-lock", type=Path, required=True)
    parser.add_argument("--eligibility-lock", type=Path, required=True)
    parser.add_argument("--metric-units", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_lock = read_json(args.input_lock)
    evaluator_lock = read_json(args.evaluator_lock)
    eligibility = read_json(args.eligibility_lock)
    raw_units = read_json(args.metric_units)
    input_binding, evaluator_binding = binding(args.input_lock), binding(args.evaluator_lock)
    eligibility_binding, raw_binding = binding(args.eligibility_lock), binding(args.metric_units)
    input_schema = input_lock.get("schema")
    train_dev = input_schema == "geometry-selection-five-metric-traindev-input-lock-v1"
    serialized_input = json.dumps(input_lock, sort_keys=True, separators=(",", ":"))
    if input_schema not in {
        "geometry-selection-five-metric-hippasus-input-lock-v2",
        "geometry-selection-five-metric-traindev-input-lock-v1",
    } or input_lock.get("evaluation_site") != "Hippasus":
        raise ValueError("input lock must be a Hippasus five-metric input lock")
    if train_dev and (
        input_lock.get("scope") != "train_dev_evaluation"
        or input_lock.get("dataset_split") != "dev"
        or "formal_validation" in input_lock
        or input_lock.get("reserved_ids_disclosed") is not False
        or input_lock.get("method_id")
        not in {
            "wan_lora_dpo_step64_base_traindev",
            "wan_lora_dpo_step64_adapted_traindev",
        }
        or input_lock.get("method")
        != TRAINDEV_METHOD_LABELS.get(input_lock.get("method_id"))
        or any(
            token in serialized_input
            for token in ("formal_validation", "validation_only", "wan_unguided_seed0")
        )
        or not isinstance(input_lock.get("traindev_reference_isolation_receipt_sha256"), str)
        or len(input_lock["traindev_reference_isolation_receipt_sha256"]) != 64
        or not isinstance(input_lock.get("bundle_reference_path"), str)
        or not Path(input_lock["bundle_reference_path"]).is_absolute()
    ):
        raise ValueError("train-dev input lock identity/isolation mismatch")
    if evaluator_lock.get("schema") != "geometry-selection-evaluator-lock-v2" or evaluator_lock.get("site") != "Hippasus" or evaluator_lock.get("input_manifest") != input_binding:
        raise ValueError("evaluator lock does not bind the exact method input")
    if train_dev and evaluator_lock.get("scope") != "train_dev":
        raise ValueError("train-dev receipt requires a train-dev evaluator lock")
    traindev_provenance = evaluator_lock.get("traindev_provenance") if train_dev else None
    if train_dev and (
        not isinstance(traindev_provenance, dict)
        or traindev_provenance.get("schema")
        != "geometry-selection-traindev-provenance-verification-v1"
        or traindev_provenance.get("status") != "verified"
        or traindev_provenance.get("case_count") != 100
        or traindev_provenance.get("record_count") != 800
        or traindev_provenance.get("expected_bundle_reference_sha256")
        != traindev_provenance.get("source_bundle_reference", {}).get("sha256")
        or traindev_provenance.get("source_bundle_reference", {}).get("path")
        != input_lock.get("bundle_reference_path")
        or traindev_provenance.get("expected_generation_receipt_sha256")
        != input_lock.get("source_generation_receipt_sha256")
        or traindev_provenance.get("generation_receipt", {}).get("sha256")
        != input_lock.get("source_generation_receipt_sha256")
        or traindev_provenance.get("reference_isolation_receipt", {}).get("sha256")
        != input_lock.get("traindev_reference_isolation_receipt_sha256")
    ):
        raise ValueError("train-dev evaluator lacks its verified bundle provenance closure")
    expected_eligibility_schema = (
        "geometry-selection-metric-traindev-eligibility-lock-v1"
        if train_dev
        else "geometry-selection-metric-eligibility-lock-v1"
    )
    if eligibility.get("schema") != expected_eligibility_schema or eligibility.get("site") != "Hippasus":
        raise ValueError("eligibility lock identity mismatch")
    if eligibility.get("scope") != input_lock.get("scope"):
        raise ValueError("method input scope differs from locked denominator")
    shared_identity = evaluator_lock.get("shared_evaluator_identity_sha256")
    if not isinstance(shared_identity, str) or shared_identity != eligibility.get("shared_evaluator_identity_sha256"):
        raise ValueError("method evaluator environment differs from the seed-0 evaluator environment")
    baseline_method = input_lock.get("method_id") in {
        "wan_unguided_seed0",
        "wan_lora_dpo_step64_base_traindev",
    }
    if baseline_method:
        if eligibility.get("baseline_input_lock") != input_binding or eligibility.get("baseline_evaluator_lock") != evaluator_binding or eligibility.get("baseline_units") != raw_binding:
            raise ValueError("baseline final receipt must use the exact units that froze the eligibility denominator")
        if train_dev and traindev_provenance.get("expected_baseline_eligibility_sha256") is not None:
            raise ValueError("train-dev base provenance unexpectedly contains adapted eligibility")
    elif input_lock.get("baseline_metric_eligibility_lock") != eligibility_binding:
        raise ValueError("compared-method receipt must bind the exact pre-existing baseline eligibility lock")
    elif train_dev and (
        input_lock.get("expected_baseline_eligibility_sha256")
        != eligibility_binding["sha256"]
        or traindev_provenance.get("expected_baseline_eligibility_sha256")
        != eligibility_binding["sha256"]
    ):
        raise ValueError("adapted receipt differs from externally pinned baseline eligibility")
    if train_dev and eligibility.get("baseline_method_id") != "wan_lora_dpo_step64_base_traindev":
        raise ValueError("train-dev denominator was not frozen by the paired base method")
    entries = input_lock.get("entries")
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        raise ValueError("method input entries are malformed")
    method_cases = {item.get("case_id") for item in entries}
    eligibility_cases = set(eligibility.get("case_ids", []))
    if method_cases != eligibility_cases or len(method_cases) != len(entries):
        raise ValueError("method input cases must exactly match the seed-0 denominator cases")
    if raw_units.get("schema") != "geometry-selection-metric-units-v1" or raw_units.get("method") != input_lock.get("method"):
        raise ValueError("metric-unit schema/method mismatch")
    if raw_units.get("input_manifest_sha256") != input_binding["sha256"] or raw_units.get("evaluator_lock_sha256") != evaluator_binding["sha256"]:
        raise ValueError("metric units bind the wrong input or evaluator lock")
    runtime_attestations = verify_runtime_attestations(
        raw_units,
        evaluator_lock,
        evaluator_binding,
        input_binding,
        train_dev=train_dev,
    )
    expected_preflight_cases = {
        (
            entry["case_id"],
            entry["video_sha256"],
            entry["metadata_sha256"],
            entry["complete_sha256"],
            entry.get("generation_lock_sha256") if train_dev else None,
        )
        for entry in entries
    }
    input_preflight_schema = (
        "geometry-selection-metric-traindev-input-preflight-receipt-v1"
        if train_dev
        else "geometry-selection-metric-input-preflight-receipt-v1"
    )
    for metric_name, receipts in runtime_attestations.items():
        input_preflight, _ = verify_bound_receipt(
            receipts["input_preflight_receipt"],
            input_preflight_schema,
            f"{metric_name} input preflight receipt",
        )
        if train_dev and (
            input_preflight.get("input_lock") != input_binding
            or input_preflight.get("evaluator_lock") != evaluator_binding
            or input_preflight.get("traindev_provenance") != traindev_provenance
        ):
            raise ValueError(
                f"{metric_name} input preflight lacks the verified train-dev provenance closure"
            )
        actual_preflight_cases = {
            (
                entry.get("case_id"),
                entry.get("video_sha256"),
                entry.get("metadata_sha256"),
                entry.get("complete_sha256"),
                entry.get("generation_lock_sha256") if train_dev else None,
            )
            for entry in input_preflight.get("case_artifacts", [])
            if isinstance(entry, dict)
        }
        if actual_preflight_cases != expected_preflight_cases:
            raise ValueError(f"{metric_name} input preflight receipt does not cover the exact metric input artifact set")
    units = raw_units.get("units")
    locked = eligibility.get("locked_units")
    if not isinstance(units, dict) or set(units) != METRIC_NAMES or not isinstance(locked, dict) or set(locked) != METRIC_NAMES:
        raise ValueError("five exact metric unit sets are required")
    maps: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for metric in sorted(METRIC_NAMES):
        expected = {(item.get("case_id"), item.get("unit_id")) for item in locked[metric] if isinstance(item, dict)}
        if not expected or any(not isinstance(case, str) or not isinstance(unit, str) for case, unit in expected):
            raise ValueError(f"locked {metric} denominator is malformed or empty")
        excluded: set[tuple[str, str]] = set()
        if baseline_method:
            exclusions = eligibility.get("baseline_exclusions")
            raw_exclusions = exclusions.get(metric) if isinstance(exclusions, dict) else None
            if not isinstance(raw_exclusions, list):
                raise ValueError(f"baseline exclusions are missing for {metric}")
            for item in raw_exclusions:
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("case_id"), str)
                    or item["case_id"] not in method_cases
                    or not isinstance(item.get("unit_id"), str)
                    or not isinstance(item.get("reason"), str)
                    or not item["reason"]
                ):
                    raise ValueError(f"baseline exclusion is malformed for {metric}")
                excluded.add((item["case_id"], item["unit_id"]))
            if len(excluded) != len(raw_exclusions):
                raise ValueError(f"baseline exclusions are duplicated for {metric}")
        maps[metric] = record_map(
            units[metric], expected, metric, allowed_extras=excluded
        )
    for record in maps["relative_total_motion_percent"].values():
        case_id = record["case_id"]
        anchors = eligibility.get("baseline_motion_anchors")
        if not isinstance(anchors, dict) or not isinstance(anchors.get(case_id), (int, float)):
            raise ValueError(f"missing locked baseline motion anchor: {case_id}")
        method_raw, baseline_raw = finite(record.get("raw_total_motion")), finite(anchors[case_id])
        expected_ratio = 100.0 * method_raw / baseline_raw
        if abs(finite(record["value"]) - expected_ratio) > 1e-8 * max(1.0, abs(expected_ratio)):
            raise ValueError(f"motion ratio does not match the locked seed-0 anchor: {case_id}")
    summary: dict[str, Any] = {}
    for metric, values in maps.items():
        numeric = [finite(record["value"]) for record in values.values()]
        if metric != "vbench_quality":
            summary[metric] = {"mean": statistics.fmean(numeric), "unit_count": len(numeric)}
    motion_values = [finite(record["value"]) for record in maps["relative_total_motion_percent"].values()]
    summary["relative_total_motion_percent"]["median"] = statistics.median(motion_values)
    quality_values = [finite(record["value"]) for key, record in maps["vbench_quality"].items() if key[1] == "official_quality"]
    component_means = {
        unit_id: statistics.fmean([finite(record["value"]) for key, record in maps["vbench_quality"].items() if key[1] == unit_id])
        for unit_id in sorted({key[1] for key in maps["vbench_quality"] if key[1] != "official_quality"})
    }
    summary["vbench_quality"] = {"official_quality_mean": statistics.fmean(quality_values), "case_count": len(quality_values), "component_means": component_means}
    payload = {
        "schema": (
            "geometry-selection-five-metric-traindev-receipt-v1"
            if train_dev
            else "geometry-selection-five-metric-receipt-v1"
        ),
        "site": "Hippasus",
        "scope": input_lock["scope"],
        "method": input_lock["method"],
        "method_id": input_lock.get("method_id"),
        "candidate_budget": input_lock.get("candidate_budget"),
        "input_lock": input_binding,
        "evaluator_lock": evaluator_binding,
        "shared_evaluator_identity_sha256": shared_identity,
        "eligibility_lock": eligibility_binding,
        "metric_worker_evidence": runtime_attestations,
        "metric_units": raw_binding,
        "summary": summary,
    }
    safe_output(args.output, args.derived_root, [args.input_lock, args.evaluator_lock, args.eligibility_lock, args.metric_units])
    publish(args.output, payload)
    print(json.dumps({"metric_receipt_sha256": hashlib.sha256(canonical_bytes(payload) + b"\n").hexdigest(), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
