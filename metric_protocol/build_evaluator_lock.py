#!/usr/bin/env python3
"""Publish a content-verified evaluator lock for smoke, train-dev, or formal scoring."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import subprocess
import tempfile
import types
from pathlib import Path
from typing import Any


def load_exact_sibling(path: Path, expected_sha256: str, module_name: str) -> types.ModuleType:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
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


_BASE = Path(__file__).resolve(strict=True).parent
_PACKAGE_RUNTIME_HELPER = _BASE / "python_package_runtime_lock.py"
_package_runtime_module = load_exact_sibling(
    _PACKAGE_RUNTIME_HELPER,
    "0a344fff0b0daf534ec46d60acb57cc041ba3527e7aea40041261b201efd0211",
    "locked_python_package_runtime_lock",
)
package_runtime_identity = _package_runtime_module.package_runtime_identity
verify_environment_fingerprint_payload = _package_runtime_module.verify_environment_fingerprint_payload
_TRAINDEV_PROVENANCE_HELPER = _BASE / "traindev_provenance_v1.py"
_traindev_provenance_module = load_exact_sibling(
    _TRAINDEV_PROVENANCE_HELPER,
    "89283249fba8e4176574b7bf0bf1cbe0cdfe94792a8433e51edb2d331b7fa7a1",
    "locked_traindev_provenance_v1",
)
verify_train_dev_provenance = _traindev_provenance_module.verify_train_dev_provenance


METRIC_NAMES = {
    "geco_fused",
    "met3r",
    "long_range_reprojection_error",
    "relative_total_motion_percent",
    "vbench_quality",
}
REFERENCE_BY_METRIC = {
    "geco_fused": "geco_eval",
    "met3r": "met3r",
    "long_range_reprojection_error": "independent_lre",
    "vbench_quality": "vbench",
}
HIPPASUS_INPUT_SCHEMA = "geometry-selection-five-metric-hippasus-input-lock-v2"
TRAINDEV_INPUT_SCHEMA = "geometry-selection-five-metric-traindev-input-lock-v1"
TRAINDEV_METHOD_IDS = {
    "wan_lora_dpo_step64_base_traindev",
    "wan_lora_dpo_step64_adapted_traindev",
}
TRAINDEV_METHOD_LABELS = {
    "wan_lora_dpo_step64_base_traindev": "Wan LoRA-DPO step-64 paired base train-dev",
    "wan_lora_dpo_step64_adapted_traindev": "Full-Graph LoRA-DPO step-64 paired adapted train-dev",
}
TRAINDEV_PARENT_PROTOCOL_SHA256 = "d706abfe58d449549558d11aa986f3f13a4bb463af68a8f6b7ec890049628048"
TRAINDEV_PARENT_SCHEDULE_SHA256 = "afe5b44f954adb28ffb40fccb9242111956be9814471adde87070a9978755612"
TRAINDEV_PARENT_SCHEDULE_OBJECT_SHA256 = "9ad3418a19f8deb60365f296383eaaf34a08c0402d3b4947b2646a1431493e06"
TRAINDEV_PRESERVED_SUBTREE_SHA256 = {
    "/metrics": "757429e13c191a2f7f295161b46def46ef7f55abac04b87de8e2450ad4407579",
    "/reference_evaluator_identities": "ede04481e7605afcc4c1d40b327de32b7b21025063bc0e86207b97c1be9dfae2",
    "/video_contract": "1e309bdab072ff27ba2aee26d0fd8e38525687197ee656932d6e0807bb4a9ef2",
    "/weight_source_requirements": "ae37cedfc430ca39aa7f0113e57487630855ff101b820324e4737315eabf37e4",
}
TRAINDEV_APPROVED_PROTOCOL_POINTERS = [
    "/candidate_budget_policy",
    "/dataset",
    "/derivation",
    "/execution_environment",
    "/prohibitions",
    "/schema",
    "/scope",
]
REQUIRED_WEIGHT_ROLES = {
    "geco_fused": {"geco_vggt_1b_checkpoint", "ufm_base_checkpoint"},
    "met3r": {
        "mast3r_config",
        "mast3r_checkpoint",
        "featup_dino16_jbu_checkpoint",
        "featup_dino_vits16_checkpoint",
    },
    "long_range_reprojection_error": {
        "megasam_camera_tracker_checkpoint",
        "depthanything_vitl14_checkpoint",
        "shared_raft_things_checkpoint",
        "unidepth_v2_vitl14_config",
        "unidepth_v2_vitl14_checkpoint",
        "ufm_base_checkpoint",
    },
    "relative_total_motion_percent": {"relative_motion_raft_large_checkpoint"},
    "vbench_quality": {
        "vbench_clip_vit_b32_checkpoint",
        "vbench_amt_s_checkpoint",
        "shared_raft_things_checkpoint",
        "vbench_dino_vitbase16_checkpoint",
        "vbench_clip_vit_l14_checkpoint",
        "vbench_aesthetic_linear_checkpoint",
        "vbench_musiq_spaq_checkpoint",
    },
}
REQUIRED_CODE_ROLES = {
    "geco_fused": {"metric_adapter_common", "geco_eval_entrypoint", "geco_eval_utils", "vggt_1b_runtime", "ufm_runtime"},
    "met3r": {"metric_adapter_common", "met3r_entrypoint", "mast3r_runtime", "featup_runtime"},
    "long_range_reprojection_error": {
        "metric_adapter_common",
        "independent_lre_reprojection",
        "independent_lre_megasam_runner",
        "independent_lre_runtime_trace",
        "megasam_camera_tracking_runtime",
        "megasam_cvd_runtime",
        "ufm_runtime",
        "lre_synthetic_geometry_certificate",
    },
    "relative_total_motion_percent": {
        "metric_adapter_common", "raft_metric_runtime", "opencv_native_extension",
        "opencv_native_dependency_inspector", "opencv_native_dependency_inspector_interpreter",
        "torchvision_runtime_guard", "torchvision_runtime", "torchvision_package_content_manifest",
        "torchvision_native_extension", "torchvision_native_dependency_inspector",
        "torchvision_native_dependency_inspector_interpreter",
    },
    "vbench_quality": {
        "metric_adapter_common", "vbench_entrypoint", "vbench_subject_consistency_runtime", "vbench_background_consistency_runtime",
        "vbench_temporal_flickering_runtime", "vbench_motion_smoothness_runtime",
        "vbench_aesthetic_quality_runtime", "vbench_imaging_quality_runtime", "vbench_dynamic_degree_runtime",
        "vbench_official_constant", "vbench_official_final_score", "vbench_dino_runtime",
        "torchvision_runtime_guard", "torchvision_runtime", "torchvision_package_content_manifest",
        "torchvision_native_extension", "torchvision_native_dependency_inspector",
        "torchvision_native_dependency_inspector_interpreter",
    },
}
COMMON_TORCH_CODE_ROLES = {
    "python_package_runtime_lock",
    "python_runtime_trace_validation",
    "torchvision_preprocessing_parity_runner",
    "torch_runtime_guard",
    "torch_runtime",
    "torch_package_content_manifest",
    "torch_native_extension",
    "torch_native_dependency_inspector",
    "torch_native_dependency_inspector_interpreter",
}
for _metric_name in REQUIRED_CODE_ROLES:
    REQUIRED_CODE_ROLES[_metric_name] |= COMMON_TORCH_CODE_ROLES
CORE_ENTRYPOINT_ROLE = {
    "geco_fused": "geco_eval_entrypoint",
    "met3r": "met3r_entrypoint",
    "long_range_reprojection_error": "independent_lre_megasam_runner",
    "relative_total_motion_percent": "raft_metric_runtime",
    "vbench_quality": "vbench_entrypoint",
}


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
    """Resolve exactly the ELF libraries used by the locked OpenCV extension."""
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
    """Return the exact native OpenCV implementation used by RAFT preprocessing."""
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


def build_shared_evaluator_identity(
    *,
    site: str,
    protocol_binding: dict[str, Any],
    environment_binding: dict[str, Any],
    source_snapshot_closure: dict[str, Any],
    decoder: dict[str, Any],
    decoder_trusted_root: dict[str, Any],
    schedule_binding: dict[str, Any],
    verified_metrics: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Build the method-neutral evaluator identity shared by a paired comparison."""
    identity = {
        "site": site,
        "offline_runtime": True,
        "metric_protocol": protocol_binding,
        "environment_fingerprint": environment_binding,
        "source_snapshot_closure": source_snapshot_closure,
        "decoder_binary": decoder,
        "decoder_trusted_root": decoder_trusted_root,
        "schedule": schedule_binding,
        "metrics": verified_metrics,
    }
    return identity, hashlib.sha256(canonical_bytes(identity) + b"\n").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA256 hex string")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    return value


def require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise ValueError(f"{label} must be a 40-character commit ID")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    return value


def verify_file_binding(binding: Any, label: str) -> dict[str, str]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} must be an object")
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}.path must be nonempty")
    path = Path(raw_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must name a regular file: {path}")
    expected = require_sha256(binding.get("sha256"), f"{label}.sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA mismatch: {path}")
    return {"path": str(path.resolve()), "sha256": actual}


def verify_trusted_binding(binding: Any, trusted_root_value: Any, label: str) -> tuple[dict[str, str], str]:
    result = verify_file_binding(binding, label)
    if not isinstance(trusted_root_value, str) or not trusted_root_value:
        raise ValueError(f"{label} trusted root is required")
    root = Path(trusted_root_value)
    path = Path(result["path"])
    if root.is_symlink() or not root.is_dir() or root.stat().st_mode & 0o222:
        raise ValueError(f"{label} trusted root must be a non-writable directory")
    resolved_root = root.resolve(strict=True)
    # Hippasus uses file-level sealing plus immediately-before-worker full
    # content rehashing; its user project hierarchy is not mount-read-only.
    if resolved_root not in path.parents:
        raise ValueError(f"{label} escapes trusted root")
    for item in (resolved_root, *path.parents):
        if item == resolved_root or resolved_root in item.parents:
            if item.stat().st_mode & 0o222:
                raise ValueError(f"{label} has writable trusted-root ancestor")
    return result, str(resolved_root)


def verify_content_manifest(binding: Any, label: str, schema: str, require_commit_field: bool, site: str) -> tuple[dict[str, str], dict[str, Any]]:
    file_binding = verify_file_binding(binding, label)
    payload = read_json(Path(file_binding["path"]))
    if payload.get("schema") != schema:
        raise ValueError(f"{label} schema mismatch")
    if payload.get("site") != site:
        raise ValueError(f"{label} site mismatch")
    if payload.get("seal_mode") != "file_level_readonly_with_preworker_rehash":
        raise ValueError(f"{label} does not use the required Hippasus file-level seal")
    if require_commit_field:
        require_commit(payload.get("repository_commit"), f"{label}.repository_commit")
    trace = payload.get("runtime_load_trace")
    if payload.get("offline_ready") is not True or payload.get("load_closure_complete") is not True or not isinstance(trace, dict):
        raise ValueError(f"{label} lacks offline load-closure attestation")
    trace_binding = verify_file_binding(trace, f"{label}.runtime_load_trace")
    trace_payload = read_json(Path(trace_binding["path"]))
    if trace_payload.get("schema") != "geometry-selection-offline-load-trace-v1" or trace_payload.get("site") != site or trace_payload.get("offline_runtime") is not True or trace_payload.get("network_access") != "disabled":
        raise ValueError(f"{label} runtime load trace is not an offline Hippasus trace")
    raw_root = payload.get("trusted_root")
    if not isinstance(raw_root, str) or not raw_root:
        raise ValueError(f"{label} trusted root is missing")
    trusted_root = Path(raw_root)
    if trusted_root.is_symlink() or not trusted_root.is_dir() or trusted_root.stat().st_mode & 0o222:
        raise ValueError(f"{label} trusted root must be a non-writable directory")
    trusted_root = trusted_root.resolve(strict=True)
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{label}.files must be nonempty")
    verified_files = []
    roles: set[str] = set()
    for index, item in enumerate(files):
        if not isinstance(item, dict) or not isinstance(item.get("role"), str) or not item["role"] or item["role"] in roles:
            raise ValueError(f"{label}.files[{index}] needs a unique role")
        roles.add(item["role"])
        verified = verify_file_binding(item, f"{label}.files[{index}]")
        content_path = Path(verified["path"])
        if content_path.stat().st_mode & 0o222:
            raise ValueError(f"{label}.files[{index}] is writable")
        if trusted_root not in content_path.parents:
            raise ValueError(f"{label}.files[{index}] escapes its trusted root")
        for parent in (trusted_root, *content_path.parents):
            if parent == trusted_root or trusted_root in parent.parents:
                if parent.stat().st_mode & 0o222:
                    raise ValueError(f"{label}.files[{index}] has a writable trusted-root ancestor")
        verified_files.append({"role": item["role"], **verified})
    normalized_loaded = sorted(
        [{"role": item.get("role"), "path": item.get("path"), "sha256": item.get("sha256")} for item in trace_payload.get("loaded_files", []) if isinstance(item, dict)],
        key=lambda item: str(item.get("role")),
    )
    expected_loaded = sorted(verified_files, key=lambda item: item["role"])
    if normalized_loaded != expected_loaded:
        raise ValueError(f"{label} runtime load trace does not attest its complete content closure")
    payload = dict(payload)
    payload["files"] = verified_files
    return file_binding, payload


def verify_opencv_native_closure(manifest_files: list[dict[str, str]]) -> None:
    """Bind every library `ldd` resolves, not just the Python extension module."""
    roles = {item["role"]: item for item in manifest_files}
    identity = opencv_runtime_identity(
        roles["opencv_native_dependency_inspector"],
        roles["opencv_native_dependency_inspector_interpreter"],
    )
    declared = [
        {"path": item["path"], "sha256": item["sha256"]}
        for item in manifest_files
        if item["role"] == "opencv_native_extension" or item["role"].startswith("opencv_native_library_")
    ]
    expected = [
        {"path": identity["native_extension_path"], "sha256": identity["native_extension_sha256"]},
        *identity["native_library_closure"],
    ]
    if sorted(declared, key=lambda item: item["path"]) != sorted(expected, key=lambda item: item["path"]):
        raise ValueError("OpenCV content manifest does not bind the exact native extension and ldd dependency closure")


def verify_environment(binding: Any, site: str) -> tuple[dict[str, str], dict[str, Any]]:
    file_binding = verify_file_binding(binding, "environment_fingerprint")
    payload = read_json(Path(file_binding["path"]))
    return file_binding, verify_environment_fingerprint_payload(payload, site)


def verify_source_snapshot_closure(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"root", "snapshot", "ready", "full_rehash_receipt"}:
        raise ValueError("source snapshot closure binding is malformed")
    raw_root = Path(value["root"])
    root_info = raw_root.stat(follow_symlinks=False)
    if raw_root.is_symlink() or not stat.S_ISDIR(root_info.st_mode) or root_info.st_mode & 0o222:
        raise ValueError("source snapshot root is not sealed")
    root = raw_root.resolve(strict=True)
    approved_parent = Path("/vol/dissolve/yz10325/evaluator_sources").resolve(strict=True)
    if root.parent != approved_parent:
        raise ValueError("source snapshot root is outside the approved Hippasus parent")
    for raw in [root, *root.rglob("*")]:
        info = raw.stat(follow_symlinks=False)
        if raw.is_symlink() or (not stat.S_ISDIR(info.st_mode) and not stat.S_ISREG(info.st_mode)) or info.st_mode & 0o222:
            raise ValueError(f"source snapshot tree is not fully immutable: {raw}")
    snapshot = verify_file_binding(value["snapshot"], "source_snapshot_closure.snapshot")
    ready = verify_file_binding(value["ready"], "source_snapshot_closure.ready")
    receipt = verify_file_binding(value["full_rehash_receipt"], "source_snapshot_closure.full_rehash_receipt")
    if Path(snapshot["path"]).resolve(strict=True) != root / "SOURCE_SNAPSHOT.json" or Path(ready["path"]).resolve(strict=True) != root / "SOURCE_READY.json":
        raise ValueError("source snapshot closure paths differ from the sealed root")
    snapshot_payload = read_json(Path(snapshot["path"]))
    ready_payload = read_json(Path(ready["path"]))
    receipt_payload = read_json(Path(receipt["path"]))
    if (
        snapshot_payload.get("schema") != "geometry-selection-hippasus-evaluator-source-snapshot-v1"
        or ready_payload.get("schema") != "geometry-selection-hippasus-evaluator-source-ready-v1"
        or ready_payload.get("snapshot") != "SOURCE_SNAPSHOT.json"
        or ready_payload.get("snapshot_sha256") != snapshot["sha256"]
        or receipt_payload.get("schema") != "geometry-selection-hippasus-evaluator-source-snapshot-rehash-receipt-v1"
        or receipt_payload.get("status") != "full_rehash_verified"
        or receipt_payload.get("source_snapshot_root") != str(root)
        or receipt_payload.get("source_snapshot_sha256") != snapshot["sha256"]
        or receipt_payload.get("source_ready_sha256") != ready["sha256"]
        or not isinstance(receipt_payload.get("file_count"), int)
        or receipt_payload["file_count"] <= 0
    ):
        raise ValueError("source snapshot READY/full-rehash receipt closure differs")
    return {
        "root": str(root),
        "snapshot": snapshot,
        "ready": ready,
        "full_rehash_receipt": receipt,
        "file_count": receipt_payload["file_count"],
    }


def verify_protocol(binding: Any) -> tuple[dict[str, str], dict[str, Any]]:
    file_binding = verify_file_binding(binding, "metric_protocol")
    payload = read_json(Path(file_binding["path"]))
    if (
        payload.get("schema")
        not in {
            "geometry-selection-five-metric-protocol-v2",
            "geometry-selection-five-metric-traindev-protocol-v1",
        }
        or payload.get("status") != "frozen"
    ):
        raise ValueError("metric protocol schema mismatch")
    return file_binding, payload


def verify_input(binding: Any, scope: str, site: str) -> tuple[dict[str, str], dict[str, Any]]:
    file_binding = verify_file_binding(binding, "input_manifest")
    payload = read_json(Path(file_binding["path"]))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("input manifest entries must be a list")
    if scope == "engineering_smoke":
        if site != "Hippasus" or payload.get("schema") != HIPPASUS_INPUT_SCHEMA or payload.get("scope") != "engineering_smoke_not_formal_result" or payload.get("evaluation_site") != "Hippasus" or not 3 <= len(entries) <= 10:
            raise ValueError("engineering smoke requires a 3–10-entry Hippasus input lock")
    elif scope == "formal":
        if site != "Hippasus" or payload.get("schema") != HIPPASUS_INPUT_SCHEMA or payload.get("scope") != "formal_validation" or payload.get("evaluation_site") != "Hippasus" or len(entries) != 100:
            raise ValueError("formal scoring requires a 100-entry Hippasus input lock")
    else:
        method_id = payload.get("method_id")
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if (
            site != "Hippasus"
            or payload.get("schema") != TRAINDEV_INPUT_SCHEMA
            or payload.get("scope") != "train_dev_evaluation"
            or payload.get("evaluation_site") != "Hippasus"
            or payload.get("dataset_split") != "dev"
            or "formal_validation" in payload
            or payload.get("reserved_ids_disclosed") is not False
            or method_id not in TRAINDEV_METHOD_IDS
            or payload.get("method") != TRAINDEV_METHOD_LABELS.get(method_id)
            or any(
                token in serialized
                for token in ("formal_validation", "validation_only", "wan_unguided_seed0")
            )
            or not isinstance(payload.get("traindev_reference_isolation_receipt_sha256"), str)
            or len(entries) != 100
            or any(
                not isinstance(entry, dict) or entry.get("split_order") != index
                for index, entry in enumerate(entries)
            )
        ):
            raise ValueError("train-dev scoring requires an isolated 100-entry Hippasus dev input lock")
        require_sha256(
            payload["traindev_reference_isolation_receipt_sha256"],
            "train-dev reference-isolation receipt",
        )
    return file_binding, payload


def verify_schedule(binding: Any, protocol_binding: dict[str, str], protocol: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    file_binding = verify_file_binding(binding, "schedule")
    payload = read_json(Path(file_binding["path"]))
    if (
        payload.get("schema")
        != (
            "geometry-selection-metric-traindev-schedule-v1"
            if protocol.get("schema") == "geometry-selection-five-metric-traindev-protocol-v1"
            else "geometry-selection-metric-schedule-v2"
        )
        or payload.get("status") != "frozen"
    ):
        raise ValueError("metric schedule schema mismatch")
    if payload.get("metric_protocol_sha256") != protocol_binding["sha256"]:
        raise ValueError("metric schedule does not bind exact protocol SHA")
    if payload.get("video_contract") != protocol.get("video_contract"):
        raise ValueError("metric schedule video contract mismatch")
    if protocol.get("schema") == "geometry-selection-five-metric-traindev-protocol-v1":
        normalized = copy.deepcopy(payload)
        normalized["schema"] = "geometry-selection-metric-schedule-v2"
        normalized["metric_protocol_sha256"] = TRAINDEV_PARENT_PROTOCOL_SHA256
        if hashlib.sha256(canonical_bytes(normalized)).hexdigest() != TRAINDEV_PARENT_SCHEDULE_OBJECT_SHA256:
            raise ValueError("train-dev schedule changed outside schema/protocol identity")
    return file_binding, payload


def verify_weight_source_identity(role: str, source: Any) -> dict[str, Any]:
    """Recheck versioned and direct-content provenance inside a sealed manifest."""

    if not isinstance(source, dict):
        raise ValueError(f"weight source identity is malformed: {role}")
    provider = source.get("provider")
    repository = source.get("repository")
    kind = source.get("kind")
    origin_class = source.get("origin_class")
    url = source.get("url")
    if not all(isinstance(item, str) and item for item in (provider, repository, kind, origin_class, url)) or not url.startswith("https://"):
        raise ValueError(f"weight source identity lacks immutable origin fields: {role}")
    if origin_class == "versioned_repository":
        require_commit(source.get("revision"), f"weight source revision {role}")
    elif origin_class == "direct_content":
        if not isinstance(source.get("source_code_repository"), str) or not source["source_code_repository"]:
            raise ValueError(f"direct-content source lacks source repository: {role}")
        require_commit(source.get("source_code_revision"), f"direct-content source revision {role}")
        labels = source.get("source_snapshot_labels")
        if not isinstance(labels, list) or not labels or len(labels) != len(set(labels)) or not all(isinstance(label, str) and label for label in labels):
            raise ValueError(f"direct-content source lacks unique source snapshot labels: {role}")
    else:
        raise ValueError(f"unrecognized weight origin class: {role}")
    return source


def verify_source_requirement(role: str, source: dict[str, Any], requirement: Any) -> None:
    """Require every frozen family field supplied by the protocol, no aliases."""

    if not isinstance(requirement, dict):
        raise ValueError(f"protocol source requirement is malformed: {role}")
    for field, expected in requirement.items():
        if source.get(field) != expected:
            raise ValueError(f"weight source identity does not match frozen family requirement: {role}.{field}")


def verify_metric(name: str, value: Any, references: dict[str, Any], weight_source_requirements: dict[str, Any], site: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"metrics.{name} must be an object")
    code_commit = require_commit(value.get("code_commit"), f"metrics.{name}.code_commit")
    adapter_entrypoint = verify_file_binding(value.get("adapter_entrypoint"), f"metrics.{name}.adapter_entrypoint")
    core_evaluator = verify_file_binding(value.get("core_evaluator"), f"metrics.{name}.core_evaluator")
    code_manifest_binding, code_manifest = verify_content_manifest(value.get("code_manifest"), f"metrics.{name}.code_manifest", "geometry-selection-code-content-manifest-v1", True, site)
    if code_manifest["repository_commit"] != code_commit:
        raise ValueError(f"metrics.{name} code commit does not match code manifest")
    manifest_files = code_manifest["files"]
    if not any(item.get("role") == "metric_adapter_entrypoint" and {"path": item["path"], "sha256": item["sha256"]} == adapter_entrypoint for item in manifest_files):
        raise ValueError(f"metrics.{name} adapter entrypoint is not covered by its required runtime role")
    core_role = CORE_ENTRYPOINT_ROLE[name]
    if not any(item.get("role") == core_role and {"path": item["path"], "sha256": item["sha256"]} == core_evaluator for item in manifest_files):
        raise ValueError(f"metrics.{name} core evaluator is not covered by its required runtime role")
    role_to_file = {item["role"]: item for item in manifest_files}
    missing_code_roles = (REQUIRED_CODE_ROLES[name] | {"metric_adapter_entrypoint", "guarded_worker_launcher", "input_preflight_verifier", "evaluator_preflight_verifier", "evaluator_source_snapshot_verifier"}) - set(role_to_file)
    if missing_code_roles:
        raise ValueError(f"metrics.{name} code manifest lacks required runtime roles: {sorted(missing_code_roles)}")
    if name == "relative_total_motion_percent":
        verify_opencv_native_closure(manifest_files)
    adapter_common = {"path": role_to_file["metric_adapter_common"]["path"], "sha256": role_to_file["metric_adapter_common"]["sha256"]}
    weight_binding, weight_manifest = verify_content_manifest(value.get("weight_manifest"), f"metrics.{name}.weight_manifest", "geometry-selection-weight-content-manifest-v1", False, site)
    roles = {item["role"]: item for item in weight_manifest["files"]}
    if set(roles) != REQUIRED_WEIGHT_ROLES[name]:
        raise ValueError(
            f"metrics.{name} weight manifest roles differ from its exact frozen contract: "
            f"expected {sorted(REQUIRED_WEIGHT_ROLES[name])}, found {sorted(roles)}"
        )
    sources = weight_manifest.get("source_identities")
    metric_source_requirements = weight_source_requirements.get(name)
    if not isinstance(sources, dict) or not isinstance(metric_source_requirements, dict):
        raise ValueError(f"metrics.{name} weight source identities are malformed")
    if set(sources) != set(roles):
        raise ValueError(f"metrics.{name} weight source identities must cover every locked weight file")
    protocol_roles = metric_source_requirements.get("required_asset_roles")
    if not isinstance(protocol_roles, list) or len(protocol_roles) != len(set(protocol_roles)) or set(protocol_roles) != REQUIRED_WEIGHT_ROLES[name]:
        raise ValueError(f"metrics.{name} protocol required_asset_roles differ from the evaluator contract")
    for role, source in sources.items():
        verify_weight_source_identity(role, source)
    for role, expected_source in metric_source_requirements.items():
        if role == "required_asset_roles" or role.endswith("_requirement") or role == "status":
            continue
        if role not in REQUIRED_WEIGHT_ROLES[name]:
            raise ValueError(f"metrics.{name} protocol names an undeclared weight source role: {role}")
        verify_source_requirement(role, sources[role], expected_source)
    preprocessing = value.get("preprocessing")
    if not isinstance(preprocessing, dict) or not preprocessing:
        raise ValueError(f"metrics.{name}.preprocessing must be a nonempty object")
    reference_name = REFERENCE_BY_METRIC.get(name)
    if reference_name and name != "long_range_reprojection_error":
        reference = references.get(reference_name)
        expected_entrypoint_name = str(reference.get("entrypoint", "")).split("::", 1)[0] if isinstance(reference, dict) else ""
        if not isinstance(reference, dict) or code_commit != reference.get("commit") or core_evaluator["sha256"] != reference.get("entrypoint_sha256") or Path(core_evaluator["path"]).name != expected_entrypoint_name:
            raise ValueError(f"metrics.{name} core evaluator does not match the fixed protocol identity")
        if name == "met3r":
            dependencies = code_manifest.get("runtime_dependency_commits")
            if not isinstance(dependencies, dict) or dependencies.get("mast3r") != reference.get("mast3r_submodule_commit"):
                raise ValueError("MEt3R code manifest does not bind the pinned MASt3R submodule commit")
    if name == "long_range_reprojection_error":
        reference = references.get("independent_lre")
        expected = {
            "megasam_commit": reference.get("megasam_commit") if isinstance(reference, dict) else None,
            "camera_tracking_entrypoint": reference.get("camera_tracking_entrypoint") if isinstance(reference, dict) else None,
            "camera_tracking_entrypoint_sha256": reference.get("camera_tracking_entrypoint_sha256") if isinstance(reference, dict) else None,
            "consistent_depth_entrypoint": reference.get("consistent_depth_entrypoint") if isinstance(reference, dict) else None,
            "consistent_depth_entrypoint_sha256": reference.get("consistent_depth_entrypoint_sha256") if isinstance(reference, dict) else None,
            "correspondence_backbone": reference.get("correspondence_backbone") if isinstance(reference, dict) else None,
        }
        if not isinstance(reference, dict) or any(not isinstance(value, str) or not value for value in expected.values()):
            raise ValueError("Independent LRE protocol reference is malformed")
        for field, required_value in expected.items():
            if preprocessing.get(field) != required_value:
                raise ValueError(f"Independent LRE preprocessing does not bind its pinned {field}")
    if name == "relative_total_motion_percent":
        raft = references.get("raft_large")
        raft_sha = roles["relative_motion_raft_large_checkpoint"]["sha256"]
        if not isinstance(raft, dict) or preprocessing.get("torchvision_version") != raft.get("torchvision_version") or preprocessing.get("weights_enum") != raft.get("weights_enum") or preprocessing.get("checkpoint_url") != raft.get("checkpoint_url") or preprocessing.get("checkpoint_sha256") != raft_sha:
            raise ValueError("RAFT preprocessing does not match the fixed protocol identity")
    runtime_identities = runtime_identities_for_metric(name, role_to_file)
    return {
        "code_commit": code_commit,
        "adapter_entrypoint": adapter_entrypoint,
        "adapter_common": adapter_common,
        "core_evaluator": core_evaluator,
        "code_manifest": code_manifest_binding,
        "weight_manifest": weight_binding,
        "weight_source_identities": {role: sources[role] for role in sorted(REQUIRED_WEIGHT_ROLES[name])},
        "preprocessing": preprocessing,
        "runtime_identities": runtime_identities,
    }


def ensure_safe_derived_output(path: Path, derived_root: Path, inputs: list[Path]) -> None:
    if derived_root.is_symlink() or not derived_root.is_dir():
        raise ValueError("derived root must be an existing regular directory")
    derived = derived_root.resolve(strict=True)
    if "validation_wan_candidates" in derived.parts:
        raise ValueError("derived root must not overlap the frozen candidate root")
    if tuple(derived.parts[-3:]) != ("outputs", "geometry-selection", "hippasus_evaluation"):
        raise ValueError("derived root must be the approved Hippasus evaluation root")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("output parent must already exist as a regular directory")
    candidate = path.parent.resolve(strict=True) / path.name
    if candidate == derived or derived not in candidate.parents:
        raise ValueError("output must be strictly below the existing derived root")
    if candidate.relative_to(derived).parts[0] != "locks":
        raise ValueError("evaluator locks may only publish below the approved locks root")
    for raw_input in inputs:
        input_path = raw_input.resolve(strict=True)
        if candidate == input_path or candidate in input_path.parents or input_path in candidate.parents:
            raise ValueError("output must not overlap an input path")


def atomic_publish(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".evaluator-lock-", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(payload) + b"\n")
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    descriptor = read_json(args.descriptor)
    if descriptor.get("schema") != "geometry-selection-evaluator-lock-descriptor-v2":
        raise ValueError("unexpected evaluator-lock descriptor schema")
    scope = descriptor.get("scope")
    site = descriptor.get("site")
    if scope not in {"engineering_smoke", "train_dev", "formal"} or site != "Hippasus":
        raise ValueError("invalid scope or site")
    if descriptor.get("offline_runtime") is not True:
        raise ValueError("offline_runtime must be true")

    protocol_binding, protocol = verify_protocol(descriptor.get("metric_protocol"))
    input_binding, input_payload = verify_input(descriptor.get("input_manifest"), scope, site)
    traindev_provenance = None
    if scope == "train_dev":
        traindev_provenance = verify_train_dev_provenance(
            input_payload=input_payload,
            input_binding=input_binding,
            source_bundle_ready=descriptor.get("source_bundle_ready"),
            expected_generation_receipt_sha256=descriptor.get(
                "expected_generation_receipt_sha256"
            ),
            expected_baseline_eligibility_sha256=descriptor.get(
                "expected_baseline_eligibility_sha256"
            ),
        )
    if scope == "train_dev":
        dataset = protocol.get("dataset")
        derivation = protocol.get("derivation")
        preserved = derivation.get("preserved_parent_subtrees_sha256") if isinstance(derivation, dict) else None
        if (
            protocol.get("schema") != "geometry-selection-five-metric-traindev-protocol-v1"
            or protocol.get("scope") != "train_dev_evaluation"
            or not isinstance(dataset, dict)
            or dataset.get("split") != "dev"
            or dataset.get("case_count") != 100
            or "formal_validation" in dataset
            or dataset.get("reserved_ids_disclosed") is not False
            or dataset.get("source_manifest_sha256") != input_payload.get("source_manifest_sha256")
            or not isinstance(derivation, dict)
            or derivation.get("parent_protocol_sha256") != TRAINDEV_PARENT_PROTOCOL_SHA256
            or derivation.get("parent_schedule_sha256") != TRAINDEV_PARENT_SCHEDULE_SHA256
            or not isinstance(derivation.get("derivation_tool_sha256"), str)
            or derivation.get("approved_changed_json_pointers")
            != TRAINDEV_APPROVED_PROTOCOL_POINTERS
            or preserved != TRAINDEV_PRESERVED_SUBTREE_SHA256
        ):
            raise ValueError("train-dev evaluator lock requires the exact isolated derived protocol")
        require_sha256(derivation["derivation_tool_sha256"], "train-dev derivation tool")
        for pointer, key in (
            ("/metrics", "metrics"),
            ("/reference_evaluator_identities", "reference_evaluator_identities"),
            ("/video_contract", "video_contract"),
            ("/weight_source_requirements", "weight_source_requirements"),
        ):
            if hashlib.sha256(canonical_bytes(protocol.get(key))).hexdigest() != TRAINDEV_PRESERVED_SUBTREE_SHA256[pointer]:
                raise ValueError(f"train-dev protocol changed frozen subtree: {pointer}")
        serialized_protocol = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
        if any(token in serialized_protocol for token in ("formal_validation", "validation_only", "wan_unguided_seed0")):
            raise ValueError("train-dev protocol contains a prohibited held-out identity")
    elif protocol.get("schema") != "geometry-selection-five-metric-protocol-v2":
        raise ValueError("non-train-dev evaluator lock requires the frozen parent protocol schema")
    environment_binding, environment_payload = verify_environment(descriptor.get("environment_fingerprint"), site)
    source_snapshot_closure = verify_source_snapshot_closure(descriptor.get("source_snapshot_closure"))
    decoder, decoder_trusted_root = verify_trusted_binding(descriptor.get("decoder_binary"), descriptor.get("decoder_trusted_root"), "decoder_binary")
    schedule_binding, _ = verify_schedule(descriptor.get("schedule"), protocol_binding, protocol)
    if input_payload.get("metric_protocol_sha256") != protocol_binding["sha256"] or input_payload.get("metric_schedule_sha256") != schedule_binding["sha256"]:
        raise ValueError("input lock does not bind the exact protocol and schedule")
    method_id = input_payload.get("method_id")
    if not isinstance(method_id, str) or input_payload.get("candidate_budget") != protocol.get("candidate_budget_policy", {}).get(method_id):
        raise ValueError("input lock does not bind the fixed candidate-budget policy")
    metrics = descriptor.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != METRIC_NAMES:
        raise ValueError("metrics must contain exactly the five fixed metric names")
    references = protocol.get("reference_evaluator_identities")
    weight_source_requirements = protocol.get("weight_source_requirements")
    if not isinstance(references, dict) or not isinstance(weight_source_requirements, dict):
        raise ValueError("protocol lacks evaluator identity or model-source requirements")
    verified_metrics = {name: verify_metric(name, metrics[name], references, weight_source_requirements, site) for name in sorted(METRIC_NAMES)}
    for name, metric in verified_metrics.items():
        for distribution, identity in metric["runtime_identities"].items():
            if distribution not in {"torch", "torchvision"}:
                continue
            if (
                identity.get("environment_closure") != environment_payload["environment_closure"]
                or identity.get("wheelhouse_closure") != environment_payload["wheelhouse_closure"]
            ):
                raise ValueError(f"{name} {distribution} runtime differs from the frozen evaluator environment")
    parity_certificate = read_json(Path(environment_payload["torchvision_preprocessing_parity_certificate"]["path"]))
    parity_runner = parity_certificate.get("runner")
    if not isinstance(parity_runner, dict):
        raise ValueError("environment parity certificate lacks its runner binding")
    for name, metric in verified_metrics.items():
        manifest = read_json(Path(metric["code_manifest"]["path"]))
        runners = [
            {"path": item["path"], "sha256": item["sha256"]}
            for item in manifest["files"]
            if item.get("role") == "torchvision_preprocessing_parity_runner"
        ]
        if runners != [parity_runner]:
            raise ValueError(f"{name} does not bind the exact preprocessing parity runner")
    shared_identity, shared_identity_sha = build_shared_evaluator_identity(
        site=site,
        protocol_binding=protocol_binding,
        environment_binding=environment_binding,
        source_snapshot_closure=source_snapshot_closure,
        decoder=decoder,
        decoder_trusted_root=decoder_trusted_root,
        schedule_binding=schedule_binding,
        verified_metrics=verified_metrics,
    )
    payload = {
        "schema": "geometry-selection-evaluator-lock-v2",
        "scope": scope,
        "site": site,
        "offline_runtime": True,
        "metric_protocol": protocol_binding,
        "input_manifest": input_binding,
        "environment_fingerprint": environment_binding,
        "source_snapshot_closure": source_snapshot_closure,
        "decoder_binary": decoder,
        "decoder_trusted_root": decoder_trusted_root,
        "schedule": schedule_binding,
        "metrics": verified_metrics,
        "shared_evaluator_identity_sha256": shared_identity_sha,
    }
    if traindev_provenance is not None:
        payload["traindev_provenance"] = traindev_provenance
    protected_inputs = [args.descriptor, Path(protocol_binding["path"]), Path(input_binding["path"]), Path(environment_binding["path"]), Path(source_snapshot_closure["snapshot"]["path"]), Path(source_snapshot_closure["ready"]["path"]), Path(source_snapshot_closure["full_rehash_receipt"]["path"]), Path(decoder["path"]), Path(schedule_binding["path"])]
    if traindev_provenance is not None:
        protected_inputs.extend(
            Path(binding["path"])
            for key, binding in traindev_provenance.items()
            if key in {
                "source_bundle_ready",
                "generation_receipt",
                "manifest",
                "reference_isolation_receipt",
                "mirror_index",
                "base_input_lock",
                "adapted_entries",
            }
        )
    ensure_safe_derived_output(args.output, args.derived_root, protected_inputs)
    atomic_publish(args.output, payload)
    lock_sha = hashlib.sha256(canonical_bytes(payload) + b"\n").hexdigest()
    print(json.dumps({"evaluator_lock_sha256": lock_sha, "output": str(args.output), "scope": scope, "site": site}, sort_keys=True))


if __name__ == "__main__":
    main()
