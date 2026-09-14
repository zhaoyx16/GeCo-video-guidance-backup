#!/usr/bin/env python3
"""Run one sealed evaluator-v13 metric on an immutable 25-case dev lock."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
UPSTREAM = Path("/vol/dissolve/yz10325/tmp/run_metric_formal_nonlre_v6.py")
UPSTREAM_SHA256 = "4eaf72f93c17457cb16353f394c3a92c169fac67e20e28af6d82cb0e2ef853dd"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_upstream():
    if sha256_file(UPSTREAM) != UPSTREAM_SHA256:
        raise RuntimeError("sealed evaluator launcher SHA mismatch")
    spec = importlib.util.spec_from_file_location("_sealed_metric_launcher", UPSTREAM)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {UPSTREAM}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    base = load_upstream()
    metrics = dict(base.METRICS)
    metrics["met3r_multiscale"] = {
        "adapter": "met3r_multiscale_adapter.py",
        "core": "met3r/met3r/met3r.py",
        "roles": list(base.METRICS["met3r"]["roles"]),
        "records_per_entry": 16,
    }
    metrics["relative_total_motion_raw"] = {
        "adapter": "relative_motion_raw_adapter.py",
        "core": str(base.RELATIVE_MOTION_CORE),
        "roles": ["relative_motion_raft_large_checkpoint"],
        "records_per_entry": 1,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("metric", choices=sorted(metrics))
    parser.add_argument("--device", required=True, help="one physical Hippasus GPU index")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-lock", type=Path, required=True)
    parser.add_argument("--input-lock-sha256", required=True)
    parser.add_argument("--input-label", required=True)
    parser.add_argument("--expected-entries", type=int, default=25)
    args = parser.parse_args()

    if not args.device.isdigit():
        raise ValueError("--device must be one physical GPU index")
    if args.expected_entries != 25:
        raise ValueError("this launcher is frozen for the 25-case development gate")
    final_binding = base.require_final_environment()
    base.require_readonly(base.SOURCE / "SOURCE_SNAPSHOT.json", base.SOURCE_SNAPSHOT_SHA)
    base.require_readonly(base.ASSET_RECEIPT, base.ASSET_RECEIPT_SHA)
    input_lock = base.require_readonly(args.input_lock, args.input_lock_sha256)
    lock_payload = json.loads(input_lock.read_text(encoding="utf-8"))
    entries = lock_payload.get("entries")
    if not isinstance(entries, list) or len(entries) != args.expected_entries:
        raise ValueError(f"input lock must contain exactly {args.expected_entries} entries")
    if lock_payload.get("scope") != "development_validation_disjoint_from_frozen_test_validation_debug":
        raise ValueError("input lock is not the frozen disjoint development scope")
    if lock_payload.get("reserved_overlap_counts") != {"test": 0, "validation": 0, "debug": 0}:
        raise ValueError("input lock does not certify zero reserved-split overlap")
    base.require_readonly(base.SCHEDULE, base.SCHEDULE_SHA)

    output_root = args.output_root
    if output_root.is_symlink():
        raise ValueError("output root cannot be a symlink")
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not output_root.is_dir() or output_root.stat().st_uid != os.getuid():
        raise ValueError("output root is not a user-owned directory")
    contracts = output_root / "contracts"
    components = output_root / "components"
    logs = output_root / "logs"
    for directory in (contracts, components, logs):
        directory.mkdir(mode=0o700, exist_ok=True)

    metric = args.metric
    metric_spec = metrics[metric]
    adapter_dir = base.SOURCE / "protocol/metric_adapters"
    adapter_source = None
    custom_adapter_names = {
        "met3r_multiscale": "met3r_multiscale_adapter.py",
        "relative_total_motion_raw": "relative_motion_raw_adapter.py",
    }
    if metric in custom_adapter_names:
        adapter_source = HERE / custom_adapter_names[metric]
        if adapter_source.is_symlink() or not adapter_source.is_file():
            raise ValueError("custom metric adapter source must be a regular file")
        adapter_copy = contracts / custom_adapter_names[metric]
        descriptor = os.open(adapter_copy, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(adapter_source.read_bytes())
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.chmod(adapter_copy, 0o444)
        adapter = base.require_readonly(adapter_copy, sha256_file(adapter_source))
    elif metric == "met3r":
        adapter = base.require_readonly(base.MET3R_FIXED_ADAPTER, base.MET3R_FIXED_ADAPTER_SHA)
    else:
        adapter = base.require_readonly(adapter_dir / metric_spec["adapter"])
    common = base.require_readonly(adapter_dir / "metric_adapter_common.py")
    core_value = Path(metric_spec["core"])
    core = base.require_readonly(core_value if core_value.is_absolute() else base.SOURCE / core_value)
    core_source_receipt = None
    if metric in {"relative_total_motion_percent", "relative_total_motion_raw"}:
        if core != base.RELATIVE_MOTION_CORE or base.sha256_file(core) != base.RELATIVE_MOTION_CORE_SHA:
            raise RuntimeError("relative-motion core differs from the frozen source")
        source_receipt = base.require_readonly(
            base.RELATIVE_MOTION_CORE_RECEIPT, base.RELATIVE_MOTION_CORE_RECEIPT_SHA
        )
        core_source_receipt = {
            "path": str(source_receipt),
            "sha256": base.RELATIVE_MOTION_CORE_RECEIPT_SHA,
        }
    decoder = base.require_readonly(
        base.ENV_ROOT / "lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
    )

    acquisition = json.loads(base.ASSET_RECEIPT.read_text(encoding="utf-8"))
    by_role = {item["role"]: item for item in acquisition.get("assets", [])}
    if set(metric_spec["roles"]) - set(by_role):
        raise RuntimeError("asset receipt lacks a metric role")
    files = []
    sources = {}
    for role in metric_spec["roles"]:
        item = by_role[role]
        path = base.require_readonly(base.ASSETS / item["path"], item["sha256"])
        files.append({"role": role, "path": str(path), "sha256": item["sha256"], "bytes": item["bytes"]})
        sources[role] = item["source"]
    weight_manifest_path = contracts / f"{metric}.weights.json"
    weight_manifest = {
        "schema": "geometry-selection-weight-content-manifest-v1",
        "site": "Hippasus",
        "offline_ready": True,
        "load_closure_complete": True,
        "files": files,
        "source_identities": sources,
        "asset_acquisition_receipt": {
            "path": str(base.ASSET_RECEIPT),
            "sha256": base.ASSET_RECEIPT_SHA,
        },
    }
    weight_sha = base.atomic_readonly_json(weight_manifest_path, weight_manifest)

    component = components / f"{metric}.json"
    run_receipt = output_root / f"{metric}.RUN.json"
    stdout_path = logs / f"{metric}.stdout.log"
    stderr_path = logs / f"{metric}.stderr.log"
    for path in (component, run_receipt, stdout_path, stderr_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)

    env = base.base_environment(args.device)
    runtime_identities = {}
    if metric in {"relative_total_motion_percent", "relative_total_motion_raw"}:
        runtime_identities = base.opencv_runtime_identity(env, adapter_dir)
        adapter_wrapper = base.relative_motion_runtime_wrapper(adapter_dir)
    elif metric == "vbench_quality":
        adapter_wrapper, runtime_identities = base.vbench_runtime_bridge()
    elif metric == "geco_fused":
        adapter_wrapper = base.geco_fused_runtime_wrapper(adapter_dir)
    elif metric == "long_range_reprojection_error":
        adapter_wrapper, runtime_identities = base.independent_lre_runtime_wrapper(adapter_dir)
    elif metric in {"met3r", "met3r_multiscale"}:
        adapter_wrapper, runtime_identities = base.met3r_runtime_wrapper(adapter_dir)
    else:
        adapter_wrapper = (
            "import runpy,sys;"
            "adapter=sys.argv[1];"
            "sys.argv=sys.argv[1:];"
            f"sys.path.insert(0,{str(adapter_dir)!r});"
            "runpy.run_path(adapter,run_name='__main__')"
        )
    env.update(
        {
            "GEOMETRY_EVAL_LOCKED_ADAPTER_ENTRYPOINT_PATH": str(adapter),
            "GEOMETRY_EVAL_LOCKED_ADAPTER_ENTRYPOINT_SHA256": base.sha256_file(adapter),
            "GEOMETRY_EVAL_LOCKED_ADAPTER_COMMON_PATH": str(common),
            "GEOMETRY_EVAL_LOCKED_ADAPTER_COMMON_SHA256": base.sha256_file(common),
            "GEOMETRY_EVAL_LOCKED_CORE_EVALUATOR_PATH": str(core),
            "GEOMETRY_EVAL_LOCKED_CORE_EVALUATOR_SHA256": base.sha256_file(core),
            "GEOMETRY_EVAL_LOCKED_WEIGHT_MANIFEST_PATH": str(weight_manifest_path),
            "GEOMETRY_EVAL_LOCKED_WEIGHT_MANIFEST_SHA256": weight_sha,
            "GEOMETRY_EVAL_LOCKED_SCHEDULE_PATH": str(base.SCHEDULE),
            "GEOMETRY_EVAL_LOCKED_SCHEDULE_SHA256": base.SCHEDULE_SHA,
            "GEOMETRY_EVAL_LOCKED_DECODER_BINARY": str(decoder),
            "GEOMETRY_EVAL_LOCKED_DECODER_SHA256": base.sha256_file(decoder),
            "GEOMETRY_EVAL_LOCKED_INPUT_LOCK_PATH": str(input_lock),
            "GEOMETRY_EVAL_LOCKED_INPUT_LOCK_SHA256": args.input_lock_sha256,
            "GEOMETRY_EVAL_METRIC_OUTPUT_PATH": str(component),
            "GEOMETRY_EVAL_LOCKED_RUNTIME_IDENTITIES_JSON": json.dumps(
                runtime_identities, sort_keys=True, separators=(",", ":")
            ),
        }
    )
    gpu = base.gpu_snapshot(env)
    command = [
        str(base.ENV_ROOT / "bin/python"),
        "-I",
        "-c",
        adapter_wrapper,
        str(adapter),
        "--input-lock",
        str(input_lock),
        "--core-evaluator",
        str(core),
        "--weight-manifest",
        str(weight_manifest_path),
        "--schedule",
        str(base.SCHEDULE),
        "--decoder",
        str(decoder),
        "--output",
        str(component),
        "--device",
        "cuda",
    ]
    started = int(time.time())
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env=env)
    os.chmod(stdout_path, 0o444)
    os.chmod(stderr_path, 0o444)
    if completed.returncode != 0:
        raise RuntimeError(f"metric failed with exit {completed.returncode}; see {stderr_path}")
    base.require_readonly(component)
    payload = json.loads(component.read_text(encoding="utf-8"))
    expected_records = args.expected_entries * metric_spec["records_per_entry"]
    if (
        payload.get("schema") != "geometry-selection-metric-component-v1"
        or payload.get("metric_id") != metric
        or len(payload.get("records", [])) != expected_records
    ):
        raise RuntimeError(f"metric component failed validation; expected {expected_records} records")

    this_launcher = Path(__file__).resolve()
    receipt = {
        "schema": "geometry-selection-evaluator-v13-disjoint-dev25-component-run-v1",
        "status": "complete",
        "metric_id": metric,
        "input_profile": args.input_label,
        "input_lock": {
            "path": str(input_lock),
            "sha256": args.input_lock_sha256,
            "entries": args.expected_entries,
        },
        "schedule": {"path": str(base.SCHEDULE), "sha256": base.SCHEDULE_SHA},
        "source_snapshot_sha256": base.SOURCE_SNAPSHOT_SHA,
        "environment_final_ready": {"path": final_binding["path"], "sha256": final_binding["sha256"]},
        "adapter": {"path": str(adapter), "sha256": base.sha256_file(adapter)},
        "adapter_source": (
            {"path": str(adapter_source.resolve()), "sha256": sha256_file(adapter_source)}
            if adapter_source is not None
            else None
        ),
        "core": {"path": str(core), "sha256": base.sha256_file(core)},
        "core_source_receipt": core_source_receipt,
        "weights": {"path": str(weight_manifest_path), "sha256": weight_sha},
        "decoder": {"path": str(decoder), "sha256": base.sha256_file(decoder)},
        "component": {"path": str(component), "sha256": base.sha256_file(component), "records": expected_records},
        "physical_cuda_visible_devices": args.device,
        "gpu_at_start": gpu,
        "runtime_identities": runtime_identities,
        "launcher": {"path": str(this_launcher), "sha256": sha256_file(this_launcher)},
        "upstream_launcher": {"path": str(UPSTREAM), "sha256": UPSTREAM_SHA256},
        "started_unix_seconds": started,
        "finished_unix_seconds": int(time.time()),
    }
    receipt_sha = base.atomic_readonly_json(run_receipt, receipt)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "metric": metric,
                "component": str(component),
                "run_receipt": str(run_receipt),
                "run_receipt_sha256": receipt_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
