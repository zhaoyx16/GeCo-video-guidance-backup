#!/usr/bin/env python3
"""Run a resumable shard of the navigation baseline manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


WAN_CONFIG = {
    "steps": 50,
    "frames": 121,
    "height": 704,
    "width": 1280,
    "fps": 24,
}

COSMOS_CONFIG = {
    "steps": 36,
    "frames": 93,
    "height": 704,
    "width": 1280,
    "fps": 16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--gpu",
        default=None,
        help=(
            "Optional CUDA_VISIBLE_DEVICES override. Leave unset under Slurm so the worker "
            "uses the GPU assigned by the scheduler."
        ),
    )
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--model", choices=("wan", "cosmos"), required=True)
    parser.add_argument("--wan-model", default=os.environ.get("WAN_MODEL_PATH"))
    parser.add_argument("--cosmos-model", default=os.environ.get("COSMOS_MODEL_PATH"))
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def expected_output(output_root: Path, job: dict) -> Path:
    base = output_root / "baselines" / job["model"] / job["dataset"] / job["case_id"]
    if job["model"] == "wan":
        config = WAN_CONFIG
        return base / (
            f"baseline_seed{job['seed']}_steps{config['steps']}_frames{config['frames']}.mp4"
        )
    config = COSMOS_CONFIG
    return base / (
        f"baseline_seed{job['seed']}_steps{config['steps']}_frames{config['frames']}_"
        f"{config['height']}x{config['width']}.mp4"
    )


def load_jobs(path: Path, model: str) -> list[dict]:
    jobs = []
    with open(path) as handle:
        for line in handle:
            item = json.loads(line)
            if item["model"] == model:
                config_path = Path(item["config_path"])
                if not config_path.is_absolute():
                    config_path = path.resolve().parent / config_path
                item["config_path"] = str(config_path)
                jobs.append(item)
    return jobs


def main() -> None:
    args = parse_args()
    repo = Path(args.repo)
    output_root = Path(args.output_root)
    all_jobs = load_jobs(Path(args.jobs), args.model)
    jobs = [
        job for index, job in enumerate(all_jobs) if index % args.num_workers == args.worker_index
    ]
    if args.limit > 0:
        jobs = jobs[: args.limit]

    state_root = output_root / "job_state"
    log_root = output_root / "logs" / args.model
    state_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    if args.gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    visible_devices = environment.get("CUDA_VISIBLE_DEVICES", "<scheduler/default>")

    print(
        f"worker model={args.model} visible_gpus={visible_devices} "
        f"shard={args.worker_index}/{args.num_workers} "
        f"jobs={len(jobs)}",
        flush=True,
    )
    failures = 0
    for ordinal, job in enumerate(jobs, start=1):
        output = expected_output(output_root, job)
        done_path = state_root / f"{job['job_id']}.done.json"
        failed_path = state_root / f"{job['job_id']}.failed.json"
        if output.exists() and output.stat().st_size > 0:
            done_path.write_text(
                json.dumps({"job": job, "output": str(output), "status": "existing"}, indent=2)
                + "\n"
            )
            print(f"[{ordinal}/{len(jobs)}] skip existing {output}", flush=True)
            continue

        case_output_root = output_root / "baselines" / args.model / job["dataset"]
        if args.model == "wan":
            command = [
                sys.executable,
                str(repo / "run_wan_geco_case_full.py"),
                "--case",
                job["case_id"],
                "--prompt_json",
                job["config_path"],
                "--output_root",
                str(case_output_root),
                "--mode",
                "baseline",
                "--height",
                str(WAN_CONFIG["height"]),
                "--width",
                str(WAN_CONFIG["width"]),
                "--frames",
                str(WAN_CONFIG["frames"]),
                "--steps",
                str(WAN_CONFIG["steps"]),
                "--fps",
                str(WAN_CONFIG["fps"]),
                "--seed",
                str(job["seed"]),
            ]
            if args.wan_model:
                command.extend(["--model", args.wan_model])
        else:
            command = [
                sys.executable,
                str(repo / "run_cosmos_geco_case.py"),
                "--case",
                job["case_id"],
                "--prompt_json",
                job["config_path"],
                "--output_root",
                str(case_output_root),
                "--mode",
                "baseline",
                "--profile",
                "model_default",
                "--seed",
                str(job["seed"]),
            ]
            if args.cosmos_model:
                command.extend(["--model", args.cosmos_model])

        log_path = log_root / f"{job['job_id']}.log"
        started = time.time()
        print(
            f"[{ordinal}/{len(jobs)}] start {job['job_id']} output={output}",
            flush=True,
        )
        with open(log_path, "w") as log:
            result = subprocess.run(
                command,
                cwd=repo,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = time.time() - started
        state = {
            "job": job,
            "command": command,
            "output": str(output),
            "log": str(log_path),
            "returncode": result.returncode,
            "elapsed_sec": elapsed,
        }
        if result.returncode == 0 and output.exists() and output.stat().st_size > 0:
            failed_path.unlink(missing_ok=True)
            done_path.write_text(json.dumps({**state, "status": "done"}, indent=2) + "\n")
            print(f"[{ordinal}/{len(jobs)}] done {job['job_id']} {elapsed:.1f}s", flush=True)
        else:
            failures += 1
            failed_path.write_text(json.dumps({**state, "status": "failed"}, indent=2) + "\n")
            print(
                f"[{ordinal}/{len(jobs)}] FAILED {job['job_id']} rc={result.returncode} "
                f"log={log_path}",
                flush=True,
            )

    if failures:
        raise SystemExit(f"{failures} jobs failed")


if __name__ == "__main__":
    main()
