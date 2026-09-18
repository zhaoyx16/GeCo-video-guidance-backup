#!/usr/bin/env python3
"""Create a self-contained navigation baseline manifest for another cluster."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--include-review",
        action="store_true",
        help="Copy trajectory contact sheets when present.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source = Path(args.manifest_dir).resolve()
    output = Path(args.output_dir).resolve()
    input_dir = output / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    cases_path = source / "navigation_cases.json"
    jobs_path = source / "jobs.jsonl"
    cases = json.load(open(cases_path))
    packaged_files = []

    for case_id, item in cases.items():
        image_source = Path(item["image_prompt"])
        if not image_source.exists():
            raise FileNotFoundError(f"Missing first frame for {case_id}: {image_source}")
        suffix = image_source.suffix.lower() or ".png"
        image_target = input_dir / f"{case_id}{suffix}"
        shutil.copy2(image_source, image_target)
        item["source_image_prompt"] = str(image_source)
        item["image_prompt"] = str(Path("inputs") / image_target.name)
        packaged_files.append(image_target)

    with open(output / "navigation_cases.json", "w") as handle:
        json.dump(cases, handle, indent=2)
        handle.write("\n")

    jobs = []
    with open(jobs_path) as handle:
        for line in handle:
            job = json.loads(line)
            job["config_path"] = "navigation_cases.json"
            jobs.append(job)
    with open(output / "jobs.jsonl", "w") as handle:
        for job in jobs:
            handle.write(json.dumps(job) + "\n")

    for name in ("summary.json", "trajectory_report.csv"):
        candidate = source / name
        if candidate.exists():
            shutil.copy2(candidate, output / name)

    if args.include_review:
        review_source = source / "trajectory_contact_sheets"
        if review_source.exists():
            shutil.copytree(
                review_source,
                output / "trajectory_contact_sheets",
                dirs_exist_ok=True,
            )

    checksums = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(packaged_files)
    }
    (output / "input_sha256.json").write_text(json.dumps(checksums, indent=2) + "\n")

    model_counts: dict[str, int] = {}
    for job in jobs:
        model_counts[job["model"]] = model_counts.get(job["model"], 0) + 1
    report = {
        "cases": len(cases),
        "jobs": len(jobs),
        "jobs_by_model": model_counts,
        "input_images": len(packaged_files),
        "portable_paths": True,
    }
    (output / "package_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
