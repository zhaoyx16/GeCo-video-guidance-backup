#!/usr/bin/env python3
"""Extract frozen high-resolution conditioning frames for a DL3DV manifest."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, get_token, hf_hub_download
from PIL import Image

from download_frozen_scenes import scene_lock, sha256_file


DEFAULT_REPO_ID = "DL3DV/DL3DV-ALL-960P"
MARKER_NAME = "conditioning.provenance.json"


def load_cases(manifest_path: Path) -> tuple[dict, list[tuple[str, dict]]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("_meta"), dict):
        raise ValueError("input manifest must contain an _meta object")
    cases: list[tuple[str, dict]] = []
    seen_scenes: set[str] = set()
    for case_id, case in payload.items():
        if case_id == "_meta":
            continue
        if not isinstance(case, dict):
            raise ValueError(f"invalid case {case_id}")
        scene_id = case.get("scene_id")
        image_prompt = case.get("image_prompt")
        transforms_path = case.get("transforms_path")
        if not isinstance(scene_id, str) or len(scene_id) != 64:
            raise ValueError(f"case {case_id} has invalid scene_id")
        if scene_id in seen_scenes:
            raise ValueError(f"manifest contains multiple cases for scene {scene_id}")
        if not isinstance(image_prompt, str) or not Path(image_prompt).name.endswith(".png"):
            raise ValueError(f"case {case_id} has invalid image_prompt")
        if not isinstance(transforms_path, str) or Path(transforms_path).name != "transforms.json":
            raise ValueError(f"case {case_id} has invalid transforms_path")
        provenance = case.get("source_scene_provenance")
        if not isinstance(provenance, dict) or not isinstance(
            provenance.get("transforms_sha256"), str
        ):
            raise ValueError(f"case {case_id} lacks source transforms provenance")
        seen_scenes.add(scene_id)
        cases.append((case_id, case))
    if not cases:
        raise ValueError("input manifest contains no cases")
    return payload, cases


def _safe_member(info: zipfile.ZipInfo) -> bool:
    member = PurePosixPath(info.filename)
    mode = info.external_attr >> 16
    return (
        bool(member.parts)
        and not member.is_absolute()
        and ".." not in member.parts
        and not stat.S_ISLNK(mode)
        and not info.is_dir()
    )


def find_frame_member(
    bundle: zipfile.ZipFile,
    scene_id: str,
    frame_name: str,
    source_image_dir: str,
    max_members: int = 10_000,
) -> zipfile.ZipInfo:
    if len(bundle.infolist()) > max_members:
        raise ValueError("conditioning archive has too many members")
    matches = []
    for info in bundle.infolist():
        if not _safe_member(info):
            if info.filename and not info.is_dir():
                raise ValueError(f"unsafe archive member {info.filename!r}")
            continue
        parts = PurePosixPath(info.filename).parts
        if (
            PurePosixPath(info.filename).name == frame_name
            and scene_id in parts
            and source_image_dir in parts
        ):
            matches.append(info)
    if len(matches) != 1:
        raise ValueError(
            f"expected one {source_image_dir}/{frame_name} member for {scene_id}, "
            f"found {len(matches)}"
        )
    return matches[0]


def validate_conditioning_image(path: Path, min_long_side: int) -> tuple[int, int, str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"invalid conditioning image {path}")
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"conditioning image is not PNG: {path}")
        image.verify()
    with Image.open(path) as image:
        image.load()
        width, height = image.size
    if max(width, height) < min_long_side:
        raise ValueError(
            f"conditioning image is below requested resolution: {(width, height)}"
        )
    return width, height, sha256_file(path)


def validate_installed_frame(
    target_dir: Path,
    *,
    frame_name: str,
    revision: str,
    repo_id: str,
    scene_id: str,
    min_long_side: int,
    require_directory_name: bool = True,
) -> dict:
    marker_path = target_dir / MARKER_NAME
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError(f"missing {MARKER_NAME}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema") != "dl3dv-conditioning-v1":
        raise ValueError("unsupported conditioning marker")
    if (
        (require_directory_name and target_dir.name != scene_id)
        or marker.get("scene_id") != scene_id
        or marker.get("repo_id") != repo_id
        or marker.get("dataset_revision") != revision
        or marker.get("frame_name") != frame_name
        or marker.get("archive_filename") != f"1K/{scene_id}.zip"
    ):
        raise ValueError("conditioning marker does not match requested source")
    width, height, image_sha256 = validate_conditioning_image(
        target_dir / frame_name, min_long_side
    )
    transforms = target_dir / "transforms.json"
    if transforms.is_symlink() or not transforms.is_file():
        raise ValueError("conditioning package lacks transforms.json")
    transforms_sha256 = sha256_file(transforms)
    expected = {
        "width": width,
        "height": height,
        "image_sha256": image_sha256,
        "transforms_sha256": transforms_sha256,
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise ValueError("conditioning image content does not match its marker")
    return marker


def extract_one(
    case_id: str,
    case: dict,
    *,
    output_root: Path,
    cache_root: Path,
    repo_id: str,
    revision: str,
    source_image_dir: str,
    min_long_side: int,
    max_archive_members: int,
) -> dict:
    scene_id = case["scene_id"]
    frame_name = Path(case["image_prompt"]).name
    source_transforms = Path(case["transforms_path"]).resolve()
    if not source_transforms.is_file():
        raise FileNotFoundError(f"source transforms not found: {source_transforms}")
    source_transforms_sha256 = sha256_file(source_transforms)
    expected_source = case.get("source_scene_provenance", {}).get("transforms_sha256")
    if expected_source != source_transforms_sha256:
        raise ValueError(f"source transforms provenance mismatch for {scene_id}")
    target_dir = output_root / "1K" / scene_id
    lock_path = output_root / ".locks" / f"{scene_id}.lock"
    with scene_lock(lock_path):
        if target_dir.exists():
            marker = validate_installed_frame(
                target_dir,
                frame_name=frame_name,
                revision=revision,
                repo_id=repo_id,
                scene_id=scene_id,
                min_long_side=min_long_side,
            )
            archive = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=f"1K/{scene_id}.zip",
                    repo_type="dataset",
                    revision=revision,
                    cache_dir=cache_root / "huggingface",
                    token=True,
                )
            )
            if (
                marker.get("archive_sha256") != sha256_file(archive)
                or marker.get("archive_bytes") != archive.stat().st_size
            ):
                raise RuntimeError(f"cached 960P archive provenance mismatch for {scene_id}")
            return {
                "case_id": case_id,
                "scene_id": scene_id,
                "status": "already_complete",
                "target": str(target_dir / frame_name),
                **marker,
            }

        archive = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=f"1K/{scene_id}.zip",
                repo_type="dataset",
                revision=revision,
                cache_dir=cache_root / "huggingface",
                token=True,
            )
        )
        archive_sha256 = sha256_file(archive)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = target_dir.parent / f".{scene_id}.conditioning-{uuid.uuid4().hex}"
        try:
            staging.mkdir(parents=True, exist_ok=False)
            target_image = staging / frame_name
            with zipfile.ZipFile(archive) as bundle:
                info = find_frame_member(
                    bundle,
                    scene_id,
                    frame_name,
                    source_image_dir,
                    max_archive_members,
                )
                if info.file_size > 100 * 1024 * 1024:
                    raise ValueError(f"conditioning frame is unexpectedly large: {info.file_size}")
                with bundle.open(info) as source, target_image.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
            width, height, image_sha256 = validate_conditioning_image(
                target_image, min_long_side
            )
            shutil.copy2(source_transforms, staging / "transforms.json")
            marker = {
                "schema": "dl3dv-conditioning-v1",
                "repo_id": repo_id,
                "dataset_revision": revision,
                "scene_id": scene_id,
                "archive_filename": f"1K/{scene_id}.zip",
                "archive_sha256": archive_sha256,
                "archive_bytes": archive.stat().st_size,
                "source_member": info.filename,
                "frame_name": frame_name,
                "width": width,
                "height": height,
                "image_sha256": image_sha256,
                "transforms_sha256": source_transforms_sha256,
            }
            (staging / MARKER_NAME).write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            validate_installed_frame(
                staging,
                frame_name=frame_name,
                revision=revision,
                repo_id=repo_id,
                scene_id=scene_id,
                min_long_side=min_long_side,
                require_directory_name=False,
            )
            os.rename(staging, target_dir)
            validate_installed_frame(
                target_dir,
                frame_name=frame_name,
                revision=revision,
                repo_id=repo_id,
                scene_id=scene_id,
                min_long_side=min_long_side,
            )
            return {
                "case_id": case_id,
                "scene_id": scene_id,
                "status": "extracted",
                "target": str(target_dir / frame_name),
                **marker,
            }
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-image-dir", default="images_4")
    parser.add_argument("--min-long-side", type=int, default=900)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-archive-members", type=int, default=10_000)
    args = parser.parse_args()
    if args.workers < 1 or args.min_long_side < 1 or args.max_archive_members < 1:
        parser.error("workers and min-long-side must be positive")
    if len(args.revision) != 40:
        parser.error("--revision must be an immutable commit SHA")

    input_manifest = args.input_manifest.resolve()
    manifest_sha256 = sha256_file(input_manifest)
    if manifest_sha256 != args.expected_manifest_sha256:
        raise ValueError(
            f"manifest SHA256 mismatch: {manifest_sha256} != {args.expected_manifest_sha256}"
        )
    payload, cases = load_cases(input_manifest)
    if get_token() is None:
        raise RuntimeError("no Hugging Face token found in the active HF_HOME")
    resolved_revision = HfApi().repo_info(
        args.repo_id, repo_type="dataset", revision=args.revision, token=True
    ).sha
    if resolved_revision != args.revision:
        raise RuntimeError("dataset revision did not resolve exactly")

    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_case = {
            executor.submit(
                extract_one,
                case_id,
                case,
                output_root=args.output_root.resolve(),
                cache_root=args.cache_root.resolve(),
                repo_id=args.repo_id,
                revision=resolved_revision,
                source_image_dir=args.source_image_dir,
                min_long_side=args.min_long_side,
                max_archive_members=args.max_archive_members,
            ): (case_id, case)
            for case_id, case in cases
        }
        for future in as_completed(future_to_case):
            case_id, case = future_to_case[future]
            try:
                result = future.result()
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
            except Exception as error:
                failure = {
                    "case_id": case_id,
                    "scene_id": case["scene_id"],
                    "error": repr(error),
                }
                failures.append(failure)
                print(json.dumps(failure, sort_keys=True), flush=True)

    report = {
        "schema": "dl3dv-conditioning-extraction-v1",
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": manifest_sha256,
        "repo_id": args.repo_id,
        "dataset_revision": resolved_revision,
        "source_image_dir": args.source_image_dir,
        "expected_cases": len(cases),
        "completed_cases": len(results),
        "failed_cases": len(failures),
        "results": sorted(results, key=lambda item: item["case_id"]),
        "failures": sorted(failures, key=lambda item: item["case_id"]),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report_tmp = args.report.with_name(f".{args.report.name}.{uuid.uuid4().hex}.tmp")
    report_tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_tmp.replace(args.report)
    if failures:
        raise SystemExit(f"{len(failures)} conditioning frame extractions failed")

    result_by_case = {result["case_id"]: result for result in results}
    output_payload = json.loads(json.dumps(payload))
    output_payload["_meta"]["source_manifest"] = str(input_manifest)
    output_payload["_meta"]["source_manifest_sha256"] = manifest_sha256
    output_payload["_meta"]["conditioning_repo_id"] = args.repo_id
    output_payload["_meta"]["conditioning_dataset_revision"] = resolved_revision
    output_payload["_meta"]["conditioning_source_image_dir"] = args.source_image_dir
    for case_id, _ in cases:
        result = result_by_case[case_id]
        output_payload[case_id]["image_prompt"] = result["target"]
        output_payload[case_id]["transforms_path"] = str(
            Path(result["target"]).parent / "transforms.json"
        )
        output_payload[case_id]["conditioning_image_provenance"] = {
            key: result[key]
            for key in (
                "repo_id",
                "dataset_revision",
                "archive_sha256",
                "source_member",
                "image_sha256",
                "width",
                "height",
                "transforms_sha256",
            )
        }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = args.output_manifest.with_name(
        f".{args.output_manifest.name}.{uuid.uuid4().hex}.tmp"
    )
    output_tmp.write_text(
        json.dumps(output_payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    output_tmp.replace(args.output_manifest)


if __name__ == "__main__":
    main()
