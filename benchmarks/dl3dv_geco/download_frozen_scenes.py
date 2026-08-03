#!/usr/bin/env python3
"""Reliably download a frozen DL3DV scene split from Hugging Face."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import stat
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from huggingface_hub import HfApi, get_token, hf_hub_download
from PIL import Image


REPO_ID = "DL3DV/DL3DV-ALL-480P"
VALID_SPLITS = {"debug", "validation", "test"}
COMPLETION_MARKER = ".dl3dv_complete.json"


@dataclass(frozen=True)
class SceneRecord:
    split: str
    split_order: int
    source_order: int
    scene_hash: str
    batch: str
    duration: float


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_expected_counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        try:
            split, raw_count = value.split("=", 1)
            count = int(raw_count)
        except ValueError as error:
            raise ValueError(f"invalid expected count {value!r}; use split=count") from error
        if split not in VALID_SPLITS or count < 0 or split in counts:
            raise ValueError(f"invalid expected count {value!r}")
        counts[split] = count
    if set(counts) != VALID_SPLITS:
        raise ValueError(f"expected counts must cover exactly {sorted(VALID_SPLITS)}")
    return counts


def load_frozen_split(
    path: Path,
    selected_splits: set[str],
    expected_counts: dict[str, int] | None = None,
) -> list[SceneRecord]:
    if not selected_splits or not selected_splits <= VALID_SPLITS:
        raise ValueError(f"selected splits must be a non-empty subset of {VALID_SPLITS}")
    all_records: list[SceneRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "split_order", "source_order", "hash", "batch", "duration"}
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError(
                f"frozen split columns must be exactly {sorted(required)}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            split = row["split"]
            if split not in VALID_SPLITS:
                raise ValueError(f"invalid split {split!r}")
            scene_hash = row["hash"]
            if len(scene_hash) != 64 or any(char not in "0123456789abcdef" for char in scene_hash):
                raise ValueError(f"invalid scene hash {scene_hash!r}")
            if row["batch"] != "1K":
                raise ValueError(f"scene {scene_hash} is not in the 1K batch")
            record = SceneRecord(
                split=split,
                split_order=int(row["split_order"]),
                source_order=int(row["source_order"]),
                scene_hash=scene_hash,
                batch=row["batch"],
                duration=float(row["duration"]),
            )
            if record.split_order < 0 or record.source_order < 1 or record.duration <= 0:
                raise ValueError(f"invalid metadata for scene {scene_hash}")
            all_records.append(record)

    hashes = [record.scene_hash for record in all_records]
    if len(hashes) != len(set(hashes)):
        raise ValueError("frozen split reuses a scene hash across one or more splits")
    source_orders = [record.source_order for record in all_records]
    if len(source_orders) != len(set(source_orders)):
        raise ValueError("frozen split reuses a source_order")
    split_orders = [(record.split, record.split_order) for record in all_records]
    if len(split_orders) != len(set(split_orders)):
        raise ValueError("frozen split reuses a split_order")

    actual_counts = {
        split: sum(record.split == split for record in all_records)
        for split in sorted(VALID_SPLITS)
    }
    if expected_counts is not None and actual_counts != expected_counts:
        raise ValueError(
            f"frozen split counts do not match expectations: "
            f"actual={actual_counts}, expected={expected_counts}"
        )

    records = [record for record in all_records if record.split in selected_splits]
    return sorted(
        records,
        key=lambda item: (
            {"debug": 0, "validation": 1, "test": 2}[item.split],
            item.split_order,
        ),
    )


def _regular_file(path: Path) -> bool:
    try:
        return not path.is_symlink() and stat.S_ISREG(path.stat(follow_symlinks=False).st_mode)
    except OSError:
        return False


def validate_scene_files(scene_root: Path) -> dict[str, object]:
    transforms = scene_root / "transforms.json"
    image_root = scene_root / "images_8"
    if not _regular_file(transforms) or not image_root.is_dir() or image_root.is_symlink():
        raise ValueError("scene lacks regular transforms.json or images_8 directory")
    try:
        payload = json.loads(transforms.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid transforms.json") from error
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) < 8:
        raise ValueError("transforms.json contains fewer than eight frames")

    required_images: list[Path] = []
    seen_names: set[str] = set()
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or not isinstance(frame.get("file_path"), str):
            raise ValueError(f"frame {index} has no valid file_path")
        matrix = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"frame {index} has no finite 4x4 transform_matrix")
        name = Path(frame["file_path"]).name
        if name in seen_names:
            raise ValueError(f"duplicate frame filename {name!r}")
        seen_names.add(name)
        required_images.append(image_root / name)

    digest = hashlib.sha256()
    image_bytes = 0
    for image_path in required_images:
        if not _regular_file(image_path) or image_path.stat().st_size <= 0:
            raise ValueError(f"missing, empty, or non-regular image {image_path.name}")
        try:
            with Image.open(image_path) as image:
                if image.format != "PNG":
                    raise ValueError(f"unexpected image format for {image_path.name}")
                image.verify()
            with Image.open(image_path) as image:
                image.load()
        except (OSError, SyntaxError) as error:
            raise ValueError(f"corrupt image {image_path.name}") from error
        size = image_path.stat().st_size
        file_hash = sha256_file(image_path)
        digest.update(f"{image_path.name}\0{size}\0{file_hash}\n".encode("utf-8"))
        image_bytes += size

    actual_names = {
        path.name
        for path in image_root.iterdir()
        if path.name.startswith("frame_") and path.suffix.lower() == ".png"
    }
    if actual_names != seen_names:
        raise ValueError(
            f"images_8 frame set differs from transforms.json: "
            f"required={len(seen_names)}, actual={len(actual_names)}"
        )
    return {
        "transforms_sha256": sha256_file(transforms),
        "frame_count": len(required_images),
        "frames_digest_sha256": digest.hexdigest(),
        "image_bytes": image_bytes,
    }


def read_completion_marker(scene_root: Path) -> dict[str, object]:
    marker = scene_root / COMPLETION_MARKER
    if not _regular_file(marker):
        raise ValueError(f"missing {COMPLETION_MARKER}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("schema") != "dl3dv-scene-complete-v1":
        raise ValueError("unsupported completion marker")
    return payload


def scene_is_complete(
    scene_root: Path,
    revision: str | None = None,
    repo_id: str | None = None,
    scene_hash: str | None = None,
) -> bool:
    try:
        marker = read_completion_marker(scene_root)
        if revision is not None and marker.get("dataset_revision") != revision:
            return False
        if repo_id is not None and marker.get("repo_id") != repo_id:
            return False
        if scene_hash is not None:
            if scene_root.name != scene_hash or marker.get("scene_hash") != scene_hash:
                return False
            if marker.get("archive_filename") != f"1K/{scene_hash}.zip":
                return False
        validation = validate_scene_files(scene_root)
        return all(marker.get(key) == value for key, value in validation.items())
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def safe_extract_zip(
    archive: Path,
    destination: Path,
    max_uncompressed_bytes: int,
    max_members: int = 10_000,
) -> None:
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        if len(infos) > max_members:
            raise ValueError(f"archive has {len(infos)} members, above limit {max_members}")
        total = sum(info.file_size for info in infos)
        if total > max_uncompressed_bytes:
            raise ValueError(
                f"archive expands to {total} bytes, above limit {max_uncompressed_bytes}"
            )
        for info in infos:
            member = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if (
                member.is_absolute()
                or not member.parts
                or ".." in member.parts
                or stat.S_ISLNK(mode)
            ):
                raise ValueError(f"unsafe zip member {info.filename!r}")
            resolved = (destination / Path(*member.parts)).resolve()
            if destination.resolve() not in resolved.parents and resolved != destination.resolve():
                raise ValueError(f"zip member escapes destination: {info.filename!r}")
        destination.mkdir(parents=True, exist_ok=False)
        bundle.extractall(destination)


def locate_extracted_scene(staging: Path, scene_hash: str) -> Path:
    direct = staging / scene_hash
    candidates = [direct] if direct.is_dir() else []
    try:
        validate_scene_files(staging)
    except ValueError:
        pass
    else:
        candidates.append(staging)
    candidates.extend(path for path in staging.rglob(scene_hash) if path.is_dir() and path != direct)
    complete: list[Path] = []
    for path in candidates:
        try:
            validate_scene_files(path)
        except ValueError:
            continue
        complete.append(path)
    unique = {path.resolve() for path in complete}
    if len(unique) != 1:
        raise ValueError(
            f"expected exactly one complete extracted scene {scene_hash}, found {len(unique)}"
        )
    return next(iter(unique))


@contextmanager
def scene_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def download_one(
    record: SceneRecord,
    *,
    output_root: Path,
    cache_root: Path,
    repo_id: str,
    revision: str,
    max_uncompressed_bytes: int,
    max_archive_members: int,
) -> dict[str, object]:
    batch_root = output_root / record.batch
    target = batch_root / record.scene_hash
    batch_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".locks" / f"{record.scene_hash}.lock"
    with scene_lock(lock_path):
        if target.exists():
            if not scene_is_complete(target, revision, repo_id, record.scene_hash):
                raise RuntimeError(f"refusing to overwrite incomplete or mismatched scene {target}")
            marker = read_completion_marker(target)
            archive = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{record.batch}/{record.scene_hash}.zip",
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
                raise RuntimeError(f"cached archive provenance mismatch for {record.scene_hash}")
            return {
                **asdict(record),
                "status": "already_complete",
                "target": str(target),
                "dataset_revision": revision,
                "archive_sha256": marker["archive_sha256"],
                "archive_bytes": marker["archive_bytes"],
            }

        staging = batch_root / f".{record.scene_hash}.extract-{uuid.uuid4().hex}"
        try:
            archive = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{record.batch}/{record.scene_hash}.zip",
                    repo_type="dataset",
                    revision=revision,
                    cache_dir=cache_root / "huggingface",
                    token=True,
                )
            )
            archive_sha256 = sha256_file(archive)
            archive_bytes = archive.stat().st_size
            safe_extract_zip(
                archive,
                staging,
                max_uncompressed_bytes,
                max_archive_members,
            )
            extracted = locate_extracted_scene(staging, record.scene_hash)
            validation = validate_scene_files(extracted)
            marker = {
                "schema": "dl3dv-scene-complete-v1",
                "repo_id": repo_id,
                "dataset_revision": revision,
                "scene_hash": record.scene_hash,
                "archive_filename": f"{record.batch}/{record.scene_hash}.zip",
                "archive_sha256": archive_sha256,
                "archive_bytes": archive_bytes,
                **validation,
            }
            (extracted / COMPLETION_MARKER).write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            if target.exists():
                raise RuntimeError(f"scene target appeared while lock was held: {target}")
            if not scene_is_complete(extracted, revision, repo_id, record.scene_hash):
                raise RuntimeError(f"scene failed pre-install validation: {extracted}")
            os.rename(extracted, target)
            if not scene_is_complete(target, revision, repo_id, record.scene_hash):
                quarantine = batch_root / f".{record.scene_hash}.invalid-{uuid.uuid4().hex}"
                os.rename(target, quarantine)
                raise RuntimeError(f"scene failed post-install validation: {target}")
            return {
                **asdict(record),
                "status": "downloaded",
                "target": str(target),
                "dataset_revision": revision,
                "archive_sha256": archive_sha256,
                "archive_bytes": archive_bytes,
            }
        finally:
            if staging.exists():
                import shutil

                shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument(
        "--expected-count",
        action="append",
        required=True,
        help="Expected full-CSV count as split=count; specify debug, validation, and test.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="Immutable Hugging Face commit SHA.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(VALID_SPLITS),
        default=["debug", "validation"],
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--max-uncompressed-gib", type=float, default=5.0)
    parser.add_argument("--max-archive-members", type=int, default=10_000)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_uncompressed_gib <= 0:
        parser.error("--max-uncompressed-gib must be positive")
    if args.max_archive_members < 1:
        parser.error("--max-archive-members must be positive")
    if len(args.revision) != 40 or any(char not in "0123456789abcdef" for char in args.revision):
        parser.error("--revision must be a lowercase 40-character commit SHA")

    run_id = uuid.uuid4().hex
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def write_report(payload: dict[str, object]) -> None:
        temporary = args.report.with_name(f".{args.report.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(args.report)

    write_report(
        {
            "schema": "dl3dv-frozen-download-v2",
            "status": "running",
            "run_id": run_id,
            "started_unix": time.time(),
        }
    )

    split_csv = args.split_csv.resolve()
    actual_split_sha256 = sha256_file(split_csv)
    if actual_split_sha256 != args.expected_split_sha256:
        raise ValueError(
            f"split CSV SHA256 mismatch: actual={actual_split_sha256}, "
            f"expected={args.expected_split_sha256}"
        )
    expected_counts = parse_expected_counts(args.expected_count)
    records = load_frozen_split(split_csv, set(args.splits), expected_counts)
    if not records:
        raise ValueError("the selected splits contain no scenes")
    if get_token() is None:
        raise RuntimeError("no Hugging Face token found in the active HF_HOME")
    resolved_revision = HfApi().repo_info(
        args.repo_id, repo_type="dataset", revision=args.revision, token=True
    ).sha
    if resolved_revision != args.revision:
        raise RuntimeError(
            f"dataset revision did not resolve exactly: {resolved_revision} != {args.revision}"
        )

    max_bytes = int(args.max_uncompressed_gib * 1024**3)
    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_record = {
            executor.submit(
                download_one,
                record,
                output_root=args.output_root.resolve(),
                cache_root=args.cache_root.resolve(),
                repo_id=args.repo_id,
                revision=args.revision,
                max_uncompressed_bytes=max_bytes,
                max_archive_members=args.max_archive_members,
            ): record
            for record in records
        }
        for future in as_completed(future_to_record):
            record = future_to_record[future]
            try:
                result = future.result()
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
            except Exception as error:  # Preserve all failures in the final report.
                failure = {"scene_hash": record.scene_hash, "error": repr(error)}
                failures.append(failure)
                print(json.dumps(failure, sort_keys=True), flush=True)

    report = {
        "schema": "dl3dv-frozen-download-v2",
        "status": "failed" if failures else "success",
        "run_id": run_id,
        "repo_id": args.repo_id,
        "dataset_revision": resolved_revision,
        "split_csv": str(split_csv),
        "split_csv_sha256": actual_split_sha256,
        "expected_split_counts": expected_counts,
        "selected_splits": sorted(set(args.splits)),
        "expected_scenes": len(records),
        "completed_scenes": len(results),
        "failed_scenes": len(failures),
        "results": sorted(results, key=lambda item: int(item["source_order"])),
        "failures": sorted(failures, key=lambda item: item["scene_hash"]),
    }
    write_report(report)
    if failures:
        raise SystemExit(f"{len(failures)} scene downloads failed; rerun to resume")


if __name__ == "__main__":
    main()
