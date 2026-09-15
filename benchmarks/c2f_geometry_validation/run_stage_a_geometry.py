#!/usr/bin/env python3
"""Compare frozen C2F matches with pinned VGGT-Omega reprojections."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_selection.backbones.vggt_omega import VGGTOmegaAdapter  # noqa: E402
from geometry_selection.projection import (  # noqa: E402
    bilinear_sample,
    camera_to_world,
    project_camera,
    world_to_camera,
)


STATUS_COLOURS = {
    "accept": (44, 196, 92),
    "reject_reprojection": (238, 76, 65),
    "reject_front_conflict": (255, 145, 45),
    "abstain_occluded": (88, 140, 220),
    "abstain_low_confidence": (155, 155, 155),
    "abstain_out_of_bounds": (115, 115, 115),
    "abstain_invalid": (90, 90, 90),
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def git_identity() -> dict:
    commit = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True).strip()
    )
    return {"commit": commit, "dirty": dirty}


def decode_frames(video: Path, indices: list[int], output_dir: Path) -> tuple[list[Path], tuple[int, int]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    selected = set(indices)
    paths = {}
    size = None
    try:
        for frame_index in range(indices[-1] + 1):
            ok, frame_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"could not decode frame {frame_index} from {video}")
            if frame_index not in selected:
                continue
            height, width = frame_bgr.shape[:2]
            size = size or (height, width)
            if size != (height, width):
                raise RuntimeError("video frame size changed")
            path = output_dir / f"frame_{frame_index:04d}.png"
            if not cv2.imwrite(str(path), frame_bgr):
                raise RuntimeError(f"failed to write {path}")
            paths[frame_index] = path
    finally:
        capture.release()
    if set(paths) != selected or size is None:
        raise RuntimeError("not all requested frames were decoded")
    return [paths[index] for index in indices], size


def sample_record_columns(record: dict, target_times: set[int]) -> list[dict]:
    columns = record["samples"]
    lengths = {key: len(value) for key, value in columns.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"sample columns have inconsistent lengths: {lengths}")
    rows = []
    for index in range(next(iter(lengths.values()), 0)):
        row = {key: value[index] for key, value in columns.items()}
        if int(row["target_time"]) in target_times:
            rows.append(row)
    return rows


def token_to_processed(value: np.ndarray, cells: int, pixels: int) -> np.ndarray:
    return (value.astype(np.float64) + 0.5) * pixels / cells - 0.5


def processed_to_original(value: float, processed_pixels: int, original_pixels: int) -> float:
    return (value + 0.5) * original_pixels / processed_pixels - 0.5


def project_rows(
    rows: list[dict],
    geometry,
    frame_to_position: dict[int, int],
    token_grid: tuple[int, int, int],
    temporal_scale: int,
    confidence_percentile: float,
    depth_tolerance: float,
    max_error_tokens: float,
) -> list[dict]:
    _, token_h, token_w = token_grid
    processed_h, processed_w = geometry.image_size_hw
    confidence_thresholds = np.percentile(geometry.confidence, confidence_percentile, axis=(1, 2))
    projected_rows = []

    grouped: dict[tuple[int, int], list[tuple[int, dict]]] = {}
    for row_index, row in enumerate(rows):
        source_time = int(row["source_time"])
        target_time = int(row["target_time"])
        grouped.setdefault((source_time, target_time), []).append((row_index, row))

    results_by_index: dict[int, dict] = {}
    for (source_time, target_time), indexed_rows in grouped.items():
        source_frame = source_time * temporal_scale
        target_frame = target_time * temporal_scale
        source_position = frame_to_position[source_frame]
        target_position = frame_to_position[target_frame]
        source = geometry.world_to_camera[source_position]
        target = geometry.world_to_camera[target_position]
        k_source = geometry.intrinsics[source_position]
        k_target = geometry.intrinsics[target_position]

        source_x_token = np.asarray([item[1]["source_x"] for item in indexed_rows], dtype=np.float64)
        source_y_token = np.asarray([item[1]["source_y"] for item in indexed_rows], dtype=np.float64)
        target_x_token = np.asarray([item[1]["target_x"] for item in indexed_rows], dtype=np.float64)
        target_y_token = np.asarray([item[1]["target_y"] for item in indexed_rows], dtype=np.float64)
        source_u = token_to_processed(source_x_token, token_w, processed_w)
        source_v = token_to_processed(source_y_token, token_h, processed_h)
        c2f_target_u = token_to_processed(target_x_token, token_w, processed_w)
        c2f_target_v = token_to_processed(target_y_token, token_h, processed_h)

        source_depth = bilinear_sample(geometry.depth[source_position], source_u, source_v)
        source_conf = bilinear_sample(geometry.confidence[source_position], source_u, source_v)
        depth = source_depth.values
        source_points = np.stack(
            [
                (source_u - k_source[0, 2]) / k_source[0, 0] * depth,
                (source_v - k_source[1, 2]) / k_source[1, 1] * depth,
                depth,
            ],
            axis=-1,
        )
        points_world = camera_to_world(source_points, source)
        points_target = world_to_camera(points_world, target)
        geometry_u, geometry_v, projected_depth = project_camera(points_target, k_target)
        target_depth = bilinear_sample(geometry.depth[target_position], geometry_u, geometry_v)
        target_conf = bilinear_sample(geometry.confidence[target_position], geometry_u, geometry_v)

        finite = (
            np.isfinite(depth)
            & np.isfinite(source_conf.values)
            & np.isfinite(geometry_u)
            & np.isfinite(geometry_v)
            & np.isfinite(projected_depth)
            & np.isfinite(target_depth.values)
            & np.isfinite(target_conf.values)
        )
        positive = (depth > 0.0) & (projected_depth > 0.0) & (target_depth.values > 0.0)
        in_bounds = source_depth.in_bounds & source_conf.in_bounds & target_depth.in_bounds & target_conf.in_bounds
        source_confident = source_conf.values >= confidence_thresholds[source_position]
        target_confident = target_conf.values >= confidence_thresholds[target_position]
        depth_ratio = (projected_depth - target_depth.values) / np.maximum(target_depth.values, 1e-8)
        dx_tokens = (geometry_u - c2f_target_u) / (processed_w / token_w)
        dy_tokens = (geometry_v - c2f_target_v) / (processed_h / token_h)
        error_tokens = np.sqrt(dx_tokens**2 + dy_tokens**2)

        for local_index, (row_index, row) in enumerate(indexed_rows):
            if not finite[local_index] or not positive[local_index]:
                status = "abstain_invalid"
            elif not in_bounds[local_index]:
                status = "abstain_out_of_bounds"
            elif not source_confident[local_index] or not target_confident[local_index]:
                status = "abstain_low_confidence"
            elif depth_ratio[local_index] > depth_tolerance:
                status = "abstain_occluded"
            elif depth_ratio[local_index] < -depth_tolerance:
                status = "reject_front_conflict"
            elif error_tokens[local_index] <= max_error_tokens:
                status = "accept"
            else:
                status = "reject_reprojection"

            result = dict(row)
            result.update(
                source_frame=source_frame,
                target_frame=target_frame,
                source_u=float(source_u[local_index]),
                source_v=float(source_v[local_index]),
                c2f_target_u=float(c2f_target_u[local_index]),
                c2f_target_v=float(c2f_target_v[local_index]),
                geometry_target_u=float(geometry_u[local_index]),
                geometry_target_v=float(geometry_v[local_index]),
                source_depth=float(depth[local_index]),
                projected_target_depth=float(projected_depth[local_index]),
                observed_target_depth=float(target_depth.values[local_index]),
                source_geometry_confidence=float(source_conf.values[local_index]),
                target_geometry_confidence=float(target_conf.values[local_index]),
                target_depth_relative_delta=float(depth_ratio[local_index]),
                reprojection_error_tokens=float(error_tokens[local_index]),
                geometry_status=status,
            )
            results_by_index[row_index] = result

    for row_index in range(len(rows)):
        projected_rows.append(results_by_index[row_index])
    return projected_rows


def finite_summary(values: list[float]) -> dict:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "median": None, "p90": None, "mean": None}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "mean": float(array.mean()),
    }


def summarize_rows(rows: list[dict]) -> dict:
    counts = Counter(row["geometry_status"] for row in rows)
    evidence = counts["accept"] + counts["reject_reprojection"] + counts["reject_front_conflict"]
    return {
        "sample_count": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "evidence_count": evidence,
        "evidence_coverage": evidence / len(rows) if rows else 0.0,
        "accept_fraction_of_evidence": counts["accept"] / evidence if evidence else None,
        "reprojection_error_tokens_all_finite": finite_summary(
            [row["reprojection_error_tokens"] for row in rows]
        ),
        "reprojection_error_tokens_evidence": finite_summary(
            [
                row["reprojection_error_tokens"]
                for row in rows
                if row["geometry_status"] in {"accept", "reject_reprojection", "reject_front_conflict"}
            ]
        ),
        "target_depth_relative_delta": finite_summary(
            [row["target_depth_relative_delta"] for row in rows]
        ),
    }


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(result, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def evenly_limit(rows: list[dict], limit: int) -> list[dict]:
    if len(rows) <= limit:
        return rows
    indices = np.linspace(0, len(rows) - 1, limit).round().astype(int)
    return [rows[index] for index in indices]


def make_visualization(
    rows: list[dict],
    frame_paths: dict[int, Path],
    token_grid: tuple[int, int, int],
    processed_hw: tuple[int, int],
    target_time: int,
    output: Path,
) -> None:
    _, token_h, token_w = token_grid
    processed_h, processed_w = processed_hw
    target_rows = [row for row in rows if int(row["target_time"]) == target_time]
    if not target_rows:
        raise RuntimeError(f"no sampled correspondences for target time {target_time}")
    source_times = sorted({int(row["source_time"]) for row in target_rows})
    temporal_scale = int(target_rows[0]["target_frame"]) // target_time
    source_panels = []
    original_size = None
    for source_time in source_times:
        frame_index = source_time * temporal_scale
        panel = cv2.imread(str(frame_paths[frame_index]), cv2.IMREAD_COLOR)
        if panel is None:
            raise RuntimeError(f"could not read {frame_paths[frame_index]}")
        panel = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        original_size = panel.shape[:2]
        group = sorted(
            [row for row in target_rows if int(row["source_time"]) == source_time],
            key=lambda row: (int(row["source_y"]), int(row["source_x"])),
        )
        for row in evenly_limit(group, 100):
            colour = STATUS_COLOURS[row["geometry_status"]]
            x = int(round((float(row["source_x"]) + 0.5) * panel.shape[1] / token_w - 0.5))
            y = int(round((float(row["source_y"]) + 0.5) * panel.shape[0] / token_h - 0.5))
            cv2.circle(panel, (x, y), 4, colour, -1, cv2.LINE_AA)
        source_panels.append(add_label(panel, f"source frame {frame_index} (lag {target_time-source_time})"))

    target_frame = target_time * temporal_scale
    target = cv2.imread(str(frame_paths[target_frame]), cv2.IMREAD_COLOR)
    if target is None:
        raise RuntimeError(f"could not read {frame_paths[target_frame]}")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    original_h, original_w = target.shape[:2]
    target_overlay = target.copy()
    ordered = sorted(target_rows, key=lambda row: (row["geometry_status"], int(row["target_y"]), int(row["target_x"])))
    for row in evenly_limit(ordered, 180):
        colour = STATUS_COLOURS[row["geometry_status"]]
        c2f_x = int(round((float(row["target_x"]) + 0.5) * original_w / token_w - 0.5))
        c2f_y = int(round((float(row["target_y"]) + 0.5) * original_h / token_h - 0.5))
        geo_x = int(round(processed_to_original(row["geometry_target_u"], processed_w, original_w)))
        geo_y = int(round(processed_to_original(row["geometry_target_v"], processed_h, original_h)))
        if 0 <= geo_x < original_w and 0 <= geo_y < original_h:
            cv2.line(target_overlay, (c2f_x, c2f_y), (geo_x, geo_y), colour, 1, cv2.LINE_AA)
            cv2.drawMarker(target_overlay, (geo_x, geo_y), colour, cv2.MARKER_TILTED_CROSS, 8, 1)
        cv2.circle(target_overlay, (c2f_x, c2f_y), 4, colour, 1, cv2.LINE_AA)
    counts = Counter(row["geometry_status"] for row in target_rows)
    target_overlay = add_label(
        target_overlay,
        f"target frame {target_frame}: circle=C2F, cross=geometry | "
        f"accept {counts['accept']} reject {counts['reject_reprojection']+counts['reject_front_conflict']} "
        f"abstain {sum(v for k,v in counts.items() if k.startswith('abstain'))}",
    )

    panel_width = max(1, original_w // len(source_panels))
    resized_sources = []
    for panel in source_panels:
        panel_height = int(round(panel.shape[0] * panel_width / panel.shape[1]))
        resized_sources.append(cv2.resize(panel, (panel_width, panel_height), interpolation=cv2.INTER_AREA))
    source_row = np.concatenate(resized_sources, axis=1)
    if source_row.shape[1] != original_w:
        source_row = cv2.resize(source_row, (original_w, source_row.shape[0]), interpolation=cv2.INTER_AREA)
    sheet = np.concatenate([source_row, target_overlay], axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=REPO_ROOT / "benchmarks/c2f_geometry_validation/STAGE_A_LOCK.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/vol/dissolve/yz10325/outputs/c2f_geometry_validation_0915/stage_a_v1"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--replay-source", choices=("dense", "sparse"), default="dense")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    if args.case_index is not None and (args.num_shards != 1 or args.shard_index != 0):
        raise ValueError("--case-index cannot be combined with sharding")

    lock_path = args.lock.resolve()
    lock = read_json(lock_path)
    if lock.get("schema") != "c2f-external-geometry-stage-a-lock-v1":
        raise RuntimeError("unexpected Stage A lock schema")
    identity = git_identity()
    if identity["dirty"] and not args.allow_dirty:
        raise RuntimeError("refusing to run from a dirty worktree")

    cases = lock["cases"]
    if args.case_index is not None:
        if not 0 <= args.case_index < len(cases):
            raise ValueError(f"case index must be in [0,{len(cases)-1}]")
        cases = [cases[args.case_index]]
    else:
        cases = [case for index, case in enumerate(cases) if index % args.num_shards == args.shard_index]

    replay_key = "dense_replay" if args.replay_source == "dense" else "sparse_p0_replay"
    for case in cases:
        replay_path = Path(case[replay_key])
        if not replay_path.is_file():
            raise FileNotFoundError(replay_path)
        if not Path(case["baseline_video"]).is_file():
            raise FileNotFoundError(case["baseline_video"])
    if args.validate_only:
        print(f"validated {len(cases)} cases from {replay_key}")
        return

    geometry_config = lock["geometry"]
    checkpoint = Path(geometry_config["checkpoint"])
    if checkpoint.stat().st_size != geometry_config["checkpoint_size"]:
        raise RuntimeError("VGGT-Omega checkpoint size differs from lock")
    adapter = VGGTOmegaAdapter(
        source_root=Path(geometry_config["source_root"]),
        checkpoint=checkpoint,
        device=args.device,
        image_resolution=int(geometry_config["image_resolution"]),
        preprocessing_mode=geometry_config["preprocessing_mode"],
        require_official_commit=True,
    )
    load_started = time.perf_counter()
    adapter.load()
    torch.cuda.synchronize(torch.device(args.device))
    model_load_seconds = time.perf_counter() - load_started
    print(f"loaded VGGT-Omega in {model_load_seconds:.2f}s", flush=True)

    observation = lock["c2f_observation"]
    evidence_rule = lock["evidence_rule"]
    expected_grid = tuple(observation["expected_token_grid_thw"])
    target_times = set(int(value) for value in observation["target_token_times"])
    frame_indices = [int(value) for value in lock["frame_indices"]]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for ordinal, case in enumerate(cases, start=1):
        case_id = case["case_id"]
        case_dir = output_root / case_id
        complete_path = case_dir / "COMPLETE.json"
        if complete_path.is_file():
            complete = read_json(complete_path)
            if complete.get("lock_sha256") == sha256_file(lock_path) and complete.get("replay_source") == args.replay_source:
                print(f"[{ordinal}/{len(cases)}] skip complete {case_id}")
                continue
            raise RuntimeError(f"stale complete output at {case_dir}")
        if case_dir.exists():
            raise RuntimeError(f"refusing to overwrite partial output at {case_dir}")
        frames_dir = case_dir / "frames"
        frames_dir.mkdir(parents=True)

        replay_path = Path(case[replay_key])
        replay = read_json(replay_path)
        if replay["replay_mode"] != observation["replay_mode"]:
            raise RuntimeError(f"wrong replay mode for {case_id}")
        if args.replay_source == "dense" and replay["sample_size"] != observation["dense_sample_size_per_layer_step"]:
            raise RuntimeError(f"wrong dense sample size for {case_id}: {replay['sample_size']}")
        matching_records = [
            record
            for record in replay["diagnostics"]["records"]
            if int(record["step"]) == observation["step"] and int(record["layer"]) == observation["layer"]
        ]
        if len(matching_records) != 1:
            raise RuntimeError(f"expected one diagnostic record for {case_id}, got {len(matching_records)}")
        record = matching_records[0]
        token_grid = tuple(record["token_grid"])
        if token_grid != expected_grid:
            raise RuntimeError(f"token grid mismatch for {case_id}: {token_grid}")
        sample_rows = sample_record_columns(record, target_times)
        if not sample_rows:
            raise RuntimeError(f"no selected target-time samples for {case_id}")

        video = Path(case["baseline_video"])
        frame_paths, original_hw = decode_frames(video, frame_indices, frames_dir)
        frame_path_by_index = dict(zip(frame_indices, frame_paths))
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        geometry_started = time.perf_counter()
        geometry = adapter.predict_image_paths(
            frame_paths,
            keyframe_indices=frame_indices,
            hash_checkpoint=False,
        )
        torch.cuda.synchronize(torch.device(args.device))
        geometry_seconds = time.perf_counter() - geometry_started
        peak_memory_mib = torch.cuda.max_memory_allocated(torch.device(args.device)) / 1024**2
        frame_to_position = {int(frame): index for index, frame in enumerate(geometry.keyframe_indices)}
        projected_rows = project_rows(
            sample_rows,
            geometry,
            frame_to_position,
            token_grid,
            int(observation["temporal_pixel_frames_per_token"]),
            float(evidence_rule["confidence_percentile"]),
            float(evidence_rule["depth_relative_tolerance"]),
            float(evidence_rule["max_reprojection_error_tokens"]),
        )
        write_rows(case_dir / "CORRESPONDENCES.csv", projected_rows)
        np.savez_compressed(
            case_dir / "GEOMETRY.npz",
            world_to_camera=geometry.world_to_camera,
            intrinsics=geometry.intrinsics,
            depth=geometry.depth,
            confidence=geometry.confidence,
            keyframe_indices=geometry.keyframe_indices,
        )
        for target_time in sorted(target_times):
            make_visualization(
                projected_rows,
                frame_path_by_index,
                token_grid,
                geometry.image_size_hw,
                target_time,
                case_dir / f"target_t{target_time:02d}_geometry_vs_c2f.png",
            )

        by_target = {
            str(target_time): summarize_rows(
                [row for row in projected_rows if int(row["target_time"]) == target_time]
            )
            for target_time in sorted(target_times)
        }
        summary = {
            "schema": "c2f-external-geometry-stage-a-case-v1",
            "case_id": case_id,
            "motion_stratum": case["motion_stratum"],
            "replay_source": args.replay_source,
            "baseline_video": str(video),
            "baseline_video_sha256": case["baseline_video_sha256"],
            "replay_path": str(replay_path),
            "replay_sha256": sha256_file(replay_path),
            "lock_path": str(lock_path),
            "lock_sha256": sha256_file(lock_path),
            "code_identity": identity,
            "token_grid_thw": list(token_grid),
            "original_image_size_hw": list(original_hw),
            "processed_image_size_hw": list(geometry.image_size_hw),
            "geometry_metadata": geometry.metadata,
            "model_load_seconds_shared": model_load_seconds,
            "geometry_forward_seconds": geometry_seconds,
            "geometry_peak_allocated_mib": peak_memory_mib,
            "all_targets": summarize_rows(projected_rows),
            "by_target_time": by_target,
        }
        atomic_json(case_dir / "STAGE_A_CASE.json", summary)
        atomic_json(
            complete_path,
            {
                "status": "complete",
                "case_id": case_id,
                "lock_sha256": sha256_file(lock_path),
                "replay_source": args.replay_source,
                "case_record_sha256": sha256_file(case_dir / "STAGE_A_CASE.json"),
                "correspondences_sha256": sha256_file(case_dir / "CORRESPONDENCES.csv"),
            },
        )
        print(
            f"[{ordinal}/{len(cases)}] {case_id} geometry={geometry_seconds:.2f}s "
            f"peak={peak_memory_mib:.1f}MiB coverage={summary['all_targets']['evidence_coverage']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
