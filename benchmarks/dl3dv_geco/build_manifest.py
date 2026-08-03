#!/usr/bin/env python3
"""Build a pose-driven DL3DV benchmark manifest.

The script selects large-camera-motion clips from DL3DV transforms.json files
and assigns camera-only prompts derived from the measured pose trajectory.
It deliberately avoids scene-object instructions and "everything is static"
language so prompt ambiguity is not baked into the benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np


PROMPTS = {
    "forward": (
        "A continuous first-person camera shot beginning from the provided image. "
        "The camera moves smoothly forward over a long distance through the environment, "
        "maintaining steady progress and a large viewpoint change. One uninterrupted take "
        "with steady speed and constant focal length, without zooms, cuts, or teleportation."
    ),
    "forward_left": (
        "A continuous first-person camera shot beginning from the provided image. "
        "The camera moves smoothly forward over a long distance, follows a broad leftward "
        "curve, and continues forward into the newly revealed area. One uninterrupted take "
        "with steady speed and constant focal length, without zooms, cuts, or teleportation."
    ),
    "forward_right": (
        "A continuous first-person camera shot beginning from the provided image. "
        "The camera moves smoothly forward over a long distance, follows a broad rightward "
        "curve, and continues forward into the newly revealed area. One uninterrupted take "
        "with steady speed and constant focal length, without zooms, cuts, or teleportation."
    ),
    "lateral_left": (
        "A continuous first-person camera shot beginning from the provided image. "
        "The camera moves smoothly left while continuing through the environment, producing "
        "a large viewpoint change. One uninterrupted take with steady speed and constant "
        "focal length, without zooms, cuts, or teleportation."
    ),
    "lateral_right": (
        "A continuous first-person camera shot beginning from the provided image. "
        "The camera moves smoothly right while continuing through the environment, producing "
        "a large viewpoint change. One uninterrupted take with steady speed and constant "
        "focal length, without zooms, cuts, or teleportation."
    ),
    "orbit_left": (
        "A continuous first-person camera shot beginning from the provided image. "
        "The camera follows a broad smooth leftward arc through the environment with a large "
        "viewpoint change. One uninterrupted take with steady speed and constant focal length, "
        "without zooms, cuts, or teleportation."
    ),
    "orbit_right": (
        "A continuous first-person camera shot beginning from the provided image. "
        "The camera follows a broad smooth rightward arc through the environment with a large "
        "viewpoint change. One uninterrupted take with steady speed and constant focal length, "
        "without zooms, cuts, or teleportation."
    ),
}


@dataclass(frozen=True)
class Candidate:
    scene_id: str
    transforms_path: Path
    image_path: Path
    start_index: int
    end_index: int
    motion_class: str
    path_length: float
    displacement: float
    rotation_deg: float
    signed_yaw_deg: float
    forward: float
    lateral: float
    straightness: float
    score: float


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(value)))


def signed_yaw_deg(relative_rotation: np.ndarray) -> float:
    # Camera-to-world convention: use the horizontal x-z components of the
    # relative camera orientation. The sign is only used to choose prompt text.
    return math.degrees(math.atan2(relative_rotation[0, 2], relative_rotation[2, 2]))


def classify_motion(
    forward: float,
    lateral: float,
    yaw_deg: float,
    rotation_deg: float,
) -> str | None:
    abs_yaw = abs(yaw_deg)
    if forward < 0 and abs(forward) >= 0.6 * abs(lateral):
        return None
    if forward > 0 and forward >= 0.6 * abs(lateral):
        if yaw_deg > 18:
            return "forward_left"
        if yaw_deg < -18:
            return "forward_right"
        return "forward"
    if abs(lateral) > max(abs(forward), 1e-8):
        return "lateral_right" if lateral > 0 else "lateral_left"
    if rotation_deg >= 25:
        return "orbit_left" if yaw_deg > 0 else "orbit_right"
    return "forward"


def resolve_image(scene_dir: Path, frame_record: dict, image_subdir: str) -> Path | None:
    file_name = Path(frame_record["file_path"]).name
    candidates = [
        scene_dir / image_subdir / file_name,
        scene_dir / frame_record["file_path"],
        scene_dir / "images" / file_name,
    ]
    return next((path for path in candidates if path.is_file()), None)


def iter_transforms(roots: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        if root.is_file() and root.name == "transforms.json":
            paths = [root]
        else:
            paths = root.rglob("transforms.json")
        for path in paths:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved


def scene_candidates(
    transforms_path: Path,
    pose_window: int,
    start_stride: int,
    image_subdir: str,
) -> list[Candidate]:
    data = json.loads(transforms_path.read_text())
    frames = data.get("frames", [])
    if len(frames) < pose_window:
        return []

    poses = np.asarray([frame["transform_matrix"] for frame in frames], dtype=np.float64)
    centers = poses[:, :3, 3]
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    nonzero_steps = steps[steps > 1e-8]
    median_step = float(np.median(nonzero_steps)) if nonzero_steps.size else 1.0
    scene_id = transforms_path.parent.name
    candidates: list[Candidate] = []

    for start in range(0, len(frames) - pose_window + 1, start_stride):
        end = start + pose_window - 1
        image_path = resolve_image(transforms_path.parent, frames[start], image_subdir)
        if image_path is None:
            continue

        segment_centers = centers[start : end + 1]
        path_length = float(np.linalg.norm(np.diff(segment_centers, axis=0), axis=1).sum())
        displacement_world = segment_centers[-1] - segment_centers[0]
        displacement = float(np.linalg.norm(displacement_world))

        r0 = poses[start, :3, :3]
        r1 = poses[end, :3, :3]
        relative_rotation = r0.T @ r1
        rotation_deg = rotation_angle_deg(relative_rotation)
        yaw_deg = signed_yaw_deg(relative_rotation)

        local_displacement = r0.T @ displacement_world
        lateral = float(local_displacement[0])
        forward = float(-local_displacement[2])
        motion_class = classify_motion(forward, lateral, yaw_deg, rotation_deg)
        if motion_class is None:
            continue

        normalized_path = path_length / max(median_step * (pose_window - 1), 1e-8)
        straightness = displacement / max(path_length, 1e-8)
        score = normalized_path + 0.01 * rotation_deg + 0.25 * straightness
        candidates.append(
            Candidate(
                scene_id=scene_id,
                transforms_path=transforms_path,
                image_path=image_path,
                start_index=start,
                end_index=end,
                motion_class=motion_class,
                path_length=path_length,
                displacement=displacement,
                rotation_deg=rotation_deg,
                signed_yaw_deg=yaw_deg,
                forward=forward,
                lateral=lateral,
                straightness=straightness,
                score=score,
            )
        )
    return candidates


def percentile_ranks(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size <= 1:
        return np.ones_like(array)
    order = np.argsort(array, kind="stable")
    ranks = np.empty_like(array)
    ranks[order] = np.arange(array.size, dtype=np.float64)
    return ranks / float(array.size - 1)


def assign_global_motion_scores(candidates: list[Candidate]) -> list[Candidate]:
    if not candidates:
        return []
    path_rank = percentile_ranks([item.path_length for item in candidates])
    displacement_rank = percentile_ranks([item.displacement for item in candidates])
    rotation_rank = percentile_ranks([item.rotation_deg for item in candidates])
    scores = 0.4 * path_rank + 0.4 * displacement_rank + 0.2 * rotation_rank
    return [
        replace(item, score=float(score))
        for item, score in zip(candidates, scores, strict=True)
    ]


def select_diverse(candidates: list[Candidate], max_clips: int, max_per_scene: int) -> list[Candidate]:
    by_class: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_class.setdefault(candidate.motion_class, []).append(candidate)
    for values in by_class.values():
        values.sort(key=lambda item: item.score, reverse=True)

    selected: list[Candidate] = []
    per_scene: dict[str, int] = {}
    classes = sorted(by_class)
    while classes and len(selected) < max_clips:
        next_classes: list[str] = []
        for motion_class in classes:
            values = by_class[motion_class]
            chosen = None
            while values:
                candidate = values.pop(0)
                if per_scene.get(candidate.scene_id, 0) < max_per_scene:
                    chosen = candidate
                    break
            if chosen is not None:
                selected.append(chosen)
                per_scene[chosen.scene_id] = per_scene.get(chosen.scene_id, 0) + 1
            if values:
                next_classes.append(motion_class)
            if len(selected) >= max_clips:
                break
        classes = next_classes
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose-window", type=int, default=121)
    parser.add_argument("--start-stride", type=int, default=24)
    parser.add_argument("--max-clips", type=int, default=100)
    parser.add_argument("--max-per-scene", type=int, default=1)
    parser.add_argument("--image-subdir", default="images_8")
    parser.add_argument("--min-path-length", type=float, default=0.0)
    parser.add_argument("--min-displacement", type=float, default=0.0)
    parser.add_argument("--min-rotation-deg", type=float, default=0.0)
    parser.add_argument(
        "--large-motion-quantile",
        type=float,
        default=0.65,
        help=(
            "Retain candidates at or above this quantile of a dataset-level "
            "path/displacement/rotation rank score."
        ),
    )
    parser.add_argument(
        "--min-straightness",
        type=float,
        default=0.25,
        help="Minimum end displacement / path length; rejects scan paths that loop back.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.large_motion_quantile <= 1.0:
        parser.error("--large-motion-quantile must be in [0, 1]")

    all_candidates: list[Candidate] = []
    paths = sorted(iter_transforms(args.roots))
    for path in paths:
        all_candidates.extend(
            scene_candidates(path, args.pose_window, args.start_stride, args.image_subdir)
        )
    threshold_filtered = [
        item
        for item in all_candidates
        if item.path_length >= args.min_path_length
        and item.displacement >= args.min_displacement
        and item.rotation_deg >= args.min_rotation_deg
        and item.straightness >= args.min_straightness
    ]
    scored = assign_global_motion_scores(threshold_filtered)
    score_cutoff = (
        float(np.quantile([item.score for item in scored], args.large_motion_quantile))
        if scored
        else float("inf")
    )
    filtered = [item for item in scored if item.score >= score_cutoff]
    selected = select_diverse(filtered, args.max_clips, args.max_per_scene)

    cases = {}
    for item in selected:
        case_id = f"dl3dv_{item.scene_id}_s{item.start_index:05d}_{item.motion_class}"
        if case_id in cases:
            raise RuntimeError(f"Duplicate benchmark case id: {case_id}")
        cases[case_id] = {
            "text_prompt": PROMPTS[item.motion_class],
            "image_prompt": str(item.image_path),
            "source": "DL3DV",
            "scene_id": item.scene_id,
            "motion_instruction": item.motion_class,
            "pose_window": [item.start_index, item.end_index],
            "pose_stats": {
                "path_length": item.path_length,
                "displacement": item.displacement,
                "rotation_deg": item.rotation_deg,
                "signed_yaw_deg": item.signed_yaw_deg,
                "forward": item.forward,
                "lateral": item.lateral,
                "straightness": item.straightness,
                "selection_score": item.score,
            },
            "transforms_path": str(item.transforms_path),
        }

    payload = {
        "_meta": {
            "selection": "pose-driven large camera motion",
            "n_transforms": len(paths),
            "n_candidates": len(all_candidates),
            "n_threshold_filtered": len(threshold_filtered),
            "n_filtered": len(filtered),
            "n_selected": len(selected),
            "pose_window": args.pose_window,
            "large_motion_quantile": args.large_motion_quantile,
            "global_motion_score_cutoff": score_cutoff if scored else None,
            "prompt_policy": (
                "camera-only motion; no object-specific path; no static/stationary instruction"
            ),
        },
        **cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["_meta"], indent=2))
    for case_id, case in cases.items():
        stats = case["pose_stats"]
        print(
            f"{case_id}: {case['motion_instruction']} "
            f"path={stats['path_length']:.3f} rot={stats['rotation_deg']:.1f}"
        )


if __name__ == "__main__":
    main()
