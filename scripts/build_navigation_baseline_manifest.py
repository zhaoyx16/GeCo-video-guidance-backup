#!/usr/bin/env python3
"""Build a pose-matched, navigation-only baseline generation manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DL3DV_SCENES = {
    "1da888bdedfc": ("temple_buddha", "quiet temple courtyard and covered walkways"),
    "20e6b8225879": ("temple_deity", "quiet ornate temple interior and connected walkways"),
    "271a624806cd": ("toy_store", "empty toy store with aisles, shelves, and a large giraffe display"),
    "2c9b3210c477": ("bronze_statues", "quiet exhibition space with bronze statues and open walkways"),
    "2cbfe28643b6": ("childrens_library", "empty children's library with shelves and reading areas"),
    "468dcc8fb389": ("empty_restaurant", "empty restaurant with tables, chairs, and open passages"),
    "a4dbca1d9579": ("furniture_store", "empty furniture showroom with dense furniture displays and aisles"),
    "b8d6efbe9ea8": ("bedding_store", "empty bedding showroom with beds, shelves, and aisles"),
    "e39057cd2f14": ("office_corridor", "empty modern office-building corridor"),
    "173ac46c9740": ("garden_path", "quiet garden pathway with rocks, shrubs, a pond, and a pavilion"),
    "9c688ec4cec6": ("traditional_corridor", "quiet traditional architectural corridor with wooden railings and lattice panels"),
    "16ceeb9e25f2": ("forest_path", "quiet forest pathway with trees and low vegetation"),
    "97a1e65b847e": ("convention_center_corridor", "empty convention-center corridor with columns, brick walls, and glass entrances"),
    "97c8843407a4": ("home_improvement_store", "empty home-improvement store with long densely stocked aisles"),
    "9eca31e5ca54": ("warehouse_aisles", "empty warehouse with tall densely stocked metal shelving and narrow aisles"),
    "019cb9f575bd": ("school_plant_corridor", "empty enclosed school corridor with railings, plants, and repeated decorations"),
    "2b68ff3e9bbd": ("hospital_corridor", "empty modern hospital corridor with repeated doors and wall panels"),
    "0e155e942fdb": ("hotel_corridor", "empty hotel corridor with repeated doors, framed art, and intersecting hallways"),
    "114adf5c2de3": ("education_corridor", "empty educational-building corridor with display boards and open passages"),
    "166336194f24": ("committee_hallway", "empty formal hallway with wooden doors, wall panels, and repeated architectural details"),
    "1a0dc4cda12b": ("pet_store_aisles", "empty pet store with long densely stocked aisles"),
    "093809e60e3b": ("apartment_hallway", "empty narrow apartment hallway with repeated doors and corners"),
    "0a41f445803e": ("administrative_hallway", "empty long administrative hallway with reflective floors and bulletin boards"),
    "4947e44121e8": ("modern_hallway", "empty modern hallway with repeated doors, patterned carpet, and exit signs"),
    "4aa389b56773": ("elevator_hallway", "empty office elevator hallway with repeated elevator doors and intersecting passages"),
}

DL3DV_CASES_PER_SCENE = {
    "1da888bdedfc": 0,
    "20e6b8225879": 0,
    "271a624806cd": 0,
    "2c9b3210c477": 0,
    "2cbfe28643b6": 4,
    "468dcc8fb389": 0,
    "a4dbca1d9579": 0,
    "b8d6efbe9ea8": 0,
    "e39057cd2f14": 8,
    "173ac46c9740": 0,
    "9c688ec4cec6": 5,
    "16ceeb9e25f2": 0,
    "97a1e65b847e": 6,
    "97c8843407a4": 4,
    "9eca31e5ca54": 4,
    "019cb9f575bd": 4,
    "2b68ff3e9bbd": 6,
    "0e155e942fdb": 4,
    "114adf5c2de3": 4,
    "166336194f24": 3,
    "1a0dc4cda12b": 3,
    "093809e60e3b": 4,
    "0a41f445803e": 4,
    "4947e44121e8": 4,
    "4aa389b56773": 3,
}

DL3DV_SPAN_PER_SCENE = {
    "e39057cd2f14": 30,
    "9eca31e5ca54": 60,
    "2b68ff3e9bbd": 50,
    "093809e60e3b": 50,
    "4947e44121e8": 50,
}

SPLIT_BY_SCENE = {
    "1da888bdedfc": "development",
    "20e6b8225879": "validation",
    "271a624806cd": "development",
    "2c9b3210c477": "test",
    "2cbfe28643b6": "development",
    "468dcc8fb389": "validation",
    "a4dbca1d9579": "development",
    "b8d6efbe9ea8": "test",
    "e39057cd2f14": "validation",
    "173ac46c9740": "development",
    "9c688ec4cec6": "development",
    "16ceeb9e25f2": "development",
    "97a1e65b847e": "validation",
    "97c8843407a4": "development",
    "9eca31e5ca54": "test",
    "019cb9f575bd": "validation",
    "2b68ff3e9bbd": "test",
    "0e155e942fdb": "development",
    "114adf5c2de3": "validation",
    "166336194f24": "test",
    "1a0dc4cda12b": "development",
    "093809e60e3b": "validation",
    "0a41f445803e": "test",
    "4947e44121e8": "development",
    "4aa389b56773": "validation",
}

SCAND_ROWS = [
    2000,
    12000,
    26000,
    80000,
    122000,
    178900,
    344000,
    346000,
    354000,
    356000,
    179000,
    179700,
    180500,
    191000,
    207600,
    208400,
    210000,
    272000,
    312000,
    340000,
    343700,
    344450,
    348000,
    350000,
    370000,
    376000,
    382000,
    388000,
    394000,
    398000,
]


@dataclass
class Segment:
    start: int
    end: int
    path_length: float
    displacement: float
    forward: float
    lateral: float
    vertical: float
    yaw_deg: float
    directness: float
    score: float
    family: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dl3dv-root",
        action="append",
        required=True,
        help="Root containing scene directories with transforms.json; repeatable.",
    )
    parser.add_argument(
        "--scand-candidate-root",
        required=True,
        help="Root containing pre-extracted row_<index>-*.jpg candidate images.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--segment-frames", type=int, default=80)
    parser.add_argument("--starts-per-scene", type=int, default=3)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--models", default="wan,cosmos")
    parser.add_argument(
        "--scand-review-status",
        default="pending",
        choices=("pending", "approved"),
        help="SCAND stays out of runnable jobs unless explicitly approved after visual review.",
    )
    return parser.parse_args()


def scene_key(path: Path) -> str:
    return path.parent.name[:12]


def resolve_frame_path(transform_path: Path, relative: str) -> Path:
    name = Path(relative).name
    scene_dir = transform_path.parent
    for folder in ("images_4", "images_8", "images"):
        candidate = scene_dir / folder / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve {relative} below {scene_dir}")


def rotation_angle_deg(r0: np.ndarray, r1: np.ndarray) -> float:
    relative = r0.T @ r1
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def classify_motion(forward: float, lateral: float, yaw_deg: float, path_length: float) -> str:
    yaw_abs = abs(yaw_deg)
    if yaw_abs >= 28.0 and forward > 0.12 * path_length:
        return "forward_then_left" if yaw_deg > 0 else "forward_then_right"
    if abs(lateral) > max(abs(forward) * 0.8, 0.25 * path_length):
        return "lateral_left" if lateral < 0 else "lateral_right"
    if forward > 0.25 * path_length:
        return "straight_forward"
    return "curved_navigation"


def segment_metrics(mats: np.ndarray, start: int, end: int) -> Segment:
    centers = mats[:, :3, 3]
    path = float(np.linalg.norm(np.diff(centers[start : end + 1], axis=0), axis=1).sum())
    delta_world = centers[end] - centers[start]
    displacement = float(np.linalg.norm(delta_world))
    local = mats[start, :3, :3].T @ delta_world
    # DL3DV/nerfstudio transforms use camera-to-world matrices. The local x axis
    # is horizontal; the capture direction is inferred from camera orientation.
    forward = float(-local[2])
    lateral = float(local[0])
    vertical = float(local[1])
    forward_end_world = -mats[end, :3, 2]
    forward_end_local = mats[start, :3, :3].T @ forward_end_world
    yaw = float(math.degrees(math.atan2(forward_end_local[0], -forward_end_local[2])))
    directness = displacement / max(path, 1e-9)
    score = path * (0.35 + 0.65 * directness) + 0.012 * abs(yaw)
    return Segment(
        start=start,
        end=end,
        path_length=path,
        displacement=displacement,
        forward=forward,
        lateral=lateral,
        vertical=vertical,
        yaw_deg=yaw,
        directness=directness,
        score=score,
        family=classify_motion(forward, lateral, yaw, path),
    )


def select_segments(mats: np.ndarray, span: int, count: int) -> list[Segment]:
    candidates: list[Segment] = []
    centers = mats[:, :3, 3]
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    positive_steps = steps[steps > 1e-8]
    median_step = float(np.median(positive_steps)) if positive_steps.size else 0.0
    for start in range(0, len(mats) - span, 5):
        end = start + span
        segment_steps = steps[start:end]
        if median_step and float(segment_steps.max(initial=0.0)) > 8.0 * median_step:
            continue
        if any(
            rotation_angle_deg(mats[idx, :3, :3], mats[idx + 1, :3, :3]) > 18.0
            for idx in range(start, end)
        ):
            continue
        item = segment_metrics(mats, start, end)
        if item.directness < 0.52 or item.displacement < 1e-4:
            continue
        candidates.append(item)

    candidates.sort(key=lambda item: item.score, reverse=True)
    selected: list[Segment] = []
    minimum_gap = max(24, span // 2)
    for item in candidates:
        if any(abs(item.start - other.start) < minimum_gap for other in selected):
            continue
        selected.append(item)
        if len(selected) >= count:
            break
    return sorted(selected, key=lambda item: item.start)


def trajectory_phrase(segment: Segment) -> str:
    angle = int(round(min(abs(segment.yaw_deg), 120.0) / 5.0) * 5)
    if segment.family == "forward_then_left":
        return (
            f"moves quickly forward for a large distance along the physically open route, then follows its "
            f"smooth left turn "
            f"of approximately {angle} degrees and continues into the newly revealed navigable area"
        )
    if segment.family == "forward_then_right":
        return (
            f"moves quickly forward for a large distance along the physically open route, then follows its "
            f"smooth right turn "
            f"of approximately {angle} degrees and continues into the newly revealed navigable area"
        )
    if segment.family == "lateral_left":
        return "moves quickly for a large distance along the physically open route with a pronounced leftward translation and continues forward"
    if segment.family == "lateral_right":
        return "moves quickly for a large distance along the physically open route with a pronounced rightward translation and continues forward"
    if segment.family == "straight_forward":
        return "moves quickly and steadily forward for a large distance along the physically open route"
    return "moves quickly for a large distance while following the same smooth curved route through the physically open scene"


def prompt_for_scene(description: str, segment: Segment) -> str:
    return (
        f"A realistic continuous navigation video starting from the given first frame in the same {description}. "
        f"The camera is the only moving element. It {trajectory_phrase(segment)}. "
        "The camera remains inside the existing walkable or navigable space and never passes through walls, "
        "railings, shelves, furniture, vegetation, glass, or other solid objects. "
        "Use one stabilized wide-angle shot with constant focal length, no zoom, no cuts, and no teleportation. "
        "All scene geometry, surfaces, fixtures, and background objects remain completely stationary and retain "
        "their shape, position, texture, and identity. No people, animals, vehicles, or independently moving objects."
    )


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_dl3dv_contact_sheet(
    case_id: str,
    frames: list[dict],
    transform_path: Path,
    segment: Segment,
    output: Path,
) -> None:
    indices = np.linspace(segment.start, segment.end, 5).round().astype(int).tolist()
    thumbs = []
    for idx in indices:
        image = Image.open(resolve_frame_path(transform_path, frames[idx]["file_path"])).convert("RGB")
        image.thumbnail((300, 190), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (310, 225), "white")
        canvas.paste(image, ((310 - image.width) // 2, 5))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 198), f"frame {idx + 1}", fill="black", font=load_font(17))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (1550, 275), "#20232a")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (10, 8),
        (
            f"{case_id} | {segment.family} | path={segment.path_length:.3f} "
            f"disp={segment.displacement:.3f} yaw={segment.yaw_deg:+.1f}"
        ),
        fill="white",
        font=load_font(18),
    )
    for column, thumb in enumerate(thumbs):
        sheet.paste(thumb, (column * 310, 45))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def scand_row(index: int) -> dict:
    query = urllib.parse.urlencode(
        {
            "dataset": "mateoguaman/vamos_navigation_only_dataset",
            "config": "default",
            "split": "train",
            "offset": index,
            "length": 1,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{query}"
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = json.load(response)
            break
        except Exception as error:
            last_error = error
            if attempt == 4:
                raise RuntimeError(f"Failed to load SCAND/VAMOS row {index}") from last_error
            time.sleep(2 ** attempt)
    return payload["rows"][0]["row"]


def find_scand_image(root: Path, row_index: int) -> Path:
    matches = sorted(root.glob(f"**/row_{row_index:06d}-c*.jpg"))
    if not matches:
        matches = sorted(root.glob(f"**/row_{row_index:06d}.jpg"))
    if not matches:
        raise FileNotFoundError(f"No image found for SCAND/VAMOS row {row_index}")
    return matches[0]


def scand_metrics(points: np.ndarray) -> Segment:
    path = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    displacement_vector = points[-1] - points[0]
    displacement = float(np.linalg.norm(displacement_vector))
    forward = float(displacement_vector[0])
    lateral = float(displacement_vector[1])
    vertical = float(displacement_vector[2])
    window = min(10, max(2, len(points) // 5))
    heading_start = points[window] - points[0]
    heading_end = points[-1] - points[-1 - window]
    yaw_start = math.atan2(float(heading_start[1]), float(heading_start[0]))
    yaw_end = math.atan2(float(heading_end[1]), float(heading_end[0]))
    yaw = math.degrees(math.atan2(math.sin(yaw_end - yaw_start), math.cos(yaw_end - yaw_start)))
    directness = displacement / max(path, 1e-9)
    return Segment(
        start=0,
        end=len(points) - 1,
        path_length=path,
        displacement=displacement,
        forward=forward,
        lateral=lateral,
        vertical=vertical,
        yaw_deg=yaw,
        directness=directness,
        score=path,
        family=classify_motion(forward, lateral, yaw, path),
    )


def make_scand_review_panel(
    case_id: str,
    image_path: Path,
    points: np.ndarray,
    segment: Segment,
    output: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((650, 390), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (1000, 460), "#20232a")
    panel.paste(image, (15, 55))
    draw = ImageDraw.Draw(panel)
    draw.text(
        (15, 12),
        f"{case_id} | {segment.family} | yaw={segment.yaw_deg:+.1f}",
        fill="white",
        font=load_font(19),
    )
    plot_box = (690, 75, 970, 390)
    draw.rectangle(plot_box, outline="white", width=2)
    xy = points[:, :2].copy()
    xy -= xy.min(axis=0)
    extent = np.maximum(xy.max(axis=0), 1e-6)
    xy /= extent
    projected = [
        (
            int(plot_box[0] + 15 + value[0] * (plot_box[2] - plot_box[0] - 30)),
            int(plot_box[3] - 15 - value[1] * (plot_box[3] - plot_box[1] - 30)),
        )
        for value in xy
    ]
    if len(projected) > 1:
        draw.line(projected, fill="#54d2d2", width=3)
    draw.ellipse(
        (projected[0][0] - 5, projected[0][1] - 5, projected[0][0] + 5, projected[0][1] + 5),
        fill="#4caf50",
    )
    draw.ellipse(
        (projected[-1][0] - 5, projected[-1][1] - 5, projected[-1][0] + 5, projected[-1][1] + 5),
        fill="#f44336",
    )
    draw.text((690, 405), "green=start, red=end", fill="white", font=load_font(16))
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)


def stable_job_id(model: str, case_id: str, seed: int) -> str:
    digest = hashlib.sha1(f"{model}:{case_id}:{seed}".encode()).hexdigest()[:10]
    return f"{model}_{case_id}_seed{seed}_{digest}"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_root = output_dir / "trajectory_contact_sheets"
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    models = [value.strip() for value in args.models.split(",") if value.strip()]

    transform_paths = []
    for root_text in args.dl3dv_root:
        transform_paths.extend(Path(root_text).glob("**/transforms.json"))
    transform_paths = sorted(
        path for path in set(transform_paths) if scene_key(path) in DL3DV_SCENES
    )
    if len(transform_paths) != len(DL3DV_SCENES):
        found = {scene_key(path) for path in transform_paths}
        missing = sorted(set(DL3DV_SCENES) - found)
        raise RuntimeError(f"Expected {len(DL3DV_SCENES)} DL3DV scenes, missing {missing}")

    cases: dict[str, dict] = {}
    trajectory_rows: list[dict] = []

    for transform_path in transform_paths:
        key = scene_key(transform_path)
        alias, description = DL3DV_SCENES[key]
        payload = json.load(open(transform_path))
        frames = payload["frames"]
        mats = np.asarray([frame["transform_matrix"] for frame in frames], dtype=np.float64)
        requested = DL3DV_CASES_PER_SCENE.get(key, args.starts_per_scene)
        if requested == 0:
            continue
        segment_span = DL3DV_SPAN_PER_SCENE.get(key, args.segment_frames)
        segments = select_segments(mats, segment_span, requested)
        if len(segments) != requested:
            raise RuntimeError(f"{key} supplied only {len(segments)} of {requested} trajectory segments")
        for segment in segments:
            frame_number = int(re.search(r"(\d+)", Path(frames[segment.start]["file_path"]).stem).group(1))
            case_id = f"dl3dv_{alias}_start{frame_number:05d}_{segment.family}"
            image_path = resolve_frame_path(transform_path, frames[segment.start]["file_path"])
            cases[case_id] = {
                "text_prompt": prompt_for_scene(description, segment),
                "image_prompt": str(image_path),
                "source": f"DL3DV/{key}",
                "dataset": "dl3dv",
                "scene_id": key,
                "split": SPLIT_BY_SCENE[key],
                "review_status": "approved",
                "motion_instruction": trajectory_phrase(segment),
                "trajectory": {
                    "start_index": segment.start,
                    "end_index": segment.end,
                    "start_frame_number": frame_number,
                    "path_length_pose_units": segment.path_length,
                    "displacement_pose_units": segment.displacement,
                    "forward_pose_units": segment.forward,
                    "lateral_pose_units": segment.lateral,
                    "vertical_pose_units": segment.vertical,
                    "yaw_change_deg": segment.yaw_deg,
                    "directness": segment.directness,
                    "family": segment.family,
                    "transforms_json": str(transform_path),
                },
            }
            make_dl3dv_contact_sheet(
                case_id,
                frames,
                transform_path,
                segment,
                contact_root / "dl3dv" / f"{case_id}.jpg",
            )
            trajectory_rows.append({"case_id": case_id, **cases[case_id]["trajectory"]})

    scand_root = Path(args.scand_candidate_root)
    for ordinal, row_index in enumerate(SCAND_ROWS):
        row = scand_row(row_index)
        points = np.asarray(row["shorter_trajectory_3d"] or row["trajectory_3d"], dtype=np.float64)
        segment = scand_metrics(points)
        image_path = find_scand_image(scand_root, row_index)
        case_id = f"scand_row{row_index:06d}_{segment.family}"
        description = "quiet robot-navigable indoor or outdoor campus route shown in the first frame"
        cases[case_id] = {
            "text_prompt": prompt_for_scene(description, segment),
            "image_prompt": str(image_path),
            "source": f"SCAND via VAMOS row {row_index}",
            "dataset": "scand",
            "scene_id": f"scand_row{row_index:06d}",
            "split": "development" if ordinal < 6 else ("validation" if ordinal < 8 else "test"),
            "review_status": args.scand_review_status,
            "motion_instruction": trajectory_phrase(segment),
            "trajectory": {
                "row_index": row_index,
                "path_length_m": segment.path_length,
                "displacement_m": segment.displacement,
                "forward_m": segment.forward,
                "lateral_m": segment.lateral,
                "vertical_m": segment.vertical,
                "yaw_change_deg": segment.yaw_deg,
                "directness": segment.directness,
                "family": segment.family,
                "curvature": row.get("curvature"),
                "horizon": row.get("horizon"),
            },
        }
        make_scand_review_panel(
            case_id,
            image_path,
            points,
            segment,
            contact_root / "scand_pending_review" / f"{case_id}.jpg",
        )
        trajectory_rows.append({"case_id": case_id, **cases[case_id]["trajectory"]})

    config_path = output_dir / "navigation_cases.json"
    config_path.write_text(json.dumps(cases, indent=2) + "\n")

    fieldnames = sorted({key for row in trajectory_rows for key in row})
    with open(output_dir / "trajectory_report.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trajectory_rows)

    approved_cases = {
        case_id: item for case_id, item in cases.items() if item["review_status"] == "approved"
    }
    jobs = []
    for case_id, item in approved_cases.items():
        for model in models:
            for seed in seeds:
                jobs.append(
                    {
                        "job_id": stable_job_id(model, case_id, seed),
                        "model": model,
                        "case_id": case_id,
                        "dataset": item["dataset"],
                        "scene_id": item["scene_id"],
                        "split": item["split"],
                        "seed": seed,
                        "config_path": str(config_path),
                    }
                )
    jobs.sort(key=lambda item: (item["model"], item["dataset"], item["case_id"], item["seed"]))
    with open(output_dir / "jobs.jsonl", "w") as handle:
        for job in jobs:
            handle.write(json.dumps(job) + "\n")

    summary = {
        "cases_total": len(cases),
        "cases_approved": len(approved_cases),
        "cases_pending_review": len(cases) - len(approved_cases),
        "jobs_runnable": len(jobs),
        "models": models,
        "seeds": seeds,
        "wan_official_config": {
            "steps": 50,
            "frames": 121,
            "height": 704,
            "width": 1280,
            "fps": 24,
        },
        "cosmos_model_default_config": {
            "steps": 36,
            "frames": 93,
            "height": 704,
            "width": 1280,
            "fps": 16,
        },
        "note": (
            "Final navigation screen: 70 DL3DV cases and 30 manually reviewed SCAND cases. "
            "Each case is paired across Wan and Cosmos with seeds 0 and 1."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
