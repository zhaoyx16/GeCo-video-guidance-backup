#!/usr/bin/env python3
"""Build a pose-driven DL3DV benchmark manifest.

The script selects large-camera-motion clips from DL3DV transforms.json files
and assigns camera-only prompts derived from the measured pose trajectory.
It deliberately avoids scene-object instructions and "everything is static"
language so prompt ambiguity is not baked into the benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np
from PIL import Image


PROMPTS = {
    "forward": (
        "A realistic continuous first-person navigation video beginning from the supplied "
        "first frame {scene}. The camera rapidly advances a long distance along the visible "
        "route, producing strong parallax and a large viewpoint change. Nearby surfaces grow "
        "in scale, sweep toward the frame edges, and pass out of view as new parts of the same "
        "environment are revealed. One uninterrupted fixed-focal-length shot with continuous "
        "physical camera translation; no zoom, cuts, or teleportation."
    ),
    "forward_left": (
        "A realistic continuous first-person navigation video beginning from the supplied "
        "first frame {scene}. The camera rapidly advances a long distance and follows one "
        "broad smooth leftward curve, producing strong parallax and a large viewpoint change. "
        "Nearby surfaces pass beside the camera and leave the frame; the final viewpoint faces "
        "clearly left into newly revealed space. One uninterrupted fixed-focal-length shot "
        "with continuous physical camera translation; no zoom, cuts, or teleportation."
    ),
    "forward_right": (
        "A realistic continuous first-person navigation video beginning from the supplied "
        "first frame {scene}. The camera rapidly advances a long distance and follows one "
        "broad smooth rightward curve, producing strong parallax and a large viewpoint change. "
        "Nearby surfaces pass beside the camera and leave the frame; the final viewpoint faces "
        "clearly right into newly revealed space. One uninterrupted fixed-focal-length shot "
        "with continuous physical camera translation; no zoom, cuts, or teleportation."
    ),
    "lateral_left": (
        "A realistic continuous first-person navigation video beginning from the supplied "
        "first frame {scene}. The camera physically travels a long distance toward the left "
        "while continuing through the visible environment, producing strong parallax and a "
        "large viewpoint change. Nearby surfaces sweep rapidly across and out of the frame. "
        "One uninterrupted fixed-focal-length shot; no zoom, cuts, or teleportation."
    ),
    "lateral_right": (
        "A realistic continuous first-person navigation video beginning from the supplied "
        "first frame {scene}. The camera physically travels a long distance toward the right "
        "while continuing through the visible environment, producing strong parallax and a "
        "large viewpoint change. Nearby surfaces sweep rapidly across and out of the frame. "
        "One uninterrupted fixed-focal-length shot; no zoom, cuts, or teleportation."
    ),
    "orbit_left": (
        "A realistic continuous camera video beginning from the supplied first frame {scene}. "
        "The camera physically travels along a broad leftward arc over a long distance, "
        "producing strong parallax, disocclusion, and a large viewpoint change. Nearby surfaces "
        "sweep rapidly across and out of the frame while new surfaces are revealed. One "
        "uninterrupted fixed-focal-length shot; no zoom, cuts, or teleportation."
    ),
    "orbit_right": (
        "A realistic continuous camera video beginning from the supplied first frame {scene}. "
        "The camera physically travels along a broad rightward arc over a long distance, "
        "producing strong parallax, disocclusion, and a large viewpoint change. Nearby surfaces "
        "sweep rapidly across and out of the frame while new surfaces are revealed. One "
        "uninterrupted fixed-focal-length shot; no zoom, cuts, or teleportation."
    ),
}

FORBIDDEN_PROMPT_PHRASES = (
    "static camera",
    "stationary camera",
    "locked camera",
    "camera remains still",
    "camera is the only moving element",
    "small camera motion",
    "gentle camera motion",
    "slow camera motion",
)
PROMPT_POLICY_VERSION = "gt-large-camera-motion-v1"
FORMAL_SPLIT_COUNTS = {"debug": 3, "validation": 100, "test": 100, "dev": 100}
FORBIDDEN_DESCRIPTION_TERMS = (
    "camera",
    " video",
    "turn",
    "turns ",
    "turning ",
    "move",
    "moves ",
    "moving ",
    "motion",
    "pan",
    "pans ",
    "zoom",
    "zooms ",
    "travel",
    "travels ",
    "walk",
    "walks ",
    "rotate",
    "rotates",
    "orbit",
    " still ",
    "shifts ",
    "changes ",
    "transitions ",
    "focuses ",
    "reveals ",
    "scene ",
    "scenes ",
    "sequence ",
    "starting ",
    "begins ",
    "ends ",
    "concludes ",
    "over time",
    "throughout",
    "is revealed",
    "are revealed",
)


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
    normalized_path: float
    normalized_displacement: float
    turn_consistency: float
    turn_monotonicity: float
    turn_smoothness: float
    turn_spread: float
    score: float


@dataclass(frozen=True)
class SplitAssignment:
    split: str
    split_order: int
    source_order: int
    scene_id: str


class FrozenSourceError(ValueError):
    """Identifier-free formal source contract failure."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_scene_relative_path(relative: PurePosixPath) -> None:
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("frozen scene path must remain relative to one allowlisted scene")


def _open_scene_regular_file_no_follow(scene_dir: Path, relative: PurePosixPath) -> int:
    """Open one allowlisted scene file without following any scene-local symlink."""

    _validate_scene_relative_path(relative)
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("this platform cannot enforce no-follow frozen-scene reads")
    # Reject the allowlisted scene entry itself before any resolve operation.
    # O_NOFOLLOW on the directory descriptor closes the check/open race.
    if scene_dir.is_symlink():
        raise ValueError("frozen scene content cannot contain symlinks")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors: list[int] = []
    try:
        current = os.open(scene_dir, directory_flags)
        descriptors.append(current)
        if not stat.S_ISDIR(os.fstat(current).st_mode):
            raise ValueError("frozen scene root must be a regular directory")
        for part in relative.parts[:-1]:
            metadata = os.stat(part, dir_fd=current, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("frozen scene content cannot contain symlinks")
            next_directory = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(next_directory)
            if not stat.S_ISDIR(os.fstat(next_directory).st_mode):
                raise ValueError("frozen scene path contains a non-directory component")
            current = next_directory
        name = relative.parts[-1]
        metadata = os.stat(name, dir_fd=current, follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("frozen scene content cannot contain symlinks")
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError("frozen scene contract input must be a regular file")
        return descriptor
    finally:
        for directory in reversed(descriptors):
            os.close(directory)


def read_scene_regular_file_bytes_no_follow(
    scene_dir: Path, relative: PurePosixPath
) -> bytes:
    descriptor = _open_scene_regular_file_no_follow(scene_dir, relative)
    try:
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 8 << 20)
            if not block:
                break
            blocks.append(block)
        return b"".join(blocks)
    finally:
        os.close(descriptor)


def _list_scene_directory_no_follow(scene_dir: Path, relative: PurePosixPath) -> set[str]:
    """List one known directory inside one allowlisted scene, without symlinks."""

    _validate_scene_relative_path(relative)
    if scene_dir.is_symlink():
        raise ValueError("frozen scene content cannot contain symlinks")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors: list[int] = []
    try:
        current = os.open(scene_dir, flags)
        descriptors.append(current)
        for part in relative.parts:
            metadata = os.stat(part, dir_fd=current, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("frozen scene content cannot contain symlinks")
            next_directory = os.open(part, flags, dir_fd=current)
            descriptors.append(next_directory)
            current = next_directory
        return set(os.listdir(current))
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def validate_scene_files_no_follow(scene_dir: Path, transforms_payload: bytes) -> dict:
    """Recompute the frozen-scene marker without path-following reads."""

    try:
        transforms = json.loads(transforms_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid transforms.json") from error
    frames = transforms.get("frames")
    if not isinstance(frames, list) or len(frames) < 8:
        raise ValueError("transforms.json contains fewer than eight frames")

    required_names: list[str] = []
    seen_names: set[str] = set()
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or not isinstance(frame.get("file_path"), str):
            raise ValueError(f"frame {index} has no valid file_path")
        matrix = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"frame {index} has no finite 4x4 transform_matrix")
        raw_name = frame["file_path"]
        if "\\" in raw_name:
            raise ValueError("frame file_path is not POSIX-compatible")
        relative_frame = PurePosixPath(raw_name)
        _validate_scene_relative_path(relative_frame)
        name = relative_frame.name
        if not name or name in {".", ".."} or name in seen_names:
            raise ValueError("transforms.json contains an invalid or duplicate frame filename")
        seen_names.add(name)
        required_names.append(name)

    digest = hashlib.sha256()
    image_bytes = 0
    for name in required_names:
        payload = read_scene_regular_file_bytes_no_follow(
            scene_dir, PurePosixPath("images_8") / name
        )
        if not payload:
            raise ValueError("frozen scene contains an empty image")
        try:
            with Image.open(io.BytesIO(payload)) as image:
                if image.format != "PNG":
                    raise ValueError("frozen scene contains an unexpected image format")
                image.verify()
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
        except (OSError, SyntaxError) as error:
            raise ValueError("frozen scene contains a corrupt image") from error
        file_hash = hashlib.sha256(payload).hexdigest()
        digest.update(f"{name}\0{len(payload)}\0{file_hash}\n".encode("utf-8"))
        image_bytes += len(payload)

    actual_names = {
        name
        for name in _list_scene_directory_no_follow(scene_dir, PurePosixPath("images_8"))
        if name.startswith("frame_") and name.lower().endswith(".png")
    }
    if actual_names != seen_names:
        raise ValueError("images_8 frame set differs from transforms.json")
    return {
        "transforms_sha256": hashlib.sha256(transforms_payload).hexdigest(),
        "frame_count": len(required_names),
        "frames_digest_sha256": digest.hexdigest(),
        "image_bytes": image_bytes,
    }


def load_frozen_assignments(
    path: Path,
    selected_splits: set[str],
    expected_counts: dict[str, int] | None = None,
    expected_sha256: str | None = None,
) -> list[SplitAssignment]:
    valid_splits = {"debug", "validation", "test", "dev"}
    if not selected_splits or not selected_splits <= valid_splits:
        raise ValueError("invalid selected split")
    if "dev" in selected_splits and selected_splits != {"dev"}:
        raise ValueError("dev split cannot be combined with another split")
    if selected_splits == {"dev"} and expected_sha256 is None:
        raise ValueError("dev split requires an expected frozen split SHA")
    if path.is_symlink() or not path.is_file():
        raise ValueError("frozen split CSV must be a regular file")
    if expected_sha256 is not None and (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("expected frozen split SHA must be lowercase SHA-256")
    payload = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("frozen split CSV SHA mismatch")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("frozen split CSV must be UTF-8") from error
    all_assignments: list[SplitAssignment] = []
    reader = csv.DictReader(io.StringIO(text, newline=""))
    required = ("split", "split_order", "source_order", "hash", "batch", "duration")
    if tuple(reader.fieldnames or ()) != required:
        raise ValueError("frozen split has unexpected columns")
    for row in reader:
        if (
            None in row
            or set(row) != set(required)
            or any(not isinstance(row.get(field), str) for field in required)
        ):
            raise ValueError("frozen split contains a malformed row")
        split = row["split"]
        if selected_splits == {"dev"} and split != "dev":
            raise ValueError("dev CSV contains a non-dev row")
        if split not in valid_splits or row["batch"] != "1K":
            raise ValueError("frozen split contains an invalid split or batch")
        scene_id = row["hash"]
        if len(scene_id) != 64 or any(char not in "0123456789abcdef" for char in scene_id):
            raise ValueError("frozen split contains an invalid scene identifier")
        try:
            split_order = int(row["split_order"])
            source_order = int(row["source_order"])
            duration = float(row["duration"])
        except ValueError as error:
            raise ValueError("frozen split contains invalid numeric metadata") from error
        assignment = SplitAssignment(
            split=split,
            split_order=split_order,
            source_order=source_order,
            scene_id=scene_id,
        )
        if (
            assignment.split_order < 0
            or not 1 <= assignment.source_order <= 1000
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError("frozen split contains invalid numeric metadata")
        all_assignments.append(assignment)

    scene_ids = [item.scene_id for item in all_assignments]
    source_orders = [item.source_order for item in all_assignments]
    split_orders = [(item.split, item.split_order) for item in all_assignments]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("frozen split reuses a scene across one or more splits")
    if len(source_orders) != len(set(source_orders)):
        raise ValueError("frozen split reuses a source_order")
    if len(split_orders) != len(set(split_orders)):
        raise ValueError("frozen split reuses a split_order")
    actual_counts = {
        split: sum(item.split == split for item in all_assignments)
        for split in sorted(selected_splits)
    }
    if expected_counts is not None:
        selected_expected = {split: expected_counts[split] for split in sorted(selected_splits)}
        if actual_counts != selected_expected:
            raise ValueError(
                f"selected frozen split counts do not match: actual={actual_counts}, "
                f"expected={selected_expected}"
            )

    assignments = [
        assignment
        for assignment in all_assignments
        if assignment.split in selected_splits
    ]
    if not assignments:
        raise ValueError("selected frozen split contains no scenes")
    if selected_splits == {"dev"} and [item.split_order for item in assignments] != list(range(100)):
        raise ValueError("dev split_order must be contiguous from zero")
    return sorted(
        assignments,
        key=lambda item: (
            {"debug": 0, "validation": 1, "test": 2, "dev": 3}[item.split],
            item.split_order,
        ),
    )


def load_scene_descriptions(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scene descriptions must be a JSON object keyed by scene id")
    descriptions: dict[str, str] = {}
    for scene_id, description in payload.items():
        if not isinstance(scene_id, str) or not isinstance(description, str):
            raise ValueError("scene descriptions must map strings to strings")
        cleaned = " ".join(description.strip().rstrip(".").split())
        if not cleaned:
            raise ValueError("scene descriptions contain an empty value")
        forbidden = find_forbidden_description_terms(cleaned)
        if forbidden:
            raise ValueError(
                f"scene descriptions contain motion/temporal terms: {forbidden}"
            )
        descriptions[scene_id] = cleaned
    return descriptions


def find_forbidden_description_terms(description: str) -> list[str]:
    lowered = description.lower()
    return [
        term
        for term in FORBIDDEN_DESCRIPTION_TERMS
        if re.search(rf"(?<!\w){re.escape(term.strip())}(?!\w)", lowered)
    ]


def render_prompt(motion_class: str, scene_description: str | None = None) -> str:
    scene = (
        f"in {scene_description}"
        if scene_description
        else "and continuing through the visible environment"
    )
    prompt = PROMPTS[motion_class].format(scene=scene)
    audit_prompt(prompt)
    return prompt


def audit_prompt(prompt: str) -> None:
    lowered = prompt.lower()
    forbidden = [phrase for phrase in FORBIDDEN_PROMPT_PHRASES if phrase in lowered]
    if forbidden:
        raise ValueError(f"prompt contains camera-motion-suppressing phrase(s): {forbidden}")
    if "strong parallax" not in lowered or "large viewpoint change" not in lowered:
        raise ValueError("prompt must make the large camera trajectory visually observable")
    if not any(
        phrase in lowered
        for phrase in ("camera rapidly advances", "camera physically travels")
    ):
        raise ValueError("prompt must explicitly request substantial physical camera motion")


def load_source_scene_provenance(candidate: Candidate, required: bool) -> dict | None:
    scene_dir = candidate.transforms_path.parent
    try:
        marker_payload = read_scene_regular_file_bytes_no_follow(
            scene_dir, PurePosixPath(".dl3dv_complete.json")
        )
    except FileNotFoundError:
        if required:
            raise ValueError("missing frozen scene marker")
        return None
    except Exception:
        raise ValueError("frozen source scene validation failed") from None
    try:
        marker = json.loads(marker_payload.decode("utf-8"))
        transforms_payload = read_scene_regular_file_bytes_no_follow(
            scene_dir, PurePosixPath("transforms.json")
        )
        expected = {
            "schema": "dl3dv-scene-complete-v1",
            "scene_hash": candidate.scene_id,
            "transforms_sha256": hashlib.sha256(transforms_payload).hexdigest(),
        }
        for key, value in expected.items():
            if marker.get(key) != value:
                raise ValueError("source marker contract mismatch")
        current_scene = validate_scene_files_no_follow(scene_dir, transforms_payload)
        for key, value in current_scene.items():
            if marker.get(key) != value:
                raise ValueError("source scene differs from frozen marker")
        conditioning_relative = PurePosixPath(*candidate.image_path.relative_to(scene_dir).parts)
        conditioning_payload = read_scene_regular_file_bytes_no_follow(
            scene_dir, conditioning_relative
        )
        result = {
            "repo_id": marker.get("repo_id"),
            "dataset_revision": marker.get("dataset_revision"),
            "archive_sha256": marker.get("archive_sha256"),
            "transforms_sha256": marker.get("transforms_sha256"),
            "conditioning_frame_480p_sha256": hashlib.sha256(
                conditioning_payload
            ).hexdigest(),
        }
    except Exception:
        # Formal split identifiers are private protocol material.  Keep all
        # marker/content failure paths identifier- and path-free.
        raise ValueError("frozen source scene validation failed") from None
    return result


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
    turn_consistency: float = 1.0,
    turn_monotonicity: float = 1.0,
    turn_smoothness: float = 1.0,
    turn_spread: float = 1.0,
) -> str | None:
    abs_yaw = abs(yaw_deg)
    reliable_turn = (
        turn_consistency >= 0.65
        and turn_monotonicity >= 0.65
        and turn_smoothness >= 0.50
        and turn_spread >= 0.15
    )
    if forward < 0 and abs(forward) >= 0.6 * abs(lateral):
        return None
    if forward > 0 and forward >= 0.6 * abs(lateral):
        if yaw_deg > 18 and reliable_turn:
            return "forward_left"
        if yaw_deg < -18 and reliable_turn:
            return "forward_right"
        return "forward"
    if abs(lateral) > max(abs(forward), 1e-8):
        return "lateral_right" if lateral > 0 else "lateral_left"
    return None


def yaw_path_metrics(rotations: np.ndarray) -> tuple[float, float, float, float, float]:
    r0 = rotations[0]
    wrapped = np.asarray(
        [math.radians(signed_yaw_deg(r0.T @ rotation)) for rotation in rotations],
        dtype=np.float64,
    )
    unwrapped = np.unwrap(wrapped)
    increments = np.diff(unwrapped)
    final_yaw_deg = math.degrees(float(unwrapped[-1]))
    total_variation = float(np.abs(increments).sum())
    consistency = abs(float(unwrapped[-1])) / max(total_variation, 1e-8)
    if abs(final_yaw_deg) < 1e-6 or increments.size == 0:
        monotonicity = 1.0
        smoothness = 1.0
        spread = 1.0
    else:
        direction = 1.0 if final_yaw_deg > 0 else -1.0
        monotonicity = float(np.mean(direction * increments >= math.radians(-0.5)))
        smoothness = 1.0 - float(np.max(np.abs(increments))) / max(total_variation, 1e-8)
        squared_sum = float(np.square(np.abs(increments)).sum())
        spread = total_variation**2 / max(increments.size * squared_sum, 1e-8)
    return (
        final_yaw_deg,
        min(consistency, 1.0),
        monotonicity,
        max(smoothness, 0.0),
        min(max(spread, 0.0), 1.0),
    )


def _regular_contained_file(scene_dir: Path, relative: PurePosixPath) -> Path | None:
    """Return a regular, symlink-free file contained in one allowlisted scene."""

    if scene_dir.is_symlink():
        raise ValueError("frozen scene content cannot contain symlinks")
    try:
        descriptor = _open_scene_regular_file_no_follow(scene_dir, relative)
    except FileNotFoundError:
        return None
    else:
        os.close(descriptor)
    return scene_dir.joinpath(*relative.parts)


def resolve_image(scene_dir: Path, frame_record: dict, image_subdir: str) -> Path | None:
    raw_file_path = frame_record.get("file_path")
    if not isinstance(raw_file_path, str) or not raw_file_path or "\\" in raw_file_path:
        raise ValueError("frame file_path must be a nonempty POSIX-relative path")
    relative = PurePosixPath(raw_file_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("frame file_path must not be absolute or traverse directories")
    image_dir = PurePosixPath(image_subdir)
    if (
        not image_subdir
        or image_dir.is_absolute()
        or len(image_dir.parts) != 1
        or image_dir.parts[0] in {"", ".", ".."}
    ):
        raise ValueError("image_subdir must be one relative directory name")
    file_name = relative.name
    candidates = [
        PurePosixPath(image_subdir) / file_name,
        relative,
        PurePosixPath("images") / file_name,
    ]
    for candidate in candidates:
        resolved = _regular_contained_file(scene_dir, candidate)
        if resolved is not None:
            return resolved
    return None


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


def resolve_frozen_transforms(
    roots: Iterable[Path], assignments: list[SplitAssignment]
) -> list[Path]:
    """Resolve only allowlisted frozen scenes without enumerating a root.

    Formal roots may contain other protocol splits.  Never use ``rglob`` or
    inspect directory entry names in this path: derive each candidate solely
    from an already SHA-verified assignment.
    """

    checked_roots: list[Path] = []
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            raise FrozenSourceError("frozen source root must be a regular directory")
        checked_roots.append(root.resolve(strict=True))
    if not checked_roots:
        raise FrozenSourceError("frozen source requires at least one root")

    resolved: list[Path] = []
    missing = 0
    duplicate = 0
    for assignment in assignments:
        matches: list[Path] = []
        for root in checked_roots:
            scene_directory = root / assignment.scene_id
            transforms = scene_directory / "transforms.json"
            try:
                descriptor = _open_scene_regular_file_no_follow(
                    scene_directory, PurePosixPath("transforms.json")
                )
            except FileNotFoundError:
                continue
            except (OSError, ValueError):
                raise FrozenSourceError("frozen source paths cannot be symlinks") from None
            else:
                os.close(descriptor)
                matches.append(transforms)
        if not matches:
            missing += 1
        elif len(matches) > 1:
            duplicate += 1
        else:
            resolved.append(matches[0])
    if missing:
        raise FrozenSourceError(f"missing downloaded data for {missing} frozen scene(s)")
    if duplicate:
        raise FrozenSourceError(f"multiple roots contain {duplicate} frozen scene(s)")
    return resolved


def scene_candidates(
    transforms_path: Path,
    pose_window: int,
    start_stride: int,
    image_subdir: str,
) -> list[Candidate]:
    scene_dir = transforms_path.parent
    transforms_payload = read_scene_regular_file_bytes_no_follow(
        scene_dir, PurePosixPath("transforms.json")
    )
    data = json.loads(transforms_payload.decode("utf-8"))
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
        (
            yaw_deg,
            turn_consistency,
            turn_monotonicity,
            turn_smoothness,
            turn_spread,
        ) = yaw_path_metrics(poses[start : end + 1, :3, :3])

        local_displacement = r0.T @ displacement_world
        lateral = float(local_displacement[0])
        forward = float(-local_displacement[2])
        motion_class = classify_motion(
            forward,
            lateral,
            yaw_deg,
            rotation_deg,
            turn_consistency,
            turn_monotonicity,
            turn_smoothness,
            turn_spread,
        )
        if motion_class is None:
            continue

        normalized_path = path_length / max(median_step * (pose_window - 1), 1e-8)
        straightness = displacement / max(path_length, 1e-8)
        normalized_displacement = displacement / max(
            median_step * (pose_window - 1), 1e-8
        )
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
                normalized_path=normalized_path,
                normalized_displacement=normalized_displacement,
                turn_consistency=turn_consistency,
                turn_monotonicity=turn_monotonicity,
                turn_smoothness=turn_smoothness,
                turn_spread=turn_spread,
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


def select_best_per_scene(
    candidates: list[Candidate], assignments: list[SplitAssignment]
) -> list[Candidate]:
    by_scene: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_scene.setdefault(candidate.scene_id, []).append(candidate)
    missing = [item.scene_id for item in assignments if not by_scene.get(item.scene_id)]
    if missing:
        raise RuntimeError(
            f"no eligible large-motion trajectory for {len(missing)} frozen scene(s)"
        )
    return [
        max(by_scene[item.scene_id], key=lambda candidate: candidate.score)
        for item in assignments
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose-window", type=int, default=81)
    parser.add_argument("--start-stride", type=int, default=8)
    parser.add_argument("--max-clips", type=int, default=100)
    parser.add_argument("--max-per-scene", type=int, default=1)
    parser.add_argument("--image-subdir", default="images_8")
    parser.add_argument("--frozen-split-csv", type=Path)
    parser.add_argument("--expected-frozen-split-sha256")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["debug", "validation", "test", "dev"],
        default=["validation"],
    )
    parser.add_argument(
        "--scene-descriptions-json",
        type=Path,
        help="Optional scene-id to content-description mapping; camera motion still comes from GT.",
    )
    parser.add_argument("--min-path-length", type=float, default=2.0)
    parser.add_argument("--min-displacement", type=float, default=0.75)
    parser.add_argument("--min-rotation-deg", type=float, default=0.0)
    parser.add_argument("--min-normalized-path", type=float, default=0.75)
    parser.add_argument("--min-normalized-displacement", type=float, default=0.25)
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
    if args.pose_window < 2 or args.start_stride < 1:
        parser.error("--pose-window must be >= 2 and --start-stride must be positive")

    selected_splits = set(args.splits)
    if "dev" in selected_splits and selected_splits != {"dev"}:
        parser.error("dev split cannot be combined with another split")
    if selected_splits == {"dev"} and args.scene_descriptions_json is not None:
        parser.error("dev manifest forbids --scene-descriptions-json")
    if selected_splits == {"dev"} and (
        args.frozen_split_csv is None or args.expected_frozen_split_sha256 is None
    ):
        parser.error(
            "dev manifest requires --frozen-split-csv and "
            "--expected-frozen-split-sha256"
        )
    assignments = (
        load_frozen_assignments(
            args.frozen_split_csv,
            selected_splits,
            FORMAL_SPLIT_COUNTS,
            args.expected_frozen_split_sha256,
        )
        if args.frozen_split_csv
        else []
    )
    # For the dev contract, load_frozen_assignments has already verified the
    # exact bytes against this digest.  Carry that verified identity forward
    # instead of reopening the path later when writing provenance metadata.
    validated_frozen_split_sha256 = (
        args.expected_frozen_split_sha256
        if args.frozen_split_csv and args.expected_frozen_split_sha256
        else sha256_file(args.frozen_split_csv)
        if args.frozen_split_csv
        else None
    )
    assignment_by_scene = {item.scene_id: item for item in assignments}
    descriptions = (
        {}
        if selected_splits == {"dev"}
        else load_scene_descriptions(args.scene_descriptions_json)
    )

    all_candidates: list[Candidate] = []
    if assignments:
        try:
            paths = resolve_frozen_transforms(args.roots, assignments)
        except FrozenSourceError:
            raise
        except Exception:
            # Keep all formal source resolution failures private even when the
            # filesystem itself includes identifiers in an exception message.
            raise ValueError("frozen source path resolution failed") from None
    else:
        paths = sorted(iter_transforms(args.roots))
    for path in paths:
        try:
            all_candidates.extend(
                scene_candidates(path, args.pose_window, args.start_stride, args.image_subdir)
            )
        except Exception:
            if assignments:
                raise ValueError("frozen source transforms validation failed") from None
            raise
    threshold_filtered = [
        item
        for item in all_candidates
        if item.path_length >= args.min_path_length
        and item.displacement >= args.min_displacement
        and item.rotation_deg >= args.min_rotation_deg
        and item.straightness >= args.min_straightness
        and item.normalized_path >= args.min_normalized_path
        and item.normalized_displacement >= args.min_normalized_displacement
    ]
    if assignments:
        scored = threshold_filtered
        filtered = threshold_filtered
        score_cutoff = None
        selected = select_best_per_scene(filtered, assignments)
    else:
        scored = assign_global_motion_scores(threshold_filtered)
        score_cutoff = (
            float(np.quantile([item.score for item in scored], args.large_motion_quantile))
            if scored
            else float("inf")
        )
        filtered = [item for item in scored if item.score >= score_cutoff]
        selected = select_diverse(filtered, args.max_clips, args.max_per_scene)

    cases = {}
    source_dataset_revisions: set[str] = set()
    for item in selected:
        case_id = f"dl3dv_{item.scene_id}_s{item.start_index:05d}_{item.motion_class}"
        if case_id in cases:
            if assignments:
                raise RuntimeError("duplicate frozen benchmark case")
            raise RuntimeError(f"Duplicate benchmark case id: {case_id}")
        assignment = assignment_by_scene.get(item.scene_id)
        text_prompt = render_prompt(item.motion_class, descriptions.get(item.scene_id))
        source_provenance = load_source_scene_provenance(item, required=bool(assignments))
        if source_provenance and isinstance(source_provenance.get("dataset_revision"), str):
            source_dataset_revisions.add(source_provenance["dataset_revision"])
        cases[case_id] = {
            "text_prompt": text_prompt,
            "scene_description": descriptions.get(item.scene_id),
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
                "normalized_path": item.normalized_path,
                "normalized_displacement": item.normalized_displacement,
                "turn_consistency": item.turn_consistency,
                "turn_monotonicity": item.turn_monotonicity,
                "turn_smoothness": item.turn_smoothness,
                "turn_spread": item.turn_spread,
                "selection_score": item.score,
            },
            "transforms_path": str(item.transforms_path),
            "source_scene_provenance": source_provenance,
        }
        if assignment is not None:
            cases[case_id]["protocol_split"] = assignment.split
            cases[case_id]["split"] = assignment.split
            cases[case_id]["split_order"] = assignment.split_order
            cases[case_id]["source_order"] = assignment.source_order

    payload = {
        "_meta": {
            "schema": "dl3dv-gt-trajectory-source-v1",
            "formal_protocol": False,
            "selection": (
                "best eligible GT-grounded trajectory per frozen scene"
                if assignments
                else "pose-driven large camera motion"
            ),
            "n_transforms": len(paths),
            "n_candidates": len(all_candidates),
            "n_threshold_filtered": len(threshold_filtered),
            "n_filtered": len(filtered),
            "n_selected": len(selected),
            "pose_window": args.pose_window,
            "start_stride": args.start_stride,
            "image_subdir": args.image_subdir,
            "minimum_motion_thresholds": {
                "path_length": args.min_path_length,
                "displacement": args.min_displacement,
                "rotation_deg": args.min_rotation_deg,
                "straightness": args.min_straightness,
                "normalized_path": args.min_normalized_path,
                "normalized_displacement": args.min_normalized_displacement,
            },
            "large_motion_quantile": args.large_motion_quantile,
            "global_motion_score_cutoff": score_cutoff,
            "frozen_split_csv": str(args.frozen_split_csv) if args.frozen_split_csv else None,
            "frozen_split_sha256": validated_frozen_split_sha256,
            "scene_descriptions_json": (
                str(args.scene_descriptions_json) if args.scene_descriptions_json else None
            ),
            "scene_descriptions_sha256": (
                sha256_file(args.scene_descriptions_json)
                if args.scene_descriptions_json
                else None
            ),
            "selected_splits": sorted(set(args.splits)) if assignments else None,
            "source_dataset_revisions": sorted(source_dataset_revisions),
            "prompt_policy": PROMPT_POLICY_VERSION,
            "prompt_template_sha256": hashlib.sha256(
                json.dumps(PROMPTS, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "prompt_policy_description": (
                "large physical camera motion derived from GT poses; strong parallax; "
                "no static-camera or motion-suppressing language"
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
