"""DL3DV parsing and preprocessing helpers for Wan latent probes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

import torch
from PIL import Image


_FRAME_ID_RE = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class DL3DVScene:
    scene_uid: str
    root: Path
    frame_paths: tuple[Path, ...]
    frame_ids: torch.Tensor
    camera_poses_w2c: torch.Tensor
    intrinsics: torch.Tensor
    source_width: int
    source_height: int
    camera_model: str
    distortion: tuple[float, float, float, float]


def _frame_id(path: Path) -> int:
    match = _FRAME_ID_RE.search(path.stem)
    if match is None:
        raise ValueError(f"Cannot parse numeric frame ID from {path}")
    return int(match.group(1))


def resolve_frame_path(scene_root: Path, file_path: str) -> Path:
    relative = Path(file_path)
    candidates = [
        scene_root / relative,
        scene_root / "images_4" / relative.name,
        scene_root / "images" / relative.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve {file_path!r} below {scene_root}")


def load_dl3dv_scene(transforms_path: str | Path, dataset_root: str | Path) -> DL3DVScene:
    transforms_path = Path(transforms_path)
    dataset_root = Path(dataset_root)
    payload = json.loads(transforms_path.read_text(encoding="utf-8"))
    scene_root = transforms_path.parent
    source_width = int(payload["w"])
    source_height = int(payload["h"])
    base_intrinsics = torch.tensor(
        [
            [float(payload["fl_x"]), 0.0, float(payload["cx"])],
            [0.0, float(payload["fl_y"]), float(payload["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    ordered = []
    for frame in payload["frames"]:
        path = resolve_frame_path(scene_root, frame["file_path"])
        ordered.append((_frame_id(path), path, torch.tensor(frame["transform_matrix"], dtype=torch.float64)))
    ordered.sort(key=lambda item: item[0])
    if len({item[0] for item in ordered}) != len(ordered):
        raise ValueError(f"Duplicate frame IDs in {transforms_path}")

    # Nerfstudio stores OpenGL c2w (x right, y up, z back). Convert to OpenCV
    # camera axes so poses and pixel intrinsics use the same convention.
    opencv_to_opengl = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64))
    poses_c2w = torch.stack([item[2] for item in ordered])
    poses_w2c = torch.linalg.inv(poses_c2w @ opencv_to_opengl).float()
    frame_ids = torch.tensor([item[0] for item in ordered], dtype=torch.long)
    intrinsics = base_intrinsics.unsqueeze(0).repeat(len(ordered), 1, 1)
    return DL3DVScene(
        scene_uid=str(scene_root.relative_to(dataset_root)),
        root=scene_root,
        frame_paths=tuple(item[1] for item in ordered),
        frame_ids=frame_ids,
        camera_poses_w2c=poses_w2c,
        intrinsics=intrinsics,
        source_width=source_width,
        source_height=source_height,
        camera_model=str(payload.get("camera_model", "UNKNOWN")),
        distortion=tuple(float(payload.get(key, 0.0)) for key in ("k1", "k2", "p1", "p2")),
    )


def discover_dl3dv_scenes(dataset_root: str | Path) -> list[DL3DVScene]:
    dataset_root = Path(dataset_root)
    transforms = sorted(dataset_root.glob("**/transforms.json"))
    if not transforms:
        raise FileNotFoundError(f"No transforms.json found below {dataset_root}")
    return [load_dl3dv_scene(path, dataset_root) for path in transforms]


def resize_cover_crop_geometry(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int, int, int]:
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Image dimensions must be positive")
    scale = max(target_width / source_width, target_height / source_height)
    resized_width = max(target_width, round(source_width * scale))
    resized_height = max(target_height, round(source_height * scale))
    crop_left = (resized_width - target_width) // 2
    crop_top = (resized_height - target_height) // 2
    return resized_width, resized_height, crop_left, crop_top


def transform_intrinsics_for_cover_crop(
    intrinsics: torch.Tensor,
    *,
    calibration_width: int,
    calibration_height: int,
    image_width: int,
    image_height: int,
    target_width: int,
    target_height: int,
) -> torch.Tensor:
    resized_width, resized_height, crop_left, crop_top = resize_cover_crop_geometry(
        image_width, image_height, target_width, target_height
    )
    sx = resized_width / calibration_width
    sy = resized_height / calibration_height
    transformed = intrinsics.clone().float()
    transformed[..., 0, 0] *= sx
    transformed[..., 1, 1] *= sy
    transformed[..., 0, 2] = transformed[..., 0, 2] * sx - crop_left
    transformed[..., 1, 2] = transformed[..., 1, 2] * sy - crop_top
    return transformed


def preprocess_frame(image: Image.Image, target_width: int, target_height: int) -> torch.Tensor:
    image = image.convert("RGB")
    resized_width, resized_height, crop_left, crop_top = resize_cover_crop_geometry(
        image.width, image.height, target_width, target_height
    )
    image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    image = image.crop((crop_left, crop_top, crop_left + target_width, crop_top + target_height))
    byte_tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    frame = byte_tensor.reshape(target_height, target_width, 3).permute(2, 0, 1).float()
    return frame.div(127.5).sub(1.0)


def source_content_sha256(frame_paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(b"dl3dv_ordered_source_frames_v1")
    for path in frame_paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def choose_clip_starts(
    num_frames: int,
    *,
    clip_frames: int,
    frame_step: int,
    clip_stride: int,
    max_clips: int | None,
) -> list[int]:
    if min(clip_frames, frame_step, clip_stride) <= 0:
        raise ValueError("clip_frames, frame_step, and clip_stride must be positive")
    span = (clip_frames - 1) * frame_step + 1
    starts = list(range(0, max(0, num_frames - span + 1), clip_stride))
    if max_clips is not None:
        if max_clips <= 0:
            raise ValueError("max_clips must be positive when provided")
        if len(starts) > max_clips:
            positions = torch.linspace(0, len(starts) - 1, max_clips).round().long().tolist()
            starts = [starts[index] for index in positions]
    return starts


def latent_anchor_rgb_indices(num_rgb_frames: int, temporal_ratio: int) -> list[int]:
    if num_rgb_frames < 1 or temporal_ratio < 1:
        raise ValueError("num_rgb_frames and temporal_ratio must be positive")
    if (num_rgb_frames - 1) % temporal_ratio:
        raise ValueError("(num_rgb_frames - 1) must be divisible by temporal_ratio")
    return list(range(0, num_rgb_frames, temporal_ratio))
