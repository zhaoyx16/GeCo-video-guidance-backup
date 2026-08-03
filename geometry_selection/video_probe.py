"""Strict video validation with an ffmpeg-independent fallback."""

from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path


def _validate_common(
    actual: dict,
    *,
    frames: int,
    height: int,
    width: int,
    fps: int,
) -> None:
    expected_shape = {"width": width, "height": height, "nb_frames": frames}
    actual_shape = {key: actual[key] for key in expected_shape}
    if actual_shape != expected_shape:
        raise ValueError(f"Video mismatch: {actual_shape} != {expected_shape}")
    actual_fps = float(Fraction(str(actual["avg_frame_rate"])))
    if abs(actual_fps - fps) > 0.01:
        raise ValueError(f"Video frame-rate mismatch: {actual_fps} != {fps}")


def _probe_with_ffmpeg(
    path: Path,
    *,
    frames: int,
    height: int,
    width: int,
    fps: int,
) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected exactly one video stream: {path}")
    stream = streams[0]
    actual = {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": stream["avg_frame_rate"],
        "nb_frames": int(stream["nb_read_frames"]),
    }
    _validate_common(actual, frames=frames, height=height, width=width, fps=fps)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
        check=True,
    )
    actual["backend"] = "ffmpeg-full-decode"
    return actual


def _probe_with_opencv(
    path: Path,
    *,
    frames: int,
    height: int,
    width: int,
    fps: int,
) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open video: {path}")
    actual_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    actual_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.shape[:2] != (height, width):
                shape = None if frame is None else frame.shape[:2]
                raise ValueError(
                    f"Decoded frame {decoded_frames} has shape {shape}; "
                    f"expected {(height, width)}"
                )
            decoded_frames += 1
    finally:
        capture.release()

    actual = {
        "width": actual_width,
        "height": actual_height,
        "avg_frame_rate": str(actual_fps),
        "nb_frames": decoded_frames,
    }
    _validate_common(actual, frames=frames, height=height, width=width, fps=fps)
    actual["backend"] = "opencv-full-decode"
    return actual


def _probe_with_imageio_ffmpeg(
    path: Path,
    *,
    frames: int,
    height: int,
    width: int,
    fps: int,
) -> dict:
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(
        str(path),
        pix_fmt="rgb24",
        bits_per_pixel=24,
    )
    decoded_frames = 0
    expected_frame_bytes = width * height * 3
    try:
        metadata = next(reader)
        size = metadata.get("size")
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError(f"Missing decoded video size in metadata: {metadata}")
        actual_width, actual_height = (int(size[0]), int(size[1]))
        actual_fps = float(metadata["fps"])
        for frame in reader:
            if len(frame) != expected_frame_bytes:
                raise ValueError(
                    f"Decoded frame {decoded_frames} has {len(frame)} bytes; "
                    f"expected {expected_frame_bytes}"
                )
            decoded_frames += 1
    finally:
        reader.close()

    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
    )

    actual = {
        "width": actual_width,
        "height": actual_height,
        "avg_frame_rate": str(actual_fps),
        "nb_frames": decoded_frames,
    }
    _validate_common(actual, frames=frames, height=height, width=width, fps=fps)
    actual["backend"] = "imageio-ffmpeg-full-decode"
    return actual


def probe_video(path: Path, *, frames: int, height: int, width: int, fps: int) -> dict:
    """Validate dimensions, frame rate, exact decoded frame count, and decodability."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if shutil.which("ffprobe") is not None and shutil.which("ffmpeg") is not None:
        return _probe_with_ffmpeg(
            path, frames=frames, height=height, width=width, fps=fps
        )
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        pass
    else:
        return _probe_with_imageio_ffmpeg(
            path, frames=frames, height=height, width=width, fps=fps
        )
    return _probe_with_opencv(path, frames=frames, height=height, width=width, fps=fps)
