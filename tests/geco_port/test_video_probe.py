from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from geometry_selection import video_probe


class _ImageIOReader:
    def __init__(self, *, frame_count=4, size=(12, 8), fps=24.0):
        self.items = iter(
            [
                {"size": size, "fps": fps},
                *([bytes(size[0] * size[1] * 3)] * frame_count),
            ]
        )
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.items)

    def close(self):
        self.closed = True


def _fake_imageio(reader):
    return SimpleNamespace(
        read_frames=lambda *args, **kwargs: reader,
        get_ffmpeg_exe=lambda: "/fake/ffmpeg",
    )


@pytest.fixture(autouse=True)
def _successful_strict_decode(monkeypatch):
    monkeypatch.setattr(
        video_probe.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )


def test_probe_video_uses_full_imageio_decode_without_system_ffmpeg(tmp_path, monkeypatch):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"placeholder")
    reader = _ImageIOReader(frame_count=4)
    monkeypatch.setattr(video_probe.shutil, "which", lambda _: None)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", _fake_imageio(reader))

    result = video_probe.probe_video(path, frames=4, height=8, width=12, fps=24)

    assert result["backend"] == "imageio-ffmpeg-full-decode"
    assert result["nb_frames"] == 4
    assert reader.closed


@pytest.mark.parametrize(
    ("reader", "message"),
    [
        (_ImageIOReader(frame_count=3), "nb_frames"),
        (_ImageIOReader(frame_count=5), "nb_frames"),
        (_ImageIOReader(size=(13, 8)), "Decoded frame 0"),
        (_ImageIOReader(fps=23.0), "frame-rate"),
    ],
)
def test_probe_video_rejects_invalid_imageio_decode(tmp_path, monkeypatch, reader, message):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"placeholder")
    monkeypatch.setattr(video_probe.shutil, "which", lambda _: None)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", _fake_imageio(reader))

    with pytest.raises(ValueError, match=message):
        video_probe.probe_video(path, frames=4, height=8, width=12, fps=24)

    assert reader.closed


def test_probe_video_rejects_nonzero_strict_ffmpeg_decode(tmp_path, monkeypatch):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"placeholder")
    reader = _ImageIOReader(frame_count=4)
    monkeypatch.setattr(video_probe.shutil, "which", lambda _: None)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", _fake_imageio(reader))

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(video_probe.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        video_probe.probe_video(path, frames=4, height=8, width=12, fps=24)
