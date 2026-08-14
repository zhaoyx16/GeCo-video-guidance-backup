#!/usr/bin/env python3
"""Build the sealed paired DPO train-dev evaluator mirror and base input lock.

The generation tree is intentionally treated as mutable.  This builder accepts
only the controller's final 200-task receipt, opens every source artifact with
``O_NOFOLLOW``, copies from the held inode, rechecks pathname/inode metadata,
and publishes one fresh read-only bundle by atomic directory rename.

The bundle contains both video methods, but only the base input lock.  The
adapted input lock is published later, after base scoring freezes the shared
Independent-LRE denominator and per-scene motion anchors.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any


GENERATION_RECEIPT_SCHEMA = "wan-lora-dpo-traindev-paired-generation-receipt-v1"
INPUT_LOCK_SCHEMA = "geometry-selection-five-metric-traindev-input-lock-v1"
BUNDLE_SCHEMA = "wan-lora-dpo-traindev-evaluator-bundle-v1"
PARENT_PROTOCOL_SCHEMA = "geometry-selection-five-metric-protocol-v2"
PARENT_SCHEDULE_SCHEMA = "geometry-selection-metric-schedule-v2"
TRAINDEV_PROTOCOL_SCHEMA = "geometry-selection-five-metric-traindev-protocol-v1"
TRAINDEV_SCHEDULE_SCHEMA = "geometry-selection-metric-traindev-schedule-v1"
BASE_METHOD_ID = "wan_lora_dpo_step64_base_traindev"
ADAPTED_METHOD_ID = "wan_lora_dpo_step64_adapted_traindev"
BASE_METHOD_LABEL = "Wan LoRA-DPO step-64 paired base train-dev"
ADAPTED_METHOD_LABEL = "Full-Graph LoRA-DPO step-64 paired adapted train-dev"
EXPECTED_MANIFEST_SHA256 = "c48bf6ab56359bd22c08b905b4be1cf2822dd7f315ad6b21d693693d5d53f579"
EXPECTED_FORMAL_MANIFEST_SHA256 = "24a4e47a42e576f3947f37e98aadfb3959447f6248d06652ca37ce2981cb4d89"
EXPECTED_FORMAL_SOURCE_MANIFEST_SHA256 = "da4c05c0ec8f6f8fd08daf3a69482c631d221f84fcb7a5d251c1e3fb1509dd9d"
PARENT_PROTOCOL_SHA256 = "d706abfe58d449549558d11aa986f3f13a4bb463af68a8f6b7ec890049628048"
PARENT_SCHEDULE_SHA256 = "afe5b44f954adb28ffb40fccb9242111956be9814471adde87070a9978755612"
EXPECTED_CONTROLLER_SHA256 = "402c6e8ac720c9166d4c6d82d982d416189470a3061442f27daf0131e1e4b62b"
EXPECTED_CONTROLLER_TESTS_SHA256 = "87461746097bc7ef56d38c80d8ca1c667bb6016b935f4bdde2657b71381c3ef6"
EXPECTED_APPROVAL_SHA256 = "cb078d3299ff0526340d893f849e5a1c1fd1f68cfc5539bc6f64b739ab947a18"
EXPECTED_GENERATION_COMMIT = "efa13ac0ce45f3e24ddd3701fc2c7ea1e1fc85a3"
EXPECTED_RUNNER_SHA256 = "c9bb9f7c200cbb8bb4a64466326fa21a2542b2381d3f0c08a0aa680eb081fdf8"
EXPECTED_LORA_RECEIPT_SHA256 = "5d9509e36d5d9d9861df254bc64bdc3342eba79980d39316375b6795e8773ef0"
EXPECTED_LORA_WEIGHT_SHA256 = "043331e8c9cc67cbc168d60256aed7cfa3f6f65a0b19ee6fb4bafd374d5a5ee3"
EXPECTED_MODEL_IDENTITY_SHA256 = "f0235d0491e5911382b65055775c859ecc5f2a3b4301a68f985f47eb7092a569"
EXPECTED_VIDEO_PROBE = {"frames": 121, "width": 1280, "height": 704, "fps": 24}
EXPECTED_STORED_VIDEO_PROBE = {
    "width": 1280,
    "height": 704,
    "avg_frame_rate": "24.0",
    "nb_frames": 121,
    "backend": "imageio-ffmpeg-full-decode",
}
EXPECTED_GENERATION_PROFILE = {
    "steps": 50,
    "frames": 121,
    "height": 704,
    "width": 1280,
    "fps": 24,
    "guidance_scale": 5.0,
    "negative_prompt": None,
    "wan_negative_prompt_mode": "none",
}
MODES = ("base", "adapted")


class ContractError(RuntimeError):
    """A source artifact or immutable identity differs from the contract."""


def canonical_bytes(value: Any, *, newline: bool = True) -> bytes:
    result = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return result + (b"\n" if newline else b"")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value, newline=False)).hexdigest()


def set_commitment(values: set[str]) -> str:
    """Commit to a private identifier set without publishing its members."""
    return hashlib.sha256(canonical_bytes(sorted(values), newline=False)).hexdigest()


def is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def require_sha256(value: Any, label: str) -> str:
    if not is_sha256(value):
        raise ContractError(f"{label} must be a SHA256")
    return value


def verify_external_payload_sha(payload: bytes, expected_sha256: Any, label: str) -> str:
    expected = require_sha256(expected_sha256, f"expected {label} SHA")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ContractError(f"{label} SHA differs from the externally supplied identity")
    return actual


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def read_regular_no_follow(path: Path, label: str) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino):
            raise ContractError(f"{label} must be one stable regular inode")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named_after = os.stat(path, follow_symlinks=False)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ContractError(f"{label} changed while it was read")
        if (after.st_dev, after.st_ino, after.st_size) != (named_after.st_dev, named_after.st_ino, named_after.st_size):
            raise ContractError(f"{label} pathname changed while it was read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise ContractError(f"{label} size changed while it was read")
        return payload, before
    finally:
        os.close(descriptor)


def require_no_symlink_ancestors(path: Path, label: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise ContractError(f"{label} traverses a symlink ancestor")


def open_absolute_directory_no_follow(path: Path, label: str) -> int:
    """Open an absolute directory through a descriptor-pinned, no-symlink chain."""
    absolute = path.absolute()
    if not absolute.is_absolute():
        raise ContractError(f"{label} must be absolute")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parts[1:]:
            parent_before = os.fstat(descriptor)
            child = os.open(part, flags, dir_fd=descriptor)
            try:
                named = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
                child_stat = os.fstat(child)
                parent_after = os.fstat(descriptor)
                if (
                    (parent_before.st_dev, parent_before.st_ino)
                    != (parent_after.st_dev, parent_after.st_ino)
                    or not stat.S_ISDIR(child_stat.st_mode)
                    or (child_stat.st_dev, child_stat.st_ino) != (named.st_dev, named.st_ino)
                ):
                    raise ContractError(f"{label} directory chain changed while opening")
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException as error:
        os.close(descriptor)
        if isinstance(error, ContractError):
            raise
        raise ContractError(
            f"{label} traverses a missing, non-directory, or symlink component"
        ) from error


def copy_stable_file(source: Path, target: Path, expected_sha256: str, label: str) -> tuple[str, int]:
    expected_sha256 = require_sha256(expected_sha256, f"{label} expected SHA")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(source, flags)
    target_fd: int | None = None
    try:
        before = os.fstat(source_fd)
        named = os.stat(source, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino):
            raise ContractError(f"{label} source is not one stable regular inode")
        target_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(source_fd, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(target_fd, view)
                view = view[written:]
            size += len(block)
        os.fsync(target_fd)
        after = os.fstat(source_fd)
        named_after = os.stat(source, follow_symlinks=False)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ContractError(f"{label} source changed during copy")
        if (after.st_dev, after.st_ino, after.st_size) != (named_after.st_dev, named_after.st_ino, named_after.st_size):
            raise ContractError(f"{label} source pathname changed during copy")
        actual = digest.hexdigest()
        if actual != expected_sha256 or size != before.st_size:
            raise ContractError(f"{label} source SHA/size mismatch")
    finally:
        os.close(source_fd)
        if target_fd is not None:
            os.close(target_fd)
    os.chmod(target, 0o444)
    return expected_sha256, size


def relative_source_path(path_value: Any, source_root: Path, label: str) -> Path:
    """Return a lexical source-root-relative path without following components."""
    if not isinstance(path_value, str) or not path_value:
        raise ContractError(f"{label} path is missing")
    path = Path(path_value)
    if not path.is_absolute():
        raise ContractError(f"{label} must be absolute")
    root = source_root.absolute()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ContractError(f"{label} is not lexically below the generation root") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ContractError(f"{label} has unsafe path components")
    return relative


def open_directory_below(root: Path, relative: Path, label: str) -> int:
    """Open every directory component with openat/O_NOFOLLOW."""
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ContractError(f"{label} has unsafe relative components")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = open_absolute_directory_no_follow(root, f"{label} source root")
    try:
        for part in relative.parts:
            parent_before = os.fstat(descriptor)
            child = os.open(part, flags, dir_fd=descriptor)
            try:
                named = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
                child_stat = os.fstat(child)
                parent_after = os.fstat(descriptor)
                if (
                    (parent_before.st_dev, parent_before.st_ino)
                    != (parent_after.st_dev, parent_after.st_ino)
                    or not stat.S_ISDIR(child_stat.st_mode)
                    or (child_stat.st_dev, child_stat.st_ino) != (named.st_dev, named.st_ino)
                ):
                    raise ContractError(f"{label} directory chain changed while opening")
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException as error:
        os.close(descriptor)
        if isinstance(error, ContractError):
            raise
        raise ContractError(
            f"{label} traverses a missing, non-directory, or symlink component"
        ) from error


def verify_directory_below_identity(
    root: Path, relative: Path, expected_fd: int, label: str
) -> None:
    reopened = open_directory_below(root, relative, label)
    try:
        expected = os.fstat(expected_fd)
        actual = os.fstat(reopened)
        if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
            raise ContractError(f"{label} path was replaced while its artifacts were copied")
    finally:
        os.close(reopened)


def read_regular_at(directory_fd: int, name: str, label: str) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino):
            raise ContractError(f"{label} must be one stable regular inode")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ContractError(f"{label} changed while it was read")
        if (after.st_dev, after.st_ino, after.st_size) != (named_after.st_dev, named_after.st_ino, named_after.st_size):
            raise ContractError(f"{label} directory entry changed while it was read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise ContractError(f"{label} size changed while it was read")
        return payload, before
    finally:
        os.close(descriptor)


def copy_regular_at(
    directory_fd: int, name: str, target: Path, expected_sha256: str, label: str
) -> tuple[str, int]:
    expected_sha256 = require_sha256(expected_sha256, f"{label} expected SHA")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(name, flags, dir_fd=directory_fd)
    target_fd: int | None = None
    try:
        before = os.fstat(source_fd)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino):
            raise ContractError(f"{label} source is not one stable regular inode")
        target_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(source_fd, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(target_fd, view)
                view = view[written:]
            size += len(block)
        os.fsync(target_fd)
        after = os.fstat(source_fd)
        named_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ContractError(f"{label} source changed during copy")
        if (after.st_dev, after.st_ino, after.st_size) != (named_after.st_dev, named_after.st_ino, named_after.st_size):
            raise ContractError(f"{label} directory entry changed during copy")
        if digest.hexdigest() != expected_sha256 or size != before.st_size:
            raise ContractError(f"{label} source SHA/size mismatch")
    finally:
        os.close(source_fd)
        if target_fd is not None:
            os.close(target_fd)
    os.chmod(target, 0o444)
    return expected_sha256, size


def hash_regular_at(directory_fd: int, name: str, label: str) -> tuple[str, int]:
    payload, metadata = read_regular_at(directory_fd, name, label)
    return hashlib.sha256(payload).hexdigest(), metadata.st_size


def acquire_generation_lock_at(
    directory_fd: int, expected_sha256: str, label: str
) -> tuple[int, tuple[int, int, int, int, int]]:
    """Hold the run's exclusive generation flock until its copy is closed."""
    expected_sha256 = require_sha256(expected_sha256, f"{label} SHA")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(".generation.lock", flags, dir_fd=directory_fd)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(descriptor)
        raise
    try:
        before = os.fstat(descriptor)
        named = os.stat(".generation.lock", dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino):
            raise ContractError(f"{label} is not one stable regular inode")
        os.lseek(descriptor, 0, os.SEEK_SET)
        payload = b""
        while True:
            block = os.read(descriptor, 4096)
            if not block:
                break
            payload += block
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if hashlib.sha256(payload).hexdigest() != expected_sha256 or len(payload) != before.st_size:
            raise ContractError(f"{label} differs from the final generation receipt")
        return descriptor, identity
    except BaseException:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        raise


def release_generation_lock_at(
    directory_fd: int,
    descriptor: int,
    identity: tuple[int, int, int, int, int],
    label: str,
) -> None:
    try:
        after = os.fstat(descriptor)
        named = os.stat(".generation.lock", dir_fd=directory_fd, follow_symlinks=False)
        actual = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if actual != identity or (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino):
            raise ContractError(f"{label} changed while the mirror copy was held")
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def hash_descriptor(descriptor: int) -> tuple[str, int]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        block = os.read(descriptor, 8 * 1024 * 1024)
        if not block:
            break
        digest.update(block)
        size += len(block)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest(), size


def full_decode_video(
    path: Path,
    expected_video_sha256: str,
    expected_video_bytes: int,
    ffmpeg: Path,
    expected_ffmpeg_sha256: str,
    label: str,
) -> dict[str, Any]:
    expected_video_sha256 = require_sha256(expected_video_sha256, f"{label} SHA")
    if not isinstance(expected_video_bytes, int) or expected_video_bytes <= 0:
        raise ContractError(f"{label} byte count is invalid")
    expected_ffmpeg_sha256 = require_sha256(
        expected_ffmpeg_sha256, "destination decoder SHA"
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    video_descriptor = os.open(path, flags)
    try:
        descriptor = os.open(ffmpeg, flags)
    except BaseException:
        os.close(video_descriptor)
        raise
    try:
        video_before = os.fstat(video_descriptor)
        video_named = os.stat(path, follow_symlinks=False)
        video_sha, video_size = hash_descriptor(video_descriptor)
        if (
            not stat.S_ISREG(video_before.st_mode)
            or (video_before.st_dev, video_before.st_ino)
            != (video_named.st_dev, video_named.st_ino)
            or (video_sha, video_size)
            != (expected_video_sha256, expected_video_bytes)
        ):
            raise ContractError(f"{label} differs before full decode")
        before = os.fstat(descriptor)
        named = os.stat(ffmpeg, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_mode & 0o111 == 0
            or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ContractError("an executable stable regular ffmpeg decoder is required")
        decoder_sha, decoder_size = hash_descriptor(descriptor)
        if decoder_sha != expected_ffmpeg_sha256 or decoder_size != before.st_size:
            raise ContractError("destination decoder differs from its external SHA binding")
        pinned_decoder = f"/proc/self/fd/{descriptor}"
        pinned_video = f"/proc/self/fd/{video_descriptor}"
        result = subprocess.run(
            [
                pinned_decoder,
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-i",
                pinned_video,
                "-map",
                "0:v:0",
                "-f",
                "null",
                "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            pass_fds=(descriptor, video_descriptor),
        )
        video_after = os.fstat(video_descriptor)
        video_named_after = os.stat(path, follow_symlinks=False)
        video_sha_after, video_size_after = hash_descriptor(video_descriptor)
        after = os.fstat(descriptor)
        named_after = os.stat(ffmpeg, follow_symlinks=False)
        decoder_sha_after, decoder_size_after = hash_descriptor(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            before_identity != after_identity
            or (after.st_dev, after.st_ino) != (named_after.st_dev, named_after.st_ino)
            or (decoder_sha_after, decoder_size_after)
            != (expected_ffmpeg_sha256, decoder_size)
        ):
            raise ContractError("destination decoder changed during full decode")
        if (
            (
                video_before.st_dev,
                video_before.st_ino,
                video_before.st_size,
                video_before.st_mtime_ns,
                video_before.st_ctime_ns,
            )
            != (
                video_after.st_dev,
                video_after.st_ino,
                video_after.st_size,
                video_after.st_mtime_ns,
                video_after.st_ctime_ns,
            )
            or (video_after.st_dev, video_after.st_ino)
            != (video_named_after.st_dev, video_named_after.st_ino)
            or (video_sha_after, video_size_after)
            != (expected_video_sha256, expected_video_bytes)
        ):
            raise ContractError(f"{label} changed during full decode")
        if result.returncode != 0:
            raise ContractError(f"{label} failed independent destination full decode")
        return {
            "decoder_path": str(ffmpeg.absolute()),
            "decoder_sha256": expected_ffmpeg_sha256,
            "decoder_bytes": decoder_size,
            "execution_path": "descriptor_pinned_proc_fd",
            "video_sha256": expected_video_sha256,
            "video_bytes": expected_video_bytes,
            "video_execution_path": "descriptor_pinned_proc_fd",
            "status": "passed",
        }
    finally:
        os.close(descriptor)
        os.close(video_descriptor)


def verify_regular_copy(path: Path, expected_sha256: str, expected_size: int, label: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino):
            raise ContractError(f"{label} is not one stable regular inode")
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
        after = os.fstat(descriptor)
        named_after = os.stat(path, follow_symlinks=False)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ContractError(f"{label} changed while it was verified")
        if (after.st_dev, after.st_ino, after.st_size) != (
            named_after.st_dev,
            named_after.st_ino,
            named_after.st_size,
        ):
            raise ContractError(f"{label} pathname changed while it was verified")
        if size != expected_size or digest.hexdigest() != expected_sha256:
            raise ContractError(f"{label} changed after destination verification")
    finally:
        os.close(descriptor)


def manifest_identifier_sets(payload: dict[str, Any], label: str) -> dict[str, set[str]]:
    rows = [(key, value) for key, value in payload.items() if not key.startswith("_")]
    if len(rows) != 100:
        raise ContractError(f"{label} must contain exactly 100 cases")
    cases, scenes, transforms = set(), set(), set()
    for case_id, item in rows:
        if not isinstance(case_id, str) or not isinstance(item, dict):
            raise ContractError(f"{label} case schema mismatch")
        scene = item.get("scene_id")
        transform = (
            item.get("transforms_sha256")
            or item.get("source_scene_provenance", {}).get("transforms_sha256")
            or item.get("conditioning_image_provenance", {}).get("transforms_sha256")
        )
        if not is_sha256(scene) or not is_sha256(transform):
            raise ContractError(f"{label} lacks scene/transform source identities")
        cases.add(case_id)
        scenes.add(scene)
        transforms.add(transform)
    if any(len(values) != 100 for values in (cases, scenes, transforms)):
        raise ContractError(f"{label} identifiers are not one-to-one")
    return {"case_ids": cases, "scene_ids": scenes, "transform_sources": transforms}


def build_isolation_receipt(
    train_manifest: dict[str, Any], formal_manifest: dict[str, Any], formal_manifest_sha256: str
) -> dict[str, Any]:
    train_sets = manifest_identifier_sets(train_manifest, "train-dev manifest")
    formal_sets = manifest_identifier_sets(formal_manifest, "formal validation manifest")
    overlap_counts = {
        label: len(train_sets[label] & formal_sets[label])
        for label in ("case_ids", "scene_ids", "transform_sources")
    }
    if overlap_counts != {"case_ids": 0, "scene_ids": 0, "transform_sources": 0}:
        raise ContractError("train-dev overlaps the frozen formal validation identities")
    return {
        "schema": "wan-lora-dpo-traindev-reference-isolation-receipt-v1",
        "status": "verified_zero_overlap",
        "train_dev": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "case_count": 100,
            "commitments": {label: set_commitment(values) for label, values in train_sets.items()},
        },
        "excluded_reference": {
            "manifest_sha256": formal_manifest_sha256,
            "source_manifest_sha256": EXPECTED_FORMAL_SOURCE_MANIFEST_SHA256,
            "case_count": 100,
            "commitments": {label: set_commitment(values) for label, values in formal_sets.items()},
        },
        "overlap_counts": overlap_counts,
        "ids_disclosed": False,
    }


def write_readonly(path: Path, payload: bytes) -> str:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444)
    return hashlib.sha256(payload).hexdigest()


def rename_noreplace(source: Path, target: Path, *, directory_fd: int | None = None) -> None:
    """Atomically publish a directory without replacement on Linux."""
    if os.uname().sysname != "Linux":
        raise ContractError("atomic RENAME_NOREPLACE publication requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ContractError("libc lacks renameat2 for no-replace publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    if directory_fd is not None:
        if source.parent.absolute() != target.parent.absolute():
            raise ContractError("descriptor-pinned publication requires one shared parent")
        source_argument = os.fsencode(source.name)
        target_argument = os.fsencode(target.name)
        source_directory_fd = target_directory_fd = directory_fd
    else:
        source_argument = os.fsencode(source)
        target_argument = os.fsencode(target)
        source_directory_fd = target_directory_fd = at_fdcwd
    rename_noreplace_flag = 1
    result = renameat2(
        source_directory_fd,
        source_argument,
        target_directory_fd,
        target_argument,
        rename_noreplace_flag,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def safe_relative_source(path_value: Any, source_root: Path, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ContractError(f"{label} path is missing")
    path = Path(path_value)
    if not path.is_absolute() or path.is_symlink():
        raise ContractError(f"{label} must be an absolute non-symlink path")
    root = source_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if root not in resolved.parents:
        raise ContractError(f"{label} escapes the generation root")
    return path


def normalized_pair_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ContractError("run config must be an object")
    result = copy.deepcopy(config)
    lora = result.get("lora_dpo")
    if not isinstance(lora, dict) or lora.get("mode") not in MODES:
        raise ContractError("run config lacks the exact LoRA mode")
    lora.pop("mode")
    return result


def validate_manifest(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    meta = payload.get("_meta")
    if not isinstance(meta, dict):
        raise ContractError("train-dev manifest metadata is missing")
    expected_meta = {
        "schema": "dl3dv-gt-trajectory-source-v1",
        "formal_protocol": False,
        "selected_splits": ["dev"],
        "n_transforms": 100,
        "n_selected": 100,
        "scene_descriptions_json": None,
        "scene_descriptions_sha256": None,
        "frozen_split_sha256": "605a34290f2a55990c8e30927afabd1b01ed1bf57920361ec7702f216cf92589",
    }
    if any(meta.get(key) != value for key, value in expected_meta.items()):
        raise ContractError("train-dev manifest metadata/isolation mismatch")
    cases = [(key, value) for key, value in payload.items() if not key.startswith("_")]
    if len(cases) != 100:
        raise ContractError("train-dev manifest must contain exactly 100 cases")
    scene_ids: set[str] = set()
    for order, (case_id, case) in enumerate(cases):
        if not isinstance(case_id, str) or not isinstance(case, dict):
            raise ContractError("train-dev manifest case schema mismatch")
        scene_id = case.get("scene_id")
        provenance = case.get("conditioning_image_provenance")
        if (
            not is_sha256(scene_id)
            or scene_id in scene_ids
            or case.get("split") != "dev"
            or case.get("protocol_split") != "dev"
            or case.get("split_order") != order
            or case.get("scene_description") is not None
            or not isinstance(case.get("text_prompt"), str)
            or "and continuing through the visible environment" not in case["text_prompt"]
            or not isinstance(provenance, dict)
            or not is_sha256(provenance.get("image_sha256"))
        ):
            raise ContractError("train-dev manifest case/order/isolation mismatch")
        scene_ids.add(scene_id)
    return cases


def validate_generation_receipt(
    receipt: dict[str, Any], source_root: Path, manifest_sha256: str, cases: list[tuple[str, dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if receipt.get("schema") != GENERATION_RECEIPT_SCHEMA or receipt.get("status") != "COMPLETE":
        raise ContractError("generation receipt is not the final COMPLETE receipt")
    contract = receipt.get("contract")
    generation_repo = receipt.get("generation_repo")
    reviewed = receipt.get("reviewed_sources")
    approval = receipt.get("independent_approval")
    model = receipt.get("model")
    lora = receipt.get("lora")
    if not all(isinstance(item, dict) for item in (contract, generation_repo, reviewed, approval, model, lora)):
        raise ContractError("generation receipt provenance is malformed")
    expected = {
        "case_count": 100,
        "task_count": 200,
        "pair_count": 100,
        "frozen_video_spec": {
            "width": 1280,
            "height": 704,
            "fps_numerator": 24,
            "fps_denominator": 1,
            "nb_frames": 121,
            "full_decode_backends": ["imageio-ffmpeg-full-decode", "ffmpeg-full-decode"],
        },
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ContractError("generation receipt denominator/video contract mismatch")
    if (
        contract.get("output_root") != str(source_root.resolve(strict=True))
        or contract.get("manifest_sha256") != manifest_sha256
        or contract.get("case_count") != 100
        or contract.get("seed") != 0
        or contract.get("steps") != 50
        or contract.get("frames") != 121
        or contract.get("height") != 704
        or contract.get("width") != 1280
        or contract.get("fps") != 24
        or contract.get("guidance_scale") != 5.0
        or generation_repo.get("commit") != EXPECTED_GENERATION_COMMIT
        or generation_repo.get("clean") is not True
        or generation_repo.get("runner_sha256") != EXPECTED_RUNNER_SHA256
        or reviewed.get("controller_sha256") != EXPECTED_CONTROLLER_SHA256
        or reviewed.get("tests_sha256") != EXPECTED_CONTROLLER_TESTS_SHA256
        or approval.get("sha256") != EXPECTED_APPROVAL_SHA256
        or model.get("identity_sha256") != EXPECTED_MODEL_IDENTITY_SHA256
        or lora.get("checkpoint_receipt_sha256") != EXPECTED_LORA_RECEIPT_SHA256
        or receipt.get("lora_weight_sha256") != EXPECTED_LORA_WEIGHT_SHA256
    ):
        raise ContractError("generation receipt exact identity mismatch")
    tasks = receipt.get("tasks")
    pairs = receipt.get("pairs")
    if not isinstance(tasks, list) or len(tasks) != 200 or not isinstance(pairs, list) or len(pairs) != 100:
        raise ContractError("generation receipt lacks exact task/pair closure")
    expected_case_ids = [case_id for case_id, _ in cases]
    for index, task in enumerate(tasks):
        case_index, mode_index = divmod(index, 2)
        if (
            not isinstance(task, dict)
            or task.get("task_index") != index
            or task.get("case_index") != case_index
            or task.get("case_id") != expected_case_ids[case_index]
            or task.get("mode") != MODES[mode_index]
            or not isinstance(task.get("run_id"), str)
            or not task["run_id"]
            or not is_sha256(task.get("metadata_sha256"))
            or not is_sha256(task.get("video_sha256"))
            or not is_sha256(task.get("pair_identity_sha256"))
            or task.get("stored_video_probe") != EXPECTED_STORED_VIDEO_PROBE
            or task.get("independent_video_probe")
            != {**EXPECTED_STORED_VIDEO_PROBE, "backend": "ffmpeg-full-decode"}
        ):
            raise ContractError(f"generation task receipt mismatch at index {index}")
    for index, pair in enumerate(pairs):
        base, adapted = tasks[index * 2 : index * 2 + 2]
        if (
            not isinstance(pair, dict)
            or pair.get("case_index") != index
            or pair.get("case_id") != expected_case_ids[index]
            or pair.get("pair_identity_sha256") != base["pair_identity_sha256"]
            or pair.get("pair_identity_sha256") != adapted["pair_identity_sha256"]
            or pair.get("base_video_sha256") != base["video_sha256"]
            or pair.get("adapted_video_sha256") != adapted["video_sha256"]
        ):
            raise ContractError(f"generation pair receipt mismatch at case {index}")
    return tasks, pairs


def derive_protocol_and_schedule(
    protocol_template: dict[str, Any],
    schedule_template: dict[str, Any],
    manifest_sha256: str,
    derivation_tool_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if protocol_template.get("schema") != PARENT_PROTOCOL_SCHEMA or protocol_template.get("status") != "frozen":
        raise ContractError("metric protocol template is not frozen v2")
    if schedule_template.get("schema") != PARENT_SCHEDULE_SCHEMA or schedule_template.get("status") != "frozen":
        raise ContractError("metric schedule template is not frozen v2")
    protocol = copy.deepcopy(protocol_template)
    protocol["schema"] = TRAINDEV_PROTOCOL_SCHEMA
    protocol["scope"] = "train_dev_evaluation"
    protocol["dataset"] = {
        "name": "DL3DV-1K independent train-dev",
        "split": "dev",
        "case_count": 100,
        "source_manifest_sha256": manifest_sha256,
        "reserved_ids_disclosed": False,
    }
    protocol["candidate_budget_policy"] = {
        BASE_METHOD_ID: {
            "candidate_count": 1,
            "selected_output_count": 1,
            "comparison_note": "paired seed-0 base generation on independent train-dev",
        },
        ADAPTED_METHOD_ID: {
            "candidate_count": 1,
            "selected_output_count": 1,
            "comparison_note": "paired seed-0 step-64 LoRA-DPO generation on independent train-dev",
        },
        "reporting_requirement": (
            "This is a development-only post-training gate. Base and adapted use one "
            "paired generation each per scene."
        ),
    }
    protocol["execution_environment"] = {
        "generation": "Godot",
        "train_dev_scoring": "Hippasus",
        "required_shared_bindings": copy.deepcopy(
            protocol_template["execution_environment"]["required_shared_bindings"]
        ),
        "scoring_policy": (
            "The paired base and adapted train-dev videos are copied to one SHA-verified, "
            "read-only Hippasus mirror and scored by the same locked evaluator."
        ),
    }
    protocol["prohibitions"] = [
        "No metric, eligibility, threshold, pair, frame index, model, preprocessing, or aggregation change may be selected from this train-dev evaluation.",
        "Train-dev results may select whether to continue post-training but are not a held-out benchmark claim.",
        "No evaluator may download or replace model weights at runtime.",
        "Base and adapted must use one shared locked Hippasus evaluator identity and the exact paired 100-case denominator.",
    ]
    protocol["derivation"] = {
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "parent_schedule_sha256": PARENT_SCHEDULE_SHA256,
        "derivation_tool_sha256": require_sha256(derivation_tool_sha256, "derivation tool SHA"),
        "approved_changed_json_pointers": [
            "/candidate_budget_policy",
            "/dataset",
            "/derivation",
            "/execution_environment",
            "/prohibitions",
            "/schema",
            "/scope",
        ],
        "preserved_parent_subtrees_sha256": {
            pointer: object_sha256(protocol_template[key])
            for pointer, key in (
                ("/metrics", "metrics"),
                ("/reference_evaluator_identities", "reference_evaluator_identities"),
                ("/video_contract", "video_contract"),
                ("/weight_source_requirements", "weight_source_requirements"),
            )
        },
    }
    schedule = copy.deepcopy(schedule_template)
    if schedule.get("video_contract") != protocol.get("video_contract"):
        raise ContractError("schedule/protocol template video contracts differ")
    schedule["schema"] = TRAINDEV_SCHEDULE_SCHEMA
    schedule["metric_protocol_sha256"] = hashlib.sha256(canonical_bytes(protocol)).hexdigest()
    return protocol, schedule


def verify_derived_contract(
    protocol_template: dict[str, Any],
    schedule_template: dict[str, Any],
    protocol: dict[str, Any],
    schedule: dict[str, Any],
    manifest_sha256: str,
    derivation_tool_sha256: str,
) -> None:
    expected_protocol, expected_schedule = derive_protocol_and_schedule(
        protocol_template,
        schedule_template,
        manifest_sha256,
        derivation_tool_sha256,
    )
    if protocol != expected_protocol or schedule != expected_schedule:
        raise ContractError("derived protocol/schedule contains an unapproved change")


def seal_tree(root: Path) -> None:
    for directory, names, files in os.walk(root, topdown=False):
        current = Path(directory)
        for name in files:
            path = current / name
            if path.is_symlink() or not path.is_file():
                raise ContractError(f"bundle contains a non-regular file: {path}")
            os.chmod(path, 0o444)
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ContractError(f"bundle contains a non-regular file: {path}")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for name in names:
            path = current / name
            if path.is_symlink() or not path.is_dir():
                raise ContractError(f"bundle contains a non-regular directory: {path}")
        os.chmod(current, 0o555)
        descriptor = os.open(
            current,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise ContractError(f"bundle contains a non-regular directory: {current}")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def validate_mirror_closure(
    staging: Path,
    output_root: Path,
    records: list[dict[str, Any]],
) -> None:
    if len(records) != 800:
        raise ContractError("mirror index must contain exactly 800 artifact records")
    expected: dict[Path, tuple[str, int]] = {}
    for record in records:
        raw_path = record.get("path")
        sha = record.get("sha256")
        size = record.get("bytes")
        if not isinstance(raw_path, str) or not is_sha256(sha) or not isinstance(size, int) or size < 0:
            raise ContractError("mirror index record is malformed")
        path = Path(raw_path)
        try:
            relative = path.relative_to(output_root)
        except ValueError as error:
            raise ContractError("mirror index path escapes the final bundle root") from error
        if not relative.parts or relative.parts[0] != "mirror" or relative in expected:
            raise ContractError("mirror index path is duplicate or outside the mirror closure")
        expected[relative] = (sha, size)

    actual: set[Path] = set()
    for path in (staging / "mirror").rglob("*"):
        relative = path.relative_to(staging)
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            raise ContractError("mirror closure contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError("mirror closure contains a non-regular artifact")
        actual.add(relative)
    if actual != set(expected):
        raise ContractError("physical mirror closure differs from the exact 800-record index")
    for relative, (sha, size) in expected.items():
        verify_regular_copy(staging / relative, sha, size, f"mirror closure {relative}")


def preserve_failed_staging(staging: Path) -> None:
    """Retain a failed build attempt as read-only audit evidence."""
    if not staging.exists():
        return
    try:
        seal_tree(staging)
    except BaseException:
        for directory, names, files in os.walk(staging, topdown=False):
            current = Path(directory)
            for name in files:
                try:
                    os.chmod(current / name, 0o444, follow_symlinks=False)
                except OSError:
                    pass
            for name in names:
                try:
                    os.chmod(current / name, 0o555, follow_symlinks=False)
                except OSError:
                    pass
        try:
            os.chmod(staging, 0o555)
        except OSError:
            pass


def build_bundle(
    *,
    generation_receipt_path: Path,
    source_root: Path,
    manifest_path: Path,
    formal_manifest_path: Path,
    protocol_template_path: Path,
    schedule_template_path: Path,
    ffmpeg_path: Path,
    expected_ffmpeg_sha256: str,
    expected_generation_receipt_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    for path, label in (
        (generation_receipt_path, "generation receipt"),
        (manifest_path, "train-dev manifest"),
        (formal_manifest_path, "formal validation manifest"),
        (protocol_template_path, "protocol template"),
        (schedule_template_path, "schedule template"),
        (ffmpeg_path, "destination decoder"),
    ):
        require_no_symlink_ancestors(path, label)
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"{label} must be a regular file")
    require_no_symlink_ancestors(source_root, "generation source root")
    if source_root.is_symlink() or not source_root.is_dir():
        raise ContractError("generation source root must be a regular directory")
    require_no_symlink_ancestors(output_root.parent, "bundle output parent")
    if output_root.exists() or output_root.is_symlink() or output_root.parent.is_symlink() or not output_root.parent.is_dir():
        raise FileExistsError(f"fresh bundle output required: {output_root}")

    manifest_bytes, _ = read_regular_no_follow(manifest_path, "train-dev manifest")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ContractError("train-dev manifest SHA differs from the frozen 100-scene dev set")
    manifest = read_json_bytes(manifest_bytes, "train-dev manifest")
    cases = validate_manifest(manifest)

    receipt_bytes, _ = read_regular_no_follow(generation_receipt_path, "generation receipt")
    receipt_sha = verify_external_payload_sha(
        receipt_bytes,
        expected_generation_receipt_sha256,
        "generation receipt",
    )
    receipt = read_json_bytes(receipt_bytes, "generation receipt")
    tasks, pairs = validate_generation_receipt(receipt, source_root, manifest_sha, cases)

    formal_bytes, _ = read_regular_no_follow(formal_manifest_path, "formal validation manifest")
    formal_sha = hashlib.sha256(formal_bytes).hexdigest()
    if formal_sha != EXPECTED_FORMAL_MANIFEST_SHA256:
        raise ContractError("formal validation manifest SHA differs from its frozen identity")
    formal_manifest = read_json_bytes(formal_bytes, "formal validation manifest")
    if formal_manifest.get("_meta", {}).get("source_manifest_sha256") != EXPECTED_FORMAL_SOURCE_MANIFEST_SHA256:
        raise ContractError("formal validation source-manifest identity differs")
    isolation_receipt = build_isolation_receipt(manifest, formal_manifest, formal_sha)

    protocol_bytes, _ = read_regular_no_follow(protocol_template_path, "protocol template")
    schedule_bytes, _ = read_regular_no_follow(schedule_template_path, "schedule template")
    protocol_template = read_json_bytes(protocol_bytes, "protocol template")
    schedule_template = read_json_bytes(schedule_bytes, "schedule template")
    if hashlib.sha256(protocol_bytes).hexdigest() != PARENT_PROTOCOL_SHA256 or hashlib.sha256(schedule_bytes).hexdigest() != PARENT_SCHEDULE_SHA256:
        raise ContractError("metric protocol/schedule templates differ from their frozen parent SHAs")
    derivation_tool_sha = sha256_file(Path(__file__).resolve(strict=True))
    expected_ffmpeg_sha256 = require_sha256(
        expected_ffmpeg_sha256, "expected destination decoder SHA"
    )
    protocol, schedule = derive_protocol_and_schedule(
        protocol_template, schedule_template, manifest_sha, derivation_tool_sha
    )
    verify_derived_contract(
        protocol_template,
        schedule_template,
        protocol,
        schedule,
        manifest_sha,
        derivation_tool_sha,
    )

    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    output_parent_fd = open_absolute_directory_no_follow(
        output_root.parent, "bundle output parent"
    )
    staging_fd: int | None = None
    try:
        staging_fd = os.open(
            staging.name,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=output_parent_fd,
        )
        named_staging = os.stat(
            staging.name, dir_fd=output_parent_fd, follow_symlinks=False
        )
        pinned_staging = os.fstat(staging_fd)
        if (named_staging.st_dev, named_staging.st_ino) != (
            pinned_staging.st_dev,
            pinned_staging.st_ino,
        ):
            raise ContractError("bundle staging directory was replaced after creation")
    except BaseException:
        if staging_fd is not None:
            os.close(staging_fd)
        os.close(output_parent_fd)
        preserve_failed_staging(staging)
        raise
    try:
        mirror_root = staging / "mirror"
        mirror_root.mkdir(mode=0o700)
        entries_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
        mirror_records: list[dict[str, Any]] = []
        for index, task in enumerate(tasks):
            case_index = index // 2
            mode = task["mode"]
            case_id, manifest_case = cases[case_index]
            video_relative = relative_source_path(task.get("video"), source_root, f"task {index} video")
            metadata_relative = relative_source_path(task.get("metadata"), source_root, f"task {index} metadata")
            if video_relative.parent != metadata_relative.parent or video_relative.name != "video.mp4" or metadata_relative.name != "metadata.json":
                raise ContractError(f"task {index} source run closure is malformed")
            run_id = task["run_id"]
            if video_relative.parent.name != f"run_{run_id}":
                raise ContractError(f"task {index} source run directory differs from its run ID")
            run_fd = open_directory_below(source_root, video_relative.parent, f"task {index} run directory")
            lock_binding = task.get("generation_lock")
            if (
                not isinstance(lock_binding, dict)
                or set(lock_binding) != {"path", "sha256"}
                or relative_source_path(lock_binding.get("path"), source_root, f"task {index} generation lock")
                != video_relative.parent / ".generation.lock"
            ):
                os.close(run_fd)
                raise ContractError(f"task {index} generation lock binding is malformed")
            generation_lock_fd, generation_lock_identity = acquire_generation_lock_at(
                run_fd, lock_binding["sha256"], f"task {index} generation lock"
            )
            try:
                metadata_bytes, _ = read_regular_at(run_fd, "metadata.json", f"task {index} metadata")
                metadata_sha = hashlib.sha256(metadata_bytes).hexdigest()
                metadata = read_json_bytes(metadata_bytes, f"task {index} metadata")
                complete_bytes, _ = read_regular_at(run_fd, "COMPLETE", f"task {index} COMPLETE")
                complete_sha = hashlib.sha256(complete_bytes).hexdigest()
                if complete_bytes != f"{run_id}\n".encode("utf-8"):
                    raise ContractError(f"task {index} COMPLETE marker mismatch")
                if metadata_sha != task["metadata_sha256"]:
                    raise ContractError(f"task {index} metadata SHA differs from final receipt")
                expected_image_sha = manifest_case["conditioning_image_provenance"]["image_sha256"]
                lora = metadata.get("lora_dpo")
                run_config = metadata.get("run_config")
                if (
                    metadata.get("case_id") != case_id
                    or metadata.get("case") != manifest_case
                    or metadata.get("manifest_sha256") != manifest_sha
                    or metadata.get("prompt") != manifest_case["text_prompt"]
                    or metadata.get("image_sha256") != expected_image_sha
                    or metadata.get("seed") != 0
                    or metadata.get("method") != "lora_dpo"
                    or metadata.get("code_identity") != {"commit": EXPECTED_GENERATION_COMMIT, "dirty": False}
                    or metadata.get("runner_sha256") != EXPECTED_RUNNER_SHA256
                    or metadata.get("generation") != EXPECTED_GENERATION_PROFILE
                    or metadata.get("video_probe") != EXPECTED_STORED_VIDEO_PROBE
                    or metadata.get("video_sha256") != task["video_sha256"]
                    or not isinstance(lora, dict)
                    or lora.get("mode") != mode
                    or lora.get("checkpoint_receipt_sha256") != EXPECTED_LORA_RECEIPT_SHA256
                    or lora.get("step") != 64
                    or not isinstance(run_config, dict)
                    or object_sha256(run_config) != metadata.get("run_config_sha256")
                    or object_sha256(normalized_pair_config(run_config)) != task["pair_identity_sha256"]
                ):
                    raise ContractError(f"task {index} metadata semantic identity mismatch")

                destination = mirror_root / mode / f"{case_index:03d}_{case_id}"
                destination.mkdir(mode=0o700, parents=True)
                video_sha, video_size = copy_regular_at(
                    run_fd, "video.mp4", destination / "video.mp4", task["video_sha256"], f"task {index} video"
                )
                copied_metadata_sha, metadata_size = copy_regular_at(
                    run_fd, "metadata.json", destination / "metadata.json", metadata_sha, f"task {index} metadata"
                )
                copied_complete_sha, complete_size = copy_regular_at(
                    run_fd, "COMPLETE", destination / "COMPLETE", complete_sha, f"task {index} COMPLETE"
                )
                copied_lock_sha, copied_lock_size = copy_regular_at(
                    run_fd,
                    ".generation.lock",
                    destination / ".generation.lock",
                    lock_binding["sha256"],
                    f"task {index} generation lock",
                )
                destination_decode = full_decode_video(
                    destination / "video.mp4",
                    video_sha,
                    video_size,
                    ffmpeg_path,
                    expected_ffmpeg_sha256,
                    f"task {index} destination video",
                )
                verify_regular_copy(
                    destination / "video.mp4",
                    video_sha,
                    video_size,
                    f"task {index} decoded destination video",
                )
                for source_name, copied_sha, copied_size in (
                    ("video.mp4", video_sha, video_size),
                    ("metadata.json", copied_metadata_sha, metadata_size),
                    ("COMPLETE", copied_complete_sha, complete_size),
                    (".generation.lock", copied_lock_sha, copied_lock_size),
                ):
                    rehash, resize = hash_regular_at(
                        run_fd, source_name, f"task {index} source {source_name} rehash"
                    )
                    if (rehash, resize) != (copied_sha, copied_size):
                        raise ContractError(f"task {index} source changed after copying {source_name}")
                verify_directory_below_identity(
                    source_root,
                    video_relative.parent,
                    run_fd,
                    f"task {index} source run directory",
                )
            finally:
                try:
                    release_generation_lock_at(
                        run_fd,
                        generation_lock_fd,
                        generation_lock_identity,
                        f"task {index} generation lock",
                    )
                finally:
                    os.close(run_fd)
            entry = {
                "case_id": case_id,
                "split_order": case_index,
                "seed": 0,
                "run_id": run_id,
                "metric_video_path": str(output_root / "mirror" / mode / destination.name / "video.mp4"),
                "metric_metadata_path": str(output_root / "mirror" / mode / destination.name / "metadata.json"),
                "metric_complete_path": str(output_root / "mirror" / mode / destination.name / "COMPLETE"),
                "metric_generation_lock_path": str(
                    output_root / "mirror" / mode / destination.name / ".generation.lock"
                ),
                "video_sha256": video_sha,
                "metadata_sha256": copied_metadata_sha,
                "complete_sha256": copied_complete_sha,
                "generation_lock_sha256": copied_lock_sha,
                "conditioning_image_sha256": expected_image_sha,
                "prompt": manifest_case["text_prompt"],
                "video_probe": EXPECTED_VIDEO_PROBE,
                "source_generation_receipt_sha256": receipt_sha,
                "pair_identity_sha256": task["pair_identity_sha256"],
                "lora_mode": mode,
                "lora_step": 64,
                "destination_full_decode": destination_decode,
            }
            entries_by_mode[mode].append(entry)
            mirror_records.extend(
                [
                    {"mode": mode, "case_id": case_id, "kind": "video", "path": entry["metric_video_path"], "sha256": video_sha, "bytes": video_size},
                    {"mode": mode, "case_id": case_id, "kind": "metadata", "path": entry["metric_metadata_path"], "sha256": copied_metadata_sha, "bytes": metadata_size},
                    {"mode": mode, "case_id": case_id, "kind": "COMPLETE", "path": entry["metric_complete_path"], "sha256": copied_complete_sha, "bytes": complete_size},
                    {"mode": mode, "case_id": case_id, "kind": "generation_lock", "path": entry["metric_generation_lock_path"], "sha256": copied_lock_sha, "bytes": copied_lock_size},
                ]
            )

        for case_index in range(100):
            base = entries_by_mode["base"][case_index]
            adapted = entries_by_mode["adapted"][case_index]
            if base["case_id"] != adapted["case_id"] or base["pair_identity_sha256"] != adapted["pair_identity_sha256"]:
                raise ContractError(f"copied pair identity mismatch at case {case_index}")
            if pairs[case_index]["pair_identity_sha256"] != base["pair_identity_sha256"]:
                raise ContractError(f"copied pair differs from final receipt at case {case_index}")

        validate_mirror_closure(staging, output_root, mirror_records)

        provenance = staging / "provenance"
        provenance.mkdir(mode=0o700)
        copied_receipt_sha = write_readonly(provenance / "PAIRED_GENERATION_RECEIPT.json", receipt_bytes)
        copied_manifest_sha = write_readonly(provenance / "dev100_manifest_960p_v2.json", manifest_bytes)
        isolation_sha = write_readonly(
            staging / "TRAINDEV_REFERENCE_ISOLATION_RECEIPT.json",
            canonical_bytes(isolation_receipt),
        )
        protocol_sha = write_readonly(staging / "five_metric_protocol_traindev_v1.json", canonical_bytes(protocol))
        if schedule["metric_protocol_sha256"] != protocol_sha:
            raise ContractError("derived schedule does not bind the written protocol")
        schedule_sha = write_readonly(staging / "metric_schedule_traindev_v1.json", canonical_bytes(schedule))
        mirror_index = {
            "schema": "wan-lora-dpo-traindev-evaluator-mirror-index-v1",
            "site": "Hippasus",
            "split": "dev",
            "case_count": 100,
            "method_count": 2,
            "record_count": len(mirror_records),
            "generation_receipt_sha256": copied_receipt_sha,
            "manifest_sha256": copied_manifest_sha,
            "records": sorted(mirror_records, key=lambda item: (item["mode"], item["case_id"], item["kind"])),
        }
        mirror_index_sha = write_readonly(staging / "MIRROR_INDEX.json", canonical_bytes(mirror_index))
        adapted_entries = {
            "schema": "wan-lora-dpo-traindev-adapted-input-entries-v1",
            "site": "Hippasus",
            "scope": "train_dev_evaluation",
            "split": "dev",
            "reserved_ids_disclosed": False,
            "method": ADAPTED_METHOD_LABEL,
            "method_id": ADAPTED_METHOD_ID,
            "case_count": 100,
            "source_manifest_sha256": manifest_sha,
            "traindev_reference_isolation_receipt_sha256": isolation_sha,
            "source_generation_receipt_sha256": receipt_sha,
            "metric_protocol_sha256": protocol_sha,
            "metric_schedule_sha256": schedule_sha,
            "mirror_root": str(output_root),
            "mirror_index_sha256": mirror_index_sha,
            "entries": entries_by_mode["adapted"],
        }
        adapted_entries_sha = write_readonly(
            staging / "ADAPTED_INPUT_ENTRIES.json", canonical_bytes(adapted_entries)
        )
        base_input_lock = {
            "schema": INPUT_LOCK_SCHEMA,
            "scope": "train_dev_evaluation",
            "evaluation_site": "Hippasus",
            "method": BASE_METHOD_LABEL,
            "method_id": BASE_METHOD_ID,
            "candidate_budget": protocol["candidate_budget_policy"][BASE_METHOD_ID],
            "dataset_split": "dev",
            "reserved_ids_disclosed": False,
            "source_manifest_sha256": manifest_sha,
            "traindev_reference_isolation_receipt_sha256": isolation_sha,
            "source_generation_receipt_sha256": receipt_sha,
            "metric_protocol_sha256": protocol_sha,
            "metric_schedule_sha256": schedule_sha,
            "video_contract": EXPECTED_VIDEO_PROBE,
            "mirror_root": str(output_root),
            "mirror_index_sha256": mirror_index_sha,
            "selection_policy": "paired base mode; one seed-0 video per frozen train-dev scene",
            "entries": entries_by_mode["base"],
        }
        base_lock_sha = write_readonly(staging / "BASE_INPUT_LOCK.json", canonical_bytes(base_input_lock))
        ready = {
            "schema": BUNDLE_SCHEMA,
            "status": "READY",
            "site": "Hippasus",
            "split": "dev",
            "reserved_ids_disclosed": False,
            "case_count": 100,
            "task_count": 200,
            "pair_count": 100,
            "mirror_file_count": len(mirror_records),
            "generation_receipt": {"path": str(output_root / "provenance" / "PAIRED_GENERATION_RECEIPT.json"), "sha256": copied_receipt_sha},
            "manifest": {"path": str(output_root / "provenance" / "dev100_manifest_960p_v2.json"), "sha256": copied_manifest_sha},
            "traindev_reference_isolation_receipt": {
                "path": str(output_root / "TRAINDEV_REFERENCE_ISOLATION_RECEIPT.json"),
                "sha256": isolation_sha,
            },
            "metric_protocol": {"path": str(output_root / "five_metric_protocol_traindev_v1.json"), "sha256": protocol_sha},
            "metric_schedule": {"path": str(output_root / "metric_schedule_traindev_v1.json"), "sha256": schedule_sha},
            "mirror_index": {"path": str(output_root / "MIRROR_INDEX.json"), "sha256": mirror_index_sha},
            "base_input_lock": {"path": str(output_root / "BASE_INPUT_LOCK.json"), "sha256": base_lock_sha},
            "adapted_input_entries": {
                "path": str(output_root / "ADAPTED_INPUT_ENTRIES.json"),
                "sha256": adapted_entries_sha,
            },
            "adapted_input_lock_status": "blocked_until_base_eligibility_lock",
            "destination_decoder": {
                "path": str(ffmpeg_path.absolute()),
                "sha256": expected_ffmpeg_sha256,
                "execution_path": "descriptor_pinned_proc_fd",
            },
        }
        ready_sha = write_readonly(staging / "BUNDLE_READY.json", canonical_bytes(ready))
        seal_tree(staging)
        named_staging = os.stat(
            staging.name, dir_fd=output_parent_fd, follow_symlinks=False
        )
        pinned_staging = os.fstat(staging_fd)
        if (named_staging.st_dev, named_staging.st_ino) != (
            pinned_staging.st_dev,
            pinned_staging.st_ino,
        ):
            raise ContractError("bundle staging directory changed before publication")
        rename_noreplace(staging, output_root, directory_fd=output_parent_fd)
        published = os.stat(
            output_root.name, dir_fd=output_parent_fd, follow_symlinks=False
        )
        if (published.st_dev, published.st_ino) != (
            pinned_staging.st_dev,
            pinned_staging.st_ino,
        ):
            raise ContractError("published bundle identity differs from the sealed staging inode")
        os.fsync(output_parent_fd)
        return {"output": str(output_root), "ready_sha256": ready_sha, "base_input_lock_sha256": base_lock_sha, "mirror_file_count": len(mirror_records)}
    except Exception:
        preserve_failed_staging(staging)
        raise
    finally:
        os.close(staging_fd)
        os.close(output_parent_fd)


def validate_cli_output(output_root: Path, derived_root: Path) -> None:
    if derived_root.is_symlink() or not derived_root.is_dir():
        raise ContractError("derived root must be a regular directory")
    derived = derived_root.resolve(strict=True)
    if tuple(derived.parts[-3:]) != ("outputs", "geometry-selection", "hippasus_evaluation"):
        raise ContractError("derived root is not the approved Hippasus evaluator root")
    target = output_root.parent.resolve(strict=True) / output_root.name
    if derived not in target.parents or target.relative_to(derived).parts[0] != "mirrors":
        raise ContractError("train-dev bundle must publish below the evaluator mirrors root")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-receipt", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--formal-validation-manifest", type=Path, required=True)
    parser.add_argument("--protocol-template", type=Path, required=True)
    parser.add_argument("--schedule-template", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--expected-ffmpeg-sha256", required=True)
    parser.add_argument("--expected-generation-receipt-sha256", required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    validate_cli_output(args.output_root, args.derived_root)
    result = build_bundle(
        generation_receipt_path=args.generation_receipt,
        source_root=args.source_root,
        manifest_path=args.manifest,
        formal_manifest_path=args.formal_validation_manifest,
        protocol_template_path=args.protocol_template,
        schedule_template_path=args.schedule_template,
        ffmpeg_path=args.ffmpeg,
        expected_ffmpeg_sha256=args.expected_ffmpeg_sha256,
        expected_generation_receipt_sha256=args.expected_generation_receipt_sha256,
        output_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
