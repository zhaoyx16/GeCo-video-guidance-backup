#!/usr/bin/env python3
"""Publish an external full-rehash closure for an existing sealed environment.

The environment itself is never copied, chmodded, or otherwise mutated.  The
root inode is held for both full scans and every descendant is opened relative
to a held directory descriptor.  A fresh external receipt is published only
when both scans are identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import tempfile
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "geometry-selection-sealed-tree-content-manifest-v1"
READY_SCHEMA = "geometry-selection-sealed-tree-ready-receipt-v1"
APPROVED_ENV_PARENT = Path("/vol/dissolve/yz10325/evaluator_envs")
RECEIPT_PARENT = Path("/vol/dissolve/yz10325/evaluator_receipts")
APPROVED_ROOT = Path("/vol/dissolve/yz10325")
APPROVED_ENVIRONMENT = APPROVED_ENV_PARENT / "evaluator_v13_tv023_r1_v1"
APPROVED_BASE_RECEIPT = (
    APPROVED_ROOT / "evaluator_staging/python_environment_v13_tv023_r1_attempt05.RECEIPT.json"
)
APPROVED_BASE_RECEIPT_SHA256 = "1347c451a8eac634d9105226e90897104fcc6aed3ceb20289d4bb407e7d8abf6"
APPROVED_BASE_RECEIPT_SIZE = 16979


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    for block in iter(lambda: os.read(descriptor, 8 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def mutable_by_current_user(info: os.stat_result) -> bool:
    groups = set(os.getgroups()) | {os.getegid()}
    return bool(
        (info.st_uid == os.geteuid() and info.st_mode & stat.S_IWUSR)
        or (info.st_gid in groups and info.st_mode & stat.S_IWGRP)
        or info.st_mode & stat.S_IWOTH
    )


def ensure_private_receipt_parent() -> tuple[Path, int, os.stat_result]:
    if RECEIPT_PARENT.parent.resolve(strict=True) != APPROVED_ROOT.resolve(strict=True):
        raise ValueError("receipt parent escapes the approved project root")
    try:
        os.mkdir(RECEIPT_PARENT, 0o700)
    except FileExistsError:
        pass
    descriptor = os.open(
        RECEIPT_PARENT,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    info = os.fstat(descriptor)
    named = os.stat(RECEIPT_PARENT, follow_symlinks=False)
    if (
        RECEIPT_PARENT.is_symlink()
        or not stat.S_ISDIR(info.st_mode)
        or stat_identity(info) != stat_identity(named)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o077
    ):
        os.close(descriptor)
        raise ValueError("receipt parent must be a private user-owned real directory")
    return RECEIPT_PARENT.resolve(strict=True), descriptor, info


def direct_existing_environment(path: Path) -> Path:
    parent = APPROVED_ENV_PARENT.resolve(strict=True)
    if (
        not path.is_absolute()
        or path.parent.resolve(strict=True) != parent
        or path != APPROVED_ENVIRONMENT
    ):
        raise ValueError("environment must be a direct approved evaluator_envs child")
    info = os.stat(path, follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISDIR(info.st_mode) or mutable_by_current_user(info):
        raise ValueError("environment root is not immutable to the evaluator account")
    return path.resolve(strict=True)


def direct_fresh_receipt(path: Path, parent: Path) -> Path:
    if not path.is_absolute() or path.parent.resolve(strict=True) != parent:
        raise ValueError("receipt root must be a direct evaluator_receipts child")
    if path.exists() or path.is_symlink():
        raise FileExistsError("receipt root must be fresh")
    return path


def same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def verify_exact_base_receipt(
    path: Path, expected_sha: str
) -> tuple[dict[str, Any], tuple[int, Path, os.stat_result, str]]:
    if path != APPROVED_BASE_RECEIPT or expected_sha != APPROVED_BASE_RECEIPT_SHA256:
        raise ValueError("base environment receipt is not the frozen approved input")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        before = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or not same_inode(before, named)
            or mutable_by_current_user(before)
            or before.st_size != APPROVED_BASE_RECEIPT_SIZE
        ):
            raise ValueError("base environment receipt binding differs")
        digest = sha256_descriptor(descriptor)
        after = os.fstat(descriptor)
        named_after = os.stat(path, follow_symlinks=False)
        identity = (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns
        )
        if (
            digest != APPROVED_BASE_RECEIPT_SHA256
            or identity != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
            )
            or not same_inode(after, named_after)
        ):
            raise RuntimeError("base environment receipt changed while held")
        binding = {
            "path": str(path.resolve(strict=True)),
            "sha256": APPROVED_BASE_RECEIPT_SHA256,
            "size": APPROVED_BASE_RECEIPT_SIZE,
        }
        return binding, (descriptor, path.resolve(strict=True), before, digest)
    except BaseException:
        os.close(descriptor)
        raise


def stat_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def require_held_file_identity(
    held: tuple[int, Path, os.stat_result, str]
) -> None:
    descriptor, path, before, expected_sha = held
    after = os.fstat(descriptor)
    named = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(after.st_mode)
        or path.is_symlink()
        or stat_identity(before) != stat_identity(after)
        or stat_identity(after) != stat_identity(named)
        or sha256_descriptor(descriptor) != expected_sha
    ):
        raise RuntimeError("base environment receipt changed while certification was publishing")


def scan_held_tree(root_descriptor: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for directory, directories, files, directory_fd in os.fwalk(
        ".", topdown=True, follow_symlinks=False, dir_fd=root_descriptor
    ):
        directories.sort()
        files.sort()
        directory_info = os.fstat(directory_fd)
        if not stat.S_ISDIR(directory_info.st_mode) or mutable_by_current_user(directory_info):
            raise ValueError(f"sealed environment contains a mutable directory: {directory}")
        relative_directory = Path(directory).as_posix()
        if relative_directory == ".":
            relative_directory = ""
        identities.append({"kind": "directory", "path": relative_directory, "stat": stat_identity(directory_info)})
        for name in directories:
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode) or mutable_by_current_user(info):
                raise ValueError(f"sealed environment contains a symlink/special/mutable directory: {directory}/{name}")
        for name in files:
            named_before_open = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(named_before_open.st_mode):
                raise ValueError(f"sealed environment contains an unsafe file: {directory}/{name}")
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory_fd,
            )
            try:
                before = os.fstat(descriptor)
                named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or not same_inode(before, named)
                    or mutable_by_current_user(before)
                    or before.st_nlink != 1
                ):
                    raise ValueError(f"sealed environment contains an unsafe file: {directory}/{name}")
                digest = sha256_descriptor(descriptor)
                after = os.fstat(descriptor)
                named_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                if identity != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
                ) or not same_inode(after, named_after):
                    raise RuntimeError(f"sealed environment file changed during rehash: {directory}/{name}")
                relative = (Path(directory) / name).as_posix()
                if relative.startswith("./"):
                    relative = relative[2:]
                records.append({"path": relative, "size": before.st_size, "sha256": digest})
                identities.append({"kind": "file", "path": relative, "stat": stat_identity(before), "sha256": digest})
            finally:
                os.close(descriptor)
    records.sort(key=lambda item: item["path"])
    if not records:
        raise ValueError("sealed environment is empty")
    identities.sort(key=lambda item: (item["path"], item["kind"]))
    return records, identities


def held_published_identity(
    held: tuple[int, Path, os.stat_result, str], parent_fd: int, name: str
) -> None:
    descriptor, path, before, expected_sha = held
    after = os.fstat(descriptor)
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        path.name != name
        or not stat.S_ISREG(after.st_mode)
        or stat_identity(before) != stat_identity(after)
        or stat_identity(after) != stat_identity(named)
        or sha256_descriptor(descriptor) != expected_sha
    ):
        raise RuntimeError(f"published receipt file changed while held: {name}")


def atomic_json_at(
    parent_fd: int,
    parent_path: Path,
    name: str,
    value: Any,
    commit_check=None,
) -> tuple[int, Path, os.stat_result, str]:
    temporary_name = f".{name}.{secrets.token_hex(16)}.tmp"
    descriptor = os.open(
        temporary_name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=parent_fd,
    )
    temporary_exists = True
    linked = False
    try:
        raw = canonical_bytes(value) + b"\n"
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        expected_sha = hashlib.sha256(raw).hexdigest()
        if commit_check is not None:
            commit_check()
        os.link(
            temporary_name, name,
            src_dir_fd=parent_fd, dst_dir_fd=parent_fd, follow_symlinks=False,
        )
        linked = True
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
            temporary_exists = False
            before = os.fstat(descriptor)
            if commit_check is not None:
                commit_check()
            held = (descriptor, parent_path / name, before, expected_sha)
            held_published_identity(held, parent_fd, name)
            os.fsync(parent_fd)
            return held
        except BaseException:
            if linked:
                try:
                    os.unlink(name, dir_fd=parent_fd)
                finally:
                    os.fsync(parent_fd)
            raise
    except BaseException:
        os.close(descriptor)
        raise
    finally:
        if temporary_exists:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--expected-base-receipt-sha256", required=True)
    parser.add_argument("--expected-self-sha256", required=True)
    args = parser.parse_args()

    self_path = Path(__file__).resolve(strict=True)
    if sha256_file(self_path) != require_sha256(args.expected_self_sha256, "self SHA"):
        raise ValueError("environment certifier differs from its reviewed SHA")
    environment = direct_existing_environment(args.environment_root)
    receipt_parent, receipt_parent_fd, initial_receipt_parent_identity = ensure_private_receipt_parent()
    receipt = direct_fresh_receipt(args.receipt_root, receipt_parent)
    base_receipt, held_base_receipt = verify_exact_base_receipt(
        args.base_receipt,
        require_sha256(args.expected_base_receipt_sha256, "base receipt SHA"),
    )
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(APPROVED_ENV_PARENT, flags)
    root_fd = -1
    try:
        root_fd = os.open(APPROVED_ENVIRONMENT.name, flags, dir_fd=parent_fd)
        held_parent = os.fstat(parent_fd)
        held_root = os.fstat(root_fd)
        named_parent = os.stat(APPROVED_ENV_PARENT, follow_symlinks=False)
        named_root = os.stat(environment, follow_symlinks=False)
        relative_root = os.stat(APPROVED_ENVIRONMENT.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(held_parent.st_mode)
            or not stat.S_ISDIR(held_root.st_mode)
            or stat_identity(held_parent) != stat_identity(named_parent)
            or stat_identity(held_root) != stat_identity(named_root)
            or stat_identity(held_root) != stat_identity(relative_root)
        ):
            raise RuntimeError("environment root changed before certification")

        def publication_check() -> None:
            current_parent = os.fstat(parent_fd)
            named_current_parent = os.stat(APPROVED_ENV_PARENT, follow_symlinks=False)
            current_root = os.fstat(root_fd)
            relative_current_root = os.stat(
                APPROVED_ENVIRONMENT.name, dir_fd=parent_fd, follow_symlinks=False
            )
            named_current_root = os.stat(environment, follow_symlinks=False)
            if (
                stat_identity(held_parent) != stat_identity(current_parent)
                or stat_identity(current_parent) != stat_identity(named_current_parent)
                or stat_identity(held_root) != stat_identity(current_root)
                or stat_identity(current_root) != stat_identity(relative_current_root)
                or stat_identity(current_root) != stat_identity(named_current_root)
            ):
                raise RuntimeError("approved environment pathname changed during certification")
            require_held_file_identity(held_base_receipt)

        def full_publication_check() -> None:
            publication_check()
            if scan_held_tree(root_fd) != (first, first_identities):
                raise RuntimeError("environment tree changed during receipt publication")

        first, first_identities = scan_held_tree(root_fd)
        second, second_identities = scan_held_tree(root_fd)
        publication_check()
        if first != second or first_identities != second_identities:
            raise RuntimeError("environment tree changed during certification")
        tree_sha = hashlib.sha256(canonical_bytes(first) + b"\n").hexdigest()
        current_receipt_parent = os.fstat(receipt_parent_fd)
        named_receipt_parent = os.stat(receipt_parent, follow_symlinks=False)
        if (
            stat_identity(initial_receipt_parent_identity) != stat_identity(current_receipt_parent)
            or stat_identity(current_receipt_parent) != stat_identity(named_receipt_parent)
        ):
            raise RuntimeError("receipt parent changed before publication")
        os.mkdir(receipt.name, 0o700, dir_fd=receipt_parent_fd)
        os.fsync(receipt_parent_fd)
        receipt_fd = os.open(receipt.name, flags, dir_fd=receipt_parent_fd)
        published_files: list[tuple[int, Path, os.stat_result, str]] = []
        parent_identity = os.fstat(receipt_parent_fd)
        root_initial = os.fstat(receipt_fd)
        sealed_root_identity: os.stat_result | None = None

        def receipt_path_check(*, sealed: bool = False) -> None:
            current_parent = os.fstat(receipt_parent_fd)
            named_parent = os.stat(receipt_parent, follow_symlinks=False)
            current_root = os.fstat(receipt_fd)
            relative_root = os.stat(receipt.name, dir_fd=receipt_parent_fd, follow_symlinks=False)
            named_root = os.stat(receipt, follow_symlinks=False)
            if (
                stat_identity(parent_identity) != stat_identity(current_parent)
                or stat_identity(current_parent) != stat_identity(named_parent)
                or (root_initial.st_dev, root_initial.st_ino)
                != (current_root.st_dev, current_root.st_ino)
                or (current_root.st_dev, current_root.st_ino)
                != (relative_root.st_dev, relative_root.st_ino)
                or (current_root.st_dev, current_root.st_ino)
                != (named_root.st_dev, named_root.st_ino)
                or (sealed and stat.S_IMODE(current_root.st_mode) != 0o555)
                or (sealed and sealed_root_identity is not None and (
                    stat_identity(sealed_root_identity) != stat_identity(current_root)
                    or stat_identity(current_root) != stat_identity(relative_root)
                    or stat_identity(current_root) != stat_identity(named_root)
                ))
            ):
                raise RuntimeError("receipt publication pathname changed")

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "kind": "environment",
            "root": str(environment),
            "files": first,
            "file_count": len(first),
            "tree_sha256": tree_sha,
            "certifier": {"path": str(self_path), "sha256": args.expected_self_sha256},
            "inputs": [{"label": "base_environment_receipt", **base_receipt}],
        }
        manifest_path = receipt / "CONTENT_MANIFEST.json"
        manifest_held = atomic_json_at(
            receipt_fd,
            receipt,
            manifest_path.name,
            manifest,
            commit_check=lambda: (publication_check(), receipt_path_check()),
        )
        published_files.append(manifest_held)
        third, third_identities = scan_held_tree(root_fd)
        publication_check()
        if third != first or third_identities != first_identities:
            raise RuntimeError("environment changed before READY publication")
        manifest_binding = {
            "path": str(manifest_path),
            "sha256": manifest_held[3],
        }
        ready = {
            "schema": READY_SCHEMA,
            "status": "full_rehash_verified",
            "root": str(environment),
            "content_manifest": manifest_binding,
            "file_count": len(first),
            "tree_sha256": tree_sha,
        }
        ready_path = receipt / "READY.json"

        def ready_commit_check() -> None:
            full_publication_check()
            receipt_path_check()
            held_published_identity(manifest_held, receipt_fd, manifest_path.name)

        ready_held = None
        try:
            ready_held = atomic_json_at(
                receipt_fd,
                receipt,
                ready_path.name,
                ready,
                commit_check=ready_commit_check,
            )
            published_files.append(ready_held)
            os.fchmod(receipt_fd, 0o555)
            os.fsync(receipt_fd)
            sealed_root_identity = os.fstat(receipt_fd)
            full_publication_check()
            receipt_path_check(sealed=True)
            held_published_identity(manifest_held, receipt_fd, manifest_path.name)
            held_published_identity(ready_held, receipt_fd, ready_path.name)
        except BaseException:
            try:
                os.fchmod(receipt_fd, 0o700)
                try:
                    os.unlink(ready_path.name, dir_fd=receipt_fd)
                except FileNotFoundError:
                    pass
                os.fsync(receipt_fd)
            finally:
                os.fchmod(receipt_fd, 0o555)
                os.fsync(receipt_fd)
            raise
        print(json.dumps({
            "root": str(environment),
            "content_manifest": manifest_binding,
            "ready": {"path": str(ready_path), "sha256": ready_held[3]},
            "file_count": len(first),
            "tree_sha256": tree_sha,
        }, sort_keys=True))
        for descriptor, _path, _before, _sha in published_files:
            os.close(descriptor)
        os.close(receipt_fd)
    finally:
        if root_fd >= 0:
            os.close(root_fd)
        os.close(parent_fd)
        os.close(held_base_receipt[0])
        os.close(receipt_parent_fd)


if __name__ == "__main__":
    main()
