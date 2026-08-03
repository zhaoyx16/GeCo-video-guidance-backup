"""Crash-safe per-output lock for generation jobs."""

from __future__ import annotations

import fcntl
import json
import os
import socket
import time
from pathlib import Path


class GenerationRunLock:
    """Hold an advisory lock whose kernel ownership disappears on process death."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.output_dir / ".generation.lock"
        self.running_path = self.output_dir / "RUNNING"
        self._fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        self._active = False
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(self._fd)
            owner = self.running_path.read_text(errors="replace") if self.running_path.exists() else "unknown"
            raise RuntimeError(
                f"Another job owns this run directory: {self.output_dir}; owner={owner}"
            ) from error

        self._active = True
        payload = {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "acquired_unix_seconds": time.time(),
        }
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
        os.ftruncate(self._fd, 0)
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.write(self._fd, encoded)
        os.fsync(self._fd)
        self.running_path.write_bytes(encoded)

    def release(self) -> None:
        if not self._active:
            return
        self._active = False
        self.running_path.unlink(missing_ok=True)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)

