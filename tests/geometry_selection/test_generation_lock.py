from __future__ import annotations

import json

import pytest

from geometry_selection.generation_lock import GenerationRunLock


def test_generation_lock_rejects_concurrent_owner_and_records_slurm(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    first = GenerationRunLock(tmp_path)
    payload = json.loads((tmp_path / "RUNNING").read_text())
    assert payload["slurm_job_id"] == "123"

    with pytest.raises(RuntimeError, match="Another job owns"):
        GenerationRunLock(tmp_path)

    first.release()
    assert not (tmp_path / "RUNNING").exists()


def test_generation_lock_recovers_after_previous_process_releases_kernel_lock(tmp_path):
    first = GenerationRunLock(tmp_path)
    first.release()
    (tmp_path / "RUNNING").write_text("stale status from killed process")

    second = GenerationRunLock(tmp_path)
    assert json.loads((tmp_path / "RUNNING").read_text())["pid"] > 0
    second.release()
    assert (tmp_path / ".generation.lock").is_file()


def test_generation_lock_release_is_idempotent(tmp_path):
    lock = GenerationRunLock(tmp_path)
    lock.release()
    lock.release()
