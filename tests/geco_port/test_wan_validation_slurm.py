from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "dl3dv_geco"
    / "slurm"
    / "wan_validation_candidates.slurm"
)


def _run(tmp_path: Path, task: int) -> subprocess.CompletedProcess:
    fake = tmp_path / "apptainer"
    fake.write_text("#!/bin/bash\nprintf '%s\\n' \"$@\"\n")
    fake.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "EXPECTED_COMMIT": "a" * 40,
        "SLURM_ARRAY_TASK_ID": str(task),
        "CANDIDATE_SEEDS": "9,09",
        "EXPECTED_CASES": "200",
    }
    return subprocess.run(
        ["bash", str(SCRIPT)],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )


def test_validation_array_maps_last_task_to_last_case_and_seed(tmp_path: Path) -> None:
    result = _run(tmp_path, 399)
    assert result.returncode == 0, result.stderr
    assert "CASE_INDEX=99" in result.stdout
    assert "CANDIDATE_SEED=3" in result.stdout


def test_validation_array_rejects_out_of_range_task(tmp_path: Path) -> None:
    result = _run(tmp_path, 400)
    assert result.returncode == 2
    assert "outside [0,399]" in result.stderr


def test_validation_array_has_fixed_bounded_array_and_concurrency() -> None:
    source = SCRIPT.read_text()
    assert "#SBATCH --array=0-399%16" in source
    assert "readonly SEEDS=(0 1 2 3)" in source
    assert "readonly EXPECTED_CASES=100" in source


def test_environment_cannot_override_frozen_seeds_or_case_count(tmp_path: Path) -> None:
    result = _run(tmp_path, 399)
    assert result.returncode == 0, result.stderr
    assert "CASE_INDEX=99" in result.stdout
    assert "CANDIDATE_SEED=3" in result.stdout


def test_validation_array_locks_inputs_and_official_wan_profile() -> None:
    source = SCRIPT.read_text()
    assert "da4c05c0ec8f6f8fd08daf3a69482c631d221f84fcb7a5d251c1e3fb1509dd9d" in source
    assert "71cd88a83a67ab59d02ec1dbf1b4f76344e3da7d290f161423b21aa18cdc7648" in source
    assert "validate_development_generation_inputs.py" in source
    for option in (
        "--steps 50",
        "--frames 121",
        "--height 704",
        "--width 1280",
        "--fps 24",
        "--guidance-scale 5.0",
    ):
        assert option in source
