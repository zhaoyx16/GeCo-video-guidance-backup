from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("build_manifest.py")
SPEC = importlib.util.spec_from_file_location("train_dev_build_manifest", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_split(path: Path, split: str, count: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("split", "split_order", "source_order", "hash", "batch", "duration"),
        )
        writer.writeheader()
        for index in range(count):
            writer.writerow(
                {
                    "split": split,
                    "split_order": index,
                    "source_order": index + 1,
                    "hash": f"{index + 1:064x}",
                    "batch": "1K",
                    "duration": "1.0",
                }
            )


def test_load_frozen_train_dev_assignments(tmp_path: Path) -> None:
    split = tmp_path / "dev.csv"
    _write_split(split, "dev", 100)
    assignments = MODULE.load_frozen_assignments(
        split, {"dev"}, MODULE.FORMAL_SPLIT_COUNTS
    )
    assert len(assignments) == 100
    assert [item.split_order for item in assignments] == list(range(100))
    assert {item.split for item in assignments} == {"dev"}


def test_train_dev_count_is_strict(tmp_path: Path) -> None:
    split = tmp_path / "dev.csv"
    _write_split(split, "dev", 99)
    with pytest.raises(ValueError):
        MODULE.load_frozen_assignments(split, {"dev"}, MODULE.FORMAL_SPLIT_COUNTS)
