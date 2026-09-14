#!/usr/bin/env python3
"""Freeze a motion-stratified 25-case development subset without using outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile_positions(size: int, count: int) -> list[int]:
    if size < count:
        raise ValueError(f"cannot select {count} unique positions from {size} records")
    if count == 1:
        return [0]
    positions = [round(index * (size - 1) / (count - 1)) for index in range(count)]
    if len(set(positions)) != count:
        raise RuntimeError(f"quantile rule produced duplicate positions: {positions}")
    return positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_path = Path(config["source_manifest"])
    actual_source_sha = sha256_file(source_path)
    if actual_source_sha != config["source_manifest_sha256"]:
        raise RuntimeError(
            f"source manifest digest mismatch: {actual_source_sha} != "
            f"{config['source_manifest_sha256']}"
        )

    source = json.loads(source_path.read_text(encoding="utf-8"))
    records = [(case_id, case) for case_id, case in source.items() if not case_id.startswith("_")]
    candidate_scene_ids = {case["scene_id"] for _, case in records}

    reserved_path = Path(config["reserved_split_csv"])
    if not reserved_path.is_absolute():
        reserved_path = config_path.parents[2] / reserved_path
    actual_reserved_sha = sha256_file(reserved_path)
    if actual_reserved_sha != config["reserved_split_sha256"]:
        raise RuntimeError(
            f"reserved split digest mismatch: {actual_reserved_sha} != "
            f"{config['reserved_split_sha256']}"
        )
    with reserved_path.open(newline="", encoding="utf-8") as handle:
        reserved_rows = list(csv.DictReader(handle))
    reserved_groups = {
        split: {row["hash"] for row in reserved_rows if row["split"] == split}
        for split in config["required_zero_overlap_splits"]
    }
    for index, first in enumerate(config["required_zero_overlap_splits"]):
        for second in config["required_zero_overlap_splits"][index + 1 :]:
            overlap = reserved_groups[first] & reserved_groups[second]
            if overlap:
                raise RuntimeError(f"reserved splits {first}/{second} overlap: {sorted(overlap)}")
    candidate_overlap = {
        split: sorted(candidate_scene_ids & reserved_ids)
        for split, reserved_ids in reserved_groups.items()
    }
    if any(candidate_overlap.values()):
        raise RuntimeError(f"development candidates overlap reserved scenes: {candidate_overlap}")

    selection = config["selection"]
    strata = selection["strata"]
    cases_per_stratum = int(selection["cases_per_stratum"])
    selected: list[tuple[str, dict, str, int, int]] = []

    for stratum in strata:
        group = [(case_id, case) for case_id, case in records if case["motion_instruction"] == stratum]
        group.sort(
            key=lambda item: (-float(item[1]["pose_stats"]["selection_score"]), item[0])
        )
        positions = quantile_positions(len(group), cases_per_stratum)
        for rank, position in enumerate(positions):
            case_id, case = group[position]
            selected.append((case_id, case, stratum, rank, position))

    selected.sort(key=lambda item: (strata.index(item[2]), item[3], item[0]))
    output: dict[str, object] = {
        "_meta": {
            "schema": "wan-c2f-dev25-manifest-v1",
            "source_manifest": str(source_path),
            "source_manifest_sha256": actual_source_sha,
            "selection_config": str(config_path),
            "selection_config_sha256": sha256_file(config_path),
            "reserved_split_csv": str(reserved_path),
            "reserved_split_sha256": actual_reserved_sha,
            "reserved_split_counts": {key: len(value) for key, value in reserved_groups.items()},
            "reserved_overlap_counts": {key: len(value) for key, value in candidate_overlap.items()},
            "selection_uses_generated_outputs": False,
            "selection_uses_metrics": False,
            "strata": strata,
            "cases_per_stratum": cases_per_stratum,
            "case_count": len(selected),
        }
    }
    for selection_order, (case_id, case, stratum, rank, source_position) in enumerate(selected):
        copied = dict(case)
        copied["c2f_dev_selection"] = {
            "selection_order": selection_order,
            "stratum": stratum,
            "stratum_quantile_rank": rank,
            "stratum_sorted_position": source_position,
        }
        output[case_id] = copied

    if args.output.exists():
        raise FileExistsError(f"refusing to replace existing frozen manifest: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"wrote {len(selected)} cases: {args.output}")
    for case_id, case, stratum, rank, _ in selected:
        pose = case["pose_stats"]
        print(
            f"{stratum:14s} q{rank} path={pose['path_length']:.2f} "
            f"rot={pose['rotation_deg']:.1f} {case_id}"
        )


if __name__ == "__main__":
    main()
