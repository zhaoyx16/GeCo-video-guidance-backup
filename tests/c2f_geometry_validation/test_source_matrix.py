from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX_DIR = ROOT / "benchmarks" / "c2f_source_rerank_val100"


def test_val100_manifest_exactly_matches_frozen_validation_split() -> None:
    manifest = json.loads(
        (MATRIX_DIR / "wan_validation_100_layout_aware_v2_formal.json").read_text(
            encoding="utf-8"
        )
    )
    with (ROOT / "benchmarks" / "c2f_validation" / "frozen_scene_split_3_100_100.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        split_rows = list(csv.DictReader(handle))

    manifest_hashes = {
        record["scene_id"]
        for case_id, record in manifest.items()
        if not case_id.startswith("_")
    }
    split_hashes = {
        split: {row["hash"] for row in split_rows if row["split"] == split}
        for split in ("test", "validation", "debug")
    }

    assert len(manifest_hashes) == 100
    assert manifest_hashes == split_hashes["validation"]
    assert manifest_hashes.isdisjoint(split_hashes["test"])
    assert manifest_hashes.isdisjoint(split_hashes["debug"])


def test_frozen_source_matrix_has_exact_preregistered_crossing() -> None:
    matrix = json.loads((MATRIX_DIR / "SOURCE_MATRIX.json").read_text(encoding="utf-8"))
    assert matrix["main_count"] == 36
    assert matrix["uniform_control_count"] == 24
    assert len(matrix["records"]) == 60
    assert len({record["method_id"] for record in matrix["records"]}) == 60

    configs = [
        json.loads((MATRIX_DIR / record["config"]).read_text(encoding="utf-8"))
        for record in matrix["records"]
    ]
    assert Counter(config["factors"]["geometry_evidence"] for config in configs) == {
        "draft": 20,
        "online_snapshot": 20,
        "online_refresh": 20,
    }
    main = [config for config in configs if config["factors"]["intervention"] == "main"]
    controls = [config for config in configs if config["factors"]["intervention"] == "uniform_norm_control"]
    assert Counter(config["factors"]["retrieval_policy"] for config in main) == {"V": 12, "P": 12, "S": 12}
    assert Counter(config["factors"]["retrieval_policy"] for config in controls) == {"V": 12, "P": 12}
    assert all(config["seed"] == 0 for config in configs)
    assert all(config["generation"]["frames"] == 121 for config in configs)
    assert all(config["generation"]["height"] == 704 for config in configs)
    assert all(config["generation"]["width"] == 1280 for config in configs)
    assert all(config["method"]["attn_avg_start"] == 20 for config in configs)
    assert all(config["method"]["attn_avg_end"] == 29 for config in configs)
    assert all(
        config["evidence"].get("refresh_steps") == [22, 25, 28]
        for config in configs
        if config["factors"]["geometry_evidence"] == "online_refresh"
    )
