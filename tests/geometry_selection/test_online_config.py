from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from geometry_selection.online import OnlineBranchConfig
from geometry_selection.online_config import load_online_selection_config


def _template() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "geometry_selection"
        / "static_online_pose_graph_pilot_v1.yaml"
    )


def test_online_config_template_is_strict_and_valid_for_wan_profile() -> None:
    config = load_online_selection_config(_template())
    config.validate(num_steps=50)
    assert config.branch.selection_steps == (32,)
    assert len(config.config_hash) == 64
    assert config.semantic_dict()["vggt_omega"] == {
        "image_resolution": 512,
        "preprocessing_mode": "balanced",
    }


def test_online_config_rejects_final_step_selection() -> None:
    config = load_online_selection_config(_template())
    invalid = replace(config, branch=OnlineBranchConfig((49,)))
    with pytest.raises(ValueError, match="final denoising step"):
        invalid.validate(num_steps=50)
