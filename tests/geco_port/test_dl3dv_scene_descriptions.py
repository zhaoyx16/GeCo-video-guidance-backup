from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "dl3dv_geco"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "prepare_scene_descriptions.py"
SPEC = importlib.util.spec_from_file_location("prepare_scene_descriptions", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_caption_is_reduced_to_content_only() -> None:
    caption = (
        "The video features a serene urban park with trees, hedges, and a stone wall. "
        "Throughout the video, the camera moves toward a playground."
    )
    assert MODULE.clean_caption(caption) == (
        "the same serene urban park with trees, hedges, and a stone wall"
    )


def test_motion_in_first_sentence_is_rejected() -> None:
    with pytest.raises(ValueError, match="motion/temporal"):
        MODULE.clean_caption("The video features a corridor while the camera moves forward.")


def test_caption_keys_are_normalized_to_scene_hash() -> None:
    scene_id = "a" * 64
    indexed = MODULE.caption_index({f"1K/{scene_id}/images_8": "A corridor."})
    assert indexed == {scene_id: "A corridor."}


def test_caption_temporal_scaffolding_is_removed() -> None:
    caption = "The video showcases a shop, starting with a view of shelves and products."
    assert MODULE.clean_caption(caption) == "the same shop, with shelves and products"


def test_promotional_does_not_false_match_motion() -> None:
    assert MODULE.clean_caption(
        "The video features a retail store with promotional banners and shelves."
    ) == "the same retail store with promotional banners and shelves"


@pytest.mark.parametrize(
    "instruction",
    ["turn left", "move forward", "pan right", "zoom in", "walk through", "orbit left"],
)
def test_description_rejects_base_form_camera_instructions(instruction: str) -> None:
    with pytest.raises(ValueError, match="motion/temporal"):
        MODULE.clean_caption(f"The video features a corridor and {instruction}.")
