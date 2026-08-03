from __future__ import annotations

import json

import pytest

from geometry_selection.protocol import file_sha256, validate_candidate_spec_against_protocol
from geometry_selection.selection import CANDIDATE_SPEC_SCHEMA


def test_candidate_spec_is_bound_to_protocol_split_prompt_and_image(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    scene = dataset / "scene-a"
    scene.mkdir(parents=True)
    image = scene / "frame.png"
    image.write_bytes(b"image")
    protocol = {
        "_meta": {"schema": "dl3dv-geometry-selection-v1"},
        "case-a": {
            "scene_uid": "dl3dv:scene-a",
            "split": "validation",
            "text_prompt": "camera moves forward",
            "dataset_relative_image": "scene-a/frame.png",
            "image_sha256": file_sha256(image),
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    spec = {
        "schema": CANDIDATE_SPEC_SCHEMA,
        "protocol_manifest_sha256": file_sha256(protocol_path),
        "split": "validation",
        "candidate_count": 2,
        "backbone": "wan",
        "generation": {"steps": 50},
        "cases": [
            {
                "case_id": "case-a",
                "scene_uid": "dl3dv:scene-a",
                "conditioning_image": str(image),
                "prompt": "camera moves forward",
                "candidates": [
                    {
                        "candidate_id": f"candidate-{index}",
                        "seed": index,
                        "video": str(video),
                        "is_incumbent": index == 0,
                    }
                    for index in range(2)
                ],
            }
        ],
    }
    validate_candidate_spec_against_protocol(spec, protocol_path, dataset)

    spec["split"] = "test"
    with pytest.raises(ValueError, match="differs from frozen protocol"):
        validate_candidate_spec_against_protocol(spec, protocol_path, dataset)
