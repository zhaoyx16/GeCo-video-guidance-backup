#!/usr/bin/env python3
"""Small geometry tests for the DL3DV manifest motion labels."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "dl3dv_geco"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "build_manifest.py"
SPEC = importlib.util.spec_from_file_location("_dl3dv_manifest", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def yaw_matrix(degrees: float) -> np.ndarray:
    theta = math.radians(degrees)
    return np.asarray(
        [
            [math.cos(theta), 0.0, math.sin(theta)],
            [0.0, 1.0, 0.0],
            [-math.sin(theta), 0.0, math.cos(theta)],
        ]
    )


class MotionLabelTest(unittest.TestCase):
    def test_positive_c2w_yaw_is_left(self) -> None:
        yaw = MODULE.signed_yaw_deg(yaw_matrix(30.0))
        self.assertEqual(MODULE.classify_motion(2.0, 0.0, yaw, 30.0), "forward_left")

    def test_negative_c2w_yaw_is_right(self) -> None:
        yaw = MODULE.signed_yaw_deg(yaw_matrix(-30.0))
        self.assertEqual(MODULE.classify_motion(2.0, 0.0, yaw, 30.0), "forward_right")

    def test_backward_clip_is_rejected(self) -> None:
        self.assertIsNone(MODULE.classify_motion(-2.0, 0.1, 0.0, 0.0))

    def test_scene_candidates_uses_c2w_forward_and_yaw_conventions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / ("a" * 64)
            images = scene / "images_8"
            images.mkdir(parents=True)
            frames = []
            for index, yaw in enumerate((0.0, 15.0, 30.0)):
                pose = np.eye(4)
                pose[:3, :3] = yaw_matrix(yaw)
                pose[2, 3] = -float(index)
                name = f"frame_{index + 1:05d}.png"
                Image.new("RGB", (8, 6)).save(images / name)
                frames.append(
                    {"file_path": f"images/{name}", "transform_matrix": pose.tolist()}
                )
            transforms = scene / "transforms.json"
            transforms.write_text(json.dumps({"frames": frames}))
            candidates = MODULE.scene_candidates(transforms, 3, 1, "images_8")
            self.assertEqual(len(candidates), 1)
            self.assertGreater(candidates[0].forward, 0.0)
            self.assertEqual(candidates[0].motion_class, "forward_left")
            self.assertGreaterEqual(candidates[0].turn_consistency, 0.99)

    def test_all_prompts_request_large_motion_without_freezing_camera(self) -> None:
        for motion_class in MODULE.PROMPTS:
            prompt = MODULE.render_prompt(motion_class)
            MODULE.audit_prompt(prompt)
            lowered = prompt.lower()
            for phrase in MODULE.FORBIDDEN_PROMPT_PHRASES:
                self.assertNotIn(phrase, lowered)

    def test_scene_description_does_not_replace_gt_motion_instruction(self) -> None:
        prompt = MODULE.render_prompt("forward_left", "the same indoor corridor")
        self.assertIn("same indoor corridor", prompt)
        self.assertIn("leftward curve", prompt)
        self.assertIn("strong parallax", prompt)

    def test_best_candidate_is_selected_for_every_frozen_scene(self) -> None:
        def candidate(scene_id: str, score: float):
            return MODULE.Candidate(
                scene_id=scene_id,
                transforms_path=Path("transforms.json"),
                image_path=Path("frame.png"),
                start_index=0,
                end_index=80,
                motion_class="forward",
                path_length=1.0,
                displacement=1.0,
                rotation_deg=0.0,
                signed_yaw_deg=0.0,
                forward=1.0,
                lateral=0.0,
                straightness=1.0,
                normalized_path=1.0,
                normalized_displacement=1.0,
                turn_consistency=1.0,
                turn_monotonicity=1.0,
                turn_smoothness=1.0,
                turn_spread=1.0,
                score=score,
            )

        assignments = [
            MODULE.SplitAssignment("validation", 0, 101, "a"),
            MODULE.SplitAssignment("validation", 1, 102, "b"),
        ]
        selected = MODULE.select_best_per_scene(
            [candidate("a", 0.1), candidate("a", 0.9), candidate("b", 0.4)],
            assignments,
        )
        self.assertEqual([item.scene_id for item in selected], ["a", "b"])
        self.assertEqual([item.score for item in selected], [0.9, 0.4])

    def test_non_monotonic_yaw_does_not_claim_one_smooth_turn(self) -> None:
        label = MODULE.classify_motion(
            2.0,
            0.0,
            30.0,
            80.0,
            turn_consistency=0.3,
            turn_monotonicity=0.4,
        )
        self.assertEqual(label, "forward")

    def test_last_frame_yaw_jump_does_not_claim_smooth_curve(self) -> None:
        rotations = np.stack([np.eye(3)] * 80 + [yaw_matrix(30.0)])
        yaw, consistency, monotonicity, smoothness, spread = MODULE.yaw_path_metrics(rotations)
        self.assertGreaterEqual(consistency, 0.99)
        self.assertGreaterEqual(monotonicity, 0.99)
        self.assertLess(smoothness, 0.1)
        self.assertEqual(
            MODULE.classify_motion(
                2.0,
                0.0,
                yaw,
                30.0,
                consistency,
                monotonicity,
                smoothness,
                spread,
            ),
            "forward",
        )

    def test_two_last_frame_yaw_jumps_do_not_claim_smooth_curve(self) -> None:
        rotations = np.stack([np.eye(3)] * 79 + [yaw_matrix(15.0), yaw_matrix(30.0)])
        yaw, consistency, monotonicity, smoothness, spread = MODULE.yaw_path_metrics(rotations)
        self.assertGreaterEqual(smoothness, 0.49)
        self.assertLess(spread, 0.1)
        self.assertEqual(
            MODULE.classify_motion(
                2.0,
                0.0,
                yaw,
                30.0,
                consistency,
                monotonicity,
                smoothness,
                spread,
            ),
            "forward",
        )


if __name__ == "__main__":
    unittest.main()
