#!/usr/bin/env python3
"""Small geometry tests for the DL3DV manifest motion labels."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/dl3dv_geco/build_manifest.py"
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


if __name__ == "__main__":
    unittest.main()
