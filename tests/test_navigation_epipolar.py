from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from navigation_benchmark.epipolar import (
    EpipolarConfig,
    evenly_spaced_starts,
    evaluate_frame_pair,
    evaluate_frames,
    parse_float_list,
    parse_video_spec,
    sampson_errors_px,
    summarize_pair_results,
)
from scripts.calibrate_epipolar_metric import preprocess_reference


class EpipolarHelpersTest(unittest.TestCase):
    def test_evenly_spaced_starts_include_interval_extremes(self) -> None:
        self.assertEqual(evenly_spaced_starts(0, 20, 5, 4), [0, 5, 10, 15])
        self.assertEqual(evenly_spaced_starts(3, 8, 5, 8), [3])
        self.assertEqual(evenly_spaced_starts(0, 3, 4, 2), [])

    def test_sampson_error_is_zero_for_exact_epipolar_points(self) -> None:
        fundamental = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        )
        points_a = np.array([[4.0, 2.0], [10.0, 7.0], [1.5, 20.0]])
        points_b = np.array([[9.0, 2.0], [3.0, 7.0], [8.5, 20.0]])
        np.testing.assert_allclose(
            sampson_errors_px(fundamental, points_a, points_b), 0.0, atol=1e-12
        )

    def test_identical_textured_pair_abstains_for_low_motion(self) -> None:
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 256, size=(256, 320, 3), dtype=np.uint8)
        result = evaluate_frame_pair(
            frame,
            frame.copy(),
            config=EpipolarConfig(max_side=320, min_matches=16),
        )
        self.assertEqual(result["status"], "low_motion_degenerate")
        self.assertEqual(result["match_motion_ratio_median"], 0.0)

    def test_blank_pair_reports_insufficient_features(self) -> None:
        frame = np.zeros((128, 192, 3), dtype=np.uint8)
        result = evaluate_frame_pair(frame, frame, config=EpipolarConfig())
        self.assertEqual(result["status"], "insufficient_features")

    def test_summary_does_not_treat_unobservable_pair_as_good_geometry(self) -> None:
        summary = summarize_pair_results(
            [
                {"status": "low_motion_degenerate", "match_motion_ratio_median": 0.0},
                {
                    "status": "ok",
                    "heldout_f_inlier_ratio": 0.8,
                    "heldout_capped_sampson_px_mean": 0.2,
                    "heldout_capped_sampson_diagonal_ratio_mean": 0.0002,
                    "failure_aware_capped_sampson_px_mean": 0.2,
                    "match_motion_ratio_median": 0.02,
                },
            ]
        )
        self.assertEqual(summary["parallax_observable_fraction"], 0.5)
        self.assertEqual(summary["geometry_model_success_fraction"], 1.0)
        self.assertAlmostEqual(summary["heldout_f_inlier_ratio_mean"], 0.8)

    def test_nonfinite_fundamental_is_counted_as_model_failure(self) -> None:
        rng = np.random.default_rng(23)
        frame = rng.integers(0, 256, size=(256, 320, 3), dtype=np.uint8)
        shifted = np.roll(frame, shift=8, axis=1)
        with patch(
            "navigation_benchmark.epipolar.cv2.findFundamentalMat",
            return_value=(
                np.full((3, 3), np.nan),
                np.ones((256, 1), dtype=np.uint8),
            ),
        ), patch(
            "navigation_benchmark.epipolar.cv2.findHomography",
            return_value=(None, None),
        ):
            result = evaluate_frame_pair(
                frame,
                shifted,
                config=EpipolarConfig(
                    max_side=320, min_matches=16, min_motion_ratio=0.001
                ),
            )
        self.assertEqual(result["status"], "model_failure")
        self.assertEqual(result["failure_aware_capped_sampson_px_mean"], 5.0)

    def test_cli_value_parsers(self) -> None:
        self.assertEqual(parse_float_list("0.25, 0.5,1"), (0.25, 0.5, 1.0))
        label, path = parse_video_spec("candidate=/tmp/candidate.mp4")
        self.assertEqual(label, "candidate")
        self.assertEqual(str(path), "/tmp/candidate.mp4")

    def test_reference_stretch_matches_requested_generator_size(self) -> None:
        frame = np.zeros((12, 30, 3), dtype=np.uint8)
        resized = preprocess_reference(
            [frame], width=20, height=16, mode="stretch"
        )
        self.assertEqual(resized[0].shape, (16, 20, 3))

    def test_inexact_cross_fps_lag_is_rejected(self) -> None:
        frames = [np.zeros((32, 48, 3), dtype=np.uint8) for _ in range(4)]
        with self.assertRaisesRegex(ValueError, "represented by 0.2s"):
            evaluate_frames(
                "ten-fps",
                frames,
                10.0,
                config=EpipolarConfig(
                    lags_sec=(0.25,),
                    max_lag_error_sec=0.001,
                    max_pairs_per_lag=1,
                ),
            )

    def test_planar_warp_is_marked_homography_degenerate(self) -> None:
        rng = np.random.default_rng(19)
        frame = rng.integers(0, 256, size=(256, 320, 3), dtype=np.uint8)
        homography = np.array(
            [[1.0, 0.02, 9.0], [0.01, 1.0, 5.0], [0.0001, 0.0002, 1.0]],
            dtype=np.float32,
        )
        warped = cv2.warpPerspective(frame, homography, (320, 256))
        result = evaluate_frame_pair(
            frame,
            warped,
            config=EpipolarConfig(
                max_side=320,
                min_matches=16,
                min_motion_ratio=0.001,
            ),
        )
        self.assertEqual(result["status"], "homography_degenerate")


if __name__ == "__main__":
    unittest.main()
