"""Independent classical epipolar-consistency diagnostics for navigation video.

The evaluator intentionally uses no VGGT, UFM, depth network, or generator
feature. SIFT correspondences are filtered with a ratio test and a fundamental
matrix is estimated with MAGSAC. Low-motion pairs abstain because epipolar
geometry is not identifiable there; their frequency is reported separately so
a method cannot receive a good geometry score by producing a static video.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


EVALUATOR_NAME = "opencv-sift-magsac-epipolar"
EVALUATOR_VERSION = "2"


@dataclass(frozen=True)
class EpipolarConfig:
    lags_sec: tuple[float, ...] = (0.5, 1.0)
    max_pairs_per_lag: int = 8
    max_lag_error_sec: float = 0.001
    max_side: int = 960
    sift_features: int = 4096
    ratio_threshold: float = 0.75
    min_matches: int = 24
    min_motion_ratio: float = 0.002
    ransac_threshold_px: float = 1.0
    ransac_confidence: float = 0.999
    ransac_max_iterations: int = 10_000
    heldout_fraction: float = 0.3
    heldout_inlier_threshold_px: float = 1.5
    capped_error_px: float = 5.0
    homography_inlier_threshold_px: float = 2.0
    homography_dominance_margin: float = 0.05
    rng_seed: int = 0

    def validate(self) -> None:
        if not self.lags_sec or any(value <= 0 for value in self.lags_sec):
            raise ValueError("lags_sec must contain positive values")
        if self.max_pairs_per_lag < 1:
            raise ValueError("max_pairs_per_lag must be positive")
        if self.max_lag_error_sec < 0:
            raise ValueError("max_lag_error_sec must be non-negative")
        if self.max_side < 64:
            raise ValueError("max_side must be at least 64")
        if self.sift_features < 32:
            raise ValueError("sift_features must be at least 32")
        if not 0 < self.ratio_threshold < 1:
            raise ValueError("ratio_threshold must be in (0, 1)")
        if self.min_matches < 16:
            raise ValueError("min_matches must be at least 16 for fit/heldout evaluation")
        if self.min_motion_ratio < 0:
            raise ValueError("min_motion_ratio must be non-negative")
        if self.ransac_threshold_px <= 0:
            raise ValueError("ransac_threshold_px must be positive")
        if not 0 < self.ransac_confidence < 1:
            raise ValueError("ransac_confidence must be in (0, 1)")
        if not 0.1 <= self.heldout_fraction <= 0.5:
            raise ValueError("heldout_fraction must be in [0.1, 0.5]")
        if self.heldout_inlier_threshold_px <= 0:
            raise ValueError("heldout_inlier_threshold_px must be positive")
        if self.capped_error_px <= 0:
            raise ValueError("capped_error_px must be positive")
        if self.homography_inlier_threshold_px <= 0:
            raise ValueError("homography_inlier_threshold_px must be positive")
        if not 0 <= self.homography_dominance_margin <= 1:
            raise ValueError("homography_dominance_margin must be in [0, 1]")


def parse_video_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("video specification must be LABEL=/absolute/path/video.mp4")
    label, path = value.split("=", 1)
    if not label:
        raise ValueError("video label must not be empty")
    if not path:
        raise ValueError("video path must not be empty")
    return label, Path(path)


def parse_float_list(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise ValueError("expected positive comma-separated values")
    return result


def read_video(path: str | Path) -> tuple[list[np.ndarray], float]:
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"video has no decodable frames: {path}")
    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"video has invalid FPS {fps}: {path}")
    return frames, fps


def evenly_spaced_starts(first: int, last: int, gap: int, max_pairs: int) -> list[int]:
    latest_start = last - gap
    if first < 0 or last < first:
        raise ValueError("invalid frame interval")
    if gap < 1 or max_pairs < 1:
        raise ValueError("gap and max_pairs must be positive")
    if latest_start < first:
        return []
    available = latest_start - first + 1
    count = min(max_pairs, available)
    return sorted(set(int(value) for value in np.linspace(first, latest_start, count)))


def sampson_errors_px(
    fundamental: np.ndarray, points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    """Return first-order point-to-epipolar-line error in pixel units."""

    fundamental = np.asarray(fundamental, dtype=np.float64)
    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)
    if fundamental.shape != (3, 3):
        raise ValueError("fundamental matrix must be 3x3")
    if points_a.shape != points_b.shape or points_a.ndim != 2 or points_a.shape[1] != 2:
        raise ValueError("point arrays must both have shape [N,2]")
    ones = np.ones((points_a.shape[0], 1), dtype=np.float64)
    homogeneous_a = np.concatenate([points_a, ones], axis=1)
    homogeneous_b = np.concatenate([points_b, ones], axis=1)
    lines_b = (fundamental @ homogeneous_a.T).T
    lines_a = (fundamental.T @ homogeneous_b.T).T
    residual = np.sum(homogeneous_b * lines_b, axis=1)
    denominator = (
        lines_b[:, 0] ** 2
        + lines_b[:, 1] ** 2
        + lines_a[:, 0] ** 2
        + lines_a[:, 1] ** 2
    )
    squared = residual**2 / np.maximum(denominator, 1e-12)
    return np.sqrt(squared)


def symmetric_homography_errors_px(
    homography: np.ndarray, points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    homography = np.asarray(homography, dtype=np.float64)
    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)
    if homography.shape != (3, 3):
        raise ValueError("homography must be 3x3")
    if points_a.shape != points_b.shape or points_a.ndim != 2 or points_a.shape[1] != 2:
        raise ValueError("point arrays must both have shape [N,2]")
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return np.full(points_a.shape[0], np.inf)
    projected_b = _project_homography(homography, points_a)
    projected_a = _project_homography(inverse, points_b)
    forward = np.linalg.norm(projected_b - points_b, axis=1)
    backward = np.linalg.norm(projected_a - points_a, axis=1)
    return 0.5 * (forward + backward)


def evaluate_frame_pair(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    *,
    config: EpipolarConfig,
) -> dict[str, Any]:
    config.validate()
    gray_a = _prepare_gray(frame_a, config.max_side)
    gray_b = _prepare_gray(frame_b, config.max_side)
    if gray_a.shape != gray_b.shape:
        raise ValueError("paired frames must have the same aspect ratio")

    cv2.setRNGSeed(config.rng_seed)
    sift = cv2.SIFT_create(nfeatures=config.sift_features)
    keypoints_a, descriptors_a = sift.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = sift.detectAndCompute(gray_b, None)
    common = {
        "keypoints_a": len(keypoints_a),
        "keypoints_b": len(keypoints_b),
        "resized_height": int(gray_a.shape[0]),
        "resized_width": int(gray_a.shape[1]),
    }
    if descriptors_a is None or descriptors_b is None:
        return {
            **common,
            "status": "insufficient_features",
            "ratio_matches_forward": 0,
            "reciprocal_matches": 0,
        }

    matches, forward_count = _reciprocal_ratio_matches(
        descriptors_a, descriptors_b, config.ratio_threshold
    )
    common["ratio_matches_forward"] = forward_count
    common["reciprocal_matches"] = len(matches)
    if len(matches) < config.min_matches:
        return {**common, "status": "insufficient_matches"}

    points_a = np.float32([keypoints_a[match.queryIdx].pt for match in matches])
    points_b = np.float32([keypoints_b[match.trainIdx].pt for match in matches])
    diagonal = math.hypot(gray_a.shape[0], gray_a.shape[1])
    displacement = np.linalg.norm(points_b - points_a, axis=1)
    motion_ratio = float(np.median(displacement) / diagonal)
    common["match_motion_ratio_median"] = motion_ratio
    common["match_displacement_px_median"] = float(np.median(displacement))
    if motion_ratio < config.min_motion_ratio:
        return {**common, "status": "low_motion_degenerate"}

    fit_indices, heldout_indices = _fit_heldout_indices(
        len(matches), config.heldout_fraction, config.rng_seed
    )
    fit_a, fit_b = points_a[fit_indices], points_b[fit_indices]
    heldout_a, heldout_b = points_a[heldout_indices], points_b[heldout_indices]
    fundamental, mask = cv2.findFundamentalMat(
        fit_a,
        fit_b,
        cv2.USAC_MAGSAC,
        config.ransac_threshold_px,
        config.ransac_confidence,
        config.ransac_max_iterations,
    )
    homography, _ = cv2.findHomography(
        fit_a,
        fit_b,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=config.homography_inlier_threshold_px,
        maxIters=config.ransac_max_iterations,
        confidence=config.ransac_confidence,
    )
    common["fit_matches"] = int(len(fit_indices))
    common["heldout_matches"] = int(len(heldout_indices))

    fundamental_valid = (
        fundamental is not None
        and mask is not None
        and np.asarray(fundamental).shape == (3, 3)
        and np.all(np.isfinite(np.asarray(fundamental)))
        and int(np.asarray(mask).reshape(-1).astype(bool).sum()) >= 8
    )
    f_errors = (
        sampson_errors_px(fundamental, heldout_a, heldout_b)
        if fundamental_valid
        else np.full(len(heldout_indices), np.inf)
    )
    f_errors = np.nan_to_num(
        f_errors,
        nan=config.capped_error_px,
        posinf=config.capped_error_px,
        neginf=config.capped_error_px,
    )
    f_inlier_ratio = float(
        np.mean(f_errors <= config.heldout_inlier_threshold_px)
    )
    homography_valid = (
        homography is not None
        and np.asarray(homography).shape == (3, 3)
        and np.all(np.isfinite(np.asarray(homography)))
    )
    h_errors = (
        symmetric_homography_errors_px(homography, heldout_a, heldout_b)
        if homography_valid
        else np.full(len(heldout_indices), np.inf)
    )
    h_errors = np.nan_to_num(h_errors, nan=np.inf, posinf=np.inf, neginf=np.inf)
    h_inlier_ratio = float(
        np.mean(h_errors <= config.homography_inlier_threshold_px)
    )
    common["heldout_f_inlier_ratio"] = f_inlier_ratio
    common["heldout_h_inlier_ratio"] = h_inlier_ratio
    common["homography_error_px_median"] = _finite_median_or_none(h_errors)

    if h_inlier_ratio >= 0.8 and (
        not fundamental_valid
        or h_inlier_ratio >= f_inlier_ratio - config.homography_dominance_margin
    ):
        return {**common, "status": "homography_degenerate"}
    if not fundamental_valid:
        return {
            **common,
            "status": "model_failure",
            "failure_aware_capped_sampson_px_mean": config.capped_error_px,
        }

    clipped_errors = np.minimum(f_errors, config.capped_error_px)
    fit_inlier_count = int(np.asarray(mask).reshape(-1).astype(bool).sum())
    return {
        **common,
        "status": "ok",
        "fit_inliers": fit_inlier_count,
        "fit_inlier_ratio": float(fit_inlier_count / len(fit_indices)),
        "heldout_sampson_error_px_median": float(np.median(f_errors)),
        "heldout_capped_sampson_px_mean": float(np.mean(clipped_errors)),
        "heldout_capped_sampson_diagonal_ratio_mean": float(
            np.mean(clipped_errors) / diagonal
        ),
        "failure_aware_capped_sampson_px_mean": float(np.mean(clipped_errors)),
    }


def evaluate_frames(
    label: str,
    frames: Sequence[np.ndarray],
    fps: float,
    *,
    config: EpipolarConfig,
    start_frame: int = 0,
    end_frame: int | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    config.validate()
    if not frames:
        raise ValueError("frames must not be empty")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    first = max(0, start_frame)
    last = len(frames) - 1 if end_frame is None else min(end_frame, len(frames) - 1)
    if first > last:
        raise ValueError(f"empty frame range [{first}, {last}] for {label}")

    lag_reports: dict[str, Any] = {}
    all_pairs: list[Mapping[str, Any]] = []
    for lag_sec in config.lags_sec:
        gap = max(1, int(round(lag_sec * fps)))
        actual_lag_sec = gap / fps
        lag_error_sec = abs(actual_lag_sec - lag_sec)
        if lag_error_sec > config.max_lag_error_sec:
            raise ValueError(
                f"requested lag={lag_sec:g}s is represented by {actual_lag_sec:g}s "
                f"at {fps:g} FPS; error {lag_error_sec:g}s exceeds "
                f"max_lag_error_sec={config.max_lag_error_sec:g}"
            )
        starts = evenly_spaced_starts(first, last, gap, config.max_pairs_per_lag)
        if not starts:
            raise ValueError(
                f"range [{first}, {last}] is too short for lag={lag_sec:g}s ({gap} frames)"
            )
        pairs = []
        for start in starts:
            end = start + gap
            result = evaluate_frame_pair(frames[start], frames[end], config=config)
            result.update({"start_frame": start, "end_frame": end})
            pairs.append(result)
        lag_reports[f"{lag_sec:g}"] = {
            "lag_sec": lag_sec,
            "actual_lag_sec": actual_lag_sec,
            "lag_error_sec": lag_error_sec,
            "gap_frames": gap,
            "summary": summarize_pair_results(pairs),
            "pairs": pairs,
        }
        all_pairs.extend(pairs)

    return {
        "label": label,
        "source": source,
        "fps": fps,
        "num_frames": len(frames),
        "evaluated_range": [first, last],
        "overall": summarize_pair_results(all_pairs),
        "lags": lag_reports,
    }


def evaluate_video(
    label: str,
    path: str | Path,
    *,
    config: EpipolarConfig,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict[str, Any]:
    frames, fps = read_video(path)
    return evaluate_frames(
        label,
        frames,
        fps,
        config=config,
        start_frame=start_frame,
        end_frame=end_frame,
        source=str(Path(path).resolve()),
    )


def summarize_pair_results(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for pair in pairs:
        status = str(pair["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    total = len(pairs)
    feature_detectable = [
        pair for pair in pairs if pair["status"] != "insufficient_features"
    ]
    matchable = [
        pair
        for pair in pairs
        if pair["status"] not in {"insufficient_features", "insufficient_matches"}
    ]
    observable = [pair for pair in pairs if pair["status"] in {"ok", "model_failure"}]
    successful = [pair for pair in pairs if pair["status"] == "ok"]
    return {
        "pair_count": total,
        "status_counts": status_counts,
        "feature_detectable_fraction": (
            len(feature_detectable) / total if total else None
        ),
        "match_coverage_fraction": len(matchable) / total if total else None,
        "parallax_observable_fraction": len(observable) / total if total else None,
        "low_motion_degenerate_fraction": (
            status_counts.get("low_motion_degenerate", 0) / total if total else None
        ),
        "homography_degenerate_fraction": (
            status_counts.get("homography_degenerate", 0) / total if total else None
        ),
        "geometry_model_success_fraction": (
            len(successful) / len(observable) if observable else None
        ),
        "heldout_f_inlier_ratio_mean": _mean_or_none(
            successful, "heldout_f_inlier_ratio"
        ),
        "heldout_f_inlier_ratio_median": _median_or_none(
            successful, "heldout_f_inlier_ratio"
        ),
        "heldout_capped_sampson_px_mean": _mean_or_none(
            successful, "heldout_capped_sampson_px_mean"
        ),
        "heldout_capped_sampson_diagonal_ratio_mean": _mean_or_none(
            successful, "heldout_capped_sampson_diagonal_ratio_mean"
        ),
        "failure_aware_capped_sampson_px_mean": _mean_or_none(
            observable, "failure_aware_capped_sampson_px_mean"
        ),
        "match_motion_ratio_median": _median_or_none(
            [
                pair
                for pair in pairs
                if pair.get("match_motion_ratio_median") is not None
            ],
            "match_motion_ratio_median",
        ),
    }


def build_report(
    videos: Sequence[tuple[str, str | Path]],
    *,
    config: EpipolarConfig,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict[str, Any]:
    return {
        "evaluator": {
            "name": EVALUATOR_NAME,
            "version": EVALUATOR_VERSION,
            "independence_policy": "independent",
            "uses": ["OpenCV SIFT", "USAC_MAGSAC fundamental matrix"],
            "does_not_use": ["VGGT", "UFM", "generator features"],
        },
        "config": asdict(config),
        "videos": [
            evaluate_video(
                label,
                path,
                config=config,
                start_frame=start_frame,
                end_frame=end_frame,
            )
            for label, path in videos
        ],
    }


def write_report(path: str | Path, report: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _prepare_gray(frame: np.ndarray, max_side: int) -> np.ndarray:
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        raise ValueError("frame must be HxW or HxWxC")
    height, width = gray.shape
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(gray)


def _reciprocal_ratio_matches(
    descriptors_a: np.ndarray, descriptors_b: np.ndarray, ratio_threshold: float
) -> tuple[list[cv2.DMatch], int]:
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = _ratio_matches(matcher.knnMatch(descriptors_a, descriptors_b, k=2), ratio_threshold)
    reverse = _ratio_matches(matcher.knnMatch(descriptors_b, descriptors_a, k=2), ratio_threshold)
    reverse_lookup = {
        match.queryIdx: match.trainIdx
        for match in reverse
    }
    reciprocal = [
        match
        for match in forward
        if reverse_lookup.get(match.trainIdx) == match.queryIdx
    ]
    reciprocal.sort(key=lambda match: (match.queryIdx, match.trainIdx))
    return reciprocal, len(forward)


def _ratio_matches(
    candidates: Sequence[Sequence[cv2.DMatch]], ratio_threshold: float
) -> list[cv2.DMatch]:
    return [
        best
        for candidate in candidates
        if len(candidate) == 2
        for best, second in [candidate]
        if best.distance < ratio_threshold * second.distance
    ]


def _fit_heldout_indices(
    count: int, heldout_fraction: float, rng_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    heldout_count = max(8, int(round(count * heldout_fraction)))
    heldout_count = min(heldout_count, count - 8)
    if heldout_count < 8:
        raise ValueError("at least 16 reciprocal matches are required for fit/heldout")
    generator = np.random.default_rng(rng_seed)
    permutation = generator.permutation(count)
    heldout = np.sort(permutation[:heldout_count])
    fit = np.sort(permutation[heldout_count:])
    return fit, heldout


def _project_homography(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate([points, ones], axis=1)
    projected = (homography @ homogeneous.T).T
    denominator = projected[:, 2:3]
    safe = np.where(np.abs(denominator) > 1e-12, denominator, np.nan)
    return projected[:, :2] / safe


def _finite_median_or_none(values: np.ndarray) -> float | None:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else None


def _mean_or_none(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return float(np.mean(values)) if values else None


def _median_or_none(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return float(np.median(values)) if values else None
