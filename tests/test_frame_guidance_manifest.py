from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from frame_guidance_manifest import (
    FrameGuidanceManifestError,
    build_frame_guidance_time_contract,
    canonical_json_hash,
    load_frame_guidance_case,
)


class FrameGuidanceManifestTest(unittest.TestCase):
    def make_manifest(
        self,
        root: Path,
        anchors: list[dict],
        frame_guidance_metadata: dict | None = None,
    ) -> Path:
        frame_guidance = {"anchors": anchors}
        if frame_guidance_metadata:
            frame_guidance.update(frame_guidance_metadata)
        manifest = {
            "cases": {
                "scene": {
                    "text_prompt": "A camera moves through the same static corridor.",
                    "image_prompt": "frames/first.png",
                    "frame_guidance": frame_guidance,
                }
            }
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def create_frames(self, root: Path) -> None:
        frames = root / "frames"
        frames.mkdir()
        for name, value in [("first.png", b"first"), ("middle.png", b"middle"), ("last.png", b"last")]:
            (frames / name).write_bytes(value)

    def test_loads_relative_first_middle_last_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_frames(root)
            manifest = self.make_manifest(
                root,
                [
                    {"frame_index": 0, "image_path": "frames/first.png"},
                    {"frame_index": 60, "image_path": "frames/middle.png"},
                    {"frame_index": 120, "image_path": "frames/last.png"},
                ],
            )
            case = load_frame_guidance_case(manifest, "scene")
            self.assertEqual([anchor["frame_index"] for anchor in case["anchors"]], [0, 60, 120])
            self.assertTrue(Path(case["condition_image_path"]).is_absolute())
            self.assertEqual(case["condition_image_sha256"], case["anchors"][0]["sha256"])

    def test_rejects_missing_frame_zero_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_frames(root)
            manifest = self.make_manifest(
                root,
                [
                    {"frame_index": 20, "image_path": "frames/middle.png"},
                    {"frame_index": 60, "image_path": "frames/middle.png"},
                    {"frame_index": 120, "image_path": "frames/last.png"},
                ],
            )
            with self.assertRaisesRegex(FrameGuidanceManifestError, "frame_index=0"):
                load_frame_guidance_case(manifest, "scene")

    def test_rejects_condition_anchor_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_frames(root)
            manifest = self.make_manifest(
                root,
                [
                    {"frame_index": 0, "image_path": "frames/middle.png"},
                    {"frame_index": 60, "image_path": "frames/middle.png"},
                    {"frame_index": 120, "image_path": "frames/last.png"},
                ],
            )
            with self.assertRaisesRegex(FrameGuidanceManifestError, "conditioning image"):
                load_frame_guidance_case(manifest, "scene")

    def test_canonical_hash_is_key_order_invariant(self) -> None:
        self.assertEqual(canonical_json_hash({"a": 1, "b": 2}), canonical_json_hash({"b": 2, "a": 1}))

    def test_builds_source_and_generated_timestamp_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_frames(root)
            manifest = self.make_manifest(
                root,
                [
                    {
                        "frame_index": 0,
                        "image_path": "frames/first.png",
                        "source_frame_index": 0,
                        "source_timestamp_seconds": 0.0,
                    },
                    {
                        "frame_index": 60,
                        "image_path": "frames/middle.png",
                        "source_frame_index": 25,
                        "source_timestamp_seconds": 2.5,
                    },
                    {
                        "frame_index": 120,
                        "image_path": "frames/last.png",
                        "source_frame_index": 50,
                        "source_timestamp_seconds": 5.0,
                    },
                ],
                {
                    "source_fps": 10,
                    "generated_fps": 24,
                    "source_clip_start_frame": 0,
                    "timestamp_tolerance_seconds": 0.01,
                },
            )
            contract = build_frame_guidance_time_contract(load_frame_guidance_case(manifest, "scene"), 24)
            self.assertEqual(contract["source_fps"], 10.0)
            self.assertEqual(contract["generated_fps"], 24.0)
            self.assertEqual(contract["anchors"][1]["generated_timestamp_seconds"], 2.5)

    def test_rejects_anchor_timestamp_that_does_not_bind_generated_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_frames(root)
            manifest = self.make_manifest(
                root,
                [
                    {
                        "frame_index": 0,
                        "image_path": "frames/first.png",
                        "source_frame_index": 0,
                        "source_timestamp_seconds": 0.0,
                    },
                    {
                        "frame_index": 60,
                        "image_path": "frames/middle.png",
                        "source_frame_index": 25,
                        "source_timestamp_seconds": 2.0,
                    },
                    {
                        "frame_index": 120,
                        "image_path": "frames/last.png",
                        "source_frame_index": 50,
                        "source_timestamp_seconds": 5.0,
                    },
                ],
                {
                    "source_fps": 10,
                    "generated_fps": 24,
                    "source_clip_start_frame": 0,
                    "timestamp_tolerance_seconds": 0.01,
                },
            )
            case = load_frame_guidance_case(manifest, "scene")
            with self.assertRaisesRegex(FrameGuidanceManifestError, "source timestamp"):
                build_frame_guidance_time_contract(case, 24)

    def test_rejects_runner_fps_that_differs_from_manifest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_frames(root)
            manifest = self.make_manifest(
                root,
                [
                    {
                        "frame_index": 0,
                        "image_path": "frames/first.png",
                        "source_frame_index": 0,
                        "source_timestamp_seconds": 0.0,
                    },
                    {
                        "frame_index": 60,
                        "image_path": "frames/middle.png",
                        "source_frame_index": 25,
                        "source_timestamp_seconds": 2.5,
                    },
                    {
                        "frame_index": 120,
                        "image_path": "frames/last.png",
                        "source_frame_index": 50,
                        "source_timestamp_seconds": 5.0,
                    },
                ],
                {
                    "source_fps": 10,
                    "generated_fps": 24,
                    "source_clip_start_frame": 0,
                    "timestamp_tolerance_seconds": 0.01,
                },
            )
            case = load_frame_guidance_case(manifest, "scene")
            with self.assertRaisesRegex(FrameGuidanceManifestError, "must equal the runner --fps"):
                build_frame_guidance_time_contract(case, 16)


if __name__ == "__main__":
    unittest.main()
