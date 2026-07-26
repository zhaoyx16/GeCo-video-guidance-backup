from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from frame_guidance_manifest import (
    FrameGuidanceManifestError,
    canonical_json_hash,
    load_frame_guidance_case,
)


class FrameGuidanceManifestTest(unittest.TestCase):
    def make_manifest(self, root: Path, anchors: list[dict]) -> Path:
        manifest = {
            "cases": {
                "scene": {
                    "text_prompt": "A camera moves through the same static corridor.",
                    "image_prompt": "frames/first.png",
                    "frame_guidance": {"anchors": anchors},
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


if __name__ == "__main__":
    unittest.main()
