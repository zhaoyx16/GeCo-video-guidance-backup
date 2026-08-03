"""Adapter for the official VGGT-Omega inference release."""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from ..schema import GeometryPrediction


OFFICIAL_UPSTREAM_COMMIT = "39a0cb8af88554f15ddcb5354cd52bde588fa014"


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit(source_root: Path) -> str | None:
    marker = source_root / "UPSTREAM_COMMIT"
    if marker.is_file():
        value = marker.read_text(encoding="utf-8").strip()
        return value or None
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _source_tree_sha256(source_root: Path) -> str:
    records = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        records.append(
            {
                "path": str(path.relative_to(source_root)),
                "sha256": file_sha256(path),
            }
        )
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _homogeneous_world_to_camera(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics)
    if extrinsics.ndim != 3 or extrinsics.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(f"VGGT-Omega extrinsics must have shape [T,3,4] or [T,4,4], got {extrinsics.shape}")
    if extrinsics.shape[-2:] == (4, 4):
        result = extrinsics.copy()
    else:
        result = np.broadcast_to(np.eye(4, dtype=extrinsics.dtype), (extrinsics.shape[0], 4, 4)).copy()
        result[:, :3, :4] = extrinsics
    return result


def geometry_from_vggt_omega_outputs(
    *,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    depth: np.ndarray,
    confidence: np.ndarray,
    keyframe_indices: Sequence[int],
    metadata: dict,
) -> GeometryPrediction:
    """Normalize public VGGT-Omega outputs into the project convention."""

    depth = np.asarray(depth)
    confidence = np.asarray(confidence)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    indices = np.asarray(keyframe_indices)
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("keyframe_indices must be integral before normalization")
    if np.any(indices < 0):
        raise ValueError("keyframe_indices must be non-negative")
    prediction = GeometryPrediction(
        world_to_camera=_homogeneous_world_to_camera(np.asarray(extrinsics)),
        intrinsics=np.asarray(intrinsics),
        depth=depth,
        confidence=confidence,
        keyframe_indices=indices.astype(np.int64, copy=False),
        metadata=metadata,
    ).as_float32()
    prediction.validate()
    return prediction


class VGGTOmegaAdapter:
    """Load one pinned VGGT-Omega checkpoint and emit canonical geometry."""

    def __init__(
        self,
        *,
        source_root: Path,
        checkpoint: Path,
        device: str = "cuda",
        image_resolution: int = 512,
        preprocessing_mode: str = "balanced",
        require_official_commit: bool = True,
    ) -> None:
        self.source_root = source_root.resolve()
        self.checkpoint = checkpoint.resolve()
        self.device = device
        self.image_resolution = int(image_resolution)
        self.preprocessing_mode = preprocessing_mode
        self.require_official_commit = require_official_commit
        self._model = None
        self._api = None
        self._checkpoint_sha256: str | None = None

        if not (self.source_root / "vggt_omega").is_dir():
            raise FileNotFoundError(f"VGGT-Omega package not found under {self.source_root}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {self.checkpoint}")
        if self.image_resolution <= 0 or self.image_resolution % 16 != 0:
            raise ValueError("image_resolution must be a positive multiple of 16")
        if self.preprocessing_mode not in {"balanced", "max_size"}:
            raise ValueError("preprocessing_mode must be 'balanced' or 'max_size'")

        commit = _source_commit(self.source_root)
        if require_official_commit and commit != OFFICIAL_UPSTREAM_COMMIT:
            raise ValueError(
                "VGGT-Omega source commit mismatch: "
                f"expected {OFFICIAL_UPSTREAM_COMMIT}, got {commit!r}"
            )

    def identity(self, *, hash_checkpoint: bool = True) -> dict:
        checkpoint_stat = self.checkpoint.stat()
        if hash_checkpoint and self._checkpoint_sha256 is None:
            self._checkpoint_sha256 = file_sha256(self.checkpoint)
        return {
            "name": "VGGT-Omega-1B-512",
            "source_root": str(self.source_root),
            "source_commit": _source_commit(self.source_root),
            "source_tree_sha256": _source_tree_sha256(self.source_root),
            "checkpoint": str(self.checkpoint),
            "checkpoint_size": checkpoint_stat.st_size,
            "checkpoint_sha256": self._checkpoint_sha256 if hash_checkpoint else None,
            "image_resolution": self.image_resolution,
            "preprocessing_mode": self.preprocessing_mode,
            "camera_convention": "opencv_world_to_camera",
            "depth_definition": "camera_z_depth",
        }

    def artifact_identity(self) -> dict:
        """Content-based model identity that is stable when artifacts are relocated."""

        identity = self.identity(hash_checkpoint=True)
        identity.pop("source_root")
        identity.pop("checkpoint")
        return identity

    def cache_identity(self) -> dict:
        """Artifact plus numerical-runtime identity used for geometry cache keys."""

        import torch

        identity = self.artifact_identity()
        cuda_available = torch.cuda.is_available()
        identity["execution_environment"] = {
            "python": platform.python_version(),
            "platform_machine": platform.machine(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if cuda_available else None,
            "device": str(self.device),
            "device_name": torch.cuda.get_device_name(self.device) if cuda_available else None,
            "inference_dtype": "float32",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
        return identity

    def _import_api(self):
        if self._api is not None:
            return self._api
        source = str(self.source_root)
        if source not in sys.path:
            sys.path.insert(0, source)
        models = importlib.import_module("vggt_omega.models")
        load_fn = importlib.import_module("vggt_omega.utils.load_fn")
        pose_enc = importlib.import_module("vggt_omega.utils.pose_enc")
        self._api = (models.VGGTOmega, load_fn.load_and_preprocess_images, pose_enc.encoding_to_camera)
        return self._api

    def load(self) -> None:
        if self._model is not None:
            return
        import torch

        VGGTOmega, _, _ = self._import_api()
        model = VGGTOmega().eval()
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        if not isinstance(state, dict):
            raise TypeError(f"unexpected VGGT-Omega checkpoint payload: {type(state)!r}")
        incompatible = model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"checkpoint incompatibility: {incompatible}")
        self._model = model.to(self.device).eval()
        self._model.requires_grad_(False)

    def predict_image_paths(
        self,
        image_paths: Sequence[Path],
        *,
        keyframe_indices: Sequence[int] | None = None,
        hash_checkpoint: bool = True,
    ) -> GeometryPrediction:
        if len(image_paths) < 2:
            raise ValueError("at least two image paths are required")
        resolved_paths = [Path(path).resolve() for path in image_paths]
        missing = [str(path) for path in resolved_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing input frames: {missing}")
        if keyframe_indices is None:
            keyframe_indices = list(range(len(resolved_paths)))
        if len(keyframe_indices) != len(resolved_paths):
            raise ValueError("keyframe_indices must match image_paths")

        from PIL import Image

        input_sizes = []
        for path in resolved_paths:
            with Image.open(path) as image:
                input_sizes.append(image.size)
        if len(set(input_sizes)) != 1:
            raise ValueError(
                "candidate video frames must share one image size; mixed-size padding is unsupported"
            )

        import torch

        self.load()
        _, load_and_preprocess_images, encoding_to_camera = self._import_api()
        images = load_and_preprocess_images(
            [str(path) for path in resolved_paths],
            mode=self.preprocessing_mode,
            image_resolution=self.image_resolution,
        ).to(self.device)
        with torch.inference_mode():
            predictions = self._model(images)
        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
        )

        def numpy_batch(value):
            result = value.detach().float().cpu().numpy()
            if result.shape[0] != 1:
                raise ValueError(f"expected batch size 1, got {result.shape}")
            return result[0]

        input_records = [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in resolved_paths
        ]
        metadata = {
            "adapter_schema_version": 1,
            "backbone": self.identity(hash_checkpoint=hash_checkpoint),
            "inputs": input_records,
            "processed_image_size_hw": list(predictions["images"].shape[-2:]),
            "official_output_convention": {
                "extrinsics": "OpenCV camera-from-world",
                "depth": "camera-space z depth",
                "confidence": "raw expp1 confidence; not a probability",
            },
        }
        return geometry_from_vggt_omega_outputs(
            extrinsics=numpy_batch(extrinsics),
            intrinsics=numpy_batch(intrinsics),
            depth=numpy_batch(predictions["depth"]),
            confidence=numpy_batch(predictions["depth_conf"]),
            keyframe_indices=keyframe_indices,
            metadata=metadata,
        )

    def identity_json(self) -> str:
        return json.dumps(self.identity(), sort_keys=True, indent=2)
