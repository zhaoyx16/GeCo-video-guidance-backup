"""Small, frozen-generator latent geometry probe utilities.

This package intentionally contains no sampling-pipeline integration.  It is
for testing whether a Wan VAE latent can support a lightweight geometry
predictor before any inference-time guidance is attempted.
"""

from .data import (
    CACHE_FORMAT_VERSION,
    LATENT_DOMAIN_NORMALIZED_DIFFUSION_Z,
    LATENT_DOMAIN_RAW_VAE_Z0,
    LATENT_DOMAIN_X0_PRED,
    MANIFEST_FORMAT_VERSION,
    CachedLatentDataset,
    LinearFlowNoiseSchedule,
    ManifestError,
    ProbeManifestRecord,
    load_manifest_records,
    save_clean_latent_record,
    tensor_sha256,
)
from .geometry import (
    make_relative_pose_target,
    pose_losses,
    pose_metrics,
    rotation_6d_to_matrix,
    rotation_matrix_to_6d,
    validate_world_to_camera_se3,
)
from .models import ConstantPoseBaseline, LinearLatentProbe, Small3DConvCritic

__all__ = [
    "CACHE_FORMAT_VERSION",
    "LATENT_DOMAIN_NORMALIZED_DIFFUSION_Z",
    "LATENT_DOMAIN_RAW_VAE_Z0",
    "LATENT_DOMAIN_X0_PRED",
    "MANIFEST_FORMAT_VERSION",
    "CachedLatentDataset",
    "ConstantPoseBaseline",
    "LinearFlowNoiseSchedule",
    "LinearLatentProbe",
    "ManifestError",
    "ProbeManifestRecord",
    "Small3DConvCritic",
    "load_manifest_records",
    "make_relative_pose_target",
    "pose_losses",
    "pose_metrics",
    "rotation_6d_to_matrix",
    "rotation_matrix_to_6d",
    "save_clean_latent_record",
    "tensor_sha256",
    "validate_world_to_camera_se3",
]
