# Latent Geometry Probe Foundation

This branch contains a small, isolated research scaffold for testing whether
Wan clean VAE latents can support geometry prediction. It does not alter a Wan
pipeline, Frame Guidance, GeCo guidance, or evaluation code.

## Scope and three latent distributions

- `z0`: a clean latent cached from a frozen Wan VAE. This is the first and
  cheapest question: can geometry be read from the VAE representation at all?
- `zt`: an online noised version of that cached latent. This branch uses the
  probe-only convention `zt = (1 - t) z0 + t epsilon`, with `t` given to the
  critic. It is useful for checking robustness to noise, but is not a claim
  that it exactly reproduces Wan's scheduler state.
- `x0_pred(t)`: a future input. It must be obtained by running frozen Wan at a
  real sampling timestep and is the distribution that matters for sampling-time
  guidance. It is deliberately not generated in this foundation because that
  would require expensive VDM inference and a separate cache/extraction plan.

Passing a `z0` probe only shows representational information. It does not prove
that a critic can guide Wan sampling. The required next stages are documented at
the end of this file.

## Manifest and cache contract

The data interface is JSONL. Every record has these required fields:

```json
{
  "format_version": 1,
  "record_id": "kitti_00_000123_pair_000_020",
  "scene_id": "kitti_odometry_00",
  "split": "train",
  "cache_path": "cache/kitti_00_clip_000123.pt",
  "source_pose_index": 0,
  "target_pose_index": 20,
  "source_latent_index": 0,
  "target_latent_index": 5,
  "static_scene": true,
  "source_dataset": "KITTI-odometry"
}
```

`source_pose_index` and `target_pose_index` index the cached RGB-camera poses.
`source_latent_index` and `target_latent_index` index the temporal axis of
`z0`. They are intentionally separate because a causal video VAE's temporal
mapping is not generally one-to-one with RGB frames.

One cached `.pt` file is a trusted PyTorch dictionary with:

```text
format_version: 1
z0:                  float tensor [C, T_latent, H_latent, W_latent]
camera_poses_w2c:    float tensor [F, 4, 4]
intrinsics:          float tensor [F, 3, 3]
frame_ids:           int tensor   [F]
metadata:
  model_family: wan
  vae_identifier: <exact frozen VAE/checkpoint identifier>
  pose_convention: world_to_camera
  z0_layout: C,T,H,W
  translation_target: unit_direction_in_target_camera
```

The loader rejects a manifest if any `scene_id` occurs in more than one of
`train`, `val`, or `test`. This is intentional: random frame-level splits would
leak route and scene appearance into validation. For DL3DV/KITTI, split by
source scene or sequence before creating latent pairs.

## Pose target convention

All cached poses must be world-to-camera transforms. For ordered source view
`s` and target view `u`, the target is:

```text
T_u_from_s = T_w2c[u] @ inverse(T_w2c[s])
```

The rotation target uses the continuous 6D representation (the first two
columns of `R_u_from_s`). The translation target is only
`normalize(t_u_from_s)`, expressed in target-camera coordinates. Its magnitude
is not supervised, avoiding monocular global-scale ambiguity. If the baseline
is effectively zero, `translation_valid=false` masks the direction loss.

## Models and metrics

- `ConstantPoseBaseline`: a learned constant prediction. It establishes the
  minimum that a latent-dependent model must beat.
- `LinearLatentProbe`: global mean of the latent pair plus scalar timestep, fed
  to one linear pose head. It is a diagnostic, not the proposed final method.
- `Small3DConvCritic`: two lightweight 3D residual blocks, timestep embedding,
  and a small pose head. This is the first nonlinear candidate critic.

Training uses chordal rotation loss plus masked translation-direction cosine
loss. Validation reports rotation geodesic error in degrees and translation
direction angular error in degrees. A useful probe must outperform the constant
baseline on held-out *scenes*, not merely training pairs.

## CPU smoke run

From the repository root:

```bash
python -m unittest discover -s tests -v
python scripts/run_latent_geometry_probe_smoke.py --steps 24
```

The smoke script writes only temporary synthetic latent records, trains the
three small heads on CPU, and removes the data afterward. It never loads Wan,
VGGT, Any4D, or a VAE.

## Gate before sampling guidance

Do not integrate this branch with Wan sampling until all of these hold:

1. A nonlinear critic beats constant and linear probes on scene-disjoint held-
   out data.
2. The same result remains true at the intended late/mid `zt` timesteps.
3. A separate cache of real Wan `x0_pred(t)` examples validates the critic on
   the distribution seen during guidance.
4. Gradients through the frozen critic to the latent are finite, nontrivial,
   and a small update decreases critic loss without collapsing predicted motion.
5. A future paired Frame-Guidance experiment shows trajectory adherence and an
   independent geometry metric improve or at least do not regress.

Only after those gates should a separate `latent-guidance-wan` branch attach a
frozen critic to a sampling loop.
