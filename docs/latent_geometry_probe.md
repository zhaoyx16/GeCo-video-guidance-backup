# Latent Geometry Probe Foundation

This branch is a strict, probe-only scaffold for testing geometry information
in cached Wan VAE latents. It does not modify a Wan sampling pipeline, Frame
Guidance, RGB GeCo guidance, or evaluation.

## Scope: three distinct latent distributions

- `raw_vae_z0`: the direct, clean output of a frozen VAE encoder. This is the
  only cache domain accepted by `CachedLatentDataset` in this branch.
- `diffusion_z0`: the raw VAE posterior mode after Wan's exact channel-wise
  `(z0 - latents_mean) / latents_std` normalization.
- `online_probe_zt`: generated in memory as
  `zt = (1 - t) diffusion_z0 + t epsilon`. It is a controlled robustness
  probe in Wan's diffusion-latent scale, not an assertion that it exactly
  equals a real Wan scheduler state.
- `x0_pred`: a future domain that must be extracted from frozen Wan at a real
  scheduler timestep under a fully recorded prompt/image/CFG condition. It is
  not generated or accepted by this branch, because treating it as ordinary z0
  would be an invalid sampler-integration claim.

For a pure `z0` control, the model always receives `t=0` through
`z0_timestep`; it never sees a nonzero noising timestep merely because the
same batch also contains an online `zt` example.

## Versioned cache and provenance contract

The v3 cache is a trusted PyTorch dictionary with:

```text
format_version: 2
record_id, scene_id
z0:                     float [C, T_latent, H_latent, W_latent]
camera_poses_w2c:       float [F, 4, 4]
intrinsics:             float [F, 3, 3]
frame_ids:              integer [F], unique source frame IDs
source_provenance:      immutable source identifiers and digests
pose_spec:              explicit W2C, SE3, right-handed convention
latent_spec:            typed latent-domain/VAE/preprocessing metadata
temporal_mapping:       explicit latent-token to RGB-anchor mapping
cache_sha256:           digest of tensors plus validated metadata
```

`source_provenance` must include:

```text
source_dataset
source_scene_uid
source_clip_uid
source_content_sha256
source_frame_ids_sha256
source_pose_sha256
```

The writer verifies the frame-ID and pose hashes against the cached tensors,
then stores a `cache_sha256` over `z0`, poses, intrinsics, frame IDs, temporal
mapping, pose spec, provenance, and latent spec. The v3 manifest mirrors the
dataset-qualified `source_scene_uid`, `source_clip_uid`, verified
`source_content_sha256`, and `cache_sha256`. Dataset construction validates
every cache in every split before selecting the requested split. Split
validation treats `source_content_sha256` as an identity key, so independently
re-caching identical source content under fresh UIDs and cache hashes cannot
place it in two splits.

`source_content_sha256` is intentionally required but cannot be recomputed by
this package without the original images/video. A future extractor must compute
it from a documented ordered source-frame list or source asset bytes. The cache
hash cryptographically binds the extractor-provided digest, and the loader
checks that its manifest copy matches exactly. Verifying that the extractor
derived that digest from the original RGB/video remains an external-data
responsibility; this probe cannot claim that without those assets.

## Explicit RGB-pose to latent mapping

`temporal_mapping` has one explicit RGB anchor per latent token:

```text
mapping_type: latent_anchor_frame_id
anchor_rule: <documented rule, e.g. causal_first_output_anchor>
is_causal: true | false
temporal_compression_ratio: positive integer
latent_to_frame_ids: integer tensor [T_latent]
```

`latent_to_frame_ids[k]` must exist in `frame_ids`. Every manifest pair supplies
both RGB pose indices and latent indices; the loader rejects it unless the pose
indices point to exactly the RGB frame IDs anchored by its selected latent
tokens. This avoids silently assuming `latent_index == RGB_frame_index`.

The cache also requires a `pose_spec` and numerically validates every stored
transform as finite, right-handed SE(3): bottom row `[0, 0, 0, 1]`, orthonormal
rotation, and determinant near `+1`. Relative pose is:

```text
T_target_from_source = T_w2c[target] @ inverse(T_w2c[source])
```

The target is 6D rotation plus translation *direction* in target-camera
coordinates. Translation magnitude is deliberately excluded because it is
ambiguous in monocular video.

## Typed latent metadata

`latent_spec` records a domain, model family, VAE identifier/revision/scaling,
preprocessing, temporal-mapping type, scheduler, and condition:

```text
domain: raw_vae_z0 | normalized_diffusion_z | x0_pred
model_family: wan
vae: {identifier, revision, latent_scaling}
preprocessing: {image_normalization, height, width, fps, frame_sampling}
temporal_mapping_type: latent_anchor_frame_id
scheduler: null for raw_vae_z0; required structured metadata otherwise
condition: null for raw_vae_z0; required for x0_pred
```

The current dataset rejects any domain other than `raw_vae_z0`. This is
intentional: `normalized_diffusion_z` and `x0_pred` require scheduler- and
condition-faithful extraction, and adding them is a later research task rather
than an implicit sampler integration.

## Models, baselines, and metrics

- `ConstantPoseBaseline` is learned and trained with the same optimizer and
  number of steps as the other probes.
- `LinearLatentProbe` reads global latent statistics plus the valid timestep.
- `Small3DConvCritic` is the small nonlinear candidate, with 3D convolution and
  timestep embedding.

Training uses chordal rotation loss and masked translation-direction cosine
loss. Evaluation reports rotation geodesic degrees and translation-direction
angular degrees on scene-disjoint held-out data. The evaluator moves the model
to the requested device before consuming the batch.

## CPU tests and smoke run

From the repository root:

```bash
python -m unittest discover -s tests -v
python scripts/run_latent_geometry_probe_smoke.py --steps 24
```

Both commands use synthetic tensors only. They do not download or load Wan,
VAE, VGGT, Any4D, or any checkpoint.

## Real DL3DV clean-latent pilot

The real extractor converts DL3DV/nerfstudio OpenGL c2w poses to OpenCV-axis
w2c, updates intrinsics for resize-cover/center-crop, encodes ordered 17-frame
clips with the frozen tiled Wan VAE, and writes a scene-disjoint manifest.
Source distortion is recorded but not corrected, so cached intrinsics must not
yet be used for a reprojection loss. The first pilot uses only adjacent latent
tokens (`--pair-gaps 1`) to avoid pair-gap leakage.

```bash
python scripts/cache_dl3dv_wan_z0.py \
  --dl3dv-root /path/to/dl3dv_0718 \
  --model /path/to/Wan2.2-TI2V-5B-Diffusers \
  --output-root /path/to/probe_cache \
  --val-scenes "one/source_scene_uid" \
  --test-scenes "another/source_scene_uid" \
  --height 704 --width 1280 \
  --clip-frames 17 --clip-stride 16 \
  --max-clips-per-scene 24 --pair-gaps 1

python scripts/run_latent_geometry_probe_real.py \
  --manifest /path/to/probe_cache/manifest.jsonl \
  --output-root /path/to/probe_results \
  --device cuda --train-domain diffusion_z0
```

The runner trains learned-constant, ordered-linear, and small 3D-Conv probes
with identical shuffle/noise streams. It reports pair-weighted and scene-macro
validation/test metrics, a controlled online-noise timestep curve, runtime,
manifest digest, code commit, and checkpoints. These clean/online-noise
results are only a gate for producing scheduler-faithful `x0_pred(t)` caches;
they are not evidence that guidance works during Wan sampling.

## Gate before any sampling-guidance branch

Do not attach this critic to sampling until all of these are demonstrated:

1. The nonlinear critic beats learned constant and linear baselines on held-out
   source scenes/sequences.
2. Results hold at the selected online-zt timesteps.
3. A separate, scheduler-faithful `x0_pred(t)` cache with full condition
   metadata validates the critic on the actual sampling distribution.
4. Gradients from a frozen critic to latent input are finite and reduce critic
   loss without collapsing motion or trajectory adherence.
5. In a separate Frame-Guidance experiment, independent geometry and trajectory
   metrics do not regress.

Only after these gates should a new `latent-guidance-wan` branch be created.
