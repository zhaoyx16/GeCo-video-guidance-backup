# Geometry-Aware Attention Branch

This worktree isolates the explicit-geometry attention experiments from the
existing latent-consistency, GeCo guidance, and draft-geometry experiments.

## Scope

- Frozen Wan2.2-TI2V-5B.
- Training-free changes during sampling only.
- Offline VGGT camera/depth estimation is used to build sparse, visibility-
  checked correspondences from trusted early frames to later target frames.
- The current intervention transports aligned Q/K/V before Wan's native
  Q/K normalization, RoPE, and full self-attention.
- No VAE decode, VGGT forward, or metric backward occurs in the guided pass.

## Main case

`wan_traditional_corridor_s1`, seed 1, 50 steps, 121 frames, 704x1280.

The pre-registered failure interval is frame 16 through frame 48. Visual
comparison of this interval is required before computing any aggregate metric.

## Files

- `run_wan_geometry_transport.py`: isolated sampling entry point.
- `external/guidance_wan/pipeline_wan_i2v_geometry_transport.py`: Wan pipeline
  with geometry-aware attention processors.
- `external/guidance_wan/geometry_transport.py`: online sparse map utilities.
- `external/guidance_wan/draft_geometry_map.py`: offline trusted-anchor map.
- `scripts/build_draft_geometry_map.py`: builds the offline map.
- `scripts/diagnose_vggt_geometry.py`: visual geometry diagnostics.

## Experimental rule

Do not claim success from GeCo-Eval alone. A candidate must first show a
specific visible repair in the registered failure region while preserving
camera motion and overall video quality. Only then run independent metrics.

## Required Method Improvements

### Separate conflict detection from corrective evidence

`hidden_input_suppress` only says that the current target token is unreliable:

`h' = (1 - alpha * M_conflict) * h`

It does not identify the correct content. Future experiments must keep two
separate masks:

- `M_conflict`: a target token is likely geometrically invalid.
- `M_transport`: a trusted reference surface has a valid, visible geometric
  correspondence to the target token.

Suppression may use `M_conflict`, but reference K/V or output transport is only
allowed under `M_transport`. A free-space conflict without a verified
background surface must suppress conservatively or abstain; it must not copy an
arbitrary source token.

### Respect Wan temporal compression

One Wan transformer time token jointly represents approximately four RGB
frames. A conflict detected in one RGB frame must not automatically apply at
full strength to the whole temporal token. Build a per-RGB-frame soft conflict
map first, then aggregate within each temporal group:

`M_token = confidence_weighted_aggregate(M_frame_0, ..., M_frame_3)`

Record support count and conflict proportion. Require multi-frame support for
strong intervention; use a lower weight or abstain for single-frame evidence.
Spatial masks should aggregate multiple geometry samples per token rather than
only its centre pixel.

### Use visibility-checked geometry-aligned evidence

Reference evidence must pass all of the following gates:

- confident source and target depth/pose;
- valid projective correspondence;
- target-view visibility and z-buffer consistency;
- static-scene validity;
- sufficient appearance/correspondence confidence;
- not a newly revealed or disoccluded region.

Only then may the target query access projected reference K/V, receive a
geometric logit bias, or use a conservative attention-output residual. Native
self-attention must remain available so the model can generate genuinely new
content. Full 3D attention propagation is not itself evidence of correct 3D
correspondence.

### Require safe abstention

Map-level quality gates must reject sparse, fragmented, border-dominated, or
transparent/depth-ambiguous geometry. A rejected map must produce an exact
baseline no-op for the same seed.

## Evaluation Funnel

### Stage 1: every run

- fixed baseline/guided keyframes and fixed failure-region crops;
- visible repair of the pre-registered failure;
- new artifacts, blur, disappearance, or ghosting;
- qualitative camera-motion retention;
- map coverage, temporal support, connected-component share, border share,
  confidence, and abstention status.

Reject visually ineffective or degraded runs before aggregate metrics.

### Stage 2: visually promising configurations

- camera path length, endpoint translation, cumulative rotation, and endpoint
  rotation from the same sampled frames;
- an independent local geometry metric such as MEt3R-0.5s/1s once its GPU
  environment is fixed;
- feature-track consistency;
- GeCo long/short only as an auxiliary metric;
- matched-coverage random mask, same-coordinate/identity map, global
  suppression, and baseline controls.

VGGT-derived map construction and VGGT-dependent GeCo evaluation are not
independent evidence.

### Stage 3: small generalization screen

Evaluate two or three suitable static scenes across three fixed seeds using
baseline, the frozen best method, and matched random-mask control. Report per
case values, mean, median, and failure count. Do not tune per-scene parameters.

### Stage 4: final benchmark

Freeze the method and hyperparameters before running the complete benchmark.
Use MEt3R short windows, overlap-aware reprojection, track consistency, camera
trajectory retention, GeCo, VBench quality/temporal metrics, blinded human
review, paired statistics, and confidence intervals.

## Acceptance Criteria

A method is considered promising only when:

1. the registered geometry failure is visibly reduced;
2. the camera trajectory and video quality are not materially weakened;
3. at least one independent geometry/track metric agrees with the visual
   result;
4. it outperforms matched random and non-geometric suppression controls;
5. the direction is reproduced across seeds or scenes.

An improved GeCo score alone is not a successful result.
