# Stage A: External Geometry Sanity Check for Frozen C2F

## Question

Does pinned external geometry provide a usable, correctly aligned signal for
validating the correspondences selected by frozen C2F?

Stage A is a coordinate and signal sanity check. It does not claim that a new
guidance method improves generated video, and it does not generate new full
videos.

## Frozen scene selection

Use exactly one scene from each of the five Dev25 motion strata. Within each
stratum, choose `stratum_quantile_rank == 2`, the middle case selected from
real DL3DV pose statistics. This rule was fixed without using C2F outcomes or
geometry outputs.

The selected Dev25 remains disjoint from the frozen test, validation, and
debug scene hashes.

## Frozen C2F observation

- trajectory: `baseline_observe` (alpha zero)
- denoising step: 20, the first active C2F step
- transformer layer: 10, the first active C2F layer
- token grid: `[31, 22, 40]`
- target temporal tokens: 10 and 25
- history: the actual source selected by frozen K=3 C2F (lags 1--3)
- dense deterministic diagnostic sample size: 8192 per layer-step

The target times provide an early and late view of the same mechanism. A C2F
target token can select a different history lag than its neighbour, so each
target time is a correspondence set rather than one artificial frame-global
pair. Wan temporal token `t` maps to output frame `4*t` for 121-frame videos.

The geometry input is the existing official Wan baseline video. This matches
the unguided trajectory observed by `baseline_observe` and avoids measuring a
video that C2F has already changed.

## Frozen geometry model

- backbone: pinned `VGGT-Omega-1B-512`
- official source commit: `39a0cb8af88554f15ddcb5354cd52bde588fa014`
- checkpoint SHA256:
  `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`
- preprocessing: `balanced`, resolution 512
- camera convention: OpenCV world-to-camera
- depth: camera-space z depth

The already-audited 512 checkpoint is used to avoid making a model upgrade a
precondition for the mechanism check. A newer geometry checkpoint is a later
robustness test, not a Stage A tuning axis.

## Coordinate mapping

C2F token `(x, y)` is mapped to the centre of its cell in the processed
VGGT-Omega image. With token grid `(Htok, Wtok)` and processed image `(H, W)`:

```text
u = (x + 0.5) * W / Wtok - 0.5
v = (y + 0.5) * H / Htok - 0.5
```

The source point is unprojected with source depth/intrinsics, transformed via
the explicit world-to-camera matrices, and projected into the target frame.

## Frozen evidence rule

For every sampled C2F match:

1. Abstain if source or target geometry confidence is below that frame's 20th
   percentile, the projection is non-finite, behind the camera, or outside the
   target image.
2. Abstain as occluded if projected source depth is more than 15% behind the
   target depth.
3. Reject as a front-surface conflict if projected source depth is more than
   15% in front of the target depth.
4. For depth-consistent visible evidence, accept when the geometry projection
   is within 1.5 token-cell units of the C2F target; otherwise reject.

Continuous reprojection and depth errors are always retained. The thresholds
are fixed for this sanity check and must not be tuned per scene.

## Outputs

For every scene and target time:

- source frames with C2F source locations;
- target frame with C2F target locations, geometry projections, and residuals;
- accepted/rejected/abstained counts;
- valid geometry coverage and continuous error distributions.

Per scene, also record VGGT-Omega forward time and peak allocated CUDA memory.

## Decision

Proceed to the 10-scene B/C/G/U experiment only if visual inspection confirms
that coordinates are aligned and the geometry evidence behaves sensibly:

- clear same-surface matches usually have small residuals;
- obvious wrong-surface matches are rejected more often;
- occluded or newly revealed regions abstain rather than receive a hard label.

If correct matches are systematically rejected, first debug preprocessing,
camera convention, and token/frame mapping. Only after those checks pass may
the same frozen examples be tested with an alternate geometry backbone.
