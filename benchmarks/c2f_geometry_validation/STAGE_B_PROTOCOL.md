# Stage B: Geometry-Gated C2F Paired Development Experiment

## Question

Does external geometry validation make frozen C2F more reliable than both the
original intervention and an equally weak but spatially uniform control?

## Outcome-independent scene selection

Use exactly two scenes from each of the five frozen Dev25 motion strata. The
selected cases have `stratum_quantile_rank` 1 and 3. Stage A used rank 2, so
Stage B contains ten distinct scenes. Selection does not inspect generated
videos, C2F outcomes, or geometry outputs. All cases remain disjoint from the
frozen validation, test, and debug splits.

## Frozen generation and C2F mechanism

- Wan2.2-TI2V-5B
- seed 0
- 121 frames, 704x1280, 24 FPS
- 50 sampling steps, CFG 5
- C2F K=3, alpha=0.025
- transformer blocks 10, 15, and 20
- sampling steps 20--29 inclusive
- conditional CFG branch only

Existing B (official baseline) and C (frozen C2F) videos are reused. Only G
and U are newly generated.

## Fixed geometry evidence

VGGT-Omega is run once on the existing baseline draft at every fourth output
frame, exactly matching Wan's 31 temporal transformer tokens. For each C2F
source-target token pair, the source point is unprojected, transformed to the
target camera, and tested with the Stage A rule:

- source and projected-target confidence at or above each frame's p20;
- positive, finite, in-bounds depth and projection;
- depth agreement within 15%;
- reprojection error at most 1.5 Wan token cells.

Only a positive visible acceptance permits transport. Rejections, occlusions,
newly revealed regions, and low-confidence geometry all abstain and produce no
C2F update. The geometry model never supplies a negative correction.

## Groups

**B: Unguided.** Existing official Wan baseline.

**C: Frozen C2F.** Existing K=3, alpha=0.025 result.

**G: Geometry-gated C2F.** For each accepted correspondence:

```text
O' = O + alpha * C * G * (V_source - O)
```

where `C` is the original C2F confidence and `G` is the binary external
geometry acceptance.

**U: Uniform-norm control.** Compute the hypothetical G update norm for the
current layer-step, then use one scalar on all original C2F residuals:

```text
s = ||alpha * C * G * residual||_2 / ||alpha * C * residual||_2
O' = O + alpha * C * s * residual
```

Thus U has the same instantaneous L2 intervention strength as G on its own
trajectory but does not know which spatial tokens passed geometry. Sampling
trajectories diverge after the first intervention, so exact cross-run energy
identity is not claimed; per-event norm mismatch is recorded.

## Staged execution

1. Run one paired G/U case and inspect contact sheets before any metrics.
2. If both videos are usable, run the remaining nine paired cases.
3. Produce visual sheets for all methods before metric evaluation.
4. Evaluate the locked primary comparisons: G-C, G-U, and G-B.

Primary consistency metric: MEt3R-1s. Report GeCo Fused and independent LRE
as secondary geometry metrics. Relative Motion and VBench Quality are safety
metrics for motion collapse and visual degradation. Also report runtime, peak
VRAM, geometry coverage, C2F survival fraction, and U/G norm-match error.

G beating C but merely returning to B means geometry reduces C2F harm; it does
not mean the method improves the frozen generator. A positive method result
requires G to improve over B while remaining better than U and preserving
motion and visual quality.
