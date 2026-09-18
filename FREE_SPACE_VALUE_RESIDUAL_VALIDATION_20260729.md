# Free-Space Value Residual: Frozen Validation Record

## Scope

This record covers only the `codex/geometry-attention-0728` worktree:

```text
/vol/dissolve/yz10325/repos/GeCo-worktrees/geometry-attention-0728
```

It does not cover or modify the persistent-surface deformation branch:

```text
/vol/dissolve/yz10325/repos/GeCo-worktrees/draft-geometry-attn-0728
```

The reviewed intervention handles one failure family:

```text
source-observed free space -> hallucinated foreground surface
```

It does not claim to repair deformation, disappearance, texture drift, or
dynamic-object failures.

## Frozen Method

For an active target token, the offline geometry map supplies:

- a free-space conflict confidence `c_conflict`;
- a source-frame observed-background token selected behind the conflict
  along the same source ray;
- a validated source Value feature `V_source`.

At block-0 conditional self-attention, steps 0--20:

```text
O' = (1 - alpha * c_conflict) * O
     + alpha * c_conflict * V_source
```

Frozen settings:

```text
mode: free_space_value_residual
reference: conditioning frame f0
layer: 0
alpha: 0.10
denoising steps: 0--20
branch: conditional CFG branch only
frames: 121
resolution: 704x1280
sampling steps: 50
```

## Canonical Case

Baseline:

```text
/vol/dissolve/yz10325/outputs/static_motion_observable_0718/baselines/wan/dl3dv/dl3dv_corridor_traditional_11k_9c688/baseline_seed1_steps50_frames121.mp4
```

Geometry-aligned result:

```text
/vol/dissolve/yz10325/outputs/geometry_attention_20260728/wan_traditional_corridor_s1/free_space_validation_20260729/free_space_value_v3_l0_a010_s0_20_seed1.mp4
```

Controls:

```text
same-2D identity source:
free_space_value_identity_control_v3_l0_a010_s0_20_seed1.mp4

random source:
free_space_value_random_control_v3_l0_a010_s0_20_seed1.mp4
```

Visual result:

- baseline: a horizontal wooden board grows across the open passage;
- geometry-aligned source: the board is removed;
- same-2D identity source: the board is removed, but the intended turn is
  strongly reduced;
- random source: the board is not repaired.

The right-side wooden railing deformation remains. That failure belongs to
the persistent-surface branch and is outside this detector's state model.

## Camera-Trajectory Diagnostic

VGGT trajectory diagnostic:

| Variant | Path/depth | End/depth | Cumulative rotation | End rotation |
|---|---:|---:|---:|---:|
| Baseline | 3.4915 | 3.1996 | 49.49 deg | 50.74 deg |
| Geometry | 2.9403 | 2.8388 | 49.32 deg | 50.28 deg |
| Same-2D identity | 3.3480 | 3.1788 | 26.33 deg | 24.03 deg |
| Random source | 3.7085 | 3.6429 | 47.63 deg | 47.42 deg |

Interpretation:

- same-2D copying removes the obstacle partly by freezing the video toward
  the first-frame coordinate system;
- geometry alignment removes the obstacle while retaining the requested
  turn;
- geometry has a residual risk of reduced translational path length, about
  16% under this normalized estimate.

## GeCo-Eval: Supplementary, Not Decisive

All four variants use identical evaluation settings and source frames.

Long-window configuration:

```text
win_sec=2.0, eval_fps=4, max_windows=4,
pair_stride=1, ufm_longside=255
```

| Variant | Motion | Depth | Fused |
|---|---:|---:|---:|
| Baseline | 0.0485 | 0.2435 | 0.1832 |
| Geometry | 0.0300 | 0.1925 | 0.1368 |
| Same-2D identity | 0.0384 | 0.1331 | 0.0948 |
| Random source | 0.0593 | 0.1984 | 0.1415 |

Short-window configuration:

```text
win_sec=0.5, eval_fps=8, max_windows=8,
pair_stride=1, ufm_longside=255
```

| Variant | Motion | Depth | Fused |
|---|---:|---:|---:|
| Baseline | 0.0234 | 0.1039 | 0.0674 |
| Geometry | 0.0151 | 0.0757 | 0.0512 |
| Same-2D identity | 0.0238 | 0.1085 | 0.0717 |
| Random source | 0.0280 | 0.0830 | 0.0584 |

GeCo-Eval alone is insufficient:

- long-window evaluation incorrectly prefers the trajectory-frozen identity
  control;
- random correspondence improves fused score without repairing the board;
- the short-window metric is more aligned with the visual and trajectory
  evidence, but remains supplementary.

The valid evidence is the conjunction:

```text
visible repair + requested turn retained + no new collapse
+ short-window metric moves in the same direction
```

## Negative-Seed Abstention

The frozen detector was applied to seed 0 and seed 2 of the same scene, where
the horizontal-board failure is absent. Parameters and target interval were
identical to the positive case.

| Seed | Active tokens | Active-frame coverage | Mean confidence | Gate |
|---|---:|---:|---:|---|
| 0 | 9 | 0.0026 | 0.0266 | abstain |
| 1 positive | 653 | 0.0824 | 0.3589 | active |
| 2 | 82 | 0.0133 | 0.0453 | abstain |

The frozen threshold is:

```text
min_active_frame_coverage = 0.03
```

Seed-0 activations are isolated. Seed-2 activations are sparse and mostly
near normal occlusion or image boundaries. Both maps are rejected before
attention modification. When the gate abstains, the wrapper delegates to the
original attention processor.

## Correctness Review

Three independent review rounds were completed without showing reviewers the
preferred visual result.

Fixed issues:

- conflict confidence now controls target blend strength;
- observed-background confidence only validates/selects the source;
- map evidence semantics are explicit and validator-matched;
- active source evidence must be in slot 0, causal, and in range;
- fractional indices are rejected before integer conversion;
- runtime attention batch is used for `num_videos_per_prompt > 1`;
- hook/processor installation and execution cleanup are exception-safe;
- alpha zero bypasses the geometry path.

Verification:

```text
16 CPU correctness tests: PASS
Python syntax compilation: PASS
alpha=0 / zero mask / repeated seed GPU smoke: byte-identical MP4 outputs
```

Remaining test gap:

- the batch-two helper is runtime tested, but a full processor invocation
  with `num_videos_per_prompt=2` has not been run. Research runs use one video
  per prompt.

## Generalization Status

Existing Wan baselines were screened across:

- DL3DV office, forest, garden, and traditional-corridor seeds;
- admin, hospital, narrow, office, school, and subway corridors;
- stress and batch-2 static scenes;
- earlier attention-study baselines.

No second clean free-space positive was found. Ambiguous cases were rejected:

- close-passing doorway foreground can be valid disocclusion;
- a storage-store stack of boxes is explicitly requested by the prompt;
- people and dynamic objects belong to a different failure family;
- scene switches and persistent deformation are not free-space violations.

Therefore the current claim must remain:

> On one canonical source-observed free-space hallucination, geometry-aligned
> value residual removes the hallucinated board while preserving the intended
> camera turn. Same-2D and random controls show that structured geometry is
> needed to avoid trajectory freezing and arbitrary transport. Cross-scene
> positive generalization is not yet established.

## Stop Condition

Do not continue alpha/layer tuning on the canonical case. The free-space
component is frozen until a second unambiguous positive holdout is available.
Use seed 0 and seed 2 as negative regression cases. Future integration with
the deformation branch must preserve this canonical repair and both negative
abstentions.
