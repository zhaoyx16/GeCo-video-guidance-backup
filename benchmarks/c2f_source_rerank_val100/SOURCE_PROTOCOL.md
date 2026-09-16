# Source-aware C2F Val100 protocol

## Frozen scope

- Split: the 100 validation cases in `wan_validation_100_layout_aware_v2_formal.json`.
- Manifest SHA-256: `ef59a10d21d7d30729a9ac24cfba7cb37c153f5ae05debaa18387d3720afb862`.
- Seed: `0`.
- No validation, test, or debug scene overlap is permitted.
- Test100 remains sealed while methods are developed and selected.
- Wan2.2-TI2V-5B generation: 121 frames, 704x1280, 24 fps, 50 steps, CFG 5.
- C2F intervention: blocks 10/15/20, zero-based steps 20-29, conditional CFG branch only, K=3 recent temporal candidates.

## Candidate policies

For feature confidence `c_ij`, pairwise geometry weight `w_ij`, and source support score `r_j`:

```text
V: j* = argmax_j c_ij;       apply w_ij* after selection
P: j* = argmax_j c_ij w_ij
S: j* = argmax_j c_ij w_ij r_j
```

Candidate order is lag 1, then lag 2, then lag 3, so exact ties preserve the most recent source. An all-zero candidate set is a no-op.

For source temporal index `s`, evidence views are `{s-1,s-2,s-3}` when valid. Each view is support, conflict, or unknown. Low confidence, out-of-bounds projection, and occlusion are unknown rather than conflict. The fixed score is:

```text
r_j = (1 + n_support) / (1 + n_support + n_conflict)
```

Source support affects selection only. After selection, every policy uses:

```text
O'_i = O_i + alpha c_ij* w_ij* (V_j* - O_i)
```

The source score is not multiplied into update magnitude.

## Geometry settings

- Confidence threshold: per-view p20.
- Relative depth tolerance: 0.15.
- Reprojection tolerance: 1.5 Wan token cells.
- Hard pair weight: 1 only when observable, depth-compatible, and within the reprojection tolerance.
- Soft pair weight: product of linearly decaying depth and reprojection terms over the same valid support region.

Evidence variants:

1. `draft`: one 31-view VGGT-Omega estimate from the complete unguided baseline video, RGB frames 0,4,...,120.
2. `online_snapshot`: one shared 31-view estimate from the unguided predicted-clean sample at sampling step 19. Intervention has not started, so this state is common to all methods with the same input and seed.
3. `online_refresh`: starts from the shared step-19 snapshot, then recomputes predicted-clean geometry after steps 22, 25, and 28. The new evidence applies from steps 23, 26, and 29 respectively. Pair and source evidence always come from the same refresh.

## Matrix and controls

Main methods are the complete crossing:

```text
alpha {0.0125, 0.025}
x pair weight {hard, soft}
x retrieval {V, P, S}
x evidence {draft, online_snapshot, online_refresh}
= 36 methods
```

Uniform controls cover V and P only:

```text
2 alpha x 2 pair x 2 retrieval x 3 evidence = 24 controls
```

For each layer and sampling step, the uniform control applies the original feature-top1 C2F residual with one scalar selected to match the target geometry method's global L2 update norm. The scalar is capped at 1.0 and norm mismatch is recorded. S has no uniform control because source support only changes candidate selection; the matched P method is its direct component ablation.

Public references are Wan unguided (B), C2F K3 alpha=0.025 (C), and C2F K3 alpha=0.0125 (F). Existing adapted-GeCo results are reused and not regenerated.

## Primary comparisons

- Source increment: the 12 paired `S - P` conditions.
- Geometry reranking increment: the 12 paired `P - V` conditions.
- Practical value: each candidate against B, C, F, and existing adapted-GeCo.

The primary scene-level source summary averages the 12 matched P-vs-S differences within each scene, then performs paired bootstrap over 100 scenes. Conditions are also reported individually; 12x100 observations are not treated as 1,200 independent samples.

## Evaluation and validity gates

Before metric aggregation, generated videos receive contact-sheet visual QA for corruption, motion collapse, and conditioning failure. Final metrics are GeCo Fused (lower), MEt3R (lower), Relative Motion relative to paired Wan baseline (guardrail near 100%), and VBench Quality (higher). Independent LRE is excluded from the main table.

Required implementation checks:

- `r_j = 1` recovers P from S exactly.
- Rigid synthetic geometry produces support.
- Explicit front-surface inconsistency produces conflict.
- Occlusion, low confidence, and out-of-bounds evidence do not produce conflict.
- All invalid candidates produce no update.
- Uniform controls report target/actual update norm mismatch.
- Every output records source evidence coverage, score distribution, S-vs-P changed-source rate, support/conflict counts, selected temporal distance, runtime, and peak memory.
