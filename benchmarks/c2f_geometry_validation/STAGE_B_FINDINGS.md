# Stage B Findings: External-Geometry-Gated C2F

Date: 2026-09-16

## Question

Does external geometric validation make frozen C2F more reliable than both
the original intervention and a spatially uniform control with the same
instantaneous update norm?

The locked experiment contains ten development scenes: two cases from each
of five motion strata. It is disjoint from the reserved validation, test, and
debug splits. All groups use the same prompt, input frame, seed, Wan settings,
and frozen C2F layer/step/history configuration.

## Visual gate

All B/C/G/U videos are decodable and usable. G and U introduce no new severe
stripe artifacts, black frames, or global collapse. This permits metric
evaluation but is not evidence of geometric improvement.

The formal visual review uses the immutable `official_same_host_reference`
lock for B and the frozen `c2f_k3_a0025` lock for C. The earlier
`stage_b_visuals_v1` overview used the pilot baseline path and must not be
cited as the formal B comparison.

## Method means

Lower is better for all consistency metrics. Higher is better for VBench-Q.
Relative Motion is shown as percent of B.

| Group | MEt3R-0.5s | MEt3R-1s | First-last | GeCo Fused | Motion/B | VBench-Q |
|---|---:|---:|---:|---:|---:|---:|
| B: official Wan | 0.255879 | 0.342292 | 0.411238 | 0.783138 | 100.00% | 9.440779 |
| C: frozen C2F | 0.253914 | 0.327689 | 0.435452 | 0.735134 | 101.54% | 9.422068 |
| G: geometry gate | 0.258427 | 0.328591 | 0.393433 | 0.742490 | 100.52% | 9.447276 |
| U: uniform norm | 0.254433 | **0.325372** | 0.409690 | 0.761008 | 100.84% | 9.433004 |

Independent LRE is not reported: the frozen baseline eligibility lock contains
zero eligible cases among these ten scenes. Evaluating candidate-specific
denominators would be unfair.

## Preregistered comparisons

The primary metric is MEt3R-1s.

| Comparison | Mean relative improvement | Median directional change | Win rate | 80% bootstrap CI |
|---|---:|---:|---:|---:|
| G vs B | +4.003% | -0.000479 | 4/10 | [-1.536%, +10.633%] |
| G vs C | -0.276% | -0.000357 | 4/10 | [-0.973%, +0.312%] |
| G vs U | -0.990% | -0.000389 | 2/10 | [-2.812%, +0.775%] |

The geometry-selection hypothesis is therefore **not supported** on Stage B.
G improves the mean relative to B, but does not improve over frozen C2F and is
worse than the update-norm-matched U control. The G-vs-B mean is also
heterogeneous: its median is negative and only four of ten scenes improve.

Additional evidence points in the same direction:

- At 0.5 seconds, G is worse than C by 1.778% and worse than U by 1.570%; it
  beats each in only 1/10 scenes.
- G has a favorable first-last mean, but this is unstable and driven largely
  by one forward-right case. G beats U on first-last in only 3/10 scenes.
- G improves mean GeCo Fused over U by 2.433%, but only 3/10 scenes improve and
  the median directional change is negative.
- G improves VBench-Q over C and U in 8/10 scenes. Sparse gating may preserve
  appearance better, but this does not translate into robust geometry gains.
- Motion is preserved: G/B is 100.52% and U/B is 100.84%.

## Mechanism result

The gate retains 23.15% of active C2F correspondences. U matches the
hypothetical G update norm with maximum relative error below `8e-8`, so the
G-vs-U comparison is a valid spatial-selection test. U is better on the
primary metric and wins seven of ten scenes against B. Therefore the current
evidence favors this interpretation:

> Reducing intervention strength is more useful than this hard spatial
> geometry selection rule; the current VGGT-Omega acceptance mask does not
> identify where C2F residual transport should be applied reliably enough.

This does not prove that external geometry is useless. U still uses geometry
to determine a global, layer-step-specific scale. Its mean scale is about
0.46 of frozen C2F, with substantial scene variation. The unresolved question
is whether U's signal comes from geometry-conditioned strength adaptation or
merely from using a smaller alpha.

## Efficiency

The attention intervention itself is cheap:

- B sampling: 352.79 seconds mean.
- U intervention sampling: 352.83 seconds mean.
- First three uncontended G runs: 352.05 seconds mean.
- Peak sampling memory: +0.69% over B.
- Geometry forward: 2.05 seconds per scene, plus a 17.11-second shared model
  load; geometry peak allocation is 6.78 GiB.

The complete external-geometry method is nevertheless two-pass:

1. Generate a baseline draft.
2. Infer external geometry.
3. Resample with attention intervention.

Its batch-amortized end-to-end time is 709.37 seconds per scene, or 201.08% of
B. Sequential peak memory is 29.27 GiB because the phases need not coexist.
The raw mean G sampling time is contaminated by an unrelated workload during
cases 4-10 and must not be used as method overhead.

## Decision

Do not scale the current hard geometry gate to Dev25, Validation100, or Test100.
Keep it as a negative spatial-selection ablation and mechanism result.

The highest-information next control is one geometry-free, one-pass C2F run
with a preregistered half-strength alpha (`0.0125`) on the same ten scenes:

- If fixed half-strength C2F matches U, the gain is intervention-strength
  calibration and external geometry is unnecessary.
- If U remains better, geometry-conditioned global strength is useful even
  though geometry-conditioned spatial selection is not.

This control changes one variable, is cheaper than another geometry method,
and directly decides whether U contains a publishable simple mechanism.

## Frozen assets

- Formal report:
  `/vol/dissolve/yz10325/outputs/c2f_geometry_validation_0915/stage_b_report_v2/STAGE_B_FOUR_WAY_REPORT.json`
- Per-case table:
  `/vol/dissolve/yz10325/outputs/c2f_geometry_validation_0915/stage_b_report_v2/STAGE_B_PER_CASE.csv`
- Formal visual overview:
  `/vol/dissolve/yz10325/outputs/c2f_geometry_validation_0915/stage_b_visuals_official_v2/STAGE_B_BCGU_OFFICIAL_OVERVIEW.png`
- Visual gate:
  `/vol/dissolve/yz10325/outputs/c2f_geometry_validation_0915/stage_b_visuals_official_v2/STAGE_B_VISUAL_GATE.json`
- Immutable evaluator inputs:
  `/vol/dissolve/yz10325/outputs/c2f_geometry_validation_0915/stage_b_evaluator_inputs_v1`
- G/U sealed metric components:
  `/vol/dissolve/yz10325/outputs/c2f_geometry_validation_0915/stage_b_metrics_v1`
