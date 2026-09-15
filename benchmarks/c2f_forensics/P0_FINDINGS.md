# C2F P0 Findings

## Scope

This is a mechanism-discovery analysis on the already-seen Dev25. It is not a validation or test result. Dev25 is disjoint from the frozen test, validation, and debug splits by scene hash.

The primary outcome is paired MEt3R-1s delta (`C2F - baseline`), where a positive value is harmful. All inferential statistics use 25 scenes as the units. Token samples are used only to construct scene-level summaries and distribution plots.

## Integrity

- 25 `baseline_observe` and 25 frozen-C2F replays completed.
- Each replay contains the expected 10 denoising steps x 3 transformer layers = 30 diagnostic records.
- Every record came from clean commit `d9c35c511826bd91fc01d6efd197579e778f59d4`.
- All records report zero overlap with frozen test, validation, and debug scenes.
- All 25 baseline/C2F pairs match exactly at step 20, layer 10 before the first intervention, after excluding only the deliberately different update field.
- Separate reference runs proved that diagnostics preserve the exact step-29 latent hash for both alpha=0 and frozen C2F.

Mean partial-replay cost was 185.78 s / 28,490.85 MiB for baseline-observe and 186.28 s / 28,556.08 MiB for C2F. The intervention itself therefore added about 0.50 s and 65 MiB over this diagnostic run.

## Preregistered Results

| Hypothesis or control | Scene-level rho | Permutation p | 95% bootstrap CI | Decision |
|---|---:|---:|---:|---|
| H1: selected source failed its own prior retrieval | -0.058 | 0.784 | [-0.469, 0.371] | Not supported |
| H2a: direct-vs-chained path-cycle error | -0.052 | 0.800 | [-0.436, 0.337] | Not supported |
| H2b: multi-lag velocity disagreement | -0.174 | 0.404 | [-0.549, 0.282] | Not supported |
| H3: lag-2/lag-3 intervention fraction | +0.124 | 0.553 | [-0.311, 0.495] | Not supported |
| H4 control: active intervention coverage | +0.015 | 0.946 | [-0.404, 0.424] | No explanatory value |

The source-risk bins are also non-monotonic. In particular, sampled interventions whose generated source had zero previous confidence were slightly more common in winners (0.268) than losers (0.251). This is the opposite of the proposed source-error-propagation mechanism.

Post-treatment changes in the matcher were numerically tiny. For example, the median C2F-minus-observe change in candidate gated-active fraction was `-0.000088`. There is no evidence here for a large runaway feedback loop in correspondence selection.

## Exploratory Candidate

The strongest preregistered secondary statistic was the mean gap between the best and second-best history confidence:

- rho = -0.358;
- permutation p = 0.080;
- 95% bootstrap CI = [-0.691, 0.075];
- motion-stratum-demeaned rho = -0.243;
- leave-one-scene-out sign agreement = 1.00.

This mean statistic is suggestive but does not pass a conventional two-sided 0.05 test and its 95% interval crosses zero.

An explicitly post-hoc tail analysis found a stronger candidate: the scene-level 90th percentile of the same top1-minus-top2 history-confidence margin.

- rho = -0.423;
- permutation p = 0.037;
- 95% bootstrap CI = [-0.698, -0.036];
- motion-stratum-demeaned rho = -0.324;
- excluding near ties: rho = -0.406;
- excluding the largest outcome: rho = -0.355;
- leave-one-scene-out sign agreement = 1.00;
- winner median = 0.465, loser median = 0.447.

The interpretation is narrow: C2F is more likely to help scenes containing a stronger upper tail of tokens for which one history lag clearly dominates the alternatives. This points to correspondence decisiveness or ambiguity, not to corruption of the chosen historical source.

Because the q90 result was discovered after inspecting Dev25, it is a hypothesis generator only. It must not be presented as confirmed evidence or used to tune a threshold on Dev25.

## Mechanism Table

| Candidate factor | Association evidence | Counterfactual evidence | P0 judgment |
|---|---|---|---|
| Historical source corruption | Null, with non-monotonic risk bins | Not run because the locked premise failed | Rejected as the primary mechanism under the current definition |
| Path/cycle inconsistency | Null | Not run | Not supported |
| History age | Weak and uncertain | Not run | Not supported |
| Intervention coverage/strength | Null; larger actual update did not predict harm | Not run | Not the main bottleneck |
| Motion regime | Some heterogeneous means, but no single regime explains the diagnostics | Not causal | Conditioning factor only |
| History retrieval decisiveness | Suggestive mean effect and robust post-hoc q90 association | Not yet tested | Best next hypothesis; requires a new holdout |

## Decision

Do not implement or evaluate the planned `source-valid` gate merely to preserve the original narrative. P0 falsified its immediate premise, so running that counterfactual would spend compute on a factor that did not predict harm.

The next efficient experiment is a preregistered, threshold-free **history-dominance gate** on a new scene-disjoint development holdout that also excludes frozen validation, test, and debug scenes. A minimal candidate is:

```text
D_i = (c1_i - c2_i) / (c1_i + epsilon)
w_i = c1_i * D_i
O_i' = O_i + alpha * w_i * (V_source_i - O_i)
```

Here `c1` and `c2` are the best and second-best history confidence. This retains clear retrievals and smoothly abstains from ties without adding a tuned threshold.

The causal comparison should contain:

1. frozen C2F at alpha 0.025;
2. history-dominance C2F at the same alpha;
3. a uniformly weakened C2F control matched to the dominance method's mean intervention mass.

The third arm is necessary to distinguish selective retrieval from the simpler explanation that any weaker intervention is safer. Freeze all three arms before generating the new holdout. Validation100 and Test100 remain untouched until this new hypothesis survives that holdout.
