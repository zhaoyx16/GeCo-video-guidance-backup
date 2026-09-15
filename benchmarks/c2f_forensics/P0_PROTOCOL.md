# P0: Why Does Frozen C2F Help Some Dev Scenes and Hurt Others?

Status: preregistered before inspecting any new C2F correspondence diagnostics.

## Scope

This is a mechanism-discovery study, not a final method benchmark. It uses the already-seen C2F Dev25 only. No scene from the frozen test, validation, or debug splits may enter P0. Any method designed from these findings must be evaluated first on a new, disjoint development holdout before Validation100 or Test100.

Frozen inputs:

- Dev25 manifest SHA256: `332b4183431868969ec7f9dfcf39207c393eb4efb7516fcec01ccea6cd5a4d67`
- reserved split CSV SHA256: `d1eed2400547d755e265149185515c3e6fdf981f63635bbbfa152d33fa53205c`
- frozen C2F config SHA256: `6f34f574306233b927ac13a69a23c536ccf86089c92dd0d97747f993cc775aca`
- completed paired report SHA256: `c89fbcf672be1c392e8ddd35e0d0c9ac674fa5aecdb823d9d3db03335e3e8aaf`
- required scene-hash overlap with `test`, `validation`, and `debug`: zero for every split

## Fixed Outcome

The primary outcome is the existing paired MEt3R-1s difference:

`delta = C2F - official Wan baseline`

Lower MEt3R is better. A scene is therefore a `winner` when `delta < 0` and a `loser` otherwise. The full ranked continuous delta is primary; the 12/13 winner/loser label is descriptive. A sensitivity analysis may exclude near ties with `abs(delta) < 0.001`, but it cannot replace the all-25 analysis.

No metric is regenerated to define these labels.

## Frozen Intervention

- Wan2.2-TI2V-5B
- seed 0, 121 frames, 704x1280, 50 steps, 24 FPS, CFG 5
- C2F value-residual memory
- K=3, alpha=0.025
- layers 10, 15, 20
- inclusive sampling steps 20--29
- coarse factor 2, fine radius 2, descriptor dimension 64
- cosine threshold 0.4, coarse reciprocal matching, conditional branch only

P0 does not tune alpha, layers, step range, K, matching radius, or the backbone.

## Diagnostic Replays

Each Dev25 case is replayed twice from the same prompt, image, and seed:

1. `baseline_observe`: alpha=0, with the C2F matcher running read-only.
2. `c2f`: the exact frozen alpha=0.025 intervention.

Only steps through the end of the active window are needed. The default replay stops after scheduler step 29 and returns a latent, avoiding the last 20 transformer steps and final VAE decode. Before the full replay, one case per mode must show exact step-29 latent equality between the frozen pipeline and the diagnostic pipeline.

The primary predictors come from `baseline_observe`, because they exist before C2F changes the trajectory. Guided-minus-observe diagnostic changes are secondary evidence about amplification or recovery.

## Locked Diagnostic Families

The diagnostic pipeline must not alter matching or blending. It records per active layer and step:

- intervention coverage and gated cosine confidence;
- selected history lag and spatial displacement;
- fine and coarse top-1/top-2 similarity margins;
- coarse reciprocal rate and coarse round-trip displacement;
- direct lag-k match versus a chain of lag-1 matches (`path_cycle_error`);
- agreement of image-plane velocity implied independently by lags 1, 2, and 3 (`velocity_rms`);
- selected source token's own prior-match confidence (`source_previous_confidence`);
- fraction of selected generated sources that fail the frozen threshold plus mutual gate (`source_unreliable_active_fraction`);
- matched-value versus current attention-context residual and implied update magnitude;
- a deterministic, evenly spaced sample of at most 256 active tokens per layer-step.

These are internal reliability proxies. They must not be described as ground-truth geometry.

## Hypotheses And Priority

Primary H1, source propagation:

Higher `source_unreliable_active_fraction` predicts a more positive MEt3R-1s delta (greater harm).

Primary H2, correspondence consistency:

Higher `path_cycle_error` or `velocity_rms` predicts a more positive delta.

Primary H3, history age:

A larger fraction of lag-2/lag-3 retrieval predicts a more positive delta after accounting for motion stratum.

Secondary H4, non-selective strength:

Coverage, confidence, or update magnitude alone will not separate winners from losers as consistently as source/correspondence reliability.

## Analysis

For every locked scene-level predictor, report:

- winner and loser median plus interquartile range;
- Spearman correlation with continuous MEt3R-1s delta;
- deterministic permutation interval/test where practical;
- leave-one-scene-out sign stability;
- motion-stratum-demeaned association;
- sensitivity after excluding near ties and the single largest absolute outcome.

Token samples are used for distributions and mechanism visualization, not as independent scene-level observations. Statistical uncertainty is clustered at scene level.

P0 is exploratory. Association can nominate a mechanism, but it cannot establish final generalization.

## Locked Counterfactual Subset

Before diagnostics are inspected, select exactly two scenes per motion stratum: the most improved and most harmed MEt3R-1s scene. This produces ten motion-balanced forensic cases.

The first causal counterfactual is deliberately simple:

`source-valid gate = keep a source if it is the conditioning frame OR its own previous retrieval passed the existing threshold and reciprocal gate`

No new confidence threshold is tuned. Compare frozen C2F against source-valid C2F on the ten locked cases. Also retain `lag1-only` as a supporting age ablation if H3 is stronger than H1/H2. Every generated counterfactual must pass visual usability review before metrics.

## Decision Rule

P0 ends with a mechanism table ranking source reliability, path/cycle consistency, history age, target residual, intervention coverage, and motion regime. A factor is promoted to P1 only when:

1. its direction is consistent in the primary and robustness analyses;
2. it is not explained solely by one scene or one motion stratum; and
3. deleting interventions identified by that factor improves the locked harmful cases without broad visual degradation.

If no factor meets these conditions, do not build a compound Risk-Retrieve-Repair method from Dev25. The correct result would be that current internal diagnostics do not explain C2F heterogeneity.
