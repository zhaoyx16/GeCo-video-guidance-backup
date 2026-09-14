# Preregistered C2F development gate

This gate is frozen before inspecting aggregate dev25 metrics. It decides
whether the existing C2F K=3, alpha=0.025 candidate is strong enough to justify
building a confidence-gated consensus successor. It is an engineering and
method-selection gate, not a final publication claim.

## Analysis population

- Use all 25 paired seed-0 scenes in `dev25_manifest.json`.
- Every pair must use the same prompt, conditioning image, seed, Wan generation
  settings, host class, and physical-GPU shard.
- Development scene hashes must have zero overlap with frozen test,
  validation, and debug scene hashes.
- A case may be marked technically invalid only before metrics are opened, for
  corrupt decode, wrong frame count/resolution/FPS, catastrophic generation
  failure, or broken pairing/provenance.
- Do not exclude a valid case because its metric is unfavorable. Report both
  the all-generated set and any pre-metric visually usable subset if they
  differ.

## Primary endpoint

MEt3R at one-second separation, aggregated first across the five frozen pairs
within each scene and then across the 25 paired scenes. Lower is better.

The candidate passes the primary development signal only if all are true:

1. paired mean relative improvement is at least 2%;
2. paired median delta favors C2F;
3. scene-level win rate is at least 60%;
4. the paired-bootstrap 80% interval for directional improvement is above 0.

The 95% interval is reported but is not required to exclude 0 on this 25-scene
development gate.

## Secondary geometry evidence

Report MEt3R at 0.5 seconds, MEt3R first-to-last, independent long-range
reprojection error, and GeCo Fused. Lower is better for all four.

To advance, at least two of MEt3R-0.5s, MEt3R-first-last, and independent LRE
must improve in paired mean, and none may worsen by more than 5% in paired mean.
GeCo Fused is reported as a diagnostic rather than a sole gate because its
static-scene assumptions can be weak under long-range navigation.

## Anti-cheating and quality guardrails

- Relative Total Motion must remain in [90%, 110%] of the same-host official
  baseline. A lower geometry score obtained by freezing motion fails.
- Mean VBench Quality may not fall by more than 1% relative to baseline.
- Visual review may not show a method-induced increase in collapse, color
  artifacts, duplicated structures, or semantic failure.
- Mean runtime and peak allocated VRAM overhead must each remain at or below
  10% relative to the same-host official baseline.

## Decision

- **Advance:** primary endpoint, secondary evidence, and all guardrails pass.
- **Borderline:** primary direction is positive but one preregistered threshold
  misses narrowly; diagnose by motion stratum before changing the method.
- **Stop/falsify current C2F:** primary paired mean is non-improving, win rate is
  at most 50%, or a quality/motion guardrail fails materially.

Only an Advance result justifies implementing the proposed
confidence-gated correspondence-consensus memory. No threshold or C2F
hyperparameter is changed after opening aggregate dev25 metrics.
