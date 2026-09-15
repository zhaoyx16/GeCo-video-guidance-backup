# Stage A Findings: External Geometry Validation of Frozen C2F

## Decision

**Conditional pass for the preregistered Stage B diagnostic experiment.**

Stage A supports the narrow claim that the pinned VGGT-Omega geometry can be
used as a selective, abstaining validator of frozen C2F correspondences. It
does not establish that the geometry model is ground truth, that geometry
gating improves generated video, or that rejected matches are always wrong.

## Locked evidence

- Five Dev25 scenes were selected without C2F outcomes, one from each frozen
  motion stratum.
- Each scene used the baseline-observe trajectory at step 20, layer 10, with
  target temporal tokens 10 and 25 and 8,192 deterministic diagnostic samples.
- The geometry checkpoint, preprocessing, camera convention, coordinate map,
  and decision thresholds were fixed before inspecting outputs.
- All five dense replays completed at about 185 seconds and 28,471 MiB peak
  allocated memory per case.
- VGGT-Omega loaded in 18.06 seconds. Geometry forward took 0.47--1.05 seconds
  per scene and peaked at 5,724.7 MiB.

Across the ten locked target sets, mean valid-evidence coverage was 57.5% and
mean acceptance among valid evidence was 34.6%. The gate therefore behaved
selectively instead of accepting every C2F retrieval or rejecting everything.

## Manual visual QA

The overview and all ten full-resolution panels were inspected.

1. There is no systematic horizontal/vertical flip, scale error, or global
   offset. Accepted C2F target circles and geometry projection crosses are
   locally aligned on visible surfaces across all five motion strata.
2. Long rejected residuals are spatially coherent with camera motion and
   commonly connect a C2F target to a clearly different projected location;
   they do not look like a single coordinate-convention failure.
3. Occlusion abstentions occur in strongly shifted or newly revealed regions,
   while low-confidence/invalid regions are not assigned a hard geometry
   label.
4. Late, strongly blurred views are a clear limitation. In forward-left t=25
   and lateral-left t=25, evidence coverage fell to 21.2% and 27.5%, and no
   correspondence was accepted. These cases must not be interpreted as proof
   that every rejected C2F match is geometrically wrong.

## Consequence for Stage B

Use the external signal only as a hard positive validation gate:

```text
accepted visible correspondence -> permit the frozen C2F value residual
reject or abstain             -> apply no value residual
```

Do not use negative geometry corrections. Report geometry coverage and the
fraction of C2F interventions that survive. Compare against a uniform-weakened
control whose instantaneous update norm is matched to the gated update, so a
gain cannot be attributed merely to reducing intervention strength.

Stage B remains a method experiment. Its required comparisons are G-C, G-U,
and G-B; a result where G only recovers baseline but does not beat it means the
gate reduced C2F harm, not that it improved the frozen generator.
