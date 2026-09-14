# C2F K3 alpha=0.025 validation

This directory freezes and validates the existing Wan C2F temporal value-memory
candidate. It does not change the C2F pipeline implementation.

## Scientific split

- Development: 25 cases selected deterministically from the existing 32-case
  DL3DV pilot. These cases have existing Wan seed-0 baselines.
- Final validation: the held-out 100-scene validation manifest is not used for
  tuning. It is touched only after the method and all thresholds are frozen.

The development subset contains five cases from each motion stratum:
`forward`, `forward_left`, `forward_right`, `lateral_left`, and
`lateral_right`. Within each stratum, five quantile positions are selected after
sorting by the precomputed GT trajectory selection score. Selection never uses
generated videos or method metrics.

## Frozen C2F candidate

- Wan2.2-TI2V-5B
- 121 frames, 704x1280, 50 steps, 24 FPS, CFG 5.0
- no negative prompt
- `c2f_value_residual_memory`
- layers 10, 15, 20
- denoising steps 20 through 29, inclusive
- alpha 0.025
- temporal lookback K=3
- coarse factor 2, local refinement radius 2
- cosine confidence threshold 0.4
- mutual correspondence required
- conditional CFG branch only
- first-frame token is never overwritten

## Gates

1. Generate one alpha=0 sample and compare it to the existing official Wan
   seed-0 baseline.
2. Generate all 25 C2F samples and create visual comparisons before metrics.
3. Reject corrupt, artifacted, semantically failed, or motion-collapsed videos.
4. Only then run the locked metric suite and paired statistics.

The final 100-scene run must compare against the existing seed-0 baseline, not
the four-seed baseline mean.
