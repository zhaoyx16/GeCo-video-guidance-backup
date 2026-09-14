# C2F K3 alpha=0.025 validation

This directory freezes and validates the existing Wan C2F temporal value-memory
candidate. It does not change the C2F pipeline implementation.

## Scientific split

- Development: 25 cases selected deterministically from the existing 32-case
  DL3DV pilot. These cases have existing Wan seed-0 baselines.
- Frozen test: scenes 1--100 in the sealed scene order.
- Final validation: scenes 101--200 in the sealed scene order. This split is
  not used for tuning and is touched only after the method is frozen.
- Debug: scenes 201--203 in the sealed scene order.

The selected development scene hashes have zero overlap with all three reserved
splits. The selection manifest records that audit and every generation/evaluation
launcher refuses a manifest that does not report exactly zero overlap.

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

1. Generate alpha=0 and official Wan controls on the same host/GPU/software.
2. Require pixel-exact equality between that alpha=0 output and official Wan.
3. Generate all 25 official/C2F pairs on the same host and physical GPU shard.
4. Create 11-frame contact sheets and full side-by-side videos before metrics.
5. Reject corrupt, artifacted, semantically failed, or motion-collapsed videos.
6. Materialize immutable evaluator locks and only then run the frozen metric
   schedule and paired statistics.

The alpha=0 control passes exactly: 121/121 frames have zero pixel difference
and the encoded MP4 SHA is identical. A historical A100 baseline differs from
the same-host official Blackwell output despite identical high-level generation
parameters. It therefore cannot be used for a strict pixel-paired development
comparison. This validation generates a fresh same-host official reference for
every dev case; this is a control for hardware/software trajectory differences,
not an extra candidate or a selected best-of-N baseline.

Before a final 100-scene run, either generate C2F in the historical baseline's
exact runtime/hardware environment or generate one same-host official reference
per case. In either design, compare paired seed-0 outputs; never treat the
four-seed baseline mean as the paired video.

## Utilities

- `run_c2f_batch.py`: provenance-checked sharded official/C2F generation.
- `make_dev25_visuals.py`: all-pair visual gate using 11 fixed time points plus
  a full-resolution side-by-side video.
- `build_dev_metric_inputs.py`: immutable 25-case locks for both paired methods.
- `run_metric_dev25.py`: 25-case wrapper around the SHA-pinned evaluator-v13
  launcher and frozen metric schedule.
