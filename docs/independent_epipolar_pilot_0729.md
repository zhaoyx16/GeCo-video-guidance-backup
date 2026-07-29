# Independent Epipolar Evaluation Pilot (2026-07-29)

## Question

Can a geometry diagnostic that does not reuse VGGT, UFM, or generator
features distinguish a real static-scene navigation clip from known failures,
while refusing to reward a frozen video?

This is evaluator calibration, not a comparison of geometry-guidance methods.

## Evaluator

`opencv-sift-magsac-epipolar`, version 2:

- SIFT descriptors with reciprocal Lowe-ratio matching;
- deterministic fit/held-out correspondence split;
- MAGSAC fundamental-matrix estimation on fit matches;
- capped Sampson error and F inlier ratio on held-out matches;
- a competing homography fit to identify planar/pure-rotation degeneracy;
- separate feature coverage, match coverage, parallax observability, and
  conditional geometry statistics.

The evaluator abstains on low-motion and homography-degenerate pairs. Failed F
estimation on an otherwise observable pair receives the configured error cap.

## Data And Controls

Reference:

- KITTI Odometry sequence 00, frames 80 through 130;
- 51 frames at 10 fps;
- stretched to 1280x704 before encoding, matching the current Wan
  `VideoProcessor.preprocess` behavior;
- H.264/yuv420p encoding, then decoded before evaluation.

Controls:

- `control_frozen`: the first reference frame repeated;
- `control_nonrigid`: a time-varying 24-pixel sinusoidal horizontal warp.

Existing Frame-Guidance pilot outputs:

- `fg_lr5_sparse`: `cfgcb7f7957430b`;
- `fg_lr15_contiguous`: `cfgd73d251043af`.

Configuration:

- lags: 0.5 and 1.0 seconds, exactly representable at 10 and 24 fps;
- 8 deterministic pairs per lag;
- maximum evaluation side: 960 pixels;
- minimum reciprocal matches: 24;
- held-out fraction: 0.3;
- held-out F inlier threshold: 1.5 pixels;
- capped Sampson error: 5 pixels.

## Calibration Result

Overall values pool the two lag-level clip summaries for this pilot only.
They are not independent statistical samples.

| Video | Parallax observable | Held-out F inlier (higher) | Failure-aware capped error px (lower) | Match motion ratio |
|---|---:|---:|---:|---:|
| Reference GT | 1.000 | 0.935 | 0.515 | 0.122 |
| Frozen control | 0.000 | n/a | n/a | 0.000 |
| Non-rigid control | 1.000 | 0.761 | 1.264 | 0.119 |
| FG lr=5 sparse | 1.000 | 0.792 | 1.127 | 0.070 |
| FG lr=15 contiguous | 1.000 | 0.810 | 1.101 | 0.068 |

The controls pass the intended calibration:

- the frozen video does not receive a good geometry score; it is declared
  unobservable;
- the non-rigid warp retains almost the same motion magnitude as GT but reduces
  held-out F inliers and increases capped error by about 2.45x;
- both generated videos score below the real reference.

The two Frame-Guidance settings are too close to rank from one clip. Their
motion ratio is also only about 56% of the real reference. A later geometry
comparison must therefore report GT-bound trajectory adherence and motion
preservation together with epipolar consistency.

## Text-Only Seed Diagnostic

Three existing Wan and three existing Cosmos text-only baselines from the same
KITTI start frame were also evaluated. This is a sensitivity check, not a fair
model comparison because the videos do not share Frame-Guidance anchors.

| Model | Seeds | Mean held-out F inlier | Mean capped error px | Mean motion ratio |
|---|---:|---:|---:|---:|
| Wan2.2 TI2V-5B | 3 | 0.756 | 1.252 | 0.0160 |
| Cosmos Predict2.5-2B | 3 | 0.940 | 0.558 | 0.0158 |

The low motion ratios make it invalid to conclude that Cosmos has better
navigation geometry from these values alone. Smoother imagery and reduced
camera motion can both make feature geometry easier.

## Cost

- final calibration (five videos including encoded controls): 16.6 seconds
  wall-clock, about 1.01 GiB peak host RAM, CPU only;
- six text-only videos: 15.4 seconds wall-clock, about 676 MiB peak host RAM,
  CPU only;
- no GPU memory was used.

## Artifacts

- `/vol/dissolve/yz10325/outputs/navigation_evaluation_0729/epipolar_v2_final_kitti80/calibration.json`
- `/vol/dissolve/yz10325/outputs/navigation_evaluation_0729/epipolar_v2_final_kitti80/calibration.csv`
- `/vol/dissolve/yz10325/outputs/navigation_evaluation_0729/epipolar_v2_final_kitti80/controls`
- `/vol/dissolve/yz10325/outputs/navigation_evaluation_0729/epipolar_v2_kitti80_textonly_seeds/report.json`

## Before A Paper Table

1. Bind evaluator inputs to the frozen benchmark manifest and require complete
   FG-only / FG+RGB-GeCo / FG+latent pairs.
2. Run scene-disjoint KITTI and DL3DV clips with multiple seeds.
3. Report GT trajectory ATE/RPE or scale-free relative-pose errors and motion
   ratio beside every consistency metric.
4. Add an independent temporal track survival/cycle metric.
5. Add multiple non-rigid severities and codec-matched blur/content-loss
   controls.
6. Aggregate by source sequence or scene with paired cluster bootstrap; never
   treat frame pairs as independent samples.
