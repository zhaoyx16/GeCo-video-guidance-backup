# Navigation Baseline Screen

This branch builds and runs a pose-matched navigation-only baseline screen.

## Experimental contract

- Every prompt is derived from a real source trajectory.
- DL3DV cases use camera poses from `transforms.json`.
- SCAND/VAMOS cases use the provided future `trajectory_3d`.
- All start frames must show a physically navigable route.
- SCAND cases remain pending until the first frame is confirmed to contain no
  people, moving vehicles, or other dominant dynamic content.
- A prompt describes camera motion only. It does not leak future RGB content.
- All start frames from the same DL3DV scene remain in the same split.

## Official generation profiles

Wan2.2-TI2V-5B:

```text
50 steps, 121 frames, 704x1280, 24 fps
```

Cosmos-Predict2.5-2B post-trained:

```text
36 steps, 93 frames, 704x1280, 16 fps
```

The same case, image, prompt, and seed are used across the two backbones. Frame
count and FPS remain model-specific rather than being artificially equalized.

## Outputs

The manifest builder writes:

```text
navigation_cases.json
jobs.jsonl
summary.json
trajectory_report.csv
trajectory_contact_sheets/
```

The worker writes:

```text
baselines/<model>/<dataset>/<case>/
logs/<model>/
job_state/*.done.json
job_state/*.failed.json
```

Workers are resumable and skip any non-empty expected output video.
