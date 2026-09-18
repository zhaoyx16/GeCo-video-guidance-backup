# DL3DV GeCo Benchmark

This directory defines a dataset-level benchmark for paired video-generation
experiments. It is intentionally separate from the hand-selected failure cases
used during early method development.

## Experimental unit

Every manifest row fixes:

- DL3DV conditioning image;
- camera-only prompt derived from the ground-truth camera trajectory;
- seed;
- backbone configuration;
- method and method parameters.

Within one backbone, baseline, GeCo, and proposed methods must use the same
image, prompt, seed, number of steps, resolution, frame count, FPS, scheduler,
CFG scale, and negative prompt.

Wan and Cosmos use their own recommended generation configurations and are
reported in separate tables. Scores must not be pooled across backbones.

## Prompt policy

Prompts describe only coarse camera motion. They do not:

- name a particular object that the camera must pass;
- claim that all objects are stationary;
- ask for a static camera;
- prescribe a turn at a guessed doorway or junction;
- contain method-specific anti-failure language.

The scene content comes from the conditioning image. The trajectory class comes
from DL3DV camera poses.

## Build a manifest

```bash
python benchmarks/dl3dv_geco/build_manifest.py \
  --roots /path/to/DL3DV/1K \
  --output benchmarks/dl3dv_geco/manifests/dl3dv_1k_dev.json \
  --pose-window 121 \
  --start-stride 24 \
  --max-clips 32 \
  --max-per-scene 1 \
  --large-motion-quantile 0.65
```

Recommended splits:

- correctness gate: 2 scenes x 1 seed;
- development: 32 scenes x 2 seeds;
- main: 100 scenes x 3 seeds.

The split and all prompts are frozen before comparing methods.

## GeCo migration gate

Run the static source lint first. It catches known source patterns but does not
prove correctness:

```bash
python tests/geco_port/validate_port_static.py --repo .
```

Then submit the isolated runtime gate. The submitter creates a fresh result
directory, pins downloaded model snapshots, runs two official controls plus
the zero-guidance custom pipeline, and wires dependencies automatically:

```bash
CORRECTNESS_IMAGE=/absolute/path/to/a/dl3dv/frame.png \
  bash benchmarks/dl3dv_geco/slurm/submit_correctness.sh
```

Large-scale GeCo jobs are blocked until:

1. zero-guidance custom latents match the official Diffusers pipeline;
2. the differentiable guidance prediction matches ordinary sampling at the
   same latent and timestep;
3. x0 uses the scheduler's current sigma;
4. a small normalized update lowers freshly recomputed loss;
5. the conditioning frame/region is not updated;
6. fixed seeds are repeatable.

Original GeCo time-travel is a DDIM-specific stabilization procedure. Its
absence in the flow-matching ports is reported as a controlled method
difference, not silently treated as equivalent.

## Dataset-level table

At minimum report mean, median, bootstrap 95% confidence interval, and paired
win rate for:

| Backbone | Method | MEt3R-0.5s | MEt3R-1s | First-last | GeCo Fused | Motion retention | VBench-Q |
|---|---|---:|---:|---:|---:|---:|---:|

The method-selection criterion is frozen before the main split. Videos shown
qualitatively are selected after the aggregate table and include both strong
improvements and representative failures.
