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

Download and validate the frozen scenes on Hippasus first. The downloader pins
the Hugging Face dataset commit and split checksum, validates every referenced
PNG, and can be resumed safely:

```bash
bash benchmarks/dl3dv_geco/download_on_hippasus.sh debug validation
```

Prompt/trajectory manifests are built on Hippasus from the downloaded
`transforms.json` files. Only the frozen experiment package is then copied to
Isambard for video generation.

The scene split is already frozen in
`splits/frozen_scene_split_3_100_100.csv`. Build one GT-grounded trajectory per
requested scene without reallocating scenes between splits:

```bash
python benchmarks/dl3dv_geco/build_manifest.py \
  --roots /vol/dissolve/yz10325/datasets/dl3dv-1k/480P/1K \
  --frozen-split-csv benchmarks/dl3dv_geco/splits/frozen_scene_split_3_100_100.csv \
  --splits validation \
  --scene-descriptions-json /path/to/reviewed_validation_descriptions.json \
  --output /path/to/dl3dv_validation_manifest_480p.json \
  --pose-window 81 \
  --start-stride 8
```

Download only debug and validation during development. Download and build the
100 held-out test scenes only after the method code and configuration are
frozen. The builder fails if any requested scene lacks an eligible large-motion
trajectory; it never substitutes a scene from another split.

The builder output is a reviewed trajectory source, not yet the immutable
formal protocol. `extract_conditioning_frames.py` creates one transferable
scene directory containing the selected 960P frame and its matching
`transforms.json`. After the method, test inputs, model lock, and all 203 cases
are frozen, create the formal protocol without reallocating any scene:

```bash
python benchmarks/dl3dv_geco/freeze_protocol_split.py \
  --source-manifest /path/to/all_203_packaged_cases.json \
  --dataset-root /path/to/transferable_dl3dv_package \
  --model-lock /path/to/model_lock.json \
  --preserve-source-splits \
  --output /path/to/dl3dv_formal_protocol.json
```

This final step converts absolute preparation paths to dataset-relative paths,
binds image/transform hashes and the model lock, and emits the schema consumed
by frozen generation jobs. Validation development must not be described as the
formal test protocol before this step is complete.

Debug cases are used only for correctness, runtime, memory, and qualitative
sanity checks. The validation set is used for all method and hyperparameter
decisions. The test set is run only after the code and configuration are
frozen. All prompts, conditioning frames, and paired seed policies are frozen
with the split.

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
