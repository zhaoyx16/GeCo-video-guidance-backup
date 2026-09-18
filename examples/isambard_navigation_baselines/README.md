# Navigation Baseline Batch on Isambard AI

This package contains 100 paired navigation cases:

- 70 DL3DV cases selected from real camera trajectories.
- 30 manually reviewed SCAND cases without visible people.
- Seeds `0` and `1`.
- 200 Wan2.2 baselines and 200 Cosmos baselines.

The prompt direction follows the source trajectory while asking the camera to
move quickly over a large distance. Scene geometry and objects are required to
remain static.

## Fixed generation settings

| Model | Steps | Frames | Resolution | FPS |
|---|---:|---:|---:|---:|
| Wan2.2-TI2V-5B | 50 | 121 | 704x1280 | 24 |
| Cosmos Predict2.5-2B Post-trained | 36 | 93 | 704x1280 | 16 |

## Transfer

Transfer both the GeCo checkout and the portable input package. The package
contains all 100 first frames, so the original DL3DV and SCAND datasets are not
needed during generation.

```bash
rsync -avP navigation_baseline_isambard_20260730.tar.gz \
  USER@login.isambard.ac.uk:/path/to/project/
```

Extract it on Isambard:

```bash
tar -xzf navigation_baseline_isambard_20260730.tar.gz
```

## Environment

Activate the environment that contains the same tested PyTorch, Diffusers,
Transformers, Accelerate, and image/video dependencies as the Hippasus GeCo
environment. Then set:

```bash
export GECO_REPO=/path/to/GeCo
export NAV_PACKAGE=/path/to/navigation_baseline_isambard_20260730
export NAV_OUTPUT=/path/to/outputs/navigation_baselines_20260730
export WAN_MODEL_PATH=/path/to/Wan2.2-TI2V-5B-Diffusers
export COSMOS_MODEL_PATH=/path/to/Cosmos-Predict2.5-2B-diffusers-base-post-trained
```

Run one smoke job for each model before submitting arrays:

```bash
python "$GECO_REPO/scripts/run_navigation_baseline_worker.py" \
  --jobs "$NAV_PACKAGE/jobs.jsonl" \
  --output-root "$NAV_OUTPUT" \
  --repo "$GECO_REPO" \
  --worker-index 0 \
  --num-workers 1 \
  --model wan \
  --wan-model "$WAN_MODEL_PATH" \
  --limit 1

python "$GECO_REPO/scripts/run_navigation_baseline_worker.py" \
  --jobs "$NAV_PACKAGE/jobs.jsonl" \
  --output-root "$NAV_OUTPUT" \
  --repo "$GECO_REPO" \
  --worker-index 0 \
  --num-workers 1 \
  --model cosmos \
  --cosmos-model "$COSMOS_MODEL_PATH" \
  --limit 1
```

Inspect both smoke videos. If they are valid, submit:

```bash
cd "$GECO_REPO"
sbatch examples/isambard_navigation_baselines/wan_array.slurm
sbatch examples/isambard_navigation_baselines/cosmos_array.slurm
```

The worker is resumable. Existing non-empty MP4 files are skipped, while each
job writes a JSON state file and a separate log.

Adjust only the site-specific Slurm account, partition/QoS, array width, and
wall time. Do not change model sampling settings if the outputs are intended
for the paired baseline benchmark.
