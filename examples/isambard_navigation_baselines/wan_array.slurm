#!/bin/bash
#SBATCH --job-name=wan-nav-base
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=slurm/wan_%A_%a.out
#SBATCH --error=slurm/wan_%A_%a.err

set -euo pipefail

: "${GECO_REPO:?Set GECO_REPO to the GeCo checkout}"
: "${NAV_PACKAGE:?Set NAV_PACKAGE to the portable manifest directory}"
: "${NAV_OUTPUT:?Set NAV_OUTPUT to the output directory}"
: "${WAN_MODEL_PATH:?Set WAN_MODEL_PATH to the Wan2.2 TI2V Diffusers checkpoint}"

mkdir -p "${NAV_OUTPUT}" slurm

python "${GECO_REPO}/scripts/run_navigation_baseline_worker.py" \
  --jobs "${NAV_PACKAGE}/jobs.jsonl" \
  --output-root "${NAV_OUTPUT}" \
  --repo "${GECO_REPO}" \
  --worker-index "${SLURM_ARRAY_TASK_ID}" \
  --num-workers "${SLURM_ARRAY_TASK_COUNT}" \
  --model wan \
  --wan-model "${WAN_MODEL_PATH}"
