#!/bin/bash
#SBATCH --job-name=wan-val-candidates
#SBATCH --account=brics.u6ph
#SBATCH --output=/projects/u6ph/yanxi.u6ph/logs/wan_val_%A_%a.log
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --array=0-399%16

set -euo pipefail

PROJECT_ROOT=/projects/u6ph/yanxi.u6ph
CONTAINER_PATH=${PROJECT_ROOT}/containers/pytorch_25.05-py3.sif
REPO_INNER=/vol/dissolve/yz10325/repos/GeCo-geometry-selection
VENV_INNER=/vol/dissolve/yz10325/envs/geco-geometry-selection
MODEL_INNER=/vol/dissolve/yz10325/geco_models/frozen/wan-b8fff7315c768468a5333511427288870b2e9635
MANIFEST_INNER=/vol/dissolve/yz10325/datasets/dl3dv-1k/metadata/validation_gt_large_motion_manifest_960p_v1.json
OUTPUT_INNER=/vol/dissolve/yz10325/outputs/geometry-selection/validation_wan_candidates
MODEL_LOCK_INNER=/vol/dissolve/yz10325/geco_models/refs/wan_geometry_model_lock_6c37182.json
MANIFEST_SHA256=da4c05c0ec8f6f8fd08daf3a69482c631d221f84fcb7a5d251c1e3fb1509dd9d
MODEL_LOCK_SHA256=71cd88a83a67ab59d02ec1dbf1b4f76344e3da7d290f161423b21aa18cdc7648

: "${EXPECTED_COMMIT:?submit with --export=ALL,EXPECTED_COMMIT=<clean commit>}"
readonly SEEDS=(0 1 2 3)
readonly EXPECTED_CASES=100
CANDIDATE_COUNT=${#SEEDS[@]}
EXPECTED_TASKS=$((EXPECTED_CASES * CANDIDATE_COUNT))
if [[ ! "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]]; then
  echo "SLURM_ARRAY_TASK_ID must be a non-negative integer" >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= EXPECTED_TASKS )); then
  echo "Task ${SLURM_ARRAY_TASK_ID} is outside [0,$((EXPECTED_TASKS - 1))]" >&2
  exit 2
fi
CASE_INDEX=$((SLURM_ARRAY_TASK_ID / CANDIDATE_COUNT))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % CANDIDATE_COUNT))
SEED=${SEEDS[SEED_INDEX]}

apptainer exec \
  --nv \
  --cleanenv \
  --bind "${PROJECT_ROOT}:/vol/dissolve/yz10325" \
  --env "PYTHONPATH=${REPO_INNER}" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  --env "CASE_INDEX=${CASE_INDEX}" \
  --env "CANDIDATE_SEED=${SEED}" \
  --env "EXPECTED_COMMIT=${EXPECTED_COMMIT}" \
  --env "MANIFEST_SHA256=${MANIFEST_SHA256}" \
  --env "MODEL_LOCK_SHA256=${MODEL_LOCK_SHA256}" \
  "${CONTAINER_PATH}" \
  /bin/bash -lc '
    set -euo pipefail
    source "$1/bin/activate"
    cd "$2"
    test "$(git rev-parse HEAD)" = "${EXPECTED_COMMIT}"
    test -z "$(git status --porcelain)"
    python scripts/validate_development_generation_inputs.py \
      --manifest "$3" \
      --manifest-sha256 "${MANIFEST_SHA256}" \
      --expected-split validation \
      --expected-cases 100 \
      --case-index "${CASE_INDEX}" \
      --model "$4" \
      --model-lock "$6" \
      --model-lock-sha256 "${MODEL_LOCK_SHA256}" \
      --model-profile Wan2.2-TI2V-5B
    python benchmarks/dl3dv_geco/run_generation_case.py \
      --backbone wan \
      --method baseline \
      --manifest "$3" \
      --protocol-mode legacy-debug \
      --case-index "${CASE_INDEX}" \
      --model "$4" \
      --output-root "$5" \
      --seed "${CANDIDATE_SEED}" \
      --steps 50 \
      --frames 121 \
      --height 704 \
      --width 1280 \
      --fps 24 \
      --guidance-scale 5.0 \
      --wan-negative-prompt-mode none
  ' bash "${VENV_INNER}" "${REPO_INNER}" "${MANIFEST_INNER}" \
    "${MODEL_INNER}" "${OUTPUT_INNER}" "${MODEL_LOCK_INNER}"
