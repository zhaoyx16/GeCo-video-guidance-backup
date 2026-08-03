#!/bin/bash
#SBATCH --job-name=geco-models
#SBATCH --account=brics.u6ph
#SBATCH --output=/projects/u6ph/yanxi.u6ph/logs/geco_models_%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_ROOT=/projects/u6ph/yanxi.u6ph
CONTAINER_PATH=${PROJECT_ROOT}/containers/pytorch_25.05-py3.sif
REPO_INNER=/vol/dissolve/yz10325/repos/GeCo-geometry-selection
VENV_INNER=/vol/dissolve/yz10325/envs/geco-geometry-selection
HF_HOME_INNER=/vol/dissolve/yz10325/cache/huggingface
HF_HUB_CACHE_INNER=${HF_HOME_INNER}/hub
MODEL_KEYS=${MODEL_KEYS:-wan}

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/cache/huggingface" \
    "${PROJECT_ROOT}/geco_models/frozen" "${PROJECT_ROOT}/geco_models/refs"

apptainer exec \
    --cleanenv \
    --bind "${PROJECT_ROOT}:/vol/dissolve/yz10325" \
    --env "HF_HOME=${HF_HOME_INNER}" \
    --env "HF_HUB_CACHE=${HF_HUB_CACHE_INNER}" \
    --env "HF_XET_HIGH_PERFORMANCE=1" \
    --env "PYTHONPATH=${REPO_INNER}" \
    "${CONTAINER_PATH}" \
    "${VENV_INNER}/bin/python" \
    "${REPO_INNER}/scripts/prefetch_models.py" \
    --models "${MODEL_KEYS}" \
    --cache-dir "${HF_HUB_CACHE_INNER}" \
    --model-root /vol/dissolve/yz10325/geco_models

echo "MODEL_PREFETCH_READY ${MODEL_KEYS}"
