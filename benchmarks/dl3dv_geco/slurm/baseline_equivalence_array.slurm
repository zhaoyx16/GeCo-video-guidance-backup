#!/bin/bash
#SBATCH --job-name=geco_equiv
#SBATCH --account=brics.u6ph
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-5%6
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT="${HOME}/projects/GeCo-71041f9"
CONTAINER="/projects/public/brics/containers/e4s/e4s-cuda90-aarch64-25.11.sif"
VENV="${HOME}/venvs/geco-e4s"
HF_HOME="/projects/u6ph/${USER}/huggingface"
MODEL_REFS="/projects/u6ph/${USER}/geco_models/model_refs.sh"
RESULTS="${RUN_DIR:?RUN_DIR must be exported by submit_correctness.sh}"
IMAGE="${CORRECTNESS_IMAGE:?CORRECTNESS_IMAGE must be exported by submit_correctness.sh}"
PROMPT="A continuous first-person camera shot beginning from the provided image. The camera moves smoothly forward over a long distance through the environment with a large viewpoint change."

mkdir -p "${RESULTS}" "${PROJECT}/logs"
test -f "${IMAGE}"
test -f "${MODEL_REFS}"
source "${MODEL_REFS}"
export HF_HOME
export HF_HUB_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        BACKBONE="wan"
        ACTION="official"
        RUN_LABEL="official_a"
        MODEL="${WAN_MODEL}"
        ;;
    1)
        BACKBONE="wan"
        ACTION="official"
        RUN_LABEL="official_b"
        MODEL="${WAN_MODEL}"
        ;;
    2)
        BACKBONE="wan"
        ACTION="custom"
        RUN_LABEL="custom"
        MODEL="${WAN_MODEL}"
        ;;
    3)
        BACKBONE="cosmos"
        ACTION="official"
        RUN_LABEL="official_a"
        MODEL="${COSMOS_MODEL}"
        ;;
    4)
        BACKBONE="cosmos"
        ACTION="official"
        RUN_LABEL="official_b"
        MODEL="${COSMOS_MODEL}"
        ;;
    5)
        BACKBONE="cosmos"
        ACTION="custom"
        RUN_LABEL="custom"
        MODEL="${COSMOS_MODEL}"
        ;;
esac

OUTPUT="${RESULTS}/${BACKBONE}_${RUN_LABEL}.pt"

apptainer exec --nv --bind "${HOME}:${HOME},/projects:/projects" "${CONTAINER}" \
    "${VENV}/bin/python" "${PROJECT}/tests/geco_port/baseline_equivalence.py" \
    "${ACTION}" \
    --backbone "${BACKBONE}" \
    --repo "${PROJECT}" \
    --model "${MODEL}" \
    --image "${IMAGE}" \
    --prompt "${PROMPT}" \
    --output "${OUTPUT}" \
    --run-label "${RUN_LABEL}" \
    --device cuda \
    --seed 123 \
    --steps 5 \
    --frames 9 \
    --height 256 \
    --width 448
