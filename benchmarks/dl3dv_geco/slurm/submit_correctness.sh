#!/bin/bash
set -euo pipefail

PROJECT="${HOME}/projects/GeCo-71041f9"
SBATCH_DIR="${PROJECT}/benchmarks/dl3dv_geco/slurm"
RESULTS_ROOT="/projects/u6ph/${USER}/geco_results/correctness"
CORRECTNESS_IMAGE="${CORRECTNESS_IMAGE:?Set CORRECTNESS_IMAGE to a staged DL3DV frame}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"

test -f "${CORRECTNESS_IMAGE}"
mkdir -p "${LOG_DIR}"

SETUP_JOB="$(
    sbatch --parsable \
        --output="${LOG_DIR}/setup_%j.out" \
        --error="${LOG_DIR}/setup_%j.err" \
        "${SBATCH_DIR}/setup_env.slurm"
)"
PREFETCH_JOB="$(
    sbatch --parsable \
        --dependency="afterok:${SETUP_JOB}" \
        --output="${LOG_DIR}/prefetch_%j.out" \
        --error="${LOG_DIR}/prefetch_%j.err" \
        "${SBATCH_DIR}/prefetch_models.slurm"
)"
EQUIV_JOB="$(
    sbatch --parsable \
        --dependency="afterok:${PREFETCH_JOB}" \
        --export="ALL,RUN_DIR=${RUN_DIR},CORRECTNESS_IMAGE=${CORRECTNESS_IMAGE}" \
        --output="${LOG_DIR}/equivalence_%A_%a.out" \
        --error="${LOG_DIR}/equivalence_%A_%a.err" \
        "${SBATCH_DIR}/baseline_equivalence_array.slurm"
)"
COMPARE_JOB="$(
    sbatch --parsable \
        --dependency="afterok:${EQUIV_JOB}" \
        --export="ALL,RUN_DIR=${RUN_DIR}" \
        --output="${LOG_DIR}/compare_%j.out" \
        --error="${LOG_DIR}/compare_%j.err" \
        "${SBATCH_DIR}/compare_equivalence.slurm"
)"

cat <<EOF
run_dir=${RUN_DIR}
setup_job=${SETUP_JOB}
prefetch_job=${PREFETCH_JOB}
equivalence_job=${EQUIV_JOB}
compare_job=${COMPARE_JOB}
EOF
