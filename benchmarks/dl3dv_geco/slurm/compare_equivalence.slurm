#!/bin/bash
#SBATCH --job-name=geco_compare
#SBATCH --account=brics.u6ph
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00

set -uo pipefail

PROJECT="${HOME}/projects/GeCo-71041f9"
CONTAINER="/projects/public/brics/containers/e4s/e4s-cuda90-aarch64-25.11.sif"
VENV="${HOME}/venvs/geco-e4s"
RESULTS="${RUN_DIR:?RUN_DIR must be exported by submit_correctness.sh}"

WAN_RC=0
COSMOS_RC=0

apptainer exec --bind "${HOME}:${HOME},/projects:/projects" "${CONTAINER}" \
    "${VENV}/bin/python" "${PROJECT}/tests/geco_port/baseline_equivalence.py" compare \
    --reference "${RESULTS}/wan_official_a.pt" \
    --control "${RESULTS}/wan_official_b.pt" \
    --custom "${RESULTS}/wan_custom.pt" \
    --report "${RESULTS}/wan_comparison.json" || WAN_RC=$?

apptainer exec --bind "${HOME}:${HOME},/projects:/projects" "${CONTAINER}" \
    "${VENV}/bin/python" "${PROJECT}/tests/geco_port/baseline_equivalence.py" compare \
    --reference "${RESULTS}/cosmos_official_a.pt" \
    --control "${RESULTS}/cosmos_official_b.pt" \
    --custom "${RESULTS}/cosmos_custom.pt" \
    --report "${RESULTS}/cosmos_comparison.json" || COSMOS_RC=$?

echo "wan_rc=${WAN_RC} cosmos_rc=${COSMOS_RC}"
if (( WAN_RC != 0 || COSMOS_RC != 0 )); then
    exit 1
fi
