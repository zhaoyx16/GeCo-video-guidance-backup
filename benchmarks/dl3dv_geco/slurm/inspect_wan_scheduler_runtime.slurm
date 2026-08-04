#!/bin/bash
# Read-only preflight: load the isolated adapted-Wan pipeline and print its scheduler identity.
#SBATCH --job-name=wan-geco-sched
#SBATCH --account=brics.u6ph
#SBATCH --output=/projects/u6ph/yanxi.u6ph/logs/wan_geco_scheduler_preflight_%j.log
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00

set -euo pipefail

readonly PROJECT_ROOT=/projects/u6ph/yanxi.u6ph
readonly CONTAINER_PATH=${PROJECT_ROOT}/containers/pytorch_25.05-py3.sif
readonly REPO_INNER=/vol/dissolve/yz10325/repos/GeCo-geco-evaluation
readonly VENV_INNER=/vol/dissolve/yz10325/envs/geco-geometry-selection
readonly MODEL_INNER=/vol/dissolve/yz10325/geco_models/frozen/wan-b8fff7315c768468a5333511427288870b2e9635
readonly HF_HOME_INNER=/vol/dissolve/yz10325/cache/huggingface

: "${EXPECTED_COMMIT:?Submit with EXPECTED_COMMIT=<clean isolated-worktree commit>}"
if [[ ! "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "EXPECTED_COMMIT must be a full 40-character Git SHA" >&2
    exit 2
fi
readonly OUTPUT_INNER="/vol/dissolve/yz10325/tmp/wan_geco_scheduler_preflight_${EXPECTED_COMMIT}.json"

test "$(git -C "${PROJECT_ROOT}/repos/GeCo-geco-evaluation" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_ROOT}/repos/GeCo-geco-evaluation" status --porcelain)"

apptainer exec \
  --nv \
  --cleanenv \
  --bind "${PROJECT_ROOT}:/vol/dissolve/yz10325" \
  --env "PYTHONPATH=${REPO_INNER}" \
  --env "HF_HOME=${HF_HOME_INNER}" \
  --env "HF_HUB_CACHE=${HF_HOME_INNER}/hub" \
  --env "TOKENIZERS_PARALLELISM=false" \
  "${CONTAINER_PATH}" \
  /bin/bash -lc '
    set -euo pipefail
    source "$1/bin/activate"
    cd "$2"
    python tests/geco_port/inspect_wan_scheduler_runtime.py \
      --repo "$2" --model "$3" --device cuda:0 --output "$4"
  ' bash "${VENV_INNER}" "${REPO_INNER}" "${MODEL_INNER}" "${OUTPUT_INNER}"
