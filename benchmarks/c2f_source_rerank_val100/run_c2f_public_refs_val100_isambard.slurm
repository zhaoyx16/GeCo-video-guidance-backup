#!/usr/bin/env bash
# Frozen Wan C2F public references on Val100, seed 0.
# Submit C/F separately with METHOD=c2f_k3_a0025 or c2f_k3_a00125.
#SBATCH --job-name=c2f-public-ref-v1
#SBATCH --account=brics.u6ph
#SBATCH --output=/projects/u6ph/yanxi.u6ph/logs/c2f_public_ref_v1_%x_%A_%a.log
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --array=0-99%1

set -euo pipefail
umask 077

readonly PROJECT=/projects/u6ph/yanxi.u6ph
readonly CANONICAL=/lus/lfs1aip2/projects/u6ph/yanxi.u6ph
readonly CONTAINER=${PROJECT}/containers/pytorch_25.05-py3.sif
readonly REPO=${PROJECT}/repos/GeCo-worktrees/c2f-source-rerank-val100-0916-1935005
readonly VENV=/vol/dissolve/yz10325/envs/geco-geometry-selection
readonly PYTHON=${VENV}/bin/python
readonly RUNNER=${REPO}/benchmarks/c2f_source_rerank_val100/run_val100_case.py
readonly PIPELINE=${REPO}/external/guidance_wan/pipeline_wan_i2v_c2f_memory_value.py
readonly MANIFEST=${REPO}/benchmarks/c2f_source_rerank_val100/wan_validation_100_layout_aware_v2_formal.json
readonly DATASET_ROOT=${PROJECT}/datasets/dl3dv-1k
readonly WAN_MODEL=${CANONICAL}/geco_models/frozen/wan-b8fff7315c768468a5333511427288870b2e9635
readonly OUTPUT_ROOT=${CANONICAL}/outputs/geometry-selection/c2f_public_refs_val100_seed0_v1

readonly EXPECTED_COMMIT=19350057f7e1a566674f742897c297c556c8f105
readonly EXPECTED_RUNNER_SHA=e8e2cfe14e100c13ba1da9ffba92c09662a7014a70cc1feae34b9bf7a6ac667a
readonly EXPECTED_PIPELINE_SHA=11a4685fb6c627577f20fd8fb1b673123e6a6f757abd9d47f2b3010d8509c950
readonly EXPECTED_MANIFEST_SHA=ef59a10d21d7d30729a9ac24cfba7cb37c153f5ae05debaa18387d3720afb862

readonly METHOD=${METHOD:?Export METHOD=c2f_k3_a0025 or METHOD=c2f_k3_a00125}
case "${METHOD}" in
  c2f_k3_a0025)
    readonly CONFIG=${REPO}/benchmarks/c2f_source_rerank_val100/c2f_k3_a0025.json
    readonly EXPECTED_CONFIG_SHA=80570ebb26f18e3df27f550cb36e1000879a6df33b87753dcba1fc922e9a0275
    ;;
  c2f_k3_a00125)
    readonly CONFIG=${REPO}/benchmarks/c2f_source_rerank_val100/c2f_k3_a00125.json
    readonly EXPECTED_CONFIG_SHA=ab2ccc303ef71f5bdc36f7e5b9567cf4fc10b6b86bda79b70a803c0de18ce7dd
    ;;
  *)
    echo "unsupported METHOD=${METHOD}" >&2
    exit 2
    ;;
esac

readonly CASE_INDEX=${SLURM_ARRAY_TASK_ID:?}
if [[ ! ${CASE_INDEX} =~ ^[0-9]+$ ]] || (( CASE_INDEX < 0 || CASE_INDEX >= 100 )); then
  echo "invalid Val100 case index: ${CASE_INDEX}" >&2
  exit 2
fi

test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${REPO}" status --porcelain)"
test "$(sha256sum "${RUNNER}" | awk '{print $1}')" = "${EXPECTED_RUNNER_SHA}"
test "$(sha256sum "${PIPELINE}" | awk '{print $1}')" = "${EXPECTED_PIPELINE_SHA}"
test "$(sha256sum "${MANIFEST}" | awk '{print $1}')" = "${EXPECTED_MANIFEST_SHA}"
test "$(sha256sum "${CONFIG}" | awk '{print $1}')" = "${EXPECTED_CONFIG_SHA}"
test -r "${CONTAINER}"
test -x "${PYTHON}"
test -r "${WAN_MODEL}/model_index.json"
test -d "${DATASET_ROOT}"

mapfile -t gpu_rows < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
test "${#gpu_rows[@]}" -ge 1
for row in "${gpu_rows[@]}"; do
  [[ ${row} == *"NVIDIA GH200 120GB"* ]]
  memory_mib=${row##*, }
  (( memory_mib >= 97000 ))
done

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=${TMPDIR:-/tmp}/c2f_public_ref_pycache_${SLURM_ARRAY_JOB_ID}_${CASE_INDEX}

apptainer exec --nv --cleanenv \
  --bind "${PROJECT}:/vol/dissolve/yz10325" \
  --bind "${PROJECT}:${PROJECT}" \
  --bind "${CANONICAL}:${CANONICAL}" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env TOKENIZERS_PARALLELISM=false --env PYTHONDONTWRITEBYTECODE=1 \
  --env "PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX}" \
  "${CONTAINER}" /bin/bash -lc 'source "$1/bin/activate"; shift; exec "$@"' \
  bash "${VENV}" "${PYTHON}" -I "${RUNNER}" \
    --config "${CONFIG}" \
    --manifest "${MANIFEST}" \
    --dataset-root "${DATASET_ROOT}" \
    --model "${WAN_MODEL}" \
    --output-root "${OUTPUT_ROOT}" \
    --case-index "${CASE_INDEX}" \
    --device cuda:0 \
    --expected-git-commit "${EXPECTED_COMMIT}"

echo "C2F_PUBLIC_REFERENCE_COMPLETE method=${METHOD} case_index=${CASE_INDEX}"
