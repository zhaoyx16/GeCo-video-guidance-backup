#!/usr/bin/env bash
set -euo pipefail

STORAGE_ROOT="${GECO_STORAGE_ROOT:-/vol/dissolve/yz10325}"
REPO_ROOT="${GECO_REPO_ROOT:-${STORAGE_ROOT}/repos/GeCo-worktrees/geometry-selection}"
PYTHON="${GECO_PYTHON:-${STORAGE_ROOT}/venvs/geco/bin/python}"
export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/cache/huggingface}"

DATASET_REVISION=5902ed6d707cc13a7779907c1e096676f7707971
SPLIT_SHA256=d1eed2400547d755e265149185515c3e6fdf981f63635bbbfa152d33fa53205c
SPLITS=("$@")
if [[ ${#SPLITS[@]} -eq 0 ]]; then
    SPLITS=(debug validation)
fi
OLD_IFS="${IFS}"
IFS=_
SPLIT_TAG="${SPLITS[*]}"
IFS="${OLD_IFS}"

cd "${REPO_ROOT}"
"${PYTHON}" benchmarks/dl3dv_geco/download_frozen_scenes.py \
    --split-csv benchmarks/dl3dv_geco/splits/frozen_scene_split_3_100_100.csv \
    --expected-split-sha256 "${SPLIT_SHA256}" \
    --expected-count debug=3 \
    --expected-count validation=100 \
    --expected-count test=100 \
    --output-root "${STORAGE_ROOT}/datasets/dl3dv-1k/480P" \
    --cache-root "${STORAGE_ROOT}/cache/dl3dv-download" \
    --revision "${DATASET_REVISION}" \
    --splits "${SPLITS[@]}" \
    --workers 2 \
    --report "${STORAGE_ROOT}/datasets/dl3dv-1k/metadata/download_${SPLIT_TAG}.json"
