#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <physical_gpu> <case> [case ...]" >&2
  exit 2
fi

GPU="$1"
shift

ROOT="/vol/dissolve/yz10325"
REPO="$ROOT/repos/GeCo"
PROMPT_JSON="$REPO/examples/cosmos_navigation_locomotion_corridors_0718.json"
OUT="$ROOT/outputs/static_motion_observable_0718/cosmos_locomotion_corridor_search"

source "$ROOT/env_hippasus.sh"
source "$ROOT/venvs/geco/bin/activate"
source "$ROOT/env_hippasus.sh"
cd "$REPO"

mkdir -p "$OUT/logs/gpu$GPU"

for CASE in "$@"; do
  for SEED in 0 1 2; do
    LOG="$OUT/logs/gpu$GPU/${CASE}_seed${SEED}.log"
    /usr/bin/time -v env CUDA_VISIBLE_DEVICES="$GPU" \
      python "$REPO/run_cosmos_geco_case.py" \
        --case "$CASE" \
        --prompt_json "$PROMPT_JSON" \
        --output_root "$OUT/baselines" \
        --mode baseline \
        --profile model_default \
        --seed "$SEED" \
        > "$LOG" 2>&1
  done
done

echo "worker complete: gpu=$GPU cases=$*"
