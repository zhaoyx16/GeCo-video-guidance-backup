#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <physical_gpu> <case>" >&2
  exit 2
fi

GPU="$1"
CASE="$2"
ROOT=/vol/dissolve/yz10325
REPO="$ROOT/repos/GeCo"
PROMPT_JSON="$REPO/examples/scand_navigation_trajectory_only_0721.json"
OUT="$ROOT/outputs/static_motion_observable_0721/scand_navigation_trajectory_only_rerun"

source "$ROOT/env_hippasus.sh"
source "$ROOT/venvs/geco/bin/activate"
source "$ROOT/env_hippasus.sh"
cd "$REPO"
mkdir -p "$OUT/logs/cosmos" "$OUT/logs/wan"

run_model() {
  local model="$1"
  local log="$OUT/logs/$model/$CASE.log"
  echo "START $model $CASE gpu=$GPU $(date -Is)" >> "$OUT/workers.log"

  if [[ "$model" == "cosmos" ]]; then
    /usr/bin/time -f "WALL_SEC=%e MAX_RSS_KB=%M" \
      env CUDA_VISIBLE_DEVICES="$GPU" python run_cosmos_geco_case.py \
        --case "$CASE" --prompt_json "$PROMPT_JSON" \
        --output_root "$OUT/baselines/cosmos" --mode baseline \
        --profile model_default --seed 42 > "$log" 2>&1
  else
    /usr/bin/time -f "WALL_SEC=%e MAX_RSS_KB=%M" \
      env CUDA_VISIBLE_DEVICES="$GPU" python run_wan_geco_case_full.py \
        --case "$CASE" --prompt_json "$PROMPT_JSON" \
        --output_root "$OUT/baselines/wan" --mode baseline \
        --height 704 --width 1280 --frames 121 --steps 50 --fps 24 --seed 42 \
        > "$log" 2>&1
  fi

  echo "DONE $model $CASE gpu=$GPU $(date -Is)" >> "$OUT/workers.log"
}

run_model cosmos
run_model wan
echo "WORKER_COMPLETE case=$CASE gpu=$GPU $(date -Is)" >> "$OUT/workers.log"
