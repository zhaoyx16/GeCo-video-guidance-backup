#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <cosmos|wan> <physical_gpu> <case> [case ...]" >&2
  exit 2
fi

MODEL="$1"
GPU="$2"
shift 2

ROOT=/vol/dissolve/yz10325
REPO="$ROOT/repos/GeCo"
PROMPT_JSON="$REPO/examples/scand_navigation_probe_0721.json"
OUT="$ROOT/outputs/static_motion_observable_0721/scand_navigation_probe"

source "$ROOT/env_hippasus.sh"
source "$ROOT/venvs/geco/bin/activate"
source "$ROOT/env_hippasus.sh"
cd "$REPO"

mkdir -p "$OUT/logs/$MODEL"

for case in "$@"; do
  log="$OUT/logs/$MODEL/$case.log"
  echo "START $MODEL $case gpu=$GPU $(date -Is)" >> "$OUT/split_workers.log"

  if [[ "$MODEL" == "cosmos" ]]; then
    /usr/bin/time -f "WALL_SEC=%e MAX_RSS_KB=%M" \
      env CUDA_VISIBLE_DEVICES="$GPU" python run_cosmos_geco_case.py \
        --case "$case" \
        --prompt_json "$PROMPT_JSON" \
        --output_root "$OUT/baselines/cosmos" \
        --mode baseline \
        --profile model_default \
        --seed 42 \
        > "$log" 2>&1
  elif [[ "$MODEL" == "wan" ]]; then
    /usr/bin/time -f "WALL_SEC=%e MAX_RSS_KB=%M" \
      env CUDA_VISIBLE_DEVICES="$GPU" python run_wan_geco_case_full.py \
        --case "$case" \
        --prompt_json "$PROMPT_JSON" \
        --output_root "$OUT/baselines/wan" \
        --mode baseline \
        --height 704 \
        --width 1280 \
        --frames 121 \
        --steps 50 \
        --fps 24 \
        --seed 42 \
        > "$log" 2>&1
  else
    echo "unknown model: $MODEL" >&2
    exit 2
  fi

  echo "DONE $MODEL $case gpu=$GPU $(date -Is)" >> "$OUT/split_workers.log"
done

echo "WORKER_COMPLETE model=$MODEL gpu=$GPU $(date -Is)" >> "$OUT/split_workers.log"
