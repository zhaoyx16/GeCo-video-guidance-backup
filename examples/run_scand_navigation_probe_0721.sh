#!/usr/bin/env bash
set -euo pipefail

ROOT=/vol/dissolve/yz10325
REPO="$ROOT/repos/GeCo"
PROMPT_JSON="$REPO/examples/scand_navigation_probe_0721.json"
OUT="$ROOT/outputs/static_motion_observable_0721/scand_navigation_probe"
GPU="${1:-2}"

source "$ROOT/env_hippasus.sh"
source "$ROOT/venvs/geco/bin/activate"
source "$ROOT/env_hippasus.sh"
cd "$REPO"

mkdir -p "$OUT/logs/cosmos" "$OUT/logs/wan"

CASES=(
  scand_campus_sidewalk_left_turn_row100000
  scand_tree_path_right_turn_row185000
  scand_brick_walkway_left_turn_row224000
)

for case in "${CASES[@]}"; do
  log="$OUT/logs/cosmos/$case.log"
  echo "START cosmos $case $(date -Is)" >> "$OUT/worker.log"
  /usr/bin/time -f "WALL_SEC=%e MAX_RSS_KB=%M" \
    env CUDA_VISIBLE_DEVICES="$GPU" python run_cosmos_geco_case.py \
      --case "$case" \
      --prompt_json "$PROMPT_JSON" \
      --output_root "$OUT/baselines/cosmos" \
      --mode baseline \
      --profile model_default \
      --seed 42 \
      > "$log" 2>&1
  echo "DONE cosmos $case $(date -Is)" >> "$OUT/worker.log"
done

for case in "${CASES[@]}"; do
  log="$OUT/logs/wan/$case.log"
  echo "START wan $case $(date -Is)" >> "$OUT/worker.log"
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
  echo "DONE wan $case $(date -Is)" >> "$OUT/worker.log"
done

echo "WORKER_COMPLETE $(date -Is)" >> "$OUT/worker.log"
