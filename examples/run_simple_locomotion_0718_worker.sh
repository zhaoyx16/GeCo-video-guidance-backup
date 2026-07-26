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
PROMPT_JSON="$REPO/examples/navigation_locomotion_corridors_simple_0718.json"
OUT="$ROOT/outputs/static_motion_observable_0718/cosmos_wan_locomotion_simpleprompt_0718"

source "$ROOT/env_hippasus.sh"
source "$ROOT/venvs/geco/bin/activate"
source "$ROOT/env_hippasus.sh"
cd "$REPO"

for MODEL in cosmos wan; do
  mkdir -p "$OUT/logs/$MODEL/gpu$GPU"
  for CASE in "$@"; do
    for SEED in 0 1 2; do
      LOG="$OUT/logs/$MODEL/gpu$GPU/${CASE}_seed${SEED}.log"
      if [[ "$MODEL" == "cosmos" ]]; then
        /usr/bin/time -v env CUDA_VISIBLE_DEVICES="$GPU" \
          python "$REPO/run_cosmos_geco_case.py" \
            --case "$CASE" \
            --prompt_json "$PROMPT_JSON" \
            --output_root "$OUT/baselines/cosmos" \
            --mode baseline \
            --profile model_default \
            --seed "$SEED" \
            > "$LOG" 2>&1
      else
        /usr/bin/time -v env CUDA_VISIBLE_DEVICES="$GPU" \
          python "$REPO/run_wan_geco_case_full.py" \
            --case "$CASE" \
            --prompt_json "$PROMPT_JSON" \
            --output_root "$OUT/baselines/wan" \
            --mode baseline \
            --height 704 \
            --width 1280 \
            --frames 121 \
            --steps 50 \
            --fps 24 \
            --seed "$SEED" \
            > "$LOG" 2>&1
      fi
    done
  done
done

echo "worker complete: gpu=$GPU cases=$*"
