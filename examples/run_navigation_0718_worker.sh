#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <dl3dv|kitti> <cosmos|wan> <physical_gpu>" >&2
  exit 2
fi

SOURCE="$1"
MODEL="$2"
GPU="$3"
ROOT="/vol/dissolve/yz10325"
REPO="$ROOT/repos/GeCo"
OUT="$ROOT/outputs/static_motion_observable_0718"

source "$ROOT/env_hippasus.sh"
source "$ROOT/venvs/geco/bin/activate"
cd "$REPO"

if [[ "$SOURCE" == "dl3dv" ]]; then
  PROMPT_JSON="$REPO/examples/navigation_0718_dl3dv.json"
else
  PROMPT_JSON="$REPO/examples/navigation_0718_kitti_seq00.json"
fi

if [[ "$MODEL" == "cosmos" ]]; then
  CASES=$(python - "$PROMPT_JSON" <<'PY'
import json, sys
print("\n".join(json.load(open(sys.argv[1])).keys()))
PY
  )
  for case in $CASES; do
    for seed in 0 1 2; do
      log="$OUT/logs/$SOURCE/cosmos/$case"_seed"$seed.log"
      mkdir -p "$(dirname "$log")"
      CUDA_VISIBLE_DEVICES="$GPU" python "$REPO/run_cosmos_geco_case.py" --case "$case" --prompt_json "$PROMPT_JSON" --output_root "$OUT/baselines/cosmos/$SOURCE" --mode baseline --profile model_default --seed "$seed" > "$log" 2>&1
    done
  done
else
  CASES=$(python - "$PROMPT_JSON" <<'PY'
import json, sys
print("\n".join(json.load(open(sys.argv[1])).keys()))
PY
  )
  for case in $CASES; do
    for seed in 0 1 2; do
      log="$OUT/logs/$SOURCE/wan/$case"_seed"$seed.log"
      mkdir -p "$(dirname "$log")"
      CUDA_VISIBLE_DEVICES="$GPU" python "$REPO/run_wan_geco_case_full.py" --case "$case" --prompt_json "$PROMPT_JSON" --output_root "$OUT/baselines/wan/$SOURCE" --mode baseline --height 704 --width 1280 --frames 121 --steps 50 --fps 24 --seed "$seed" > "$log" 2>&1
    done
  done
fi

echo "worker complete: source=$SOURCE model=$MODEL gpu=$GPU"
