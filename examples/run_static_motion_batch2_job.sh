#!/usr/bin/env bash
set -u

MODEL="$1"
CASE="$2"
GPU="$3"

ROOT=/vol/dissolve/yz10325
REPO="$ROOT/repos/GeCo"
RUN="$ROOT/outputs/static_motion_observable_batch2_20260717"
CONFIG="$REPO/examples/static_motion_observable_batch2.json"

source "$ROOT/env_hippasus.sh"
source "$ROOT/venvs/geco/bin/activate"
source "$ROOT/env_hippasus.sh"
cd "$REPO"

echo "START model=$MODEL case=$CASE gpu=$GPU wall=$(date -Is)"
if [[ "$MODEL" == "cosmos" ]]; then
  /usr/bin/time -v env CUDA_VISIBLE_DEVICES="$GPU" python run_cosmos_geco_case.py \
    --case "$CASE" \
    --prompt_json "$CONFIG" \
    --output_root "$RUN/baselines/cosmos" \
    --mode baseline \
    --profile model_default \
    --seed 42
else
  /usr/bin/time -v env CUDA_VISIBLE_DEVICES="$GPU" python run_wan_geco_case_full.py \
    --case "$CASE" \
    --prompt_json "$CONFIG" \
    --output_root "$RUN/baselines/wan" \
    --mode baseline \
    --height 704 \
    --width 1280 \
    --frames 121 \
    --steps 50 \
    --fps 24 \
    --seed 42
fi
STATUS=$?
echo "END model=$MODEL case=$CASE status=$STATUS wall=$(date -Is)"
exit "$STATUS"
