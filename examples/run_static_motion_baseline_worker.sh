#!/usr/bin/env bash
set -u

MODEL="$1"
GPU="$2"
shift 2

ROOT=/vol/dissolve/yz10325
RUN="$ROOT/outputs/static_motion_observable_6case_20260717"
JOB="$ROOT/repos/GeCo/examples/run_static_motion_baseline_job.sh"
mkdir -p "$RUN/run_logs/$MODEL" "$RUN/baselines/$MODEL"

for CASE in "$@"; do
  "$JOB" "$MODEL" "$CASE" "$GPU" \
    > "$RUN/run_logs/$MODEL/${CASE}.log" 2>&1
done
