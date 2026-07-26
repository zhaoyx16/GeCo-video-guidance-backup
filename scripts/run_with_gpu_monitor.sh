#!/usr/bin/env bash
# Record the command, elapsed time, and per-GPU memory while leaving model behavior unchanged.
set -uo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 OUTPUT_DIR COMMAND [ARG ...]" >&2
    exit 2
fi

out_dir="$1"
shift
mkdir -p "$out_dir"

printf 'command: ' > "$out_dir/run_metadata.txt"
printf '%q ' "$@" >> "$out_dir/run_metadata.txt"
printf '\nstarted_utc: %s\n' "$(date -u +%FT%TZ)" >> "$out_dir/run_metadata.txt"

monitor_pid=""
cleanup() {
    if [ -n "$monitor_pid" ]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

(
    while true; do
        nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits >> "$out_dir/gpu_memory.csv"
        sleep 2
    done
) &
monitor_pid=$!

set +e
/usr/bin/time -v "$@" 2> "$out_dir/time_verbose.log"
status=$?
set -e

printf 'finished_utc: %s\nexit_status: %s\n' "$(date -u +%FT%TZ)" "$status" >> "$out_dir/run_metadata.txt"
exit "$status"
