#!/bin/bash
# Run several A/B arms back to back, one service at a time.
#   bash run_series.sh lru:swereq:p1 incl:swereq:p1 excl:swereq:p1 ...
# Both datasets use run_ab.sh's defaults (the utils/trace_order capacities);
# HBM_BLOCKS / DRAM_GIB / MAX_OUT from the environment apply to every arm.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
for spec in "$@"; do
    IFS=: read -r arm ds tag <<< "$spec"
    echo "===== $spec (hbm=${HBM_BLOCKS:-default} dram=${DRAM_GIB:-100}GiB max_out=${MAX_OUT:-0}) ====="
    bash "$HERE/run_ab.sh" "$arm" "$ds" "$tag"
done
echo "===== series done ====="
