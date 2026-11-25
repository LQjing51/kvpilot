#!/bin/bash
# Sequential: ToolAgent, SWE, TraceLab (prefix>0 only).
#   RAW=raw/ssdcap KVPILOT_SSD_READ_GIBPS=1.6 bash raw/run_all.sh
set -euo pipefail
HERE=/root/kvpilot/paper-figure/Figure3-request-safe
RAW=${RAW:-raw}
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
for ds in toolagent swe tracelab; do
    say "===== $ds ====="
    mkdir -p $HERE/$RAW/$ds
    RAW=$RAW bash $HERE/raw/run_one.sh $ds > $HERE/$RAW/$ds/run.out 2>&1
    say "$ds finished"
done
say "ALL DONE"
