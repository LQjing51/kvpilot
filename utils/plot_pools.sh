#!/bin/bash
# Plot the HBM pool distribution for one or more runs, and verify the result is
# actually a PDF.
#
#   bash plot_pools.sh origin-pool kvpilot-pool kvpilot-measured-pool
#
# The verification is not paranoia: plot_hbm_pool_distribution.py also writes a
# text report, argparse matches `--out` as a prefix of `--output-file`, and a
# text report written to a .pdf name exits 0 and looks like success until
# someone opens it.
set -uo pipefail

if [[ -z "${KVPILOT_ROOT:-}" ]]; then
    _self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    KVPILOT_ROOT="$(cd "${_self}/.." 2>/dev/null && pwd)"
    [[ -d "${KVPILOT_ROOT}/vllm-dev" ]] || KVPILOT_ROOT=/root/kvpilot
fi
export PYTHONPATH="${KVPILOT_ROOT}/vllm-dev:${KVPILOT_ROOT}/lmcache-dev"

B="${LOG_ROOT:-${KVPILOT_ROOT}/vllm-dev/examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/dynamo-log/demo-tests}"
RATIOS="${RATIOS:-${KVPILOT_ROOT}/datasets/agent-traces/swe-smith-trajectories/req_ratio.json}"
cd "${KVPILOT_ROOT}/utils" || exit 1

for name in "$@"; do
    L="$B/${name}-logs"
    bytes=$(ls -la "$L"/hbm-pool-*.jsonl 2>/dev/null | awk '{s+=$5} END {print s+0}')
    echo "=== ${name}: census ${bytes} bytes ==="
    if [ "$bytes" -lt 1000 ]; then
        # An empty census means GET_HBM_POOL_DISTRIBUTION was off, or the writer
        # buffered without flushing (see kvpilot_hbm_snapshot._drain).
        echo "  SKIP: census is empty"
        continue
    fi
    rm -f "$L/pool_distribution.pdf"
    python3 plot_hbm_pool_distribution.py \
        --census "$L/hbm-pool-*.jsonl" \
        --ratios "$RATIOS" \
        --pdf-output "$L/pool_distribution.pdf" \
        --csv-output "$L/pool_distribution.csv" \
        --output-file "$L/pool_distribution.log" 2>&1 | tail -10

    if [ -f "$L/pool_distribution.pdf" ] \
       && [ "$(head -c 4 "$L/pool_distribution.pdf")" = "%PDF" ]; then
        echo "  PDF OK: $(wc -c < "$L/pool_distribution.pdf" | tr -d ' ') bytes"
    else
        echo "  PDF MISSING OR NOT A PDF"
    fi
    echo
done
