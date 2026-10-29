#!/bin/bash
# Drive one demo_test.md experiment end to end: clean, wait for GPU memory,
# setup, verify the repo code is actually live, benchmark, clean. Configuration
# comes from the environment, so one script serves every variant and the
# workload stays identical across them.
#
#   NAME=demo-tests/noprefix DISABLE_LOCAL_PREFIX_CACHE_LOOKUP=true bash run_demo.sh
#
# Results and the meaning of each switch: docs/evalution/demo_test_results.md.
# Runs on the GPU host; copy it there (scp is blocked, use base64 over ssh).
set -uo pipefail

# KVPILOT_ROOT defaults to the checkout this script lives in, so the same file
# works from the repo and from a stray copy in the home directory.
if [[ -z "${KVPILOT_ROOT:-}" ]]; then
    _self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    KVPILOT_ROOT="$(cd "${_self}/.." 2>/dev/null && pwd)"
    [[ -d "${KVPILOT_ROOT}/vllm-dev" ]] || KVPILOT_ROOT=/root/kvpilot
fi

# A non-interactive ssh does not read .bashrc, where this normally lives, so
# without it the workers silently import the stock vLLM from dist-packages and
# none of the KVPilot code runs. The check after setup exists for the same
# reason.
export PYTHONPATH="${KVPILOT_ROOT}/vllm-dev:${KVPILOT_ROOT}/lmcache-dev:${KVPILOT_ROOT}/dynamo-dev/components/src:${KVPILOT_ROOT}/dynamo-dev/lib/bindings/kvbm/python"

SD="${KVPILOT_ROOT}/vllm-dev/examples/disaggregated/lmcache/disagg_prefill_lmcache_v1"
cd "$SD" || exit 1

NAME="${NAME:?set NAME, e.g. demo-tests/origin}"
LOG_DIR="$SD/dynamo-log/${NAME}-logs"

# Workload. Identical across every variant so the runs stay comparable; see
# docs/benchmark.md for how these were calibrated.
export BM_MAX_INFLIGHT_REQ="${BM_MAX_INFLIGHT_REQ:-16}"
export BM_MAX_INFLIGHT_SESSION="${BM_MAX_INFLIGHT_SESSION:-400}"
export BM_NUM_REQUESTS="${BM_NUM_REQUESTS:-4000}"
export BM_SESSION_ARRIVAL_RATE="${BM_SESSION_ARRIVAL_RATE:-4.0}"
export BM_TOOL_TIME="${BM_TOOL_TIME:-lognormal:0,1}"

export KVPILOT_LOAD_STALL="${KVPILOT_LOAD_STALL:-true}"
export GET_HBM_POOL_DISTRIBUTION="${GET_HBM_POOL_DISTRIBUTION:-false}"
# nsys costs ~10 minutes of finalization per run and can wedge in Collection
# without writing anything. The metrics these experiments need (load-stall
# share, TTFT) come from logs, so it is off unless asked for.
export NSYS_PROFILE="${NSYS_PROFILE:-0}"

export DISABLE_LOCAL_PREFIX_CACHE_LOOKUP="${DISABLE_LOCAL_PREFIX_CACHE_LOOKUP:-false}"
export ENABLE_MOONCAKE_CPU_BACKEND="${ENABLE_MOONCAKE_CPU_BACKEND:-false}"
export ENABLE_LOCAL_MOONCAKE_HBM="${ENABLE_LOCAL_MOONCAKE_HBM:-false}"
export ENABLE_SHM_MOONCAKE_HBM="${ENABLE_SHM_MOONCAKE_HBM:-false}"
export MOONCAKE_PRELOAD_EXPECTED_ENGINES="${MOONCAKE_PRELOAD_EXPECTED_ENGINES:-2}"
export KVPILOT_ONLINE_DRAM_SAFE="${KVPILOT_ONLINE_DRAM_SAFE:-false}"
export KVPILOT_OFFLINE_DRAM_SAFE="${KVPILOT_OFFLINE_DRAM_SAFE:-false}"

say() { echo "[$(date -u +%H:%M:%S)] $*"; }

say "=== ${NAME} ==="
say "prefix_lookup_disabled=${DISABLE_LOCAL_PREFIX_CACHE_LOOKUP} mooncake_cpu=${ENABLE_MOONCAKE_CPU_BACKEND} shm_hbm=${ENABLE_SHM_MOONCAKE_HBM} online_safe=${KVPILOT_ONLINE_DRAM_SAFE} measured=${KVPILOT_MEASURED_DRAM_SAFE_PATH:-}"
say "workload: req=${BM_MAX_INFLIGHT_REQ} sess=${BM_MAX_INFLIGHT_SESSION} n=${BM_NUM_REQUESTS} rate=${BM_SESSION_ARRIVAL_RATE} tool=${BM_TOOL_TIME}"

bash dynamo_script.sh clean >/dev/null 2>&1
# `clean` gives GPU memory only 20 s to drain, and with the Mooncake HBM
# backends registered a worker can take longer than that to release. setup's own
# precheck then aborts on the leftover, so wait for the release here instead.
for _ in $(seq 1 60); do
    max=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            | tr -d ' ' | sort -rn | head -1)
    [ "${max:-0}" -le 100 ] && break
    sleep 5
done
say "GPU memory settled at ${max:-0} MiB"

say "setup..."
if ! bash dynamo_script.sh setup "$NAME"; then
    say "SETUP FAILED"
    exit 1
fi

# The repo is on PYTHONPATH rather than installed, so a missing export yields a
# healthy-looking run of upstream vLLM with none of the KVPilot code in it.
n=$(grep -c 'Free blocks distribution' "$LOG_DIR/prefill-0.log" 2>/dev/null || echo 0)
say "repo-code check: 'Free blocks distribution' lines = ${n}"
if [ "$n" -eq 0 ]; then
    say "ABORT: running stock vLLM, not the repo"
    bash dynamo_script.sh clean >/dev/null 2>&1
    exit 1
fi

say "benchmark..."
bash dynamo_script.sh benchmark "$NAME"
say "benchmark done"

bash dynamo_script.sh clean >/dev/null 2>&1
say "=== ${NAME} complete ==="
