#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e

# Graceful cleanup
SHUTDOWN_TIMEOUT_SECS="${SHUTDOWN_TIMEOUT_SECS:-60}"
_cleanup() {
    trap - EXIT
    local pids=()
    mapfile -t pids < <(jobs -pr)
    (( ${#pids[@]} )) || return
    echo "Cleaning up: sending SIGTERM to workers (timeout ${SHUTDOWN_TIMEOUT_SECS}s)..."
    kill -TERM "${pids[@]}" 2>/dev/null || true
    local deadline=$((SECONDS + SHUTDOWN_TIMEOUT_SECS))
    while (( SECONDS < deadline )); do
        mapfile -t pids < <(jobs -pr)
        (( ${#pids[@]} )) || { echo "Workers exited cleanly."; wait 2>/dev/null || true; return; }
        sleep 1
    done
    echo "Workers still alive after ${SHUTDOWN_TIMEOUT_SECS}s; sending SIGKILL." >&2
    kill -KILL "${pids[@]}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap _cleanup EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# The four repos sit side by side under one root; everything but the model is
# addressed relative to it, so the tree can be checked out anywhere.
KVPILOT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0
export DYN_LOG="info"

# Common configuration
export DYN_TCP_RESPONSE_STREAM_HOST=lo
HTTP_PORT="${DYN_HTTP_PORT:-8000}"

MODEL="/cfs_cloud_code/kiry/model/Qwen/Qwen3-30B-A3B-Instruct-2507"
BLOCK_SIZE=512
GPU_MEM_UTIL=0.95
export UCX_TLS=self,tcp,cuda_copy,cuda_ipc
export LMCACHE_TRACK_USAGE=false
export MC_TCP_ENABLE_CONNECTION_POOL=1

export DISABLE_LOCAL_PREFIX_CACHE_LOOKUP="${DISABLE_LOCAL_PREFIX_CACHE_LOOKUP:-false}"
export LMCACHE_FIRST_LAYER_LOOKUP="${LMCACHE_FIRST_LAYER_LOOKUP:-true}"
# Offline dram-safe: label blocks from a pre-annotated trace.
export KVPILOT_OFFLINE_DRAM_SAFE="${KVPILOT_OFFLINE_DRAM_SAFE:-false}"
export KVPILOT_TRACE_METADATA_PATH="${KVPILOT_TRACE_METADATA_PATH:-${KVPILOT_ROOT}/datasets/mooncake-trace/conversation_trace_turn4_half.jsonl}"

# Online dram-safe: the decode worker judges each turn and reports the verdict
# to the prefill worker, which re-tiers the blocks. Independent of the offline
# switch above; enable exactly one of the two for a meaningful A/B.
export KVPILOT_ONLINE_DRAM_SAFE="${KVPILOT_ONLINE_DRAM_SAFE:-false}"
export KVPILOT_MACHINE="${KVPILOT_MACHINE:-h20}"
# The tool-response-length lookup table, produced by the validator of whichever
# trace will be replayed. Its values take the percentile that validator was run
# with (--predict-percentile), recorded inside the file. It is required and has
# no default: label namespaces do not carry across traces (SWE-smith emits
# `bash:python`, TraceLab emits `Read` — measured overlap is exactly zero), so
# the wrong table matches nothing, every lookup falls back to the global figure,
# and the verdict silently degrades to a constant while the logs still look
# healthy. The workers read this at startup, so the trace has to be chosen here,
# not at benchmark time — switching traces means restarting the service.
KVPILOT_TOOL_RESP_LEN_PATH="${KVPILOT_TOOL_RESP_LEN_PATH:-${KVPILOT_ROOT}/datasets/agent-traces/swe-smith-trajectories/tool_resp_len.json}"
if [[ "$KVPILOT_ONLINE_DRAM_SAFE" == "true" && ! -f "$KVPILOT_TOOL_RESP_LEN_PATH" ]]; then
    echo "ERROR: KVPILOT_TOOL_RESP_LEN_PATH not found: ${KVPILOT_TOOL_RESP_LEN_PATH}" >&2
    echo "       Generate it by running the validator for the trace you plan" >&2
    echo "       to replay (validate_premises.py / validate_tracelab.py)." >&2
    exit 1
fi
export KVPILOT_TOOL_RESP_LEN_PATH

# Oracle arm of the online experiment: take each turn's dram-safe verdict from a
# table of what a prior run measured, keyed by the trace line in the request id,
# instead of from the cost model. The feedback link and re-tiering are unchanged,
# so comparing this against the predicted arm separates a wrong prediction from a
# mechanism that does not pay off. Unset (the default) keeps the prediction.
export KVPILOT_MEASURED_DRAM_SAFE_PATH="${KVPILOT_MEASURED_DRAM_SAFE_PATH:-}"
if [[ -n "$KVPILOT_MEASURED_DRAM_SAFE_PATH" && ! -f "$KVPILOT_MEASURED_DRAM_SAFE_PATH" ]]; then
    echo "ERROR: KVPILOT_MEASURED_DRAM_SAFE_PATH not found: ${KVPILOT_MEASURED_DRAM_SAFE_PATH}" >&2
    exit 1
fi

export ENABLE_MOONCAKE_CPU_BACKEND="${ENABLE_MOONCAKE_CPU_BACKEND:-false}"
export ENABLE_LOCAL_MOONCAKE_HBM="${ENABLE_LOCAL_MOONCAKE_HBM:-false}"
export ENABLE_SHM_MOONCAKE_HBM="${ENABLE_SHM_MOONCAKE_HBM:-false}"
export MOONCAKE_PRELOAD_EXPECTED_ENGINES="${MOONCAKE_PRELOAD_EXPECTED_ENGINES:-2}"


# Per-block census of the HBM pool, for the placement distribution plot. Purely
# observational and independent of the dram-safe switches above: it records only
# each block's idle time and last holder, so the same run can be compared
# against a plain-LRU baseline. Lands as hbm-pool-<role>-<pid>.jsonl in LOG_DIR,
# about 100 KB per second per worker. Sampling interval is fixed in
# kvpilot_hbm_snapshot.py so two runs are always comparable.
export GET_HBM_POOL_DISTRIBUTION="${GET_HBM_POOL_DISTRIBUTION:-false}"

# Per-prefill load-stall accounting: how much of each prefill was spent blocked
# waiting for KV to load rather than computing. With layerwise loading each
# attention layer blocks until its own KV is resident, and the sum of those
# waits over one forward is the extent to which loading failed to hide behind
# computation -- the quantity this project exists to drive to zero. Writes
# load-stall-<role>-<pid>.jsonl into LOG_DIR, one line per finished prefill.
# Independent of the dram-safe switches, so a KVPilot run and an LRU baseline
# can be compared on it.
export KVPILOT_LOAD_STALL="${KVPILOT_LOAD_STALL:-false}"

# Per-worker log directory. Defaults to current dir so the script remains
# usable standalone; callers (e.g. dynamo_script.sh) can override LOG_DIR.
LOG_DIR="${LOG_DIR:-.}"
mkdir -p "$LOG_DIR"
# The census writer resolves its output path from LOG_DIR inside the worker, so
# it has to reach the workers as an env var, not just as a shell local.
export LOG_DIR

NSYS_PROFILE="${NSYS_PROFILE:-1}"
# Skip startup (model load + cudagraph capture, ~40-90s) before sampling so the
# report isn't dominated by startup noise. Tune to your warmup time.
NSYS_DELAY="${NSYS_DELAY:-0}"
# Sampling duration in seconds; size it to cover the request window you care
# about.
NSYS_DURATION="${NSYS_DURATION:-5000}"
# vLLM env units differ: RPC timeout is milliseconds; execute_model timeout is seconds.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"

nsys_prefix() {
    [[ "$NSYS_PROFILE" == "1" ]] || return 0
    local out="$1"
    local args=(nsys profile
        --trace=cuda,nvtx
        --cuda-event-trace=false
        --cuda-graph-trace=node
        --sample=none
        --cpuctxsw=none
        --trace-fork-before-exec=true
        --delay="$NSYS_DELAY"
        --duration="$NSYS_DURATION"
        --kill=none
        --force-overwrite=true
        --output="$LOG_DIR/$out")
    printf '%s ' "${args[@]}"
}

if [[ "$NSYS_PROFILE" == "1" ]]; then
    # spawn (not fork) is required for nsys to capture vLLM's worker subprocs.
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
fi

# LMCache config files (mooncake backend). LMCache reads these via the
# LMCACHE_CONFIG_FILE env var; without it LMCache silently falls back to
# default local-CPU mode and never connects to the mooncake master.
LMCACHE_PREFILL_CFG="${LMCACHE_PREFILL_CFG:-${KVPILOT_ROOT}/vllm-dev/examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/mooncake-prefiller-config.yaml}"
LMCACHE_DECODE_CFG="${LMCACHE_DECODE_CFG:-${KVPILOT_ROOT}/vllm-dev/examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/mooncake-decoder-config.yaml}"
for cfg in "$LMCACHE_PREFILL_CFG" "$LMCACHE_DECODE_CFG"; do
    [[ -f "$cfg" ]] || { echo "ERROR: LMCache config not found: $cfg" >&2; exit 1; }
done

NUM_PREFILL_WORKERS="${NUM_PREFILL_WORKERS:-2}"
NUM_DECODE_WORKERS="${NUM_DECODE_WORKERS:-2}"
TP_SIZE="${TP_SIZE:-2}"

TOTAL_GPUS=$(((NUM_PREFILL_WORKERS + NUM_DECODE_WORKERS) * TP_SIZE))
if (( TOTAL_GPUS > 8 )); then
    echo "ERROR: requested ${TOTAL_GPUS} GPUs, but this launcher maps only GPUs 0..7." >&2
    exit 1
fi

gpu_list() {
    local first=$1
    local n=$2
    local out=""
    for ((g=first; g<first+n; g++)); do
        [[ -n "$out" ]] && out+=","
        out+="$g"
    done
    echo "$out"
}

# run ingress with kv router / round-robin router
python -m dynamo.frontend \
    --router-mode kv \
    --router-reset-states \
    > "$LOG_DIR/frontend.log" 2>&1 &

for ((i=0; i<NUM_DECODE_WORKERS; i++)); do
    gpu_start=$(((NUM_PREFILL_WORKERS + i) * TP_SIZE))
    VLLM_PORT=$((51000 + i * 100)) \
    LMCACHE_CONFIG_FILE="$LMCACHE_DECODE_CFG" \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$((20096 + i)) \
    CUDA_VISIBLE_DEVICES="$(gpu_list "$gpu_start" "$TP_SIZE")" python3 -m dynamo.vllm \
        --model $MODEL \
        --tensor-parallel-size "$TP_SIZE" \
        --enable-expert-parallel \
        --block-size $BLOCK_SIZE \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --disaggregation-mode decode \
        --disable-hybrid-kv-cache-manager \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
        --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$((20082 + i))\",\"enable_kv_cache_events\":true}" \
        > "$LOG_DIR/decode-${i}.log" 2>&1 &
done

for ((i=0; i<NUM_PREFILL_WORKERS; i++)); do
    gpu_start=$((i * TP_SIZE))
    VLLM_PORT=$((51400 + i * 100)) \
    LMCACHE_CONFIG_FILE="$LMCACHE_PREFILL_CFG" \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$((20096 + NUM_DECODE_WORKERS + i)) \
    CUDA_VISIBLE_DEVICES="$(gpu_list "$gpu_start" "$TP_SIZE")" $(nsys_prefix prefill-$i) python3 -m dynamo.vllm \
        --model $MODEL \
        --tensor-parallel-size "$TP_SIZE" \
        --enable-expert-parallel \
        --block-size $BLOCK_SIZE \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --disaggregation-mode prefill \
        --kv-transfer-config '{"kv_connector":"PdConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"use_layerwise":true}},{"kv_connector":"NixlConnector","kv_role":"kv_both"}]},"kv_connector_module_path":"kvbm.vllm_integration.connector"}' \
        --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$((20090 + i))\",\"enable_kv_cache_events\":true}" \
        > "$LOG_DIR/prefill-${i}.log" 2>&1 &
done

# In profiling mode, wait for all nsys wrappers to finish finalizing reports.
# Otherwise, the first completed nsys process can trigger cleanup and interrupt
# another worker while it is still writing its .nsys-rep.
if [[ "$NSYS_PROFILE" == "1" ]]; then
    wait
else
    # Exit on first worker failure; kill 0 in the EXIT trap tears down the rest.
    wait_any_exit
fi
