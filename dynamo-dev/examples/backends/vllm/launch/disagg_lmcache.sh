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

MODEL="${MODEL:-/cfs_cloud_code/kiry/model/Qwen/Qwen3-30B-A3B-Instruct-2507}"
TP_SIZE="${TP_SIZE:-2}"
# Hybrid linear-attention models (Qwen3.5 / 3.6: linear_attention in
# layer_types) keep a recurrent state per request. vLLM's hybrid KV cache
# manager stores one state per page, so an attention page must hold one layer's
# state on a TP rank: the block is the smallest multiple of 512 that does
# (Qwen3.6, fp32 state: 1536 at TP 2, 1024 at TP 4). Prefix caching is off by
# default for them; the prefill side turns it on (mamba "align" mode). NIXL moves
# the state to decode only with the DS conv layout.
# Every model runs with expert parallelism. Under EP a block-FP8 MoE (Qwen3.8,
# Qwen3.6-FP8) picks FlashInfer CUTLASS, whose grouped GEMM TensorRT-LLM
# compiles with nvcc on first use (cached in ~/.tensorrt_llm): nvcc must be on
# PATH, else "Assertion failed: !cubin.empty() || isPathValid(path_)".
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
read -r HYBRID KV_LAYERS HYBRID_BLOCK < <(python3 - "$MODEL" "$TP_SIZE" <<'EOF'
import json, sys
c = json.load(open(f"{sys.argv[1]}/config.json"))
tp = int(sys.argv[2])
t = c.get("text_config", c)
kinds = t.get("layer_types") or ["full_attention"] * t["num_hidden_layers"]
block = 0
if "linear_attention" in kinds:
    if t.get("mamba_ssm_dtype") != "float32":
        sys.exit(f"block size for mamba_ssm_dtype={t.get('mamba_ssm_dtype')} not worked out")
    kv_token = 2 * max(1, t["num_key_value_heads"] // tp) * t["head_dim"] * 2
    conv = (2 * t["linear_num_key_heads"] * t["linear_key_head_dim"]
            + t["linear_num_value_heads"] * t["linear_value_head_dim"]) // tp
    state = (conv * (t["linear_conv_kernel_dim"] - 1) * 2
             + t["linear_num_value_heads"] // tp * t["linear_value_head_dim"]
             * t["linear_key_head_dim"] * 4)
    block = -(-state // kv_token // 512) * 512
print(int("linear_attention" in kinds), kinds.count("full_attention"), block)
EOF
)
EP_ARGS=(--enable-expert-parallel)
if [[ "$HYBRID" == 1 ]]; then
    BLOCK_SIZE=$HYBRID_BLOCK
    export VLLM_SSM_CONV_STATE_LAYOUT=DS
    MODEL_ARGS=(--language-model-only)
    PREFILL_MODEL_ARGS=(--enable-prefix-caching)
    DECODE_MODEL_ARGS=()
else
    BLOCK_SIZE=512
    MODEL_ARGS=()
    # One KV cache group; keep vLLM's non-hybrid manager on both sides.
    PREFILL_MODEL_ARGS=(--disable-hybrid-kv-cache-manager)
    DECODE_MODEL_ARGS=(--disable-hybrid-kv-cache-manager)
fi
# LMCache chunks are vLLM blocks (KVPilot tracks one per block).
export LMCACHE_CHUNK_SIZE=$BLOCK_SIZE
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
export UCX_TLS=self,tcp,cuda_copy,cuda_ipc
export LMCACHE_TRACK_USAGE=false
export MC_TCP_ENABLE_CONNECTION_POOL=1

# KVPilot placement (vllm/v1/core/kvpilot_placement.py): empty = LRU (on a
# hybrid model, given KVPILOT_BELADY_TRACE, it holds end states as Belady
# does), belady (needs KVPILOT_BELADY_TRACE, the replayed trace, and no signal),
# inclusive-safe-reqwise, inclusive-penalty-reqwise, exclusive-penalty-reqwise,
# exclusive-penalty-reqwise-unblocking or exclusive-penalty-blockwise-unblocking
# (all but LRU need KVPILOT_TIME_COEFFS_PATH on the prefill side too: tiers
# and penalties come from the time model). The prefill engine,
# LMCache and mooncake_master (dynamo_script.sh starts it with this env) all
# read it. With a strategy set, the decode side reports each finished turn's
# signal {tier, time: [t_hbm, t_dram, t_ssd]} (vllm/v1/core/kvpilot_signal.py):
# looked up in KVPILOT_OFFLINE_TABLE (utils/placement_ab/make_req_table.py),
# or with KVPILOT_ONLINE_PREDICT=1 predicted from the turn's tool calls with
# KVPILOT_TOOL_RESP_DISTRIBUTION_PATH and KVPILOT_TIME_COEFFS_PATH.
export KVPILOT_PLACEMENT="${KVPILOT_PLACEMENT:-}"
case "$KVPILOT_PLACEMENT" in
    ""|belady|inclusive-safe-reqwise) ;;
    inclusive-penalty-reqwise|exclusive-penalty-reqwise) ;;
    exclusive-penalty-reqwise-unblocking|exclusive-penalty-blockwise-unblocking) ;;
    *) echo "ERROR: unknown KVPILOT_PLACEMENT=$KVPILOT_PLACEMENT" >&2; exit 1 ;;
esac
export KVPILOT_ONLINE_PREDICT="${KVPILOT_ONLINE_PREDICT:-0}"
case "$KVPILOT_ONLINE_PREDICT" in
    0) kvpilot_inputs=(KVPILOT_OFFLINE_TABLE) ;;
    1) kvpilot_inputs=(KVPILOT_TOOL_RESP_DISTRIBUTION_PATH KVPILOT_TIME_COEFFS_PATH) ;;
    *) echo "ERROR: KVPILOT_ONLINE_PREDICT must be 0 or 1" >&2; exit 1 ;;
esac
if [[ -n "$KVPILOT_PLACEMENT" && "$KVPILOT_ONLINE_PREDICT" == 0 ]]; then
    kvpilot_inputs+=(KVPILOT_TIME_COEFFS_PATH)
fi
if [[ "$KVPILOT_PLACEMENT" == belady ]]; then
    kvpilot_inputs=(KVPILOT_BELADY_TRACE)
fi
if [[ -n "$KVPILOT_PLACEMENT" ]]; then
    for v in "${kvpilot_inputs[@]}"; do
        if [[ ! -f "${!v:-}" ]]; then
            echo "ERROR: KVPILOT_PLACEMENT=$KVPILOT_PLACEMENT needs $v (not found: '${!v:-}')" >&2
            exit 1
        fi
        export "$v"
    done
fi

# KVPilot L4: Mooncake SSD offload (empty = off). Each prefill process keeps
# its bucket files in a pid<pid>/ directory under this path; dynamo_script.sh
# starts mooncake_master with --enable_offload and clears the directory.
# LRU and inclusive write through (every put is queued for SSD); exclusive
# writes only the chunks placed on SSD.
export KVPILOT_SSD_OFFLOAD_PATH="${KVPILOT_SSD_OFFLOAD_PATH:-}"
if [[ -n "$KVPILOT_SSD_OFFLOAD_PATH" ]]; then
    # The Mooncake fork runs the offload heartbeat every 100 ms; this upstream
    # knob only sets the client buffer GC interval now. A bucket is written
    # once it holds this many keys; one chunk's KV layers (48 for Qwen3-30B-A3B,
    # the 10 full-attention layers for Qwen3.6) fill it exactly while a chunk
    # stays in one segment, and the Mooncake fork writes any leftover on the
    # next heartbeat without new keys.
    # MOONCAKE_OFFLOAD_USE_URING must stay off: Mooncake's io_uring path uses
    # IORING_OP_READ, which kernel 5.4 lacks.
    export MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS="${MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS:-1}"
    export MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT="${MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT:-$KV_LAYERS}"
fi


export GET_HBM_POOL_DISTRIBUTION="${GET_HBM_POOL_DISTRIBUTION:-false}"

export KVPILOT_LOAD_STALL="${KVPILOT_LOAD_STALL:-true}"

# Per-worker log directory. Defaults to current dir so the script remains
# usable standalone; callers (e.g. dynamo_script.sh) can override LOG_DIR.
LOG_DIR="${LOG_DIR:-.}"
mkdir -p "$LOG_DIR"
# The census writer resolves its output path from LOG_DIR inside the worker, so
# it has to reach the workers as an env var, not just as a shell local.
export LOG_DIR

# The frontend parses the chat template with minijinja, which has no
# {% generation %} block (HF's assistant-token mask, in Qwen3.6's template):
# the model never registers. The tags render nothing, so drop them.
TEMPLATE_ARGS=()
if grep -qE '\{%-? *generation *-?%\}' "$MODEL/chat_template.jinja" 2>/dev/null; then
    sed -E 's/\{%-? *(end)?generation *-?%\}//g' "$MODEL/chat_template.jinja" > "$LOG_DIR/chat_template.jinja"
    TEMPLATE_ARGS=(--custom-jinja-template "$LOG_DIR/chat_template.jinja")
fi

NSYS_PROFILE="${NSYS_PROFILE:-0}"
# Skip startup (model load + cudagraph capture, ~40-90s) before sampling so the
# report isn't dominated by startup noise. Tune to your warmup time.
NSYS_DELAY="${NSYS_DELAY:-0}"
# Sampling duration in seconds; size it to cover the request window you care
# about.
NSYS_DURATION="${NSYS_DURATION:-5000}"
# vLLM env units differ: RPC timeout is milliseconds; execute_model timeout is seconds.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"

# vLLM sizes cudagraphs from max_num_seqs (min(2 * max_num_seqs, 512)), which
# with --max-num-seqs 1 is 2: every prefill then runs eager. Set e.g.
# '{"cudagraph_capture_sizes":[64,128,256,512,1024]}' to let suffix-sized
# prefills replay PIECEWISE graphs. Prefill workers only.
PREFILL_COMPILATION_CONFIG="${PREFILL_COMPILATION_CONFIG:-}"
# Extra vLLM args for prefill workers only, word-split, e.g.
# "--num-gpu-blocks-override 768" to shrink the HBM prefix cache.
PREFILL_EXTRA_ARGS="${PREFILL_EXTRA_ARGS:-}"
# Extra vLLM args for decode workers only, word-split (same rules as prefill).
DECODE_EXTRA_ARGS="${DECODE_EXTRA_ARGS:-}"

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
# Mooncake shm preload waits for every prefill engine's workers.
export MOONCAKE_PRELOAD_EXPECTED_ENGINES="$NUM_PREFILL_WORKERS"
NUM_DECODE_WORKERS="${NUM_DECODE_WORKERS:-2}"

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

# Runs a worker on its GPUs' NUMA node (KVPILOT_NUMA_BIND, default 1). Its
# pinned buffers (LMCache SSD read slots, ...) are allocated at start-up on
# whatever node its thread runs on; on the other socket every load crosses
# the socket link (2026-11-24 Eval-EXP1: random TP 4 ranks read SSD 3-4x
# slower, a different set every run).
gpu_numa_node() {
    local bus
    bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$1")
    cat "/sys/bus/pci/devices/$(echo "${bus:4}" | tr 'A-F' 'a-f')/numa_node"
}
numa_prefix() {
    [[ "${KVPILOT_NUMA_BIND:-1}" == 1 ]] || return 0
    local first=$1 n=$2 node
    node=$(gpu_numa_node "$first")
    if [[ "$(gpu_numa_node $((first + n - 1)))" != "$node" ]]; then
        echo "ERROR: GPUs $first..$((first + n - 1)) span NUMA nodes" >&2
        exit 1
    fi
    echo "numactl --cpunodebind=$node --preferred=$node"
}

# run ingress with kv router / round-robin router
python -m dynamo.frontend \
    --router-mode kv \
    --router-reset-states \
    > "$LOG_DIR/frontend.log" 2>&1 &

for ((i=0; i<NUM_DECODE_WORKERS; i++)); do
    gpu_start=$(((NUM_PREFILL_WORKERS + i) * TP_SIZE))
    numa=$(numa_prefix "$gpu_start" "$TP_SIZE") || exit 1
    VLLM_PORT=$((51000 + i * 100)) \
    LMCACHE_CONFIG_FILE="$LMCACHE_DECODE_CFG" \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$((20096 + i)) \
    CUDA_VISIBLE_DEVICES="$(gpu_list "$gpu_start" "$TP_SIZE")" $numa python3 -m dynamo.vllm \
        --model $MODEL \
        --tensor-parallel-size "$TP_SIZE" \
        "${EP_ARGS[@]}" \
        --block-size $BLOCK_SIZE \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --disaggregation-mode decode \
        "${MODEL_ARGS[@]}" "${DECODE_MODEL_ARGS[@]}" "${TEMPLATE_ARGS[@]}" \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
        --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$((20082 + i))\",\"enable_kv_cache_events\":true}" \
        $DECODE_EXTRA_ARGS \
        > "$LOG_DIR/decode-${i}.log" 2>&1 &
done

for ((i=0; i<NUM_PREFILL_WORKERS; i++)); do
    gpu_start=$((i * TP_SIZE))
    numa=$(numa_prefix "$gpu_start" "$TP_SIZE") || exit 1
    VLLM_PORT=$((51400 + i * 100)) \
    LMCACHE_CONFIG_FILE="$LMCACHE_PREFILL_CFG" \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$((20096 + NUM_DECODE_WORKERS + i)) \
    CUDA_VISIBLE_DEVICES="$(gpu_list "$gpu_start" "$TP_SIZE")" $numa $(nsys_prefix prefill-$i) python3 -m dynamo.vllm \
        --model $MODEL \
        --tensor-parallel-size "$TP_SIZE" \
        "${EP_ARGS[@]}" \
        --block-size $BLOCK_SIZE \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --disaggregation-mode prefill \
        "${MODEL_ARGS[@]}" "${PREFILL_MODEL_ARGS[@]}" "${TEMPLATE_ARGS[@]}" \
        --kv-transfer-config '{"kv_connector":"PdConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"use_layerwise":true}},{"kv_connector":"NixlConnector","kv_role":"kv_both"}]},"kv_connector_module_path":"kvbm.vllm_integration.connector"}' \
        --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$((20090 + i))\",\"enable_kv_cache_events\":true}" \
        --max-num-seqs 1 \
        ${PREFILL_COMPILATION_CONFIG:+--compilation-config "$PREFILL_COMPILATION_CONFIG"} \
        $PREFILL_EXTRA_ARGS \
        > "$LOG_DIR/prefill-${i}.log" 2>&1 &
done

# In profiling mode, wait for all nsys wrappers to finish finalizing reports.
# Otherwise, the first completed nsys process can trigger cleanup and interrupt
# another worker while it is still writing its .nsys-rep.
if [[ "$NSYS_PROFILE" == "1" ]]; then
    wait
else
    # Exit on first worker failure; the EXIT trap tears down the rest.
    wait_any_exit
fi
