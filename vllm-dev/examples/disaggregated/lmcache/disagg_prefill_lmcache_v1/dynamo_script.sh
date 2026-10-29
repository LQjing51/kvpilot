#!/bin/bash
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "setup" && "$MODE" != "clean" && "$MODE" != "cleanforce" && "$MODE" != "benchmark" ]]; then
    echo "Usage: $0 <setup|clean|cleanforce|benchmark> [log_name]"
    echo "  log_name: optional tag for log dir (default: 'logs')."
    echo "           Logs go under dynamo-log/."
    echo "           e.g. 'bash dynamo_script.sh setup longbench' writes to dynamo-log/longbench-logs/,"
    echo "                'bash dynamo_script.sh benchmark longbench' also writes there."
    exit 1
fi

# Optional second arg picks the log directory under dynamo-log/.
#   <log_name> -> dynamo-log/<log_name>-logs/   (used by both setup and benchmark)
#   omitted    -> dynamo-log/logs/
LOG_NAME="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The four repos sit side by side under one root; everything but the model is
# addressed relative to it, so the tree can be checked out anywhere.
KVPILOT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
DYNAMO_LOG_ROOT="${SCRIPT_DIR}/dynamo-log"
if [[ -n "$LOG_NAME" ]]; then
    LOG_DIR="${DYNAMO_LOG_ROOT}/${LOG_NAME}-logs"
else
    LOG_DIR="${DYNAMO_LOG_ROOT}/logs"
fi
MODEL="/cfs_cloud_code/kiry/model/Qwen/Qwen3-30B-A3B-Instruct-2507"

PROXY_PORT=8000

# Mooncake master listen port — must match the port in
# configs/mooncake-{prefiller,decoder}-config.yaml's master_server_address.
MOONCAKE_PORT="${MOONCAKE_PORT:-50052}"

ROUTER_SCRIPT="${KVPILOT_ROOT}/dynamo-dev/examples/backends/vllm/launch/disagg_lmcache.sh"
NUM_PREFILL_WORKERS="${NUM_PREFILL_WORKERS:-2}"
NUM_DECODE_WORKERS="${NUM_DECODE_WORKERS:-2}"

ROUTER_LOG_NAME="disagg_router.log"
MOONCAKE_WORKER_LOG_PREFIX="prefill"
MOONCAKE_READY_LOG_PATTERN="Mooncake store setup completed successfully"
EXPECTED_MOONCAKE_WORKERS="$NUM_PREFILL_WORKERS"

export PYTHONHASHSEED=${VLLM_PYTHON_HASH_SEED:-123}
export VLLM_LOGGING_LEVEL=INFO # DEBUG/INFO/WARNING/ERROR
export LMCACHE_LOG_LEVEL=INFO # DEBUG/INFO/WARNING/ERROR/CRITICAL

# Force UTC so every log in a run shares one clock. LMCache, vLLM, mooncake and
# the benchmark all render timestamps via localtime while dynamo emits ISO-8601
# UTC, so on this UTC+8 container they disagreed by 8 hours and cross-log timing
# broke silently (queue wait came out 0 for all 4000 requests, and the nsys
# correlation in analyze_lp_ttft_breakdown.py was off by the same amount).
# This is the only place TZ is set: every worker, mooncake_master and benchmark
# process is started from here and inherits it.
#
# Must be the POSIX literal "UTC0", not "UTC" or "Etc/UTC": in this image
# /usr/share/zoneinfo/UTC and /usr/share/zoneinfo/Etc/UTC both hold
# Asia/Shanghai data (embedded POSIX string "CST-8"), so any name resolved
# through a zoneinfo file still yields UTC+8. "UTC0" is parsed directly by libc
# with no file lookup. Verify with:
#   TZ=UTC0 python3 -c 'import time;time.tzset();print(time.strftime("%H:%M:%S %Z"))'
export TZ=UTC0

cleanup_orphaned_mooncake_shm() {
    local shm_path basename without_index owner_pid process_state
    local removed=0 reclaimed_bytes=0 allocated_bytes

    shopt -s nullglob
    for shm_path in /dev/shm/mooncake_*; do
        [[ -f "$shm_path" ]] || continue
        basename=${shm_path##*/}
        without_index=${basename%_*}
        owner_pid=${without_index##*_}
        [[ "$owner_pid" =~ ^[0-9]+$ ]] || continue

        # A zombie no longer owns mappings or file descriptors, so its SHM is
        # safe to unlink even though /proc/<pid> remains until it is reaped.
        if [[ -d "/proc/${owner_pid}" ]]; then
            process_state=$(
                awk '$1 == "State:" {print $2}' \
                    "/proc/${owner_pid}/status" 2>/dev/null || true
            )
            if [[ "$process_state" != "Z" && -n "$process_state" ]]; then
                echo "Keeping active Mooncake SHM: ${shm_path} (pid=${owner_pid}, state=${process_state})"
                continue
            fi
        fi

        allocated_bytes=$(du -B1 "$shm_path" 2>/dev/null | awk '{print $1}')
        allocated_bytes=${allocated_bytes:-0}
        rm -f -- "$shm_path"
        reclaimed_bytes=$((reclaimed_bytes + allocated_bytes))
        removed=$((removed + 1))
        echo "Removed orphaned Mooncake SHM: ${shm_path}"
    done
    shopt -u nullglob

    echo "Mooncake SHM cleanup: removed=${removed}, reclaimed_bytes=${reclaimed_bytes}"
}


warm_all_nixl_pairs() {
    python3 "${SCRIPT_DIR}/discover_dynamo_workers.py" \
        --namespace dynamo \
        --store-backend etcd \
        --request-plane tcp \
        --expected-prefill "$NUM_PREFILL_WORKERS" \
        --expected-decode "$NUM_DECODE_WORKERS" \
        --timeout 600 \
        --warm-nixl-pairs \
        --model "$MODEL" \
        --base-url "http://localhost:${PROXY_PORT}" \
        --log-dir "$LOG_DIR"
}

# ============================================================
# clean
# ============================================================
if [[ "$MODE" == "clean" || "$MODE" == "cleanforce" ]]; then
    if [[ "$MODE" == "cleanforce" ]]; then
        echo "Force killing Dynamo/Mooncake/Nsight processes..."
        pkill -KILL -f "^bash ${ROUTER_SCRIPT}$" 2>/dev/null || true
        for pid in $(lsof -t /dev/nvidia* 2>/dev/null | sort -u || true); do
            if grep -qE "VLLM|vllm|dynamo" "/proc/$pid/cmdline" 2>/dev/null; then
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
        pkill -KILL -f "dynamo\.(frontend|vllm)" 2>/dev/null || true
        pkill -KILL -f "^VLLM::" 2>/dev/null || true
        pkill -KILL -x nsys 2>/dev/null || true
        pkill -KILL -f "nsys-launcher|nsys-tee" 2>/dev/null || true
        pkill -KILL -f "mooncake_master" 2>/dev/null || true
        sleep 1
        echo "Cleaning orphaned Mooncake shared-memory segments..."
        cleanup_orphaned_mooncake_shm
        df -h /dev/shm
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits || true
        echo "Done."
        exit 0
    fi

    active_sessions="$(nsys sessions list 2>/dev/null || true)"
    for launcher_pid in $(pgrep -f "^bash ${ROUTER_SCRIPT}$" || true); do
        for nsys_pid in $(pgrep -P "$launcher_pid" -x nsys || true); do
            session="profile-${nsys_pid}"
            state="$(awk -v name="$session" '$NF == name { print $3 }' <<<"$active_sessions")"
            [[ -z "$state" ]] && continue
            echo "Stopping Nsight session ${session}..."
            if [[ "$state" != "Launched" ]] && ! nsys stop --session="$session"; then
                echo "WARNING: failed to finalize Nsight session ${session}; continuing cleanup." >&2
            fi
            nsys shutdown --kill=sigterm --session="$session" || true
        done
    done

    echo "Stopping Dynamo processes..."
    pkill -TERM -f "^bash ${ROUTER_SCRIPT}$" 2>/dev/null || true
    echo "Waiting up to 20s for graceful Dynamo worker shutdown..."
    term_deadline=$((SECONDS + 20))
    while (( SECONDS < term_deadline )); do
        if ! pgrep -f 'dynamo\.(frontend|vllm)|^VLLM::' >/dev/null 2>&1; then
            echo "All Dynamo workers exited gracefully."
            break
        fi
        sleep 2
    done
    if (( SECONDS >= term_deadline )); then
        echo "WARNING: Dynamo workers still running after 20s; sending SIGKILL." >&2
    fi
    for pid in $(lsof -t /dev/nvidia* 2>/dev/null | sort -u || true); do
        if grep -qE 'VLLM|vllm|dynamo' "/proc/$pid/cmdline" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    pkill -KILL -f 'dynamo\.(frontend|vllm)' 2>/dev/null || true
    pkill -KILL -f '^VLLM::' 2>/dev/null || true
    pkill -KILL -f 'nsys-launcher|nsys-tee' 2>/dev/null || true
    pkill -TERM -f 'mooncake_master' 2>/dev/null || true
    sleep 2
    pkill -KILL -f 'mooncake_master' 2>/dev/null || true

    echo "Cleaning orphaned Mooncake shared-memory segments..."
    cleanup_orphaned_mooncake_shm

    echo "Waiting for GPU memory to be released..."
    mem_deadline=$((SECONDS + 20))
    while (( SECONDS < mem_deadline )); do
        max_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
                     | tr -d ' ' | sort -rn | head -1)
        if [[ -z "$max_used" || "$max_used" -le 100 ]]; then
            echo "All GPU memory released (max ${max_used:-0} MiB)."
            break
        fi
        sleep 2
    done
    if (( SECONDS >= mem_deadline )); then
        echo "WARNING: GPU memory still in use (max ${max_used} MiB) after 20s." >&2
    fi

    # Workers may still exist (or be zombies) during the first SHM pass and
    # disappear only after their parent reaps them. Always run a final pass
    # after GPU teardown so those newly orphaned segments are reclaimed.
    echo "Final Mooncake shared-memory cleanup..."
    cleanup_orphaned_mooncake_shm
    df -h /dev/shm

    echo "Done."
    exit 0
fi

# ============================================================
# benchmark
# ============================================================
if [[ "$MODE" == "benchmark" ]]; then
    mkdir -p "$LOG_DIR"

    echo "Running benchmark (log: ${LOG_DIR}/benchmark.log)..."
    #   SWE 回放（有 prompt 文本，走 /v1/chat/completions）
    #   session 起点按 --session-arrival-rate 排成泊松到达，session 内部按照正态分布随机加 tool call 耗时；
    #   --max-inflight-req 在途请求数上限，对于 swe/tracelab 是每个会话一个协程，它限制同时有多少个请求在飞，给高到达率兜底。
    #   --max-inflight-session 同时活跃的会话数上限（0 = 不限）。会话从首轮开始占位、末轮结束才释放（含轮间 tool time），
    #      约束的是服务端要同时保温多少条不同前缀。不设时高到达率会放进远多于能服务的会话，绝大多数只发了首轮，
    #      请求摊薄在大量冷前缀上；设成接近 --max-inflight-req 可让后续轮（有 KV 要加载）占比更高。
    #   -n 总请求数上限，达到就停。
    #   --session-arrival-rate 每秒新开会话数，按泊松过程抽起点。这是开环到达：会话到点就发，不看服务端忙不忙，所以调高它就能把队列压起来。
    #      0 = 所有会话 t=0 就绪，节奏只由 --max-inflight-req 决定。同时活跃的会话数约等于 rate × 会话平均时长。
    #   --tool-time 两轮之间的工具耗时 Z，即 下一轮到达 = 上一轮完成 + Z。可填 0、常量（1s / 250ms）、或 lognormal[:mu,sigma]（默认 mu=0 sigma=1，p50 0.97s、p99 10.3s）。
    python3 "${KVPILOT_ROOT}/vllm-dev/benchmarks/multi_turn/benchmark_serving_multi_turn.py" \
        -i "${KVPILOT_ROOT}/datasets/agent-traces/swe-smith-trajectories/swe_trace.json"  \
        -m "$MODEL" \
        -u "http://localhost:${PROXY_PORT}" \
        --max-inflight-req "${BM_MAX_INFLIGHT_REQ:-0}" \
        --max-inflight-session "${BM_MAX_INFLIGHT_SESSION:-0}" \
        -n "${BM_NUM_REQUESTS:-4000}" \
        --session-arrival-rate "${BM_SESSION_ARRIVAL_RATE:-0.1}" \
        --tool-time "${BM_TOOL_TIME:-lognormal:0,1}" \
        --forced-output \
        2>&1 | tee "${LOG_DIR}/benchmark.log"

    # TraceLab 回放（只有长度，没有 prompt 文本，走 /v1/completions）。
    # session 起点按 --session-arrival-rate 重排成泊松到达，session 内部保留 trace 的真实轮间隔并按 --time-scale 缩放；
    # python3 "${KVPILOT_ROOT}/vllm-dev/benchmarks/multi_turn/replay_tracelab.py" \
    #     -m "$MODEL" \
    #     -u "http://localhost:${PROXY_PORT}" \
    #     --max-inflight-req 32 \
    #     --max-inflight-session 32 \
    #     -n 4000 \
    #     --session-arrival-rate 0.8 \
    #     --time-scale 1 \
    #     --forced-output \
    #     2>&1 | tee "${LOG_DIR}/benchmark.log"

    # Qwen3-30B-A3B-Instruct-2507 vocab_size = 151936 (from HF config.json).
    # python3 "${KVPILOT_ROOT}/vllm-dev/benchmarks/multi_turn/mooncake_trace_replay.py" \
    #     -i "${KVPILOT_ROOT}/datasets/mooncake-trace/conversation_trace_turn4_half.jsonl" \
    #     -m "$MODEL" \
    #     -u "http://localhost:${PROXY_PORT}" \
    #     -p 32 \
    #     --vocab-size 151936 \
    #     --time-scale 1 \
    #     2>&1 | tee "${LOG_DIR}/benchmark.log"
fi

# ============================================================
# setup
# ============================================================
if [[ "$MODE" == "setup" ]]; then
    # Check target GPUs are free. Edit this line when changing GPU placement.
    for gpu in 0 1 2 3 4 5 6 7; do
        used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
        used=${used:-0}
        echo "GPU ${gpu}: ${used} MiB used"
        if [[ "$used" -gt 100 ]]; then
            echo "ERROR: GPU ${gpu} has ${used} MiB memory in use. Run '$0 clean' first."
            exit 1
        fi
    done

    # Prepare logs directory
    rm -rf "$LOG_DIR"
    mkdir -p "$LOG_DIR"

    # Start mooncake_master. Address/ports must match the
    # master_server_address and metadata_server fields in
    # configs/mooncake-{prefiller,decoder}-config.yaml.
    echo "Starting mooncake_master on port ${MOONCAKE_PORT}..."
    mooncake_master \
        -rpc_port "$MOONCAKE_PORT" \
        -rpc_thread_num 64 \
        -metrics_port 9005 \
        --enable_http_metadata_server=true \
        --http_metadata_server_host=0.0.0.0 \
        --http_metadata_server_port=8081 \
        -eviction_high_watermark_ratio=0.9 \
        -eviction_ratio=0.1 \
        > "${LOG_DIR}/mooncake_master.log" 2>&1 &
    sleep 2

    echo "Starting dynamo disagg_router (proxy port ${PROXY_PORT})..."
    echo "  Script: ${ROUTER_SCRIPT}"
    echo "  Logs:   ${LOG_DIR}/ (${ROUTER_LOG_NAME} + per-worker logs)"
    DYN_HTTP_PORT="$PROXY_PORT" \
    LOG_DIR="$LOG_DIR" \
        bash "$ROUTER_SCRIPT" \
        > "${LOG_DIR}/${ROUTER_LOG_NAME}" 2>&1 &

    # Wait for frontend /v1/models to list registered workers.
    # The frontend opens its port immediately but /v1/models stays empty
    # until prefill/decode workers finish loading and register themselves.
    echo "Waiting for frontend /v1/models to list registered workers..."
    waited=0
    models_ready=0
    while [ $waited -lt 600 ]; do
        if curl -s "http://localhost:${PROXY_PORT}/v1/models" | grep -q '"id"'; then
            echo "  /v1/models is populated. (${waited}s)"
            curl -s "http://localhost:${PROXY_PORT}/v1/models"
            echo ""
            models_ready=1
            break
        fi
        sleep 5
        waited=$((waited + 5))
    done
    if [ $models_ready -eq 0 ]; then
        echo "  ERROR: /v1/models still empty after ${waited}s. Frontend did not register any worker."
        exit 1
    fi

    # This is also the real serving-readiness probe: each request is pinned to
    # one exact prefill/decode pair and must complete a NIXL metadata exchange.
    if ! warm_all_nixl_pairs; then
        echo "ERROR: Not all prefill x decode NIXL pairs became ready." >&2
        exit 1
    fi

    # Verify workers registered memory segments with mooncake_master.
    # Without this, KV transfers will silently fall back / fail later.
    # NOTE: segment mount is async and lags the first successful inference
    # probe by a few seconds, so poll instead of grepping once (avoids a
    # race where the check runs before mount_segment is logged).
    echo "Waiting for workers to mount segments with mooncake_master..."
    mount_waited=0
    mount_ready=0
    while [ $mount_waited -lt 120 ]; do
        mounted_count=$(grep -h "$MOONCAKE_READY_LOG_PATTERN"             "${LOG_DIR}"/${MOONCAKE_WORKER_LOG_PREFIX}-*.log 2>/dev/null | wc -l)
        mounted_count=${mounted_count//[[:space:]]/}
        if [[ -n "$mounted_count" && "$mounted_count" -ge "$EXPECTED_MOONCAKE_WORKERS" ]]             || grep -q 'action=mount_segment' "${LOG_DIR}/mooncake_master.log" 2>/dev/null; then
            echo "  mooncake segment registration OK. (${mount_waited}s)"
            mount_ready=1
            break
        fi
        sleep 5
        mount_waited=$((mount_waited + 5))
    done
    if [ $mount_ready -eq 0 ]; then
        echo "  ERROR: no segment mount evidence after ${mount_waited}s." >&2
        echo "         Checked mooncake_master.log and ${MOONCAKE_WORKER_LOG_PREFIX} worker setup logs." >&2
        exit 1
    fi

    echo ""
    echo "============================================================"
    echo " Dynamo disagg_router running!"
    echo " Proxy:           http://localhost:${PROXY_PORT}"
    echo " Logs:            ${LOG_DIR}/"
    echo " Stop:            $0 clean"
    echo " Benchmark:       $0 benchmark"
    echo "============================================================"
    echo ""
fi

# curl http://localhost:8000/v1/chat/completions \
#     -H 'Content-Type: application/json' \
#     -d '{
#         "model": "/host-workspace/Qwen/Qwen3-30B-A3B-Instruct-2507",
#         "messages": [{"role": "user", "content": "Hello, who are you?"}],
#         "max_tokens": 50
#     }'
