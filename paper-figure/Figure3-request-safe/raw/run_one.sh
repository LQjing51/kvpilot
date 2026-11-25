#!/bin/bash
# Run one dataset: toolagent | swe | swe_hbm1to2 | swe_hbm1to2_ssdmax | tracelab
#                  | swe_reqprio | tracelab_reqprio
#   bash raw/run_one.sh swe
#   RAW=raw/ssdcap KVPILOT_SSD_READ_GIBPS=1.6 bash raw/run_one.sh swe
# A dataset that make_trace split (<RAW>/<ds>/part*/) runs part by part, each on
# a fresh service (cleanforce also empties the SSD offload dir). A run dir that
# already holds its logs is skipped.
set -euo pipefail
DS=${1:?usage: run_one.sh toolagent|swe|swe_hbm1to2|swe_hbm1to2_ssdmax|tracelab|swe_reqprio|tracelab_reqprio|swe_blockprio_k160}
case $DS in toolagent|swe|swe_hbm1to2|swe_hbm1to2_ssdmax|tracelab|swe_reqprio|tracelab_reqprio|swe_blockprio_k160) ;; *) echo "bad ds $DS"; exit 1 ;; esac
# Repo root on the GPU machine (another checkout: ROOT=/root/q38).
ROOT=${ROOT:-/root/kvpilot}
export PYTHONPATH=$ROOT/vllm-dev:$ROOT/lmcache-dev:$ROOT/dynamo-dev/components/src:$ROOT/dynamo-dev/lib/bindings/kvbm/python
export PYTHONUNBUFFERED=1
export NUM_PREFILL_WORKERS=1 NUM_DECODE_WORKERS=1 MOONCAKE_PRELOAD_EXPECTED_ENGINES=1
# TP ranks per worker; the trace's rows must say the same (make_trace --model).
export TP_SIZE=${TP_SIZE:-2}
export KVPILOT_LOAD_STALL=true
export KVPILOT_SSD_OFFLOAD_PATH=${KVPILOT_SSD_OFFLOAD_PATH:-/root/kvpilot-ssd}
export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=${MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES:-4398046511104}
export MODEL=${MODEL:-/cfs_cloud_code/kiry/model/Qwen/Qwen3-30B-A3B-Instruct-2507}
export PREFILL_COMPILATION_CONFIG=${PREFILL_COMPILATION_CONFIG-'{"cudagraph_capture_sizes":[64,128,256,512,1024]}'}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
HERE=$ROOT/paper-figure/Figure3-request-safe
RAW=${RAW:-raw}
# Per-process SSD read cap (GiB/s); peak / GPUs per host models a full machine.
TAG=${KVPILOT_SSD_READ_GIBPS:+-ssd${KVPILOT_SSD_READ_GIBPS}}
SD=$ROOT/vllm-dev/examples/disaggregated/lmcache/disagg_prefill_lmcache_v1
VOCAB=$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1]+"/config.json")); print(c.get("text_config", c)["vocab_size"])' "$MODEL")
REPLAY="python3 $ROOT/vllm-dev/benchmarks/multi_turn/mooncake_trace_replay.py -m $MODEL -u http://localhost:8000 -p 1 --vocab-size $VOCAB --time-scale 1"
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

run() {  # run dir, service name
    local RUN=$1 NAME=$2
    local L=$SD/dynamo-log/${NAME}-logs
    if [ -f $RUN/prefill-0.log.gz ] && [ -f $RUN/benchmark.log ]; then
        say "skip $NAME (logs already in $RUN)"
        return
    fi
    cd $SD
    bash dynamo_script.sh cleanforce >/dev/null 2>&1
    for i in $(seq 1 60); do
        [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -le 100 ] && break
        sleep 5
    done
    local tp=$(head -1 $RUN/trace.jsonl | python3 -c 'import json,sys; print(json.load(sys.stdin).get("tp", 2))')
    [ "$tp" = "$TP_SIZE" ] || { say "ABORT: trace rows are for TP $tp, TP_SIZE=$TP_SIZE"; exit 1; }
    say "setup $NAME (1P1D TP$TP_SIZE, GPU_MEM_UTIL=$GPU_MEM_UTIL, KVPILOT_SSD_READ_GIBPS=${KVPILOT_SSD_READ_GIBPS:-off})"
    bash dynamo_script.sh setup $NAME > /tmp/$NAME-setup.out 2>&1 || { say "SETUP FAILED"; exit 1; }
    n=$(cat $L/prefill-*.log | grep -c "KVPilot placement="); say "repo-code check: $n"
    [ "$n" -gt 0 ] || { say "ABORT: stock vLLM"; bash dynamo_script.sh cleanforce; exit 1; }
    say "mooncake: $(grep -ho 'Starting periodic task with interval: [0-9]*ms' $L/prefill-*.log | sort | uniq -c | tr '\n' ' ')"
    say "disk free $(df -h /root | tail -1 | awk '{print $4}')"
    say "trace $(wc -l < $RUN/trace.jsonl) requests"
    $REPLAY -i $RUN/trace.jsonl > $L/benchmark.log 2>&1
    say "main done ($(grep -c '^\[ok \]' $L/benchmark.log) ok, $(grep -c '^\[ERR\]' $L/benchmark.log) err)"
    (cd $HERE && python3 retry.py "$L" "$RUN")
    n=$(wc -l < $RUN/trace_retry.jsonl)
    if [ "${n:-0}" -gt 0 ]; then
        $REPLAY -i $RUN/trace_retry.jsonl > $L/benchmark_retry.log 2>&1
        say "retry done ($(grep -c '^\[ok \]' $L/benchmark_retry.log) ok, $(grep -c '^\[ERR\]' $L/benchmark_retry.log) err)"
    fi
    say "errors: $(cat $L/prefill-*.log | grep -c ' ERROR ' || true)"
    say "mooncake: $(tail -1 $L/mooncake_master.log | grep -o 'Mem Storage: [^|]*') $(tail -1 $L/mooncake_master.log | grep -o 'Eviction: [^,]*')"
    say "ssd used: $(du -sh $KVPILOT_SSD_OFFLOAD_PATH | cut -f1)"
    bash dynamo_script.sh cleanforce >/dev/null 2>&1
    gzip -c $L/prefill-0.log > $RUN/prefill-0.log.gz
    cp -f $L/benchmark.log $L/mooncake_master.log $RUN/
    [ -f $L/benchmark_retry.log ] && cp -f $L/benchmark_retry.log $RUN/
    say "part done ($NAME)"
}

parts=$(find $HERE/$RAW/$DS -mindepth 1 -maxdepth 1 -type d -name 'part*' | sort)
if [ -n "$parts" ]; then
    for p in $parts; do
        run $p agent-request-safe$TAG-$DS-$(basename $p)
    done
else
    run $HERE/$RAW/$DS agent-request-safe$TAG-$DS
fi
say "all done ($DS)"
