#!/bin/bash
# Replay one reordered trace (make_order.py) on 1P1D: Qwen3-30B-A3B BF16, TP2,
# HBM prefix cache + Mooncake DRAM + SSD offload. Runs on the GPU host.
#
#   TRACE=datasets/agent-traces/swe-smith-trajectories/swe_mooncake_interleaved.jsonl \
#       NAME=order-swe-end bash run.sh
#   TRACE=datasets/agent-traces/tracelab/tracelab_mooncake_interleaved.jsonl \
#       NAME=order-tracelab-end CLIENTS=16 bash run.sh
#
#   TRACE       a relative path is taken from the repo root (/root/kvpilot)
#   DRAM_GIB    Mooncake global_segment_size per TP rank, default 100 (the
#               shared config file says 200)
#   HBM_BLOCKS  prefill KV pool in 512-token blocks (--num-gpu-blocks-override);
#               empty = whatever GPU_MEM_UTIL leaves (2361 at 0.9)
#   CLIENTS     requests in flight (mooncake_trace_replay.py -p), default 32
#
# Lines leave in file order and a round waits for the previous round of its
# session to finish (the replay always does). Logs stay in
# dynamo-log/$NAME-logs; compressed copies and the tier_hits.py report go to
# logs/$NAME/ next to this script.
set -uo pipefail
export PYTHONPATH=/root/kvpilot/vllm-dev:/root/kvpilot/lmcache-dev:/root/kvpilot/dynamo-dev/components/src:/root/kvpilot/dynamo-dev/lib/bindings/kvbm/python
export PYTHONUNBUFFERED=1
export NUM_PREFILL_WORKERS=1 NUM_DECODE_WORKERS=1
export KVPILOT_LOAD_STALL=true
export KVPILOT_SSD_OFFLOAD_PATH=${KVPILOT_SSD_OFFLOAD_PATH:-/root/kvpilot-ssd}
export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=${MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES:-4398046511104}
export MODEL=${MODEL:-/cfs_cloud_code/kiry/model/Qwen/Qwen3-30B-A3B-Instruct-2507}
export PREFILL_COMPILATION_CONFIG=${PREFILL_COMPILATION_CONFIG-'{"cudagraph_capture_sizes":[64,128,256,512,1024]}'}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}

HERE=/root/kvpilot/utils/trace_order
SD=/root/kvpilot/vllm-dev/examples/disaggregated/lmcache/disagg_prefill_lmcache_v1
TRACE=${TRACE:?set TRACE}
[[ "$TRACE" = /* ]] || TRACE=/root/kvpilot/$TRACE
NAME=${NAME:?set NAME}
CLIENTS=${CLIENTS:-32}
DRAM_GIB=${DRAM_GIB:-100}
L=$SD/dynamo-log/${NAME}-logs
OUT=$HERE/logs/$NAME
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

if [[ -n "${HBM_BLOCKS:-}" ]]; then
    export PREFILL_EXTRA_ARGS="--num-gpu-blocks-override $HBM_BLOCKS"
fi
export LMCACHE_PREFILL_CFG=/tmp/${NAME}-prefiller.yaml
sed -E "s/^(  global_segment_size:).*/\1 $((DRAM_GIB * 1024 * 1024 * 1024))/" \
    $SD/configs/mooncake-prefiller-config.yaml > $LMCACHE_PREFILL_CFG

cd $SD
bash dynamo_script.sh cleanforce >/dev/null 2>&1
for i in $(seq 1 60); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -le 100 ] && break
    sleep 5
done
say "setup $NAME: trace $(basename $TRACE) ($(wc -l < $TRACE) lines), clients $CLIENTS," \
    "hbm_blocks ${HBM_BLOCKS:-default}, dram $DRAM_GIB GiB/rank"
bash dynamo_script.sh setup $NAME > /tmp/$NAME-setup.out 2>&1 || { say "SETUP FAILED"; exit 1; }
n=$(grep -c "KVPilot placement=" $L/prefill-0.log); say "repo-code check: $n"
[ "$n" -gt 0 ] || { say "ABORT: stock vLLM"; bash dynamo_script.sh cleanforce; exit 1; }
say "$(grep -m1 -o 'GPU KV cache size: [0-9,]* tokens' $L/prefill-0.log)," \
    "segment $(grep -m1 -o "'global_segment_size': '[0-9]*'" $L/prefill-0.log)"

python3 /root/kvpilot/vllm-dev/benchmarks/multi_turn/mooncake_trace_replay.py \
    -i $TRACE -m $MODEL -u http://localhost:8000 -p $CLIENTS \
    --vocab-size 151936 > $L/benchmark.log 2>&1
say "replay done ($(grep -c '^\[ok \]' $L/benchmark.log) ok, $(grep -c '^\[ERR\]' $L/benchmark.log) err)"
say "errors in prefill log: $(grep -c ' ERROR ' $L/prefill-0.log)"
say "mooncake: $(tail -1 $L/mooncake_master.log | grep -o 'Mem Storage: [^|]*')" \
    "$(tail -1 $L/mooncake_master.log | grep -o 'Eviction: [^,]*')"
say "ssd used: $(du -sh $KVPILOT_SSD_OFFLOAD_PATH | cut -f1)"
bash dynamo_script.sh cleanforce >/dev/null 2>&1

mkdir -p $OUT
python3 $HERE/tier_hits.py $L $TRACE ${HBM_BLOCKS:+--hbm $HBM_BLOCKS} \
    --dram-gib $DRAM_GIB | tee $OUT/tier_hits.txt
for f in prefill-0.log mooncake_master.log benchmark.log; do
    gzip -c $L/$f > $OUT/$f.gz
done
say "done $NAME"
