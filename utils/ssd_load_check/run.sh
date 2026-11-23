#!/bin/bash
# End-to-end SSD load check: one 1P2D service with Mooncake SSD offload, the
# trace from make_trace.py (per group: warm, then the same prompt loading its
# prefix from HBM [1], DRAM [2] and SSD [4]). Logs land in
# dynamo-log/$NAME-logs; analyze.py reads prefill-0.log.
export PYTHONPATH=/root/kvpilot/vllm-dev:/root/kvpilot/lmcache-dev:/root/kvpilot/dynamo-dev/components/src:/root/kvpilot/dynamo-dev/lib/bindings/kvbm/python
export PYTHONUNBUFFERED=1
export KVPILOT_LOAD_STALL=true
export NUM_PREFILL_WORKERS=1
export MOONCAKE_PRELOAD_EXPECTED_ENGINES=1
export KVPILOT_SSD_OFFLOAD_PATH=${KVPILOT_SSD_OFFLOAD_PATH:-/root/kvpilot-ssd}
export MODEL=${MODEL:-/root/models/Qwen/Qwen3-30B-A3B-Instruct-2507}
export PREFILL_COMPILATION_CONFIG=${PREFILL_COMPILATION_CONFIG-'{"cudagraph_capture_sizes":[64,128,256,512,1024]}'}
HERE=/root/kvpilot/utils/ssd_load_check
SD=/root/kvpilot/vllm-dev/examples/disaggregated/lmcache/disagg_prefill_lmcache_v1
TRACE=${TRACE:-$HERE/trace.jsonl}
NAME=${NAME:-ssd-load-check}
L=$SD/dynamo-log/${NAME}-logs
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
cd $SD
bash dynamo_script.sh cleanforce >/dev/null 2>&1
for i in $(seq 1 60); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -le 100 ] && break
    sleep 5
done
say "setup $NAME (ssd at $KVPILOT_SSD_OFFLOAD_PATH)"
bash dynamo_script.sh setup $NAME || { say "SETUP FAILED"; exit 1; }
n=$(grep -c "KVPilot placement=" $L/prefill-0.log); say "repo-code check: $n"
[ "$n" -gt 0 ] || { say "ABORT: stock vLLM"; bash dynamo_script.sh cleanforce; exit 1; }
say "ssd offload: $(grep -c 'KVPilot: Mooncake SSD offload enabled' $L/prefill-0.log) clients"
python3 /root/kvpilot/vllm-dev/benchmarks/multi_turn/mooncake_trace_replay.py \
    -i $TRACE -m $MODEL \
    -u http://localhost:8000 -p 1 --vocab-size 151936 --time-scale 1 \
    > $L/benchmark.log 2>&1
say "benchmark done ($(grep -c '^\[ok \]' $L/benchmark.log) ok, $(grep -c '^\[ERR\]' $L/benchmark.log) err)"
say "retrieves with ssd: $(grep 'KVPilot prelookup retrieve' $L/prefill-0.log | grep -vc 'ssd_values=0 '), errors: $(grep -c ' ERROR ' $L/prefill-0.log)"
du -sh $KVPILOT_SSD_OFFLOAD_PATH
bash dynamo_script.sh cleanforce >/dev/null 2>&1
say "all done"
