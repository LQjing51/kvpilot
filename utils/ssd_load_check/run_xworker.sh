#!/bin/bash
# Cross-worker SSD check: 2P2D with Mooncake SSD offload, the trace from
# make_trace_xworker.py with each request pinned to prefill worker A or B
# (nvext.prefill_worker_id). analyze_xworker.py reads prefill-{0,1}.log.
export PYTHONPATH=/root/kvpilot/vllm-dev:/root/kvpilot/lmcache-dev:/root/kvpilot/dynamo-dev/components/src:/root/kvpilot/dynamo-dev/lib/bindings/kvbm/python
export PYTHONUNBUFFERED=1
export KVPILOT_LOAD_STALL=true
export NUM_PREFILL_WORKERS=2
export MOONCAKE_PRELOAD_EXPECTED_ENGINES=2
export KVPILOT_SSD_OFFLOAD_PATH=${KVPILOT_SSD_OFFLOAD_PATH:-/root/kvpilot-ssd}
export MODEL=${MODEL:-/root/models/Qwen/Qwen3-30B-A3B-Instruct-2507}
export PREFILL_COMPILATION_CONFIG=${PREFILL_COMPILATION_CONFIG-'{"cudagraph_capture_sizes":[64,128,256,512,1024]}'}
HERE=/root/kvpilot/utils/ssd_load_check
SD=/root/kvpilot/vllm-dev/examples/disaggregated/lmcache/disagg_prefill_lmcache_v1
NAME=${NAME:-ssd-xworker}
L=$SD/dynamo-log/${NAME}-logs
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
cd $SD
bash dynamo_script.sh cleanforce >/dev/null 2>&1
for i in $(seq 1 60); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -le 100 ] && break
    sleep 5
done
say "setup $NAME"
bash dynamo_script.sh setup $NAME | tee /tmp/$NAME-setup.out || { say "SETUP FAILED"; exit 1; }
n=$(grep -c "KVPilot placement=" $L/prefill-0.log); say "repo-code check: $n"
[ "$n" -gt 0 ] || { say "ABORT: stock vLLM"; bash dynamo_script.sh cleanforce; exit 1; }
read -r A B < <(grep -m1 "Prefill IDs:" /tmp/$NAME-setup.out | sed 's/.*Prefill IDs: *//')
[ -n "$B" ] || { say "ABORT: need two prefill ids, got '$A $B'"; bash dynamo_script.sh cleanforce; exit 1; }
say "prefill A=$A B=$B"
python3 - "$A" "$B" "$HERE/trace_xworker.jsonl" > $L/trace.jsonl <<'PY'
import json, sys
ids = {"A": int(sys.argv[1]), "B": int(sys.argv[2])}
for line in open(sys.argv[3]):
    r = json.loads(line)
    r.setdefault("extra_body", {})["nvext"] = {"prefill_worker_id": ids[r["worker"]]}
    print(json.dumps(r))
PY
python3 /root/kvpilot/vllm-dev/benchmarks/multi_turn/mooncake_trace_replay.py \
    -i $L/trace.jsonl -m $MODEL \
    -u http://localhost:8000 -p 1 --vocab-size 151936 --time-scale 1 \
    > $L/benchmark.log 2>&1
say "benchmark done ($(grep -c '^\[ok \]' $L/benchmark.log) ok, $(grep -c '^\[ERR\]' $L/benchmark.log) err)"
say "errors: $(cat $L/prefill-*.log | grep -c ' ERROR ')"
bash dynamo_script.sh cleanforce >/dev/null 2>&1
say "all done"
