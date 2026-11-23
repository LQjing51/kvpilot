#!/bin/bash
# A/B one arm of the KVPilot placement experiment on a fresh 1P1D service.
#   bash run_ab.sh <lru|belady|incl|incp|excl|exclu|exclb> <tracelab|swe|swe1to2b|swereq|swereqhbm|sweblock|sweblock160|swefile320|tlreq|toolagent> [tag]
# Every arm replays the same trace (*_mooncake_interleaved.jsonl,
# utils/trace_order; swe1to2b = swe_hbm1to2_budget220.jsonl; swereq / tlreq =
# datasets/req-priority-trace/{swe,tracelab}_reqprio.jsonl; swereqhbm = swereq with
# every signal HBM, signal_swe_reqprio_allhbm.json; sweblock =
# datasets/block-priority-trace/swe_blockprio.jsonl, HBM 600 blocks by default;
# sweblock160 = swe_blockprio_k160.jsonl, search_trace.py's pick for the default
# HBM, the first 160 lines warm-up; swefile320 = swe_fileorder_k320.jsonl, swe_full's
# sessions in file order, 320 at a time, the first 320 lines warm-up) in file order, one
# request in flight, with forced decode output, under the same HBM pool,
# Mooncake DRAM size and SSD offload; only KVPILOT_PLACEMENT differs.
#   lru         upstream HBM free queue (FIFO), Mooncake evicts whole chunk
#               groups in access order; SSD write-through, SSD hits loaded
#               through into DRAM.
#   belady      same copies as lru, but HBM and DRAM evict the block whose next
#               use in the replayed trace is farthest (Belady's MIN).
#   incl        inclusive-safe-reqwise: same copies as lru, but HBM and DRAM
#               evict by the next turn's safe tier (none < SSD < DRAM < HBM =
#               pending), FIFO / LRU within a tier.
#   incp        inclusive-penalty-reqwise: same copies as lru, but HBM and DRAM
#               evict by the request's penalty in that tier.
#   excl        exclusive-penalty-reqwise: one copy per block, moved to the
#               signalled tier; full tiers demote by penalty (the load the
#               move down would expose).
#   exclu       exclusive-penalty-reqwise-unblocking: excl, but a signal also
#               keeps one prefill chunk of HBM free and an admission short of
#               HBM borrows held blocks for its loaded prefix instead of
#               waiting for their write to DRAM.
#   exclb       exclusive-penalty-blockwise-unblocking: exclu, but each block
#               goes to its own tier (fewest HBM, then DRAM blocks that keep
#               the next turn safe) and is evicted by the expected TTFT its
#               move down adds.
# Every arm evicts a request from its tail in HBM and DRAM. All arms but lru
# take tiers and penalties from the time model (KVPILOT_TIME_COEFFS_PATH,
# default Figure3 fit_times2_swe.json) and the signal's suffix distribution.
# The signal is the next turn's tier and [t_hbm, t_dram, t_ssd] from the
# signal table signal_<trace>.json (make_req_table.py / toolagent_prep.py); with
# KVPILOT_ONLINE_PREDICT=1 it is predicted from the turn's tool calls instead.
#   HBM_BLOCKS  prefill KV pool in 512-token blocks; empty = what GPU_MEM_UTIL
#               leaves (2361 at 0.9), as in utils/trace_order/run.sh
#   DRAM_GIB    Mooncake global_segment_size per TP rank, default 100
#   MAX_OUT     cap on each request's output tokens (0 = trace value)
set -euo pipefail
ARM=${1:?usage: run_ab.sh lru|belady|incl|incp|excl|exclu|exclb tracelab|swe|swe1to2b|swereq|swereqhbm|sweblock|sweblock160|swefile320|tlreq|toolagent [tag]}
DS=${2:?usage: run_ab.sh lru|belady|incl|incp|excl|exclu|exclb tracelab|swe|swe1to2b|swereq|swereqhbm|sweblock|sweblock160|swefile320|tlreq|toolagent [tag]}
TAG=${3:-r1}
case $ARM in lru|belady|incl|incp|excl|exclu|exclb) ;; *) echo "bad arm $ARM"; exit 1 ;; esac
case $DS in tracelab|swe|swe1to2b|swereq|swereqhbm|sweblock|sweblock160|swefile320|tlreq|toolagent) ;; *) echo "bad dataset $DS"; exit 1 ;; esac

ROOT=/root/kvpilot
HERE=$ROOT/utils/placement_ab
SD=$ROOT/vllm-dev/examples/disaggregated/lmcache/disagg_prefill_lmcache_v1
export PYTHONPATH=$ROOT/vllm-dev:$ROOT/lmcache-dev:$ROOT/dynamo-dev/components/src:$ROOT/dynamo-dev/lib/bindings/kvbm/python
export PYTHONUNBUFFERED=1
export NUM_PREFILL_WORKERS=1 NUM_DECODE_WORKERS=${NUM_DECODE_WORKERS:-1}
export KVPILOT_LOAD_STALL=true
export KVPILOT_SSD_OFFLOAD_PATH=${KVPILOT_SSD_OFFLOAD_PATH:-/root/kvpilot-ssd}
export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=${MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES:-4398046511104}
export MODEL=${MODEL:-/cfs_cloud_code/kiry/model/Qwen/Qwen3-30B-A3B-Instruct-2507}
# Same compute regime the time model was fitted under (Figure3 request-safe).
export PREFILL_COMPILATION_CONFIG=${PREFILL_COMPILATION_CONFIG-'{"cudagraph_capture_sizes":[64,128,256,512,1024]}'}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}

# sweblock is built so HBM and DRAM stay full under blockwise with 600 HBM blocks
# (gen_trace_capacity.py).
[ "$DS" = sweblock ] && HBM_BLOCKS=${HBM_BLOCKS:-600}
HBM_BLOCKS=${HBM_BLOCKS:-}
DRAM_GIB=${DRAM_GIB:-100}
export PREFILL_EXTRA_ARGS="${HBM_BLOCKS:+--num-gpu-blocks-override $HBM_BLOCKS} ${PREFILL_EXTRA_ARGS_MORE:-}"
CFG=$HERE/mooncake-prefiller-${DRAM_GIB}g.yaml
sed "s/^  global_segment_size: .*/  global_segment_size: $((DRAM_GIB * 1024 * 1024 * 1024))  # ${DRAM_GIB}GiB per TP worker/" \
    $SD/configs/mooncake-prefiller-config.yaml > $CFG
export LMCACHE_PREFILL_CFG=$CFG

D=$ROOT/datasets/agent-traces
# RESP: the tool_resp distribution online prediction loads (swe / tracelab).
case $DS in
    tracelab) INPUT=$D/tracelab/tracelab_mooncake_interleaved.jsonl RESP=tracelab ;;
    swe) INPUT=$D/swe-smith-trajectories/swe_mooncake_interleaved.jsonl RESP=swe ;;
    swe1to2b) INPUT=$D/swe-smith-trajectories/swe_hbm1to2_budget220.jsonl RESP=swe ;;
    swereq) INPUT=$ROOT/datasets/req-priority-trace/swe_reqprio.jsonl RESP=swe ;;
    # Every request HBM safe (make_allhbm_table.py): HBM stays full of held blocks.
    swereqhbm) INPUT=$ROOT/datasets/req-priority-trace/swe_reqprio.jsonl RESP=swe
               TABLE=signal_swe_reqprio_allhbm.json ;;
    sweblock) INPUT=$ROOT/datasets/block-priority-trace/swe_blockprio.jsonl RESP=swe ;;
    sweblock160) INPUT=$ROOT/datasets/block-priority-trace/swe_blockprio_k160.jsonl RESP=swe ;;
    swefile320) INPUT=$ROOT/datasets/block-priority-trace/swe_fileorder_k320.jsonl RESP=swe ;;
    tlreq) INPUT=$ROOT/datasets/req-priority-trace/tracelab_reqprio.jsonl RESP=tracelab ;;
    # First 4000 requests of the Mooncake toolagent trace, sessions and tiers
    # from toolagent_prep.py; it has no tool labels, so offline table only.
    toolagent) INPUT=$ROOT/datasets/mooncake-trace/toolagent_sessions.jsonl RESP=swe ;;
esac
# AB_INPUT: replay another file (e.g. the first lines of the trace, for a smoke
# run); the signal table is still signal_<its basename>.json unless the dataset
# names one (TABLE).
INPUT=${AB_INPUT:-$INPUT}

unset KVPILOT_PLACEMENT KVPILOT_OFFLINE_TABLE KVPILOT_TOOL_RESP_DISTRIBUTION_PATH KVPILOT_BELADY_TRACE
case $ARM in
    belady) export KVPILOT_PLACEMENT=belady KVPILOT_BELADY_TRACE=$INPUT ;;
    incl) export KVPILOT_PLACEMENT=inclusive-safe-reqwise ;;
    incp) export KVPILOT_PLACEMENT=inclusive-penalty-reqwise ;;
    excl) export KVPILOT_PLACEMENT=exclusive-penalty-reqwise ;;
    exclu) export KVPILOT_PLACEMENT=exclusive-penalty-reqwise-unblocking ;;
    exclb) export KVPILOT_PLACEMENT=exclusive-penalty-blockwise-unblocking ;;
esac
case $ARM in incl|incp|excl|exclu|exclb)
    export KVPILOT_TIME_COEFFS_PATH=${KVPILOT_TIME_COEFFS_PATH:-$ROOT/paper-figure/Figure3-request-safe/fit_times2_swe.json} ;;
esac
export KVPILOT_ONLINE_PREDICT=${KVPILOT_ONLINE_PREDICT:-0}
if [ "$ARM" != lru ] && [ "$ARM" != belady ]; then
    if [ "$KVPILOT_ONLINE_PREDICT" = 1 ]; then
        export KVPILOT_TOOL_RESP_DISTRIBUTION_PATH=$ROOT/paper-figure/algorithm/resp_length_dist_$RESP.json
        export KVPILOT_TIME_COEFFS_PATH=${KVPILOT_TIME_COEFFS_PATH:-$ROOT/paper-figure/Figure3-request-safe/fit_times2_swe.json}
    else
        export KVPILOT_OFFLINE_TABLE=$HERE/${TABLE:-signal_$(basename "$INPUT" .jsonl).json}
        [ -f "$KVPILOT_OFFLINE_TABLE" ] || { echo "missing $KVPILOT_OFFLINE_TABLE"; exit 1; }
    fi
fi

NAME=pab-$DS-$ARM-$TAG
L=$SD/dynamo-log/${NAME}-logs
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

cd $SD
bash dynamo_script.sh cleanforce >/dev/null 2>&1 || true
for i in $(seq 1 60); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -le 100 ] && break
    sleep 5
done
say "setup $NAME arm=$ARM ds=$DS hbm_blocks=${HBM_BLOCKS:-default} dram=${DRAM_GIB}GiB/rank max_out=${MAX_OUT:-0} ssd_read_gibps=${KVPILOT_SSD_READ_GIBPS:-off} coeffs=${KVPILOT_TIME_COEFFS_PATH:-default}"
# Worker ports (e.g. 51400) sit inside the ephemeral range, so a setup can
# lose a bind race to an unrelated outgoing connection; retry on a clean slate.
ok=0
for attempt in 1 2 3; do
    if bash dynamo_script.sh setup $NAME > /tmp/$NAME-setup.out 2>&1; then ok=1; break; fi
    say "setup attempt $attempt failed: $(grep -ho 'EADDRINUSE[^,]*\|RuntimeError: [^\n]*' $L/prefill-0.log 2>/dev/null | head -1)"
    bash dynamo_script.sh cleanforce >/dev/null 2>&1 || true
    sleep 20
done
[ "$ok" = 1 ] || { say "SETUP FAILED"; tail -30 /tmp/$NAME-setup.out; exit 1; }
n=$(grep -c "KVPilot placement=" $L/prefill-0.log || true)
say "repo-code check: $n"
[ "$n" -gt 0 ] || { say "ABORT: stock vLLM"; bash dynamo_script.sh cleanforce; exit 1; }
say "prefill: $(grep -ho 'KVPilot placement=[a-z-]*\|KVPilot placement off[^,]*\|KVPilot heap: [A-Za-z+]*' $L/prefill-0.log | sort | uniq -c | tr '\n' ' ')"
say "master: $(grep -ho 'KVPilot placement=[a-z-]*' $L/mooncake_master.log | head -1)"
say "$(grep -m1 -o 'GPU KV cache size: [0-9,]* tokens' $L/prefill-0.log)"
{
    echo "arm=$ARM ds=$DS tag=$TAG hbm_blocks=${HBM_BLOCKS:-default} dram_gib=$DRAM_GIB max_out=${MAX_OUT:-0}"
    echo "trace=$INPUT"
    env | grep -E '^(KVPILOT|MOONCAKE|LMCACHE|PREFILL|GPU_MEM|NUM_)' | sort
} > $L/run.out
# -p 1 released at the first token → one prefill at a time, no P queue;
# TTFT ≈ prefill + fixed P→D. A session's next round waits for the previous one.
python3 $ROOT/vllm-dev/benchmarks/multi_turn/mooncake_trace_replay.py \
    -i $INPUT -m $MODEL -u http://localhost:8000 -p 1 --release-at-first-token \
    --vocab-size 151936 --forced-output \
    ${MAX_OUT:+--max-output-tokens $MAX_OUT} \
    > $L/benchmark.log 2>&1 || say "replay exited non-zero"
say "replay done ($(grep -c '^\[ok \]' $L/benchmark.log) ok, $(grep -c '^\[ERR\]' $L/benchmark.log) err)"
say "placement applied: $(grep -c 'KVPilot placement applied' $L/prefill-0.log || true)  kvp-evict: $(grep -c 'KVPILOT-EVICT' $L/mooncake_master.log || true)  writebacks: $(grep -c 'KVPilot writeback req=.* values=' $L/prefill-0.log || true)  excl jobs: $(grep -c 'KVPilot exclusive job=' $L/prefill-0.log || true)"
say "mooncake: $(tail -1 $L/mooncake_master.log | grep -o 'Mem Storage: [^|]*') $(tail -1 $L/mooncake_master.log | grep -o 'Eviction: [^,]*')"
say "ssd used: $(du -sh $KVPILOT_SSD_OFFLOAD_PATH | cut -f1)"
bash dynamo_script.sh cleanforce >/dev/null 2>&1 || true
say "done $NAME"
