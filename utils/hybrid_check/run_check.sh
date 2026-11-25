#!/bin/bash
# On the GPU machine: start a 1P1D service of a hybrid model (LRU, SSD offload,
# the A/B runner's settings), run check_state_reuse.py against it, tear down.
#   bash run_check.sh [name]
# Logs stay in dynamo-log/<name>-logs/.
set -euo pipefail
NAME=${1:-q36-check}
ROOT=${ROOT:-/root/kvpilot}
SD=$ROOT/vllm-dev/examples/disaggregated/lmcache/disagg_prefill_lmcache_v1
export PYTHONPATH=$ROOT/vllm-dev:$ROOT/lmcache-dev:$ROOT/dynamo-dev/components/src:$ROOT/dynamo-dev/lib/bindings/kvbm/python
export PYTHONUNBUFFERED=1
export NUM_PREFILL_WORKERS=1 NUM_DECODE_WORKERS=1
export TP_SIZE=${TP_SIZE:-2}
# Qwen3.6's and Qwen3.8's block at this TP (disagg_lmcache.sh works it out
# from the config).
BLOCK=$((TP_SIZE == 4 ? 1024 : 1536))
export KVPILOT_LOAD_STALL=true
export KVPILOT_SSD_OFFLOAD_PATH=${KVPILOT_SSD_OFFLOAD_PATH:-/root/kvpilot-ssd}
export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=${MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES:-4398046511104}
export MODEL=${MODEL:-/cfs_cloud_code/kiry/model/Qwen/Qwen3.6-35B-A3B}
export PREFILL_COMPILATION_CONFIG=${PREFILL_COMPILATION_CONFIG-'{"cudagraph_capture_sizes":[64,128,256,512,1024]}'}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
export SETUP_MODELS_WAIT_S=${SETUP_MODELS_WAIT_S:-1800}
DRAM_GIB=${DRAM_GIB:-100}
CFG=$ROOT/utils/placement_ab/mooncake-prefiller-${DRAM_GIB}g.yaml
sed "s/^  global_segment_size: .*/  global_segment_size: $((DRAM_GIB * 1024 * 1024 * 1024))  # ${DRAM_GIB}GiB per TP worker/" \
    $SD/configs/mooncake-prefiller-config.yaml > $CFG
export LMCACHE_PREFILL_CFG=$CFG

cd $SD
bash dynamo_script.sh cleanforce >/dev/null 2>&1 || true
bash dynamo_script.sh setup "$NAME"
# Qwen3.6's attention kernel splits the block (NIXL logs the mismatch);
# Qwen3.8's does not.
grep -qE "logical block size \($BLOCK\)|NIXL local agent metadata prepared: .* block_size=$BLOCK " \
    dynamo-log/${NAME}-logs/prefill-0.log \
    || { echo "prefill does not run block $BLOCK"; bash dynamo_script.sh cleanforce; exit 1; }
for seed in ${SEEDS:-0}; do
    echo "== seed $seed"
    python3 $ROOT/utils/hybrid_check/check_state_reuse.py --model "$MODEL" --url http://localhost:8000 \
        --block $BLOCK --seed $seed || echo "check exited non-zero"
done
bash dynamo_script.sh cleanforce >/dev/null 2>&1 || true
