#!/bin/bash
# Qwen3.8 (TP 4) part of the MoE matrix at GPU_UTIL 0.9, each config retried once
# (its first start recompiles, and the compile peak can leave no KV memory);
# then Triton configs tuned for the expert-parallel shapes of both models.
#   bash run_q38_tune.sh
set -uo pipefail
ROOT=${ROOT:-/root/kvpilot}
OUT=${OUT:-/root/moe_bench}
Q36=${Q36:-/cfs_cloud_code/kiry/model/Qwen/Qwen3.6-35B-A3B-FP8}
Q38=${Q38:-/root/models/Qwen/Qwen3.8-Flash-Next-FP8}
export PYTHONPATH=$ROOT/vllm-dev
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=/usr/local/cuda/bin:$PATH
unset KVPILOT_PLACEMENT
gpu_free() {
    for i in $(seq 1 60); do
        [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -le 100 ] && return
        sleep 5
    done
}
run() {  # name gpus args...
    local name=$1 gpus=$2; shift 2
    for attempt in 1 2; do
        gpu_free
        echo "== $name attempt $attempt $(date -u +%T)"
        GPU_UTIL=0.9 CUDA_VISIBLE_DEVICES=$gpus timeout 1500 python3 $ROOT/utils/moe_bench/bench_kernels.py "$@" > $OUT/$name.log 2>&1
        rc=$?
        echo "   rc=$rc $(grep -E '^(RESULT|PROFILED)' $OUT/$name.log | tr '\n' ';')"
        [ $rc -eq 0 ] && return
    done
}
run q38_ep_ficutlass 0,1,2,3 $Q38 4 1 auto
run q38_ep_triton 0,1,2,3 $Q38 4 1 triton
run q38_tp_triton 0,1,2,3 $Q38 4 0 auto
run q38_ep_ficutlass_prof 0,1,2,3 $Q38 4 1 auto $OUT/prof_q38_ep_ficutlass
gpu_free
cd $ROOT/vllm-dev
for spec in "q36 $Q36 2" "q38 $Q38 4"; do
    set -- $spec
    echo "== tune $1 EP $(date -u +%T)"
    timeout 5400 python3 benchmarks/kernels/benchmark_moe.py --model $2 --tp-size $3 --enable-expert-parallel \
        --dtype fp8_w8a8 --tune --save-dir $OUT/tuned \
        --batch-size 64 128 256 512 1024 1536 2048 3072 4096 6144 8192 > $OUT/tune_$1.log 2>&1
    echo "   rc=$? $(ls $OUT/tuned 2>/dev/null | tr '\n' ' ')"
done
echo "== done $(date -u +%T)"
