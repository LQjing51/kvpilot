#!/bin/bash
# MoE kernel / expert-parallel matrix for Qwen3.6-FP8 (TP 2) and Qwen3.8 (TP 4),
# one config at a time on an otherwise idle machine. Logs to $OUT/<name>.log.
#   bash run_all.sh
set -uo pipefail
ROOT=${ROOT:-/root/kvpilot}
OUT=${OUT:-/root/moe_bench}
Q36=${Q36:-/cfs_cloud_code/kiry/model/Qwen/Qwen3.6-35B-A3B-FP8}
Q38=${Q38:-/root/models/Qwen/Qwen3.8-Flash-Next-FP8}
export PYTHONPATH=$ROOT/vllm-dev
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=/usr/local/cuda/bin:$PATH
unset KVPILOT_PLACEMENT
mkdir -p $OUT
gpu_free() {
    for i in $(seq 1 60); do
        [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -le 100 ] && return
        sleep 5
    done
}
run() {  # name gpus args...
    local name=$1 gpus=$2; shift 2
    gpu_free
    echo "== $name $(date -u +%T)"
    CUDA_VISIBLE_DEVICES=$gpus timeout 1500 python3 $ROOT/utils/moe_bench/bench_kernels.py "$@" > $OUT/$name.log 2>&1
    echo "   rc=$? $(grep -E '^(RESULT|PROFILED)' $OUT/$name.log | tr '\n' ';')"
}
run q36_tp_triton 0,1 $Q36 2 0 auto
run q36_ep_ficutlass 0,1 $Q36 2 1 auto
run q36_ep_triton 0,1 $Q36 2 1 triton
run q36_tp_ficutlass 0,1 $Q36 2 0 flashinfer_cutlass
run q38_ep_ficutlass 0,1,2,3 $Q38 4 1 auto
run q38_ep_triton 0,1,2,3 $Q38 4 1 triton
run q38_tp_triton 0,1,2,3 $Q38 4 0 auto
run q36_tp_triton_prof 0,1 $Q36 2 0 auto $OUT/prof_q36_tp_triton
run q38_ep_ficutlass_prof 0,1,2,3 $Q38 4 1 auto $OUT/prof_q38_ep_ficutlass
echo "== done $(date -u +%T)"
