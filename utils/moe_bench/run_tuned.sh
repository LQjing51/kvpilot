#!/bin/bash
# The Triton expert-parallel configs again with the tuned tables
# (VLLM_TUNED_CONFIG_FOLDER=$OUT/tuned, from run_q38_tune.sh).
#   bash run_tuned.sh
set -uo pipefail
ROOT=${ROOT:-/root/kvpilot}
OUT=${OUT:-/root/moe_bench}
Q36=${Q36:-/cfs_cloud_code/kiry/model/Qwen/Qwen3.6-35B-A3B-FP8}
Q38=${Q38:-/root/models/Qwen/Qwen3.8-Flash-Next-FP8}
export PYTHONPATH=$ROOT/vllm-dev
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=/usr/local/cuda/bin:$PATH
export VLLM_TUNED_CONFIG_FOLDER=$OUT/tuned
unset KVPILOT_PLACEMENT
gpu_free() {
    for i in $(seq 1 60); do
        [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -le 100 ] && return
        sleep 5
    done
}
run() {  # name gpus util args...
    local name=$1 gpus=$2 util=$3; shift 3
    for attempt in 1 2; do
        gpu_free
        echo "== $name attempt $attempt $(date -u +%T)"
        GPU_UTIL=$util CUDA_VISIBLE_DEVICES=$gpus timeout 1500 python3 $ROOT/utils/moe_bench/bench_kernels.py "$@" > $OUT/$name.log 2>&1
        rc=$?
        echo "   rc=$rc $(grep -E '^RESULT' $OUT/$name.log | tr '\n' ';')"
        grep -m1 -oE "Using configuration from [^ ]+" $OUT/$name.log
        [ $rc -eq 0 ] && return
    done
}
run q36_ep_triton_tuned 0,1 0.85 $Q36 2 1 triton
run q38_ep_triton_tuned 0,1,2,3 0.9 $Q38 4 1 triton
# Keyed by the engine's tokens per local expert (shift_tuned.py).
rm -rf $OUT/tuned_shift
python3 $ROOT/utils/moe_bench/shift_tuned.py "$OUT/tuned/E=128,N=512,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json" 2 $OUT/tuned_shift
python3 $ROOT/utils/moe_bench/shift_tuned.py "$OUT/tuned/E=128,N=640,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json" 4 $OUT/tuned_shift
export VLLM_TUNED_CONFIG_FOLDER=$OUT/tuned_shift
run q36_ep_triton_shift 0,1 0.85 $Q36 2 1 triton
run q38_ep_triton_shift 0,1,2,3 0.9 $Q38 4 1 triton
echo "== done $(date -u +%T)"
