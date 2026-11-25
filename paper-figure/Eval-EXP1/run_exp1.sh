#!/bin/bash
# Eval-EXP1 on the GPU machine: A/B arms of one model on one trace, one after
# another, each on a fresh 1P1D service (utils/placement_ab/run_ab.sh), logs
# archived to raw/<ds>/<model>/<arm>/.
#   [REP=r2] bash run_exp1.sh <exp1swe|exp1tl|exp1swr|exp1tlr> <q30|q36|q36f8|q38> <arm> [<arm> ...]
# REP: a repeat of arms already run, archived to raw/<ds>/<model>/<arm>-<REP>/.
# Every model sees the same trace with the same KV capacity in tokens:
# Qwen3-30B-A3B's defaults at TP 2 (HBM 2361 blocks of 512 = 1.21M tokens,
# DRAM 100 GiB/rank = 1.97M tokens at the 0.9 watermark); the TP 4 hybrids get
# 1180 blocks of 1024 plus their sessions' linear-attention states in HBM (3
# per session for Qwen3.6, 4 for Qwen3.8, K sessions, K from the trace's
# spec), and the DRAM GiB per rank that holds ~1.97M tokens of their KV.
# Within a model only KVPILOT_PLACEMENT differs between arms.
set -uo pipefail
DS=${1:?usage: run_exp1.sh <exp1swe|exp1tl|exp1swr|exp1tlr> <q30|q36|q36f8|q38> <arm>...}
M=${2:?usage: run_exp1.sh <exp1swe|exp1tl|exp1swr|exp1tlr> <q30|q36|q36f8|q38> <arm>...}
shift 2
export ROOT=${ROOT:-/root/kvpilot}
HERE=$ROOT/paper-figure/Eval-EXP1
SD=$ROOT/vllm-dev/examples/disaggregated/lmcache/disagg_prefill_lmcache_v1
# Active sessions of each trace (its spec's K, README).
case $DS in
    exp1swe) K=176 NAME=swe ;;
    exp1tl) K=${EXP1_TL_K:?set EXP1_TL_K to the TraceLab trace spec K} NAME=tracelab ;;
    exp1swr) K=144 NAME=swe_rand ;;
    exp1tlr) K=160 NAME=tl_rand ;;
    *) echo "bad dataset $DS"; exit 1 ;;
esac
case $M in
    q30) export MODEL=/cfs_cloud_code/kiry/model/Qwen/Qwen3-30B-A3B-Instruct-2507 TP_SIZE=2
         export HBM_BLOCKS=2361 DRAM_GIB=100 ;;
    q36) export MODEL=/cfs_cloud_code/kiry/model/Qwen/Qwen3.6-35B-A3B TP_SIZE=4
         export HBM_BLOCKS=$((1180 + 3 * K)) DRAM_GIB=21 ;;
    # Qwen3.6 FP8 at TP 2 (no replicated KV heads; KV stays BF16): blocks of
    # 1536, 787 of them hold Qwen3-30B's 1.21M tokens.
    q36f8) export MODEL=/cfs_cloud_code/kiry/model/Qwen/Qwen3.6-35B-A3B-FP8 TP_SIZE=2
           export HBM_BLOCKS=$((787 + 3 * K)) DRAM_GIB=21 ;;
    q38) export MODEL=/root/models/Qwen/Qwen3.8-Flash-Next-FP8 TP_SIZE=4
         export HBM_BLOCKS=$((1180 + 4 * K)) DRAM_GIB=27 ;;
    *) echo "bad model $M"; exit 1 ;;
esac
export SETUP_MODELS_WAIT_S=${SETUP_MODELS_WAIT_S:-1800}
REP=${REP:-}
TAG=$M${REP:+-$REP}
for arm in "$@"; do
    echo "===== $DS $M $arm${REP:+ ($REP)} (HBM $HBM_BLOCKS blocks, DRAM $DRAM_GIB GiB/rank) $(date -u +%H:%M:%S)"
    bash $ROOT/utils/placement_ab/run_ab.sh "$arm" "$DS" "$TAG"
    L=$SD/dynamo-log/pab-$DS-$arm-$TAG-logs
    OUT=$HERE/raw/$NAME/$M/$arm${REP:+-$REP}
    mkdir -p "$OUT"
    gzip -c "$L/prefill-0.log" > "$OUT/prefill-0.log.gz"
    gzip -c "$L/decode-0.log" > "$OUT/decode-0.log.gz"
    cp -f "$L/benchmark.log" "$L/run.out" "$OUT/" 2>/dev/null
    grep -E "KVPILOT-EVICT|KVPILOT-EXCL|placement=" "$L/mooncake_master.log" | gzip -c > "$OUT/mooncake_master.kvpilot.log.gz"
    echo "archived $OUT: $(grep -c '^\[ok \]' "$OUT/benchmark.log") ok, $(grep -c '^\[ERR\]' "$OUT/benchmark.log") err"
done
echo "===== done $DS $M $(date -u +%H:%M:%S)"
