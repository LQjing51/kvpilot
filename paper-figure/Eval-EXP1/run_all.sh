#!/bin/bash
# Eval-EXP1 on the GPU machine, every model and trace in turn (one service at
# a time), after any run_exp1.sh already going:
#   nohup bash run_all.sh > /root/q38/exp1-all.log 2>&1 &
# Each model gets LRU, belady and exclusive blockwise unblocking; Qwen3-30B on
# SWE also the two arms between them (inclusive req, exclusive req unblocking),
# which its required order needs, before TraceLab.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
while pgrep -f "Eval-EXP1/[r]un_exp1.sh" > /dev/null; do sleep 30; done
# These ran before workers were bound to their GPUs' NUMA node
# (disagg_lmcache.sh KVPILOT_NUMA_BIND); on TP 4 a random set of ranks read
# SSD 3-4x slower each run (README).
bash $HERE/run_exp1.sh exp1swe q38 exclb lru belady
bash $HERE/run_exp1.sh exp1swe q36 exclb lru belady
bash $HERE/run_exp1.sh exp1swe q30 exclb lru belady incp exclu
REP=r2 bash $HERE/run_exp1.sh exp1swe q38 belady
# NUMA-bound from here; TP 2 (Qwen3-30B) was not affected.
REP=numa bash $HERE/run_exp1.sh exp1swe q38 exclb lru belady
REP=numa bash $HERE/run_exp1.sh exp1swe q36 exclb lru belady
export EXP1_TL_K=40
bash $HERE/run_exp1.sh exp1tl q38 exclb lru belady
bash $HERE/run_exp1.sh exp1tl q36 exclb lru belady
bash $HERE/run_exp1.sh exp1tl q30 exclb lru belady
echo "===== all done $(date -u +%H:%M:%S)"
