# 请求粒度优先级淘汰 vs LRU：prefill e2e

- `req_priority_e2e.pdf`：SWE、TraceLab 两条 req-priority trace 上，LRU 与请求粒度优先级（`KVPILOT_REQ_PRIORITY_OFFLINE`）
  两臂的平均 prefill 延迟（拆成露出加载与其余部分）、prefix token 从 HBM / DRAM / SSD 读的比例，以及该
  trace 下请求实测 safe tier 按 input token 加权的比例。
- `lru_breakdown.pdf`（`plot_breakdown.py`）：只看 LRU 臂，SWE、TraceLab 各一根柱子，平均 prefill 延迟拆成
  计算与露出加载。

## 配置

- 2026-11-11 / 12，Taiji-GPU（H20 × 8，11-11 重新分配后的机器，`/root` 下 2 块 NVMe），1P1D，Qwen3-30B-A3B-Instruct-2507，
  TP=2，block 512，prefill `--max-num-seqs 1` + cudagraph `[64,128,256,512,1024]`，`GPU_MEM_UTIL=0.9`（HBM 2361 块），
  Mooncake DRAM 每 rank 100 GiB（4266 块），SSD offload 开，SSD 命中全写回 DRAM（默认）
- 模型从本地拷贝加载（`MODEL=/root/models/Qwen/Qwen3-30B-A3B-Instruct-2507`，与 CFS 上同一份权重）：这台机器从 CFS
  冷加载超过 setup 的 900 s 上限
- 2026-11-12 TAG=`r5`：四级 `REQ_CLASS_MS` + DRAM/HBM 同档、`BatchUnpin` refcnt（含 offload 读路径 unpin、
  `InstrumentedRemoteConnector` 转发）。闭环 A/B：SWE e2e −13.9%，TraceLab −9.3%。
- trace：`datasets/req-priority-trace/{swe,tracelab}_reqprio.jsonl`，回放
  `mooncake_trace_replay.py -p 1 --release-at-first-token --forced-output`（同 session 的下一轮等上一轮返回），一次只有一个 prefill，两臂回放同一个文件
  （闭环节拍约 3.3 req/s，prefill 节点无排队）
- 两臂只差淘汰策略（`utils/placement_ab/run_ab.sh`）：
  - LRU：vLLM 原版空闲队列 + Mooncake 原版淘汰
  - Req-prio：`KVPILOT_PLACEMENT=penalty`、`KVPILOT_REQ_PRIORITY_OFFLINE=req_<trace>.json`、`KVPILOT_PENALTY_TIE=lru`、
    `KVPILOT_DRAM_EXCLUSIVE=0`、`KVPILOT_MOONCAKE_PRIORITY_EVICTION=1`、`KVPILOT_MOONCAKE_PENDING_MS=10.0`、
    HBM/DRAM 同四级类（`REQ_CLASS_MS` 0 / 0.1 / 1 / 10）、读路径 refcnt pin（`BatchUnpinMemoryReplicas`）
  - 以上是 r5 跑的时候的代码和开关。重构后 Req-prio 对应 `run_ab.sh incl`（`KVPILOT_PLACEMENT=inclusive-safe-reqwise`，
    同样四级、同样查表），LRU 臂的 Mooncake 也改成按 chunk 组淘汰（见 [docs/kv-placement.md](../../docs/kv-placement.md)）。
    下面「复现」和 `raw/*.sh` 是 r5 当时的命令和驱动脚本，原样保留（目录当时叫 `Eval-req-priority`）；用现在的代码重跑
    直接 `bash utils/placement_ab/run_series.sh incl:swereq:<tag> lru:swereq:<tag>`（判据表
    `utils/placement_ab/signal_swe_reqprio.json`；TraceLab 的判据表需要重测 request-safe），结果不保证和 r5 逐位一致
- 分档表 `raw/req_<trace>.json`：同一条 trace 先测 request-safe（`paper-figure/Figure3-request-safe/raw/<trace>/`，
  每行 `[1]` / `[2]` / `[4]` 三测），再 `utils/placement_ab/make_req_table.py` 取同一 session 下一轮的实测分档

## 复现

```bash
# 本机
cd paper-figure/Figure3-request-safe
python3 make_trace.py swe_reqprio --stop 5252        # -> raw/swe_reqprio/part{0,1}/trace.jsonl
python3 make_trace.py tracelab_reqprio --stop 2267   # -> raw/tracelab_reqprio/part{0,1}/trace.jsonl
# 同步到 GPU 后，在 GPU 上（request-safe → 分档表 → 两臂 A/B，SWE 再 TraceLab，约 9 小时）
cd /root/kvpilot/paper-figure/Eval-req-priority
MODEL=/root/models/Qwen/Qwen3-30B-A3B-Instruct-2507 nohup bash raw/run_all.sh > raw/run_all.out 2>&1 &
# 拷回 raw/ 后
python3 plot.py
python3 plot_breakdown.py
```

## 口径

- 只统计两臂都完成的请求，按 trace 行号（请求 id 开头 8 位，prefill 日志的 `line=`）配对
- Prefill latency = `KVPilot prefill ... e2e`（调度到 prefill 算完，不含排队），对请求取均值
- Exposed load = 同一行的 `gpu_stall`；Compute = e2e − `gpu_stall`
- 柱顶百分比 = 逐请求配对的 e2e 差的均值 ÷ LRU 的 e2e 均值
- TTFT（只打印）= prefill 日志的 `done` − 客户端 `[ok ]` 行的 `sent`
- prefix token 来源：HBM = `hit`；SSD = `KVPilot prelookup retrieve` 的 `ssd_tokens`（两个 TP rank 取大）；DRAM = `dram` − SSD
- 错读（只打印）：对照分档表里请求自己的实测分档（`tier`），从更慢一层读的 token 占比 / 从更快一层读的占比
- `lru_breakdown.pdf`：LRU 臂全部请求（不含预热 `line=-1`），柱顶百分比 = `gpu_stall` 均值 ÷ e2e 均值

## 注意

- 两臂各跑一次
- 两条 trace 是按模型估计的收益挑出来的（`datasets/req-priority-trace/README.md`），不是数据集的原貌
- 默认开 SSD 命中写回后，request-safe 的 `[4]` 读的 value DRAM 里往往还在，master 拒绝追加副本并打一行
  `object_already_exists`，每段 `mooncake_master.log` 约 1.5 GB，本地已 gzip
