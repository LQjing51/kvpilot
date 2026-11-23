# 默认 HBM 下五种 KV placement 的 TTFT（swe_blockprio_k160）

- `block_priority_ttft.pdf`：LRU、Belady 和三个 KVPilot placement（inclusive-penalty-reqwise、
  exclusive-penalty-reqwise-unblocking、exclusive-penalty-blockwise-unblocking）的平均 TTFT；三个 KVPilot 各跑了 oracle
  判据和在线预测判据两组。另一张子图是 oracle 判据下 prompt token 从 HBM / DRAM / SSD 命中、需要重算的比例。

## 配置

- oracle 五组：2026-11-22 23:00 – 11-23 00:11；在线预测三组：11-23 00:57 – 01:41（北京时间）。Taiji-GPU（H20 × 8）。
- 1P1D，Qwen3-30B-A3B-Instruct-2507，TP=2，block 512，prefill cudagraph `[64,128,256,512,1024]`，`GPU_MEM_UTIL=0.9`
  （HBM 2361 块），Mooncake DRAM 每 rank 100 GiB，淘汰水位 0.9，SSD offload 开。
- 代码：HEAD `56977ec` 加未提交的 KVPilot 改动，主要是 `vllm-dev/vllm/v1/core/kvpilot_*.py`、`sched/scheduler.py`、
  `kv_cache_manager.py`、`kv_cache_utils.py`、`lmcache-dev/lmcache/v1/kvpilot_*.py`、`Mooncake/mooncake-store/src/master_service.cpp`、
  `dynamo-dev/components/src/dynamo/vllm/{kvpilot,worker_factory}.py`。Mooncake 的 `store` 和 `mooncake_master` 在 11-22 22:58
  用当时的工作区重编；unblocking 和 blockwise 用的是简化后的借块规则。
- trace：`datasets/block-priority-trace/swe_blockprio_k160.jsonl`（`search_trace.py --write pool=10,k=160,mid=1,seed=1 --n 1300`）。
  它从 `swe_full` 里选 blockwise HBM 需求前 10% 的 session，160 个同时轮转；开头 160 个 session 从随机一轮中途加入，共 1300 条。
- 回放：`mooncake_trace_replay.py -p 1 --release-at-first-token --forced-output`，一次只有一个 prefill；同一 session 的下一轮
  要等上一轮返回。
- 各组只差 `KVPILOT_PLACEMENT`（`utils/placement_ab/run_ab.sh`）：

  | 组 | 开关 |
  |---|---|
  | LRU | 不设 |
  | Belady | `belady`，`KVPILOT_BELADY_TRACE` = 回放的 trace |
  | Incl-pen / Unblock / Blockwise | `inclusive-penalty-reqwise` / `exclusive-penalty-reqwise-unblocking` / `exclusive-penalty-blockwise-unblocking`，时间模型系数 `paper-figure/Figure3-request-safe/fit_times2_swe.json` |

  oracle（`r1`）的判据查表 `utils/placement_ab/signal_swe_blockprio_k160.json`；在线预测（`on1`）用 `KVPILOT_ONLINE_PREDICT=1`，
  工具响应长度分布 `paper-figure/algorithm/resp_length_dist_swe.json`。

## 复现

```bash
# GPU 上，每组约 15 分钟（含起服务）
cd /root/kvpilot/utils/placement_ab
bash run_series.sh belady:sweblock160:r1 exclb:sweblock160:r1 exclu:sweblock160:r1 incp:sweblock160:r1 lru:sweblock160:r1
KVPILOT_ONLINE_PREDICT=1 bash run_series.sh exclb:sweblock160:on1 exclu:sweblock160:on1 incp:sweblock160:on1
# 把 dynamo-log/pab-sweblock160-<arm>-<tag>-logs/ 拷进 raw/，prefill-0.log 先 gzip；runner 输出 /root/pab-sweblock160-{r1,on1}-series.out
python3 plot.py
```

## 口径

- 统计 trace 第 160–1299 行，共 1140 条，各组按行号配对。前 160 行是中途加入的 session 的冷首轮，作为预热不计。
- TTFT = prefill 日志 `KVPilot prefill` 行的 `done` − 客户端 `[ok ]` 行的 `sent`。
- prefill e2e = 同一行的 `e2e`；等 HBM 的阻塞 = 同一行的 `blocked`。
- prompt token 来源：HBM = `hit`；SSD = `KVPilot prelookup retrieve` 的 `ssd_tokens`（两个 TP rank 取较大的一个，且不超过 `dram`）；
  DRAM = `dram` − SSD；重算 = `prompt` − 前三者。各项都除以 `prompt` 之和。
- 配对差：逐请求 TTFT 差每 50 条一批取均值，95% 区间 = 1.96 × 批均值的标准差 ÷ √批数。
- 在线预测中不在分布里的工具调用组合：取 `decode-0.log` 最后一行的 `label_misses`。

## 注意

- 每组只跑了一次。
- trace 是按 blockwise HBM 需求挑出来的 session，不是数据集原貌，挑法见 `datasets/block-priority-trace/README.md`。
- `prefill-0.log` 里每条请求有几千行 `batch_get_replica failed ... OBJECT_NOT_FOUND`：lookup 一次查整段 prompt 的所有 key，
  不存在的 key 由 Mooncake 打 ERROR。各组都有，所以 `prefill-0.log` 很大。
- `raw/` 被 `.gitignore`（`paper-figure/*/raw/`）排除，共 152 MB。
