# 每个 prefix 下，DRAM-safe / SSD-safe 的最小 suffix

- `safe_suffix_boundary.pdf`：横轴 prefix，纵轴 suffix。两条线是 DRAM-safe、SSD-safe 的分界点，把平面分成三块：DRAM 线以下 HBM safe（只有 prefix 在 HBM 才不拖慢），两线之间 DRAM safe，SSD 线以上 SSD safe

## 配置

- 2026-11-03，Taiji-GPU（H20 × 8），HEAD `56977ec` + 未提交改动（SSD 逐层加载、跨 worker SSD 定位、同机 segment 全映射 `mooncakestore_connector.py:_preload_shared_memory_peers`、Mooncake 空心跳写出剩余 key `storage_backend.cpp:GroupOffloadingKeysByBucket`、`disagg_lmcache.sh` 的 `GPU_MEM_UTIL` 可由环境变量覆盖）
- Qwen3-30B-A3B-Instruct-2507，TP=2，2P2D，block 512，prefill `--max-num-seqs 1` + cudagraph `[64,128,256,512,1024]`（suffix > 1024 走 eager），`--gpu-memory-utilization 0.9`，`KVPILOT_PRELOOKUP_VIEWS=true`、`LMCACHE_FIRST_LAYER_LOOKUP=false`、`KVPILOT_LOAD_STALL=true`，Mooncake SSD offload 开（`/root/kvpilot-ssd`，心跳 1 s），Mooncake DRAM 默认每 TP worker 200 GiB，全程无淘汰
- 0.9 而不是默认 0.95：全映射后每个进程在 vLLM 分完 KV cache 之后注册 4 × 200 GiB host 内存，GPU 页表约 1.6 GiB，0.95 时第一次 forward OOM。`raw/run.out` 自查 `peers=3 mapped_segments=3`（8 个进程）
- **搜索**（`search.py`，`raw/run.sh`，一次服务）：prefix 4K–64K 步长 4K。每个 prefix 在 prefill worker A 发一次只含 prefix 的 warm，等 10 s，然后 DRAM、SSD 各搜一次。一次测量 = 一对请求，各用自己的 suffix hash：`[1]` 在 A（prefix 在 A 的 HBM）、`[2]` / `[4]` 在 B（只从 DRAM / SSD 取），`cache_use_max_tokens=prefix`，`nvext.prefill_worker_id` 指定 worker，串行。粗筛从第一版网格（`raw/v1`）的分界点起，×1.25 / ÷1.25 走到一个 unsafe、一个 safe 夹住，再二分到相差 4 token。每点测一次
- **复测**（`recheck.py`，`raw/recheck.sh`，另起一次服务）：分界点比每个相邻 prefix（相差 4K）都高 20% 以上或都低 20% 以上的算异常。每个异常点重新 warm，在「搜索时紧挨分界点下面的 unsafe 点、分界点、相邻分界点均值、后两者中点」4 个 suffix 上轮流各测 5 次

## 复现

```bash
# GPU 上，先搜索、再复测
bash /root/kvpilot/paper-figure/Figure2b-safe-suffix-boundary/raw/run.sh
bash /root/kvpilot/paper-figure/Figure2b-safe-suffix-boundary/raw/recheck.sh
# 拉回：dynamo-log/safe-suffix-search-logs/ 的 prefill-*.log（gzip）、mooncake_master.log，以及 raw/{run.out,requests.jsonl,search.jsonl} 到 raw/；
#       dynamo-log/safe-suffix-recheck-logs/ 同样的日志和 raw/recheck/{requests,search}.jsonl、raw/recheck.out 到 raw/recheck/
python3 plot.py
```

## 口径

- compute = `[1]` 的 `KVPilot prefill ... e2e`；delay / compute =（`[2]` 或 `[4]` 的 e2e − compute）/ compute，同一对配对
- 来源核对，不符的那对重测（最多 3 次）：`[1]` hit ≥ prefix 且 dram = 0；`[2]` dram ≥ prefix 且两个 rank `ssd_values=0`；`[4]` dram ≥ prefix 且两个 rank `ssd_values = 48 × chunks`。delay / compute < −3% 也重测
- `plot.py` 从 prefill 日志重算每一对，`requests.jsonl` 只用来知道每条请求是什么；搜索的分界点和 `search.jsonl` 逐点核对
- 分界点 = 最高的 unsafe suffix 之上最低的 safe suffix，都是测过的点，不插值。复测过的 (prefix, 介质) 只用复测数据，每个 suffix 按 5 次的中位数判；其余按搜索的单次测量判

## 注意

- **单次测量的噪声和 5% 的余量同量级**：delay 基本是常数（DRAM 5–8 ms、prefix ≥ 40K 的 SSD 17–21 ms），偶尔一次多出 6–10 ms；compute 随 suffix 分段跳变且不单调（如 64K：suffix 200 为 69 ms、224 为 106 ms、280 为 90 ms、320 为 144 ms），delay / compute 在很宽的 suffix 区间里落在 4–7%。精确到 4 token 的分界点只代表那一次测量，没复测的点也可能偏几十到几百 token
- 复测的 5 个点：4K DRAM 1412、52K DRAM 748 是单次偏高，复测后图上画的是复测里最低的 safe 点（640、536），更低的 suffix 没复测，真实分界点不高于它；4K SSD、44K SSD 那一段中位数 3–5%、5 次里常有 2 次 > 5%，按中位数画 1252、1960，按「5 次全 < 5%」是 1616、定不出；64K DRAM 236 是单次偏低，复测 232–320 全部 5/5 unsafe，分界点 > 320 未测到，图上没有 64K
- `raw/v1/` 是第一版固定网格（4K–64K 翻倍 × 15 档 suffix，3 次），只用来给搜索定起点；`raw/g0`–`raw/g6` 是中途放弃的分组网格的 trace，不进图

## 1P1D（`raw/1p1d/`，`safe_suffix_boundary_1p1d.pdf`）

和 request-safe（`../Figure3-request-safe`）同一拓扑：warm、`[1]`、`[2]`/`[4]` 都在唯一的 prefill worker 上，`cache_use_control` 让 `[2]`/`[4]` 不用 HBM 里那份。

- 2026-11-05，配置同上，但 1P1D、Mooncake 心跳 100 ms、LMCache 用上游 save 逻辑（撤回了 SSD-only 写回，见 `docs/pitfalls.md`）
- **每对测量前先发一个同形状的预热请求**，预热、`[1]`、`[2]`/`[4]` 背靠背，对与对之间空闲 1 s：空闲后的第一个请求慢约 2 ms（见 `docs/pitfalls.md`），`[1]` 和 `[2]`/`[4]` 都要在热状态下测
- `raw/run_1p1d.sh` 依次起三次新服务：
  - **search**（`search.py` 不给 `--b`）：同上，每点测一次 → `raw/1p1d/`
  - **recheck**（`recheck.py --raw raw/1p1d`）：同上，异常点 4 个 suffix × 5 次 → `raw/1p1d/recheck/`
  - **confirm**（`confirm.py`）：**每个**分界点从前两段的读数出发，在分界点和它下面的 unsafe 点上各测 5 次按中位数判；判反了就往上 / 往下挪，两头都成立后按中位数二分到间距 ≤ 10% → `raw/1p1d/confirm/`
- 同一脚本 `OUT_NAME=1p1d-ext PREFIXES=512,...,163840` 跑 trace 用到的其它 prefix（512–3K、72K–160K）→ `raw/1p1d-ext/`，只给 request-safe 比对用，不进图
- `python3 plot.py raw/1p1d`：每个 (prefix, 介质) 只用跑过的最后一段的数据，按中位数判
- `raw/1p1d-idle/`、`raw/1p1d-ext-idle/`：同一流程但没有预热（每个请求前空闲 1 s）的上一轮，留档
- `raw/1p1d-ext/v0-norecheck/`：扩展那轮最初的 recheck / confirm。当时 recheck 只认 4K 间隔的邻居，扩展的 prefix 没有一个被当成异常点；改成「相邻的已测 prefix」后重跑了 recheck + confirm（`raw/1p1d-ext/run-recheck-confirm.out`）。复测证实 80K / 160K 的 DRAM 在 301 附近有一段 unsafe（212 safe、301 unsafe、305 safe，各 5 次），3K 的 512 是整块特例（312 / 416 safe）
- `raw/idle-ab/`：加预热的依据。9 组 (prefix, suffix) × 6 轮，「前面紧跟一个请求」vs「先空闲 1 s」：空闲后 t1 慢 1.2–2.3 ms，d2 不变；只有 `[4]` 前空闲时 d4 多 1.3–2.6 ms

### 结果（`raw/1p1d/` + `raw/1p1d-ext/`，带预热）

32 + 24 个点全部确认（`confirm` 无 note）；表中是分界点，紧挨其下的 unsafe 点与它各 5 次中位数：

| prefix | 512 | 1K | 2K | 2.5K | 3K | 4K | 8K | 12K | 16K–48K | 52K | 56K–64K | 72K–144K | 160K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DRAM | 260 | 264 | 264 | 168 | 516 | 264 | 152 | 176 | 212 | 224 | 212 | 212（80K 305） | 305 |
| SSD | 2292 | 2040 | 1888 | 1828 | 1836 | 1484 | 1564 | 1304 | 1504–2536 | 1876 | 1928–1984 | 1876–2061 | 2412 |

（16K–48K 的 SSD：16K 1504，20K 2228，24K 2536，28K 1872，32K 2052，36K 2024，40K 1932，44K 1884，48K 1876。）

无预热的上一轮（`raw/1p1d-idle/`）每个请求前空闲 1 s，compute 被抬高约 2 ms，分界点普遍更低（如 DRAM 20K–32K 为 116、SSD 8K–20K 为 1252）。

### 注意

- **safe 与否由 compute 和加载时长 L(prefix) 决定，不是由 suffix 单调决定**：加载逐层流水，加载露出来的那些对的 e2e 在同一 prefix 下几乎恒定（= L），且 compute ≥ L 的对全部 safe。L 随 prefix 线性增长：DRAM 约 22 ms（≤8K）→ 77 ms（64K）→ 199 ms（160K），SSD 85 ms（4K）→ 572 ms（64K）→ 1.3 s（160K）
- **eager 段（suffix 1024–2048）**：DRAM 多出 2–4 ms 固定开销，比值贴着 5%，request-safe 里同形状的两行在这一段只有约 40% 档位一致
- **compute 对 suffix 不单调**（attention kernel 的选择）：60K 时 suffix 144 / 148 / 180 / 212 的 compute 为 58 / 84 / 67 / 100 ms，于是 safe / unsafe 来回翻（各 5 次，5/5 一致）；SSD 在 36K–64K 都落在 1876，是因为 compute 在 suffix 1872→1876 跳了约 23%。所以「每个 prefix 一个 suffix 分界点」只是近似：分界点之上仍可能有 unsafe 的 suffix
- 4K 的 suffix 512（prompt 恰好落在 512 整块）比 516 多约 4 ms 固定延迟，DRAM 分界因此是 516
