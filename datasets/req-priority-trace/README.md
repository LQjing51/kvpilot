# req-priority-trace

给请求粒度优先级淘汰（当时的 `KVPILOT_REQ_PRIORITY_OFFLINE`，现在的 `KVPILOT_PLACEMENT=inclusive-safe-reqwise`）对 LRU 的 A/B 构造的 trace：1P1D、一次只服务一条请求
（`mooncake_trace_replay.py -p 1 --release-at-first-token --forced-output`），LRU 臂约 30 分钟
回放完，在这个时长内尽量拉大两臂的 prefill e2e 差距。

| 文件 | 内容 |
|---|---|
| `gen_trace.py` | 按最优配置写出回放 trace |
| `explore_gain.py` | 扫 lease / 预算 / seed，只报模型 e2e 收益，不写 trace |
| `swe_reqprio.jsonl` | 从 `agent-traces/swe-smith-trajectories/swe_full.jsonl` 生成 |
| `tracelab_reqprio.jsonl` | 从 `agent-traces/tracelab/tracelab_all.jsonl` 生成 |

## 结果（模型估计，未上 GPU；含 Mooncake 5 s kv lease）

`dram_miss_sim` 默认按 Mooncake `default_kv_lease_ttl=5000`：lookup / put 刷新 lease，BatchEvict
只踢已过期对象；wall clock 按每条请求的估计 prefill e2e 推进。Lease 让 DRAM 更难立刻腾出
「不该留」的块，请求优先级从 SSD 读的比例上升，对请求优先级的伤害大于 LRU，最优 e2e 收益
相对无 lease 模型大约少 3–4 个点（SWE）/ 1–2 个点（TraceLab）；最优 seed / 预算 / 请求数不变。

| | `swe_reqprio.jsonl` | `tracelab_reqprio.jsonl` |
|---|---|---|
| 请求数 / session 数（其中结尾被截断的） | 5252 / 142（76） | 2267 / 106（46） |
| 候选 session | 收益率前 25%：5786 个 | 收益率前 25% 且不超过 50 轮：349 个 |
| 选中的进场顺序 / 活跃 KV 预算 | seed 2 / 3900 块 | seed 1 / 4000 块 |
| prefill e2e 均值 LRU → 请求优先级 | 166.6 → 130.2 ms（−21.8%） | 489.7 → 403.8 ms（−17.6%） |
| 请求优先级下 prefix 从 HBM / DRAM / SSD 读 | 22 / 59 / 19% | 11 / 64 / 25% |
| 估计回放时长 LRU / 请求优先级 | 30.0 / 27.9 min | 30.0 / 27.8 min |
| prompt token 合计（request-safe 测量要写的 KV） | 1.1 亿（约 11 TB） | 0.99 亿（约 10 TB） |

每条 trace 是 4 个进场顺序 × 多档预算里模型收益最高的那组，有挑选偏差。最优预算 3900 / 4000
下 4 个进场顺序的收益：SWE 18.6 / 20.1 / 21.8 / 17.6%（平均 19.5%），TraceLab 16.2 / 17.6 /
16.5 / 14.2%（平均 16.1%），对真机按平均看更稳。无 lease（`--lease-ms 0`）时同一套搜索峰值约
SWE −25.2%、TraceLab −19.3%。

## 生成逻辑

### 为什么这样排

- **收益几乎全部来自「LRU 从 SSD 读、请求优先级从 DRAM 读」**：下一轮要 DRAM 的请求，LRU 已经把它的 prefix 踢出
  DRAM，只能从 SSD 读（比 DRAM 慢 5–8 倍）；请求优先级把它留在 DRAM。HBM 和 DRAM 之间的差别带来的收益不到 1%。
- **这只在一段很窄的活跃 KV 区间里成立**。活跃 KV = 所有正在进行的 session 当前上下文的块数之和。它要大到 LRU 在
  同一 session 两轮之间就把它踢出 DRAM，又不能大到「下一轮要 DRAM / HBM 的块」在请求优先级的 DRAM 里也装不下。
  当前容量（HBM 2361 块，DRAM 每 rank 100 GiB ≈ 4266 块）下大约是 3900–4300 块，偏 300 块收益就掉 4–8 个点。
- **轮转**让每个 session 两轮之间隔着所有其他活跃 session 的上下文，是 LRU 最差的访问模式；请求优先级按请求定档，
  不受影响。
- **开头和结尾没有收益**：开头所有 session 都从第 0 轮起步，活跃 KV 要涨上来；结尾 session 一个个结束，活跃 KV
  回落，而且只剩少数 session 时每一轮都要等自己上一轮 decode 完才发，回放器大量空等。开头
  省不掉（跳过 session 的前几轮要把历史当冷 prompt 算，太贵），结尾用截断去掉。

### 步骤

1. **估计每个请求的 t1 / t2 / t4 和分档**（prefix 全在 HBM / DRAM / SSD 时的 prefill e2e）。数据来自
   `paper-figure/Figure3-request-safe` 的两次 SWE request-safe 实测（`raw/swe`、`raw/swe_hbm1to2`，共 10733 个请求）。
   在 (log prefix, log suffix) 上按 1/8 倍频程分格，取最近的至少 15 个实测点，用它们 t1 / 拟合 t_hbm、t2 / t1、
   t4 / t1 的中位数。prefix > 64K 或 suffix > 6.5K（超出实测覆盖）时直接用拟合时间模型 `fit_times_swe.json`。
   分档阈值同 request-safe：(t4 − t1) / t1 < 5% 为 SSD safe，否则 (t2 − t1) / t1 < 5% 为 DRAM safe，否则 HBM safe。
   不直接用拟合模型，是因为它把实测 HBM safe 的请求一半以上判成 DRAM safe。
2. **选 session**。每个 session 的收益率 = 各轮省下的时间之和 ÷ LRU 全从 SSD 读时的 e2e 之和；省时 HBM safe 取
   t4 − t1、DRAM safe 取 t4 − t2、SSD safe 记 0，第一轮（没有 prefix）记 t_hbm(0, n)。prefix 按 session 内此前
   出现过的整块算。取收益率前 25%：长上下文、每轮工具输出短的 session。不取更靠前的 10%：那批 session 下一轮要
   DRAM / HBM 的块占活跃 KV 的比例太高（SWE 是 0.915），没有一段活跃 KV 能同时让 LRU 落到 SSD、又让请求优先级装得下。
   TraceLab 另外去掉超过 50 轮的 session：长度长尾严重（中位数 13 轮，最长 568 轮），超长 session 吃掉请求名额、
   每条请求又贵。
3. **发送顺序**：候选 session 随机打乱成进场顺序；同时活跃的 session 轮转，每轮一个请求，session 内保序；一个
   session 结束就按进场顺序补新的，直到活跃 KV 估计值达到预算。活跃 KV 估计值 = 每个活跃 session 整个生命周期的
   平均上下文块数之和。按个数（固定 K）控制不住：trace 越长、越接近稳态，同样 K 的活跃 KV 越大。
4. **结尾截断**：按下面的模型估计 LRU 臂的回放时间，到 30 分钟（`--minutes`）就停。此时还在进行的 session 只保留
   已经写进 trace 的轮次。两臂回放同一个文件，被截掉的轮次两边都不跑。
5. **搜索**：对每个进场顺序（`--seeds`）× 每档预算（`--budgets`）重复 3–4，用 `utils/placement_ab/dram_miss_sim.py`
   （HBM 2361 块，DRAM 4266 块，SSD 命中全写回，判据不延迟，默认 5 s Mooncake lease）模拟两臂，留收益最高的一组。

### 回放时间怎么估

一条请求从发出到客户端收到首 token 就放下一条（`--release-at-first-token`），但同 session 的下一轮要等这一轮
decode 完（回放器总是这样，等的时候后面的行也不发）：

| 部分 | 取值 | 来源 |
|---|---|---|
| 发出到开始 prefill | 17 ms | request-safe 实测日志的中位数 |
| prefill | 上面模拟的逐请求 e2e | `dram_miss_sim.py` |
| D 从 Mooncake 拉整段 prompt KV | 75 ms + 1.9 ms / 1K token | [kv-placement.md](../../docs/kv-placement.md) 里 A/B 实测的 78→188 ms，线性外推 |
| 首 token、decode | 每 token SWE 20 ms、TraceLab 10 ms | `utils/trace_order/logs` 里两次回放（并发 16–32）的 TPOT |

不含起服务和清理。LRU 臂平均每条约 0.34 s（SWE）、0.79 s（TraceLab）。

### 输出格式

每行：`timestamp`（全 0，按文件顺序发）、`session_id`、`round_index`（从 0 连续）、`input_length`、
`prefix_length`（这个文件里在它之前出现过的整块前缀，至少留 1 块计算，同 `make_trace.py`）、`output_length`、
`hash_ids`、`tool_calls`（照抄源 trace）、`src_line`（本文件行号，request-safe 测量和 `make_req_table.py` 按它编号）、
`orig_line`（源 trace 行号）。

`hash_ids` 重新编号：回放把一块填成同一个 token（hash_id 对词表取模），而后面块的 cache key 串着前一块，所以只要
各请求的首块 token 互不相同就不会误共享。出现在第 0 位的 id 编成 0..F−1（F ≤ 几百，远小于词表），其余接着往后
编。源 trace 里本来就共用同一首块的 session（SWE 142 个 session 共 54 种首块）重编号后仍共用。

## 复现

```bash
cd datasets/req-priority-trace
python3 gen_trace.py swe        # 写出 swe_reqprio.jsonl；默认 lease 5s
python3 gen_trace.py tracelab

# 只探究理论收益（不写 trace）；默认扫 lease 0 / 1s / 5s
python3 explore_gain.py swe
python3 explore_gain.py swe --lease-ms 1000
python3 explore_gain.py tracelab --lease-ms 0,1000,5000
```

需要本机有 `paper-figure/Figure3-request-safe/raw/{swe,swe_hbm1to2}` 的实测日志。`--minutes`、`--top`、`--max-turns`、
`--budgets`、`--seeds`、`--tpot`、`--lease-ms` 可改，见各脚本 `-h`。

## 上 GPU

两条 trace 都已登记：request-safe 测量 `paper-figure/Figure3-request-safe/make_trace.py swe_reqprio / tracelab_reqprio`
（各两段），A/B `run_ab.sh` 的数据集 `swereq` / `tlreq`。判据表现在只有 SWE 的 `utils/placement_ab/signal_swe_reqprio.json`；
TraceLab 当时的分档表（`paper-figure/Eval-three-req-priority/raw/req_tracelab_reqprio.json`）没有 t1 / t2 / t4，现在的代码用不了，
见 [docs/kv-placement.md](../../docs/kv-placement.md)「A/B」。
整套流程（request-safe → 分档表 → 两臂 A/B）和真机结果在 [`paper-figure/Eval-three-req-priority/`](../../paper-figure/Eval-three-req-priority/)。

## 可信度

- 收益和回放时间都是模型估计。分档和时间来自 SWE 的实测点；它们由硬件和模型决定，和数据集无关，但 TraceLab 约
  1/5 的请求 prefix 超出实测覆盖，只能用拟合模型外推。
- 回放时间里 D 拉 KV 是按 SWE 实测外推的；TraceLab 的 prompt 平均约 4.4 万 token，如果实际更慢，30 分钟能跑的
  条数会少。TPOT 取的是并发 16–32 下的值，`-p 1` 时同时 decode 的请求更少，实际可能更快。
- 最优预算很尖；加 lease 后请求优先级 SSD 占比升到约 19%（SWE）/ 25%（TraceLab），更接近 11-10 真机观测。
  结果偏离预期时，换 `--budgets` / `--lease-ms` 重新生成。
