# 发送顺序：让 SWE / TraceLab 的复用落到 DRAM 和 SSD

按文件顺序发，同一 session 的相邻轮挨得太近，复用几乎全在 HBM（离线模拟：SWE 92.4%、
TraceLab 87.3% HBM，DRAM / SSD 都是 0）。这里把 trace 重排成跨 session 的轮转，让两轮之间
插进足够多别的 session，复用间隔拉长到 HBM 装不下、一部分 DRAM 也装不下。

| 文件 | 作用 |
|---|---|
| `tier_sim.py` | 离线三层命中模型：给定发送顺序，逐请求算 HBM / DRAM / SSD / 新算的 token |
| `make_order.py` | 重排：`--align end`（默认）、`--align start --window W`、`--align budget --budget-gib G`（按 KV 预算并发、结束一个补一个），或 `--align poisson`（真实时间戳：session 起点泊松到达，session 内间隔服从指数分布） |
| `run.sh` | GPU 上起 1P1D（Qwen3-30B-A3B BF16，TP2，HBM + Mooncake DRAM + SSD）回放一条重排后的 trace |
| `tier_hits.py` | 从 prefill 日志统计实测的逐层命中，并和 `tier_sim.py` 对同一顺序的预测逐窗口对照 |
| `logs/<name>/` | 每轮实验的 `tier_hits.txt` 和压缩日志 |

重排后的 trace 和原 trace 放在一起：`datasets/agent-traces/swe-smith-trajectories/swe_mooncake_interleaved.jsonl`、
`datasets/agent-traces/tracelab/tracelab_mooncake_interleaved.jsonl`（实验时叫 `swe_end.jsonl` /
`tracelab_end.jsonl`，`logs/*.out` 里还是旧名字，内容相同）。

`tracelab_mooncake_poisson.jsonl`（`--align poisson --session-rate 0.2 --mean-gap 120`）同理，
离线模拟 HBM 44.4% / DRAM 18.6% / SSD 24.4%，未实测。

`swe_mooncake_poisson.jsonl`（`--align poisson`，默认参数：每秒 0.5 个 session、轮间隔均值
60 s、seed 0）是按时间戳回放的版本，未实测。离线模拟 HBM 68.0% / DRAM 18.6% / SSD 5.9%。
扫过 session 到达率 0.2–1/s、间隔均值 30–90 s，SSD 都在 1.4%–6.3%：session 一起开始、结束
时间相差很大，任一时刻的工作集远小于整条 trace（400 GiB，只有 DRAM 的 2 倍）。按 session
结束时刻泊松排布（所有 session 差不多同时结束）能到 SSD 约 15%，但没有采用。

## 按 KV 预算并发：`swe_hbm1to2_budget220.jsonl`

`--align budget --budget-gib 220`，输入是 HBM safe : DRAM safe ≈ 1:2 的子集 `swe_hbm1to2.jsonl`
（368 个 session / 420 GiB，见 [utils/trace_select](../trace_select/README.md)），未实测。活跃 session
轮转，每个 session 按**自己各轮 prompt 的平均大小**（整块）占预算；按文件顺序进场，直到预算放不下
下一个；一个 session 发完最后一轮就释放它的预算，后面的接着进场。重排不改变 HBM : DRAM safe 比例
（仍是 1:2.03）。

离线模拟（HBM 2361 块、DRAM 每 rank 100 GiB、SSD 不限），按 prompt token 计：

| 预算 | HBM | DRAM | SSD | 新算 | 活跃 session 的 KV p50 / p90 / max |
|---|---|---|---|---|---|
| 200 GiB | 39.6% | 44.6% | 6.3% | 9.6% | 145 / 190 / 196 GiB |
| **220 GiB** | **39.0%** | **41.1%** | **10.3%** | 9.6% | 149 / 208 / 214 GiB |
| 240 GiB | 36.8% | 39.7% | 13.9% | 9.6% | 164 / 225 / 231 GiB |
| 260 GiB | 34.8% | 40.1% | 15.5% | 9.6% | 177 / 240 / 246 GiB |

活跃 session 的 KV = 每个还没发完的 session 最近一次 prompt 的整块，跨 session 去重。220 GiB 时：

- **开头 1,547 行**：前 269 个 session 一起进场，一起变长
- **1,547–3,479 行（32%）是补位阶段**：活跃 KV 187–210 GiB（p10–p90），这一段 SSD 占 20.7%
- **之后**：没有 session 可补，剩下的逐个发完，活跃 KV 一路降

**`swe_hbm1to2_budget260.jsonl`**（`--budget-gib 260`，SSD 15.5%）：开头一起进场 354 个 session，之后只补进
14 个，接近全部同时开始。1,797–3,076 行（21%）是补位阶段，活跃 KV 216–243 GiB，这一段 SSD 28.7%。

**为什么按平均大小占预算**：按当前大小算，开场时每个 session 的第一轮都很小，368 个会一次全进场
（等于 `--align start` 不设 window，SSD 19.2%），之后再也没有补位。按最终大小算，稳态活跃 KV 只有预算
的一半左右（350 GiB 才到 SSD 10.4%）。按平均大小算，各 session 处在不同阶段，合计贴着预算。

**活跃 KV 到不了 HBM + DRAM（311 GiB）**：368 个 session 的平均大小加起来只有 271 GiB，结束一个补一个
的顺序会把各 session 的阶段错开，最多 254 GiB（全部同时开始）。只有结尾对齐（`--align end`）能让活跃 KV
在最后涨到整条 trace。但 SSD 命中不需要超过 311 GiB：DRAM 也存着 HBM 里的那份（LMCache 每次新算都
put），且用量过 90% 就淘汰，所以活跃 KV 到 150–200 GiB 就开始有 SSD 命中。要让活跃 KV 长时间停在
311 GiB 以上，池子得是它的几倍（1:2 的最大集合也只有 932 GiB）。

## 顺序

所有 session 轮转：每一步发某个 session 的下一轮，两轮之间其他活跃 session 各发一轮。
**session 内严格保序，不同 session 交错。** end-aligned：有 R 轮的 session 在第 `R_max − R`
个周期加入，所有 session 在最后一个周期发各自的最后一轮。

为什么是 end-aligned 而不是所有 session 一起开始：复用距离 ≈ 其他活跃 session 的 prompt
长度之和。一起开始时短 session 先结束，工作集在中途就开始缩；end-aligned 让所有 session 同时
处在最长的时候，工作集在结尾达到整条 trace。离线模拟（HBM 2361 块、DRAM 每 rank 100 GiB）：

| | 文件顺序 | 一起开始 | end-aligned | end-aligned，DRAM 多 40% |
|---|---|---|---|---|
| SWE SSD | 0 | 12.8%（DRAM 多 20% 就掉到 6.8%） | 26.4% | 11.7% |
| TraceLab DRAM / SSD | 0 / 0 | 24.5% / 48.4% | 22.7% / 52.3% | 33.0% / 42.1% |

回放：`mooncake_trace_replay.py`。按文件顺序发，一轮要等同 session 上一轮
返回之后才发（等的时候后面的行也不发）；重排后时间戳全是 0，节奏只由 `-p` 决定。开头几个周期
只有最长的几个 session，几乎是串行的。

## 结果（2026-11-06，`logs/order-{swe,tracelab}-end/`）

按 prompt token 计，分母是全部请求：

| trace | 请求 / session | 唯一 KV | HBM | DRAM | SSD | 新算 | 有 DRAM 加载的请求 | 有 SSD 加载的请求 |
|---|---|---|---|---|---|---|---|---|
| SWE 实测 | 5140 / 198 | 400.1 GiB | 19.4% | **45.5%** | **27.5%** | 7.6% | 68.9% | 36.2% |
| SWE 模拟 | | | 19.6% | 46.4% | 26.4% | 7.6% | 64.1% | 24.7% |
| TraceLab 实测 | 2000 / 266 | 899 GiB | 12.1% | **21.8%** | **53.4%** | 12.7% | 50.6% | 55.8% |
| TraceLab 模拟 | | | 12.3% | 22.7% | 52.3% | 12.7% | 44.1% | 50.6% |

- 两轮都 0 错误，prefill 日志 0 条 ERROR。SWE `-p 32` 跑 20.6 min（4.2 req/s），TraceLab
  `-p 16` 跑 27.4 min（1.2 req/s）。SSD 分别写了 401 GiB / 900 GiB，Mooncake 淘汰 9 / 17 次。
- **保序核对**（`benchmark.log` 的 `[req]` / `[ok ]` 行）：没有一条请求在同 session 上一轮返回前
  发出；发送顺序与文件顺序完全一致。P 上实际 prefill 顺序相对文件顺序最多错位 30 / 12 位
  （平均 1.4 / 0.4），来自并发的客户端。
- 模拟与实测的 token 占比按 trace 窗口逐段吻合（各窗口差 ≤ 5 个百分点，见 `tier_hits.txt`），
  改顺序或容量时可以先离线算总量。**逐请求不完全一致**：实测有 SSD 加载的请求更多，SWE 有
  689 条（TraceLab 199 条）模拟为 0 而实测有 SSD，中位 7K（15K）token，占全部 SSD token 的
  35%（11%）——被挤出 DRAM 的块在实测里分散到更多 session 上。原因没查；Mooncake 按 shard
  选近似最老的对象、粒度是单层 chunk，且 5 s 内拿过 lease 的不淘汰，都和模型的整块 LRU 不同。
- 开头几个窗口只有最长的几个 session，全是 HBM；SSD 占比随工作集增长一路升到最后一个窗口的
  59%（SWE）/ 81%（TraceLab）。

## 三层命中模型（`tier_sim.py`）

1P、`--max-num-seqs 1` 下 prefill 一条一条跑，缓存状态按发送顺序演化：

- **HBM**：vLLM free-block LRU，请求结束时块按「末块先入队」释放，所以一段前缀从尾部开始掉。
  容量取整个池子 2361 块（`GPU KV cache size: 1,208,832 tokens`，`GPU_MEM_UTIL=0.9`）。
- **DRAM**：Mooncake 按 lease 时间 LRU。PutEnd 发 lease，**LMCache 每次 lookup 也给整段 prompt 的
  每个 chunk 发 lease**（`_lookup_views` 查全部 chunk，含 HBM 已覆盖的部分），所以每个请求都把
  自己的整段前缀刷新一次。用量过 90% 淘汰最老的 10%（`-eviction_high_watermark_ratio=0.9`、
  `-eviction_ratio=0.1`）。
- **SSD**：每次 put 都 write-through，不淘汰。**从 SSD 读回的 chunk 不写回 DRAM**——所以工作集
  超过 DRAM 时 DRAM 不会整体抖掉，而是稳定留住装得下的那部分，其余一直从 SSD 读。这正是两层
  都能拿到可观占比的原因。

**工作集必须大于 DRAM 才可能有 SSD 命中**，与顺序无关。DRAM 是 1P × TP2 × 每 rank 100 GiB
= 200 GiB，所以两条 trace 都要求不同的整块 ≥ 400 GiB：TraceLab 2000 条有 899 GiB；SWE 用
`prepare_swe_trace.py --min-unique-kv-gib 400` 生成（5140 条 / 198 条轨迹 / 400.1 GiB）。

## 复现

```bash
# 本机，在仓库根目录
D=datasets/agent-traces
python3 utils/trace_order/make_order.py $D/swe-smith-trajectories/swe_mooncake.jsonl \
    $D/swe-smith-trajectories/swe_mooncake_interleaved.jsonl
python3 utils/trace_order/make_order.py $D/tracelab/tracelab_mooncake.jsonl \
    $D/tracelab/tracelab_mooncake_interleaved.jsonl
python3 utils/trace_order/make_order.py $D/swe-smith-trajectories/swe_hbm1to2.jsonl \
    $D/swe-smith-trajectories/swe_hbm1to2_budget220.jsonl --align budget --budget-gib 220
python3 utils/trace_order/make_order.py $D/swe-smith-trajectories/swe_hbm1to2.jsonl \
    $D/swe-smith-trajectories/swe_hbm1to2_budget260.jsonl --align budget --budget-gib 260
# GPU（同步 utils/trace_order、两个 *_interleaved.jsonl 与 mooncake_trace_replay.py 之后）
TRACE=$D/swe-smith-trajectories/swe_mooncake_interleaved.jsonl NAME=order-swe-end \
    bash utils/trace_order/run.sh
TRACE=$D/tracelab/tracelab_mooncake_interleaved.jsonl NAME=order-tracelab-end CLIENTS=16 \
    bash utils/trace_order/run.sh
```

`run.sh` 的配置：1P1D、TP2、`GPU_MEM_UTIL=0.9`、HBM 默认、`DRAM_GIB=100`（生成一份只改
`global_segment_size` 的 LMCache 配置，共享的 `mooncake-prefiller-config.yaml` 仍是 200）、
SSD offload 在 `/root/kvpilot-ssd`、`KVPILOT_PRELOOKUP_VIEWS=true`、prefill cudagraph
`[64,128,256,512,1024]`。

## 口径

逐请求取 `KVPilot prefill` 的 `hit`（HBM）、`dram`（connector 提供）、`new`，和
`KVPilot prelookup retrieve` 的 `ssd_tokens`（两个 TP rank 取大）；DRAM = `dram − ssd_tokens`。
占比的分母是全部请求的 prompt token。
