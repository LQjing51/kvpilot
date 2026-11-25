# Eval-EXP1：同一个 trace，三个模型上的 placement A/B

目标：一个 trace 在 Qwen3-30B-A3B（TP 2）、Qwen3.6-35B-A3B（TP 4）、Qwen3.8-Flash-Next（TP 4）上都测出
KVPilot 的收益，TTFT 满足：

- Qwen3-30B：LRU > belady > exclusive blockwise unblocking，以及 inclusive req（`inclusive-penalty-reqwise`）>
  exclusive req unblocking > exclusive blockwise unblocking；belady 和 inclusive req 之间不比（2026-11-24 改，
  原先要求五级全序）；
- Qwen3.6、Qwen3.8：LRU > belady > exclusive blockwise unblocking。

LRU 最好不是接近 100% 从 SSD 读，每组回放约 30 分钟以内（含预热）。先 SWE，再 TraceLab（分布与 SWE 不同）。

## 文件

| 文件 | 内容 |
|---|---|
| `make_traces.py` | 按选定参数生成 `exp1_<ds>.jsonl` 和三个模型的判据表 |
| `exp1_swe.jsonl` | SWE trace：池 20、K=176、seed 0、`h24f25`，3000 条，306 个 session，预热 176 条 |
| `exp1_tracelab.jsonl` | TraceLab trace：池 20、K=40、seed 1、`h6f25`，1250 条，92 个 session，预热 250 条 |
| `run_all.sh` | GPU 上实际的运行顺序：SWE（Qwen3.8、Qwen3.6 三组，Qwen3-30B 五组），Qwen3.8 belady 重跑，绑 NUMA 后两个混合模型 SWE 重跑，再 TraceLab（三个模型各三组） |
| `signal_exp1_<ds>{,_b1024,_q38}.json` | 判据表：Qwen3-30B / Qwen3.6 TP4 / Qwen3.8（`make_model_table.py`，各自的时间模型） |
| `exp1_swe_rand.jsonl`、`exp1_tl_rand.jsonl` | 第二轮的 SWE / TraceLab trace（随机发送，见「第二轮的 trace」） |
| `signal_exp1_{swe,tl}_rand{,_q36f8,_q38}.json` | 第二轮判据表：Qwen3-30B / Qwen3.6 FP8 TP2 / Qwen3.8 开专家并行 |
| `run_exp1.sh` | GPU 上按模型设好容量后依次跑各组（`utils/placement_ab/run_ab.sh`），日志归档到 `raw/<ds>/<model>/<arm>/`；`REP=<后缀>` 的重跑归档到 `<arm>-<后缀>/` |
| `summarize.py` | 汇总：各组 TTFT、前缀来源、SSD 第 0 层延迟、相邻两组的差及 95% 区间；完整的重跑集合（如 `-numa`）单独一张表 |
| `plot.py` | 第二轮的图（阈值 5%），只读 `raw/`：`exp1-ttft.pdf`（TTFT 拆成 prefill 前、prefill 计算、exposed load）、`exp1-exposed-load.pdf`（每条请求的 exposed load）、`exp1-token-source.pdf`（prompt token 来自 HBM / DRAM / SSD / 重算）；SWE Qwen3-30B 的两个 exclusive 组用修后的 `-wait` 重跑。TTFT = prefill 结束 − 客户端发出，prefill 前 = TTFT − prefill e2e，exposed load = `gpu_stall`，重算 = `new` |
| `raw/<ds>/<model>/<arm>/` | P / D 日志、回放日志、master 的 KVPilot 行 |

## 第二轮（2026-11-25 起）：随机发送、全部开 EP、Qwen3.6 FP8 TP2、exclusive 保留 SSD 副本

和第一轮（下文「SWE 结果」）相比改了四处，各组之间仍然只差 `KVPILOT_PLACEMENT`：

| | 第一轮 | 第二轮 |
|---|---|---|
| 发送方式 | 快慢两类 session 各自轮转（`h24f25`） | K 个活跃 session，每条请求随机挑一个（`rand`） |
| Qwen3.6 | BF16，TP 4（2 个 KV 头被复制到 4 张卡） | 官方 FP8 权重（`Qwen3.6-35B-A3B-FP8`），TP 2，KV 仍是 BF16，块 1536；模型键 `q36f8` |
| 专家并行 | Qwen3-30B、Qwen3.6 开，Qwen3.8 不开 | 全部开（FP8 的 MoE 走 FlashInfer CUTLASS，`docs/pitfalls.md`「FP8 模型开专家并行」） |
| exclusive / blockwise | 换层删 SSD 副本；没有下一轮把只在 HBM 的块写到 SSD | SSD 副本写过就留着；没有下一轮直接释放 HBM（`docs/kv-placement.md`「exclusive-penalty-reqwise」） |
| Qwen3-30B 的要求 | 五级全序 | LRU > belady > blockwise 且 inclusive req > req unblocking > blockwise |

时间模型：Qwen3-30B 仍是 `fit_times2_swe.json`（本来就开 EP）；Qwen3.6 FP8 TP2 和 Qwen3.8（开 EP）重测 request-safe
（`paper-figure/Figure3-request-safe/raw/qwen3.6-fp8`、`raw/qwen3.8-ep`），系数 `fit_times2_qwen3.6-fp8_k160.json`、
`fit_times2_qwen3.8-ep_k160.json`。容量照旧按 token 对齐到 Qwen3-30B（Qwen3.6 FP8：HBM 787 块 + 3 × K 状态块，DRAM 21 GiB）。
数据集键 `exp1swr`（`exp1_swe_rand.jsonl`，结果 `raw/swe_rand/`）、`exp1tlr`（`exp1_tl_rand.jsonl`，`raw/tl_rand/`）。

### 第二轮的 trace

| | SWE（`exp1_swe_rand.jsonl`） | TraceLab（`exp1_tl_rand.jsonl`） |
|---|---|---|
| 参数 | 池 5、K=144、mid 1、seed 2、`rand` | 池 30、K=160、mid 1、seed 0、`rand` |
| 请求 / session | 2800 / 218（每 session 中位 13 条） | 1000 / 215（中位 4 条） |
| prompt 均值 / 中位 / p90 | 30.0K / 27.6K / 53.0K | 49.3K / 41.1K / 94.3K |
| 每轮新增 token 中位 / p90 | 184 / 1984 | 658 / 6758 |
| 输出中位 / p90 | 59 / 396 | 126 / 633 |
| 工具 | SWE-agent：`str_replace_editor`、`bash`、`submit` | Claude Code：`Bash`、`Read`、`Grep`、`Glob`、`Edit` |
| 预热（`summarize.py` 的 `WARM`，`run_exp1.sh` 的 K） | 144 条 | 160 条 |

```bash
python3 make_traces.py swe pool=5,k=144,mid=1,seed=2,order=rand --n 2800 --name swe_rand
python3 make_traces.py tracelab pool=30,k=160,mid=1,seed=0,order=rand --n 1000 --name tl_rand
```

TraceLab 换过两次，都在开跑之前：起初是池 20、K=80、seed 1（下表），修正 `pre` / `excl_save` 后它的 Qwen3.8 belady →
blockwise 余量只剩 +2.0，换成池 30、K=128、seed 3；再修正写 SSD 的时延（见「第二轮 SWE 结果」）后，后者 Qwen3-30B 的
req unblocking → blockwise 只剩 +0.4，换成上面这条（`make_traces.py --n` 就是搜索的 `--cap`，两者不同 trace 也不同）。

选法：`search_trace_models.py --orders rand`，三个模型用各自第二轮的时间模型和容量（`--hbm qwen3-30b=2361,qwen3.6-fp8=787+s,qwen3.8=1180+s
--dram-gib qwen3-30b=100,qwen3.6-fp8=21,qwen3.8=27`），SWE 扫池 5/10/20、K 128–224、seed 0–5，TraceLab
（`--cache /tmp/gen_trace_tracelab.pkl --cap 1100`）扫池 20/30、K 64–128、seed 0–2。余量 = 模拟的配对差减 95% 区间（ms）：

| 模拟 | LRU TTFT | LRU 前缀 H/D/S | 回放 | LRU→belady | belady→blockwise | incl req→req unbl | req unbl→blockwise |
|---|---|---|---|---|---|---|---|
| SWE Qwen3-30B | 350.6 | 30/16/54% | 33 分钟 | +30.1 | +16.4 | +17.7 | +7.3 |
| SWE Qwen3.6 FP8 | 196.1 | 20/26/54% | 24 分钟 | +3.0 | +3.1 | | |
| SWE Qwen3.8 | 323.7 | 22/24/54% | 31 分钟 | +14.1 | +2.7 | | |
| TraceLab Qwen3-30B | 708.3 | 29/15/57% | 27 分钟 | +40.7 | +24.8 | +18.5 | +14.3 |
| TraceLab Qwen3.6 FP8 | 338.0 | 25/19/56% | 19 分钟 | +4.9 | +6.1 | | |
| TraceLab Qwen3.8 | 556.2 | 26/19/56% | 23 分钟 | +28.4 | +6.4 | | |

（回放时间按 3000 / 1100 条模拟；SWE 实际取前 2800 条，约 31 分钟。上表是修正前的模拟器。）

现用 TraceLab trace 在最终的模拟器上（`--pools 30 --ks 160 --seeds 0 --cap 1000`）：

| 修正后模拟 | LRU TTFT | LRU 前缀 H/D/S | 回放 | LRU→belady | belady→blockwise | incl req→req unbl | req unbl→blockwise |
|---|---|---|---|---|---|---|---|
| TraceLab Qwen3-30B | 1193.6 | 17/8/76% | 35 分钟 | +35.7 | +25.7 | +6.3 | +9.9 |
| TraceLab Qwen3.6 FP8 | 457.2 | 16/9/75% | 18 分钟 | +6.2 | +4.6 | | |
| TraceLab Qwen3.8 | 855.9 | 16/9/75% | 27 分钟 | +25.6 | +7.3 | | |

（Qwen3-30B 一行是非混合模型也等写完之后的模拟。）挑这条时 Qwen3-30B 的 exclusive 还会重算 SSD 尾段，结果被少数几次长
prompt 的整段重算主导：同一个候选截到 950 条时 req unblocking → blockwise 是 −59.9，1000 条时 +5.9，所以主要按两个混合
模型的余量挑；修好以后（见「第二轮 SWE 结果」）模拟里 Qwen3-30B 四对都为正。

修正后的模拟器在 SWE 上扫池 2/3/5/10、K 64–224、seed 0–3，Qwen3.8 的 belady → blockwise 余量全在 −3.3 到 +0.4 之间（两组的
TTFT 只差 0.3–0.5%），没有一条 SWE trace 能把它们分开；TraceLab 的 prompt 长、DRAM 压力大，exclusive 不存重复副本的
好处才显出来（池 30 的候选多在 +2.5 到 +5.3）。

**SWE 上 Qwen3.6 FP8 的余量最小**：每一级只差约 1.5%（3 ms 上下）。TP 2 下 SSD 加载每千 token 1.7 ms，只是计算的 6%
（`Figure3-request-safe/README.md` 末节），SWE 的 prompt 又短，load-bound 的请求少。池 10 / 20 的候选 Qwen3.6 FP8 的
belady→blockwise 余量在 −15 到 +2.4 之间；池 5（每轮 HBM 需求最高的 5% session，prompt 长一些）抬到 +2.5–3.0。把 Qwen3.6 FP8
的计算整体乘 0.85 或 0.7 再模拟，余量也只到 +2.5 / +2.7，所以换更快的 MoE kernel 帮不上忙。TraceLab 的 prompt 长 1.7 倍，
Qwen3.6 FP8 的余量是 SWE 的两倍。

运行顺序（GPU 上 `/root/queue_exp1.sh` 按 `/root/exp1_queue.txt` 逐行跑）：SWE 上
Qwen3.6 FP8（exclb、lru、belady）→ Qwen3.8（同）→ Qwen3-30B（exclb、lru、belady、incp、exclu），再 TraceLab 同样的 11 组。

### 第二轮 SWE 结果（2026-11-25）

TTFT 均值（ms，第 144 条以后 2656 条，括号内相对 LRU），`python3 summarize.py swe_rand`：

| 模型 | LRU | belady | inclusive req | excl req unblocking | excl blockwise | LRU 前缀 HBM / DRAM / SSD |
|---|---|---|---|---|---|---|
| Qwen3.6 FP8 TP2 | 170.5 | 166.8（−2.2%） | | | 164.6（−3.5%） | 15 / 32 / 54% |
| Qwen3.8 TP4 | 297.6 | 282.0（−5.2%） | | | 283.3（−4.8%） | 15 / 31 / 54% |
| Qwen3-30B TP2 | 350.8 | 321.2（−8.4%） | 325.0（−7.4%） | 修后 311.0（−11.3%） | 321.8（−8.3%）→ 修后 309.4（−11.8%） | 29 / 19 / 51% |

Qwen3-30B 的两个 exclusive 组在跑的过程中改了两处代码（见下「Qwen3-30B 的 exclusive」），blockwise 修后重跑归档在
`raw/swe_rand/q30/exclb-wait/`，req unblocking 只有修后的 `exclu-wait/`（修前那次在第 1222 条挂了）。

相邻两组的配对差（ms，95% 区间按 50 条一批的批均值算）：

- Qwen3.6 FP8：belady 比 LRU 快 3.8 ± 0.4（模拟约 3.3），blockwise 再快 2.2 ± 0.5（模拟约 3.5），两级都显著，**顺序成立**。
  每组回放约 20 分钟（UTC 19:09–20:12 三组）。blockwise 的前缀 47% 从 DRAM、53% 从 SSD、几乎不占 HBM：这个配置下 DRAM
  加载接近免费，判据表里下一轮该放 HBM 的只有 40 行；暴露的加载时间 LRU 17.3 s、belady 7.7 s、blockwise 8.3 s。
- Qwen3.8：belady 比 LRU 快 15.6 ± 1.7（模拟 15.5），blockwise 比 belady **慢** 1.3 ± 1.3（模拟快 4.2），**顺序不成立**。
  每组回放约 30 分钟（UTC 20:15–21:40 三组）。
- Qwen3-30B（修后的两个 exclusive 组）：belady 比 LRU 快 29.6 ± 2.9，blockwise 比 belady 快 11.8 ± 4.0；req unblocking
  比 inclusive req 快 13.9 ± 3.2，blockwise 比 req unblocking 快 1.6 ± 2.8（**在噪声内**）。四对按均值都成立，前三对显著。
  修前 blockwise 和 belady 打平（慢 0.6 ± 12.6）。每组回放约 27 分钟（UTC 21:42–01:35）。

**SWE 小结**：Qwen3.6 FP8 两级都显著成立；Qwen3-30B 修好 exclusive 的两处问题后四对按均值成立（req unblocking →
blockwise 在噪声内）；Qwen3.8 的 belady 和 blockwise 打平，修正后的模拟器认为 SWE 上分不开（见下）。

**Qwen3-30B 的 exclusive**：修前 blockwise 和 belady 打平，是 56 条请求（2%）把前缀尾段整段重算了：上一轮的信号到达后，
下层已有的中段立刻放掉 HBM 副本，尾段排队写 SSD；下一轮在写完前到达时，HBM 的前缀匹配停在中段前、LMCache 读到中段为止，
尾段只能重算，平均 6318 token，每条比 belady 慢 863 ms，合计 48.4 s，把其余 2600 条平均快的 18.0 ms 抵光。混合模型
在同样的情况下会等写完（`KVPilot hybrid wait`），所以没受影响。改成非混合模型也等（`docs/kv-placement.md`「下一轮等还在
写的尾段」）后重跑：2800 条里等了 51 次（中位 109 ms、p90 548 ms、最长 1.2 s，合计 12 s），比 LRU 多算 512 token 以上的
请求从 56 条降到 5 条，blockwise 309.4 ms。等待条件第一版是「HBM 前缀之后 HBM 里还有的块」，req unblocking 修后第一次跑
在第 2354 条等满 30 s 报错（held 块隔着被借走的块留在 HBM，不会被写下去），改成只等 job 正在写的块后重跑；blockwise
那次没碰到这种情况，结果不受影响。另一处是 req unblocking
在第 1222 条挂掉：借块时算 DRAM penalty 对空序列取了 `max`（共用的 system prompt 块的 HBM head 被替换后，没有 head 再
需要它在 DRAM 或更快的层），改成按定义取 0。

**Qwen3.8 为什么没成立：模拟器高估了 exclusive 在 prefill 之外的优势**。把 TTFT 拆成 prefill e2e 和之前的等待
（TTFT − e2e，客户端发出到 prefill 被接纳，排队几乎为零）：

| 实测 | LRU | belady | blockwise |
|---|---|---|---|
| Qwen3.8 prefill e2e / 等待 / 暴露加载（ms） | 281.7 / 15.9 / 26.4 | 265.9 / 16.1 / 11.0 | 266.7 / 16.6 / 13.9 |
| Qwen3.6 FP8 prefill e2e / 等待 / 暴露加载 | 156.9 / 13.6 / 6.5 | 153.1 / 13.7 / 2.9 | 150.6 / 14.0 / 3.1 |

模拟器的 Qwen3.8 blockwise prefill e2e 本来就比 belady 慢 2.2 ms（270.9 对 268.7，和实测的 0.8 ms 同向），它在 TTFT 上的领先
全来自等待那一段：`sim_placement.default_cost` 的 `pre` 是在 Qwen3-30B 上拟合的，inclusive 每千命中 token 1.26–1.43 ms、
exclusive 0.86–0.99 ms，30K 的 prompt 下 exclusive 凭空便宜约 10 ms。实测两个混合模型的等待和放置无关：Qwen3.8 约
5.1 ms + 每千 prompt token 0.37 ms，Qwen3.6 FP8 约 5.0 + 0.27 ms，三组一样（Qwen3-30B 第一轮实测确实是 exclusive 少 3–4 ms，
那套 `pre` 对它是对的）。

另一处没建模的是 exclusive 在 prefill 时不 write-through：时间模型是在 inclusive 下测的，exclusive 组的实测 prefill 比
模型快。按「实测 − 模型」对新算 token 和前缀回归，exclusive 减 inclusive：Qwen3.8 每千新 token −1.6 ms，Qwen3.6 FP8
−0.65 ms − 每千新 token 0.83 ms − 每千前缀 0.08 ms，Qwen3-30B（第一轮五组）每千新 token −1.6 ms。

两处都加进了 `sim_placement.PROFILES`（`pre`、`excl_save`）。拿 Qwen3-30B 的真机数据对照时又查出 exclusive 的三处
时序没建模，一并改了：

- **信号先于下一轮的 lookup 到达**。模拟器原来在请求发出那一刻（上一轮 decode 结束）就判定命中，那时上一轮的信号还没到，
  块全 pinned 在 HBM，于是同 session 紧接着的下一轮全 HBM 命中；真机上信号先到、HBM 副本已放掉（同 session 间隔 1 条的
  请求，模拟 HBM 命中 30.8K token、真机 6.5K）。改后判定命中前先送达「发出 + `signal_ms`」以内的信号，三个模型 blockwise
  的 HBM 占比都和真机对上（7 / 3 / 1%）。
- **写往 SSD 的块写完才在 SSD 上**。真机上信号一到，中段（DRAM 已有）的 HBM 副本马上放掉，尾段排队写 SSD（Qwen3-30B
  每块约 21 ms）；下一轮若在写完前到达，HBM 只能命中开头，接上 DRAM 的中段后尾段既不在 SSD、HBM 的又因前缀断开用不上，
  **只能重算**。Qwen3-30B blockwise 有 56 条（2%）这样，平均多算 6318 个 token，摊到每条约 8 ms。模拟器原来在决定写的那一刻
  就把块算到 SSD 上。
- **混合模型会等写完**（LMCache `KVPilot hybrid wait`：命中必须落在 HBM 里的状态上，KV 不够就下一步再查，Qwen3.8 一次约
  20 ms），所以两个混合模型一条都没重算。模拟器按此区分：非混合模型重算，混合模型推迟到写完。写盘耗时也换成各模型实测的
  （`job`：Qwen3-30B 每块 DRAM 1.6 / SSD 21.4 ms，Qwen3.8 0.55 / 13.3，Qwen3.6 FP8 0.61 / 10.5）。

另外 exclusive 超水位时 master 实测也是一次清掉约十分之一（`used_ratio` 0.81–0.84 时触发），模拟器原来是贴着水位小步溢出。

全部修正后在这条 trace 上重放（`sim_placement.py --trace exp1_swe_rand.jsonl --skip 144`）：

| 模拟 vs 实测，TTFT（ms） | LRU | belady | blockwise |
|---|---|---|---|
| Qwen3-30B | 355.0 vs 350.8 | 321.8 vs 321.2 | 311.5 vs 321.8（重算尾段）；305.4 vs 309.4（等写完） |
| Qwen3.8 | 301.6 vs 297.6 | 285.0 vs 282.0 | 283.8 vs 283.3 |
| Qwen3.6 FP8 | 168.4 vs 170.5 | 163.9 vs 166.8 | 161.6 vs 164.6 |

Qwen3.8 的 belady → blockwise 仍差约 2.5 ms（模拟 blockwise 快 1.2，实测慢 1.3）；Qwen3-30B 的 blockwise 修前仍乐观约
10 ms，剩下的主要是写 SSD 对同时进行的读的干扰（加载期间撞上写盘的 526 条平均慢 13 ms，第一轮见下文）。

**exclusive 的 HBM 大半闲着**：块写到下层就丢 HBM 副本，HBM 里只有 held、pinned 和在写的块。P 日志 `KVPilot exclusive heads=`
行的中位数（后 80% 时间）：Qwen3-30B 2361 块里 free 1581（67%），Qwen3.8 1756 块里 501（29%，另有状态 248），Qwen3.6 FP8
1219 块里 338（28%）。belady 把整块 HBM 当缓存，Qwen3.8 上 53% 的前缀从 HBM 读，blockwise 只有 3%。修正后的模拟器在 SWE
上扫池 3/5/10、K 128–224、seed 0–3，Qwen3.8 的 belady → blockwise 余量都在 0 附近，没有一条 SWE trace 能分开；要让
blockwise 在 SWE 的 Qwen3.8 上赢 belady，可能得让闲着的 HBM 继续缓存已经在下层的块（要空间时再丢），这改的是 exclusive
「每块一份」的语义，没有动。

### 第二轮 TraceLab 结果（2026-11-25）

TTFT 均值（ms，第 160 条以后 840 条），`python3 summarize.py tl_rand`。Qwen3-30B 在 TraceLab 上没有跑（2026-11-25 决定先不跑；
5% 的 blockwise 跑到一半停掉，没有归档）。

| 模型 | LRU | belady | inclusive req | excl req unblocking | excl blockwise | LRU 前缀 HBM / DRAM / SSD |
|---|---|---|---|---|---|---|
| Qwen3.6 FP8 TP2 | 463.0 | 458.3（−1.0%） | | | 450.8（−2.6%） | 12 / 13 / 74% |

- Qwen3.6 FP8：belady 比 LRU 快 4.7 ± 1.3（模拟 5.9），blockwise 再快 7.5 ± 2.7（模拟 6.4），两级都显著，**顺序成立**。
  每组回放约 18 分钟（UTC 00:48–02:16）。
- Qwen3.8：LRU 856.0、belady 795.9（−7.0%）、blockwise 789.7（−7.7%）；belady 比 LRU 快 60.1 ± 40.2，blockwise 再快
  6.2 ± 4.8，两级都显著，**顺序成立**（模拟 LRU 855.9）。LRU 第一次在第 298 条挂了：Mooncake 丢了 TP0 上一块 inclusive
  的 KV（其余三个 rank 都在），混合模型的命中要落在 HBM 的状态上，等这块等满 30 s 报错。改成缺口里没有块在写就不等、
  命中退到 KV 能够到的状态（`docs/kv-placement.md`「混合模型」的「命中」）后重跑，1000 条全过，退回触发 2 次（又丢了
  两块），这两条的重算是 belady 那一对区间宽的原因。

### 阈值从 5% 降到 1%（2026-11-25 起重跑）

blockwise 先找最少的 HBM 头部块，使其余全放 DRAM 时变慢不到 `thresh`，再尽量多放 SSD 块、只要总变慢仍不到 `thresh`；
reqwise 的层也按它定。原来的 5% 让 SSD 尾部总是推到接近 5%：SWE 上 Qwen3-30B blockwise 剩下的 53.8 s 暴露加载里，
42 s 来自要求 DRAM 却把尾部放到 SSD 的 1769 条请求。阈值来自时间模型文件的 `thresh`，不改代码，复制一份 `thresh` 为 0.01 的系数（`fit_times2_*_t01.json`），
用 `KVPILOT_TIME_COEFFS_PATH` 指定，结果归档在 `<arm>-t01/`。0% 做不到：只要有一块不在 HBM，模型预测就严格大于全 HBM，
等于所有块都要进 HBM。修正后的模拟器（TTFT，ms）：

| 模拟 | 5% | 2% | 1% | 0.5% |
|---|---|---|---|---|
| SWE Qwen3-30B blockwise | 305.4 | 298.2 | 295.1 | 295.3 |
| SWE Qwen3-30B req unblocking | 312.0 | 309.0 | 310.3 | 311.0 |
| SWE Qwen3.8 blockwise（belady 285.0） | 283.8 | 280.7 | 281.3 | 282.6 |
| SWE Qwen3.6 FP8 blockwise | 161.6 | 161.0 | 160.3 | 160.5 |
| TraceLab Qwen3-30B blockwise | 1120.4 | 1116.0 | 1111.1 | 1112.2 |
| TraceLab Qwen3.8 blockwise | 813.3 | 807.8 | 808.1 | 810.3 |

收益来自 HBM 被用起来：1% 时 Qwen3-30B blockwise 的 HBM 命中从 7% 到 24%，SSD 从 45% 到 27%（阈值小了，更多请求要把头部
留在 HBM，用上了原本闲着的 HBM）。inclusive req 按 penalty 排、不看阈值，模拟结果不变。**1% 的真机重跑已取消**（2026-11-25，
先测怎么加快两个混合模型的计算；SWE Qwen3-30B 的 1% blockwise 跑到一半停掉、没有归档，排着的几组记在 GPU 的
`/root/exp1_queue_t01_cancelled.txt`），上表只有模拟。

## SWE 结果（2026-11-23 / 24）

**结论**：同一个 SWE trace 上，Qwen3.6 的三级顺序成立且两两显著；Qwen3.8 的顺序按均值成立，但 blockwise 只比
belady 快 2.5 ms，在噪声内；Qwen3-30B 前四级成立，blockwise 比 exclusive req unblocking 慢 4.7 ms，五级顺序不成立。
LRU 的前缀 82–83% 从 SSD 读，不是全 SSD。

TTFT 均值（ms，第 176 条以后 2824 条，括号内相对 LRU）。两个混合模型用绑 NUMA 后的重跑（`-numa`，见下）：

| 模型 | LRU | belady | inclusive req | excl req unblocking | excl blockwise | LRU 前缀 HBM / DRAM / SSD |
|---|---|---|---|---|---|---|
| Qwen3-30B TP2 | 250.2 | 220.1（−12.0%） | 217.3（−13.1%） | 208.4（−16.7%） | 213.1（−14.8%） | 3 / 19 / 79% |
| Qwen3.6 TP4 | 120.4 | 106.8（−11.3%） | | | 102.2（−15.2%） | 0 / 16 / 83% |
| Qwen3.8 TP4 | 270.0 | 263.2（−2.5%） | | | 260.7（−3.5%） | 0 / 18 / 82% |

相邻两组的配对差（ms，95% 区间按 50 条一批的批均值算；模拟值见下文「SWE 的选择」）：

- Qwen3-30B：belady 比 LRU 快 30.1 ± 4.3（模拟 30.4）；inclusive req 比 belady 快 2.8 ± 4.6，**在噪声内**（模拟 4.2 ± 4.7，
  也在噪声内）；req unblocking 比 inclusive req 快 9.0 ± 2.1（模拟 13.3）；blockwise 比 req unblocking **慢** 4.7 ± 2.2
  （模拟快 5.3 ± 0.9）。
- Qwen3.6：belady 快 13.6 ± 2.1（模拟 17.4），blockwise 再快 4.7 ± 2.2（模拟 8.3）。
- Qwen3.8：belady 快 6.8 ± 1.1（模拟 8.4），blockwise 再快 2.5 ± 3.9，**不显著**（模拟 6.2 ± 1.8）。

`python3 summarize.py swe` 出完整的表（含 p50 / p90、暴露的加载时间、SSD 第 0 层延迟）。

**Qwen3-30B 的 blockwise 为什么没赢 req unblocking**：blockwise 把 542 条在 req unblocking 下整段从 DRAM 读的请求的
尾部放到了 SSD（平均 34K token 里 12.9K，约 25 块），这些请求的 GPU 空等从 5.4 ms 涨到 45.2 ms，合计 −22.5 s；
两边都读 SSD 的 1672 条上它快 9.5 ms（+15.9 s），抵不过。blockwise 按时间模型选块，认为这样每条最多慢 5%，真机慢了
约 20%。拿每条请求的实际放置（HBM / DRAM / SSD 各多少 token）代入时间模型、和实测 prefill e2e 比
（`fit_times2_swe.json`，第 176 条以后）：

| Qwen3-30B，实测 − 模型 | 纯 SSD | HBM + SSD | DRAM + SSD | HBM + DRAM + SSD | 冷算 |
|---|---|---|---|---|---|
| LRU / belady | +5 / +2 ms | +4 / +2 ms | +13 / +4 ms | +5 / +5 ms | −10 ms |
| req unblocking | −16 ms | −20 ms | +15 ms（+9%） | +14 ms（+10%） | −18 ms |
| blockwise | −15 ms | −19 ms | **+22 ms（+13%）** | **+20 ms（+15%）** | −20 ms |

inclusive 各组合都在 ±5% 内；exclusive 下 DRAM + SSD 混合的组合被低估 14–22 ms，而 blockwise 有 2025 条请求正是这种
组合（req unblocking 1327 条）。exclusive 的冷算和纯 SSD 反而比模型快 15–20 ms（prefill 时不 write-through，模型是在
inclusive 下测的）。

**低估来自 exclusive 自己的写盘，不是拟合**：混合公式单独测过（`utils/ssd_load_check`，一次一条请求、后台不写盘，
e2e 误差中位 2.2–2.5%）。按加载期间有没有 exclusive writer 在写 SSD（P 日志的 `KVPilot exclusive job=... ssd=N ms=...`，
N > 0 的 job 与这条 prefill 的时间段重叠）把 DRAM + SSD 混合的请求分开：

| Qwen3-30B，实测 − 模型 | 加载时没有 SSD 写 | 加载时有 SSD 写（重叠占 e2e 84–88%） |
|---|---|---|
| req unblocking | +4.3 ms（932 条） | +37.5 ms（395 条） |
| blockwise | +5.7 ms（842 条） | +30.5 ms（1183 条） |

blockwise 把更多尾巴放 SSD，写 SSD 的 job 更多，撞上的请求是 req unblocking 的 3 倍，所以输。

慢在盘上，不在线程、锁或 GIL：SSD 读是 loader 自己的 64 个 C++ 线程 `pread`（O_DIRECT）；exclusive 的写在 writer 的
后台线程里调 Mooncake `BatchOffload`（释放 GIL），写 bucket + `fdatasync` 时不持锁，读前的 `BatchLocate` 只拿共享锁。
按加载量分档，撞上写盘时**加载时间本身**变长（blockwise：<20K token 55→79 ms，20–40K 120→154 ms，≥40K 195→281 ms；
每个 SSD 块约 5→7 ms），GPU 空等随之增加。写得多、成批、时间不错开：

| | 写 SSD 的块（每 rank） | 含 SSD 写的 job（同一 job 里也写 DRAM） | 写的时机 |
|---|---|---|---|
| LRU / inclusive write-through | 16201 块，380 GiB（每块只在第一次算出时写一次） | —（每条请求中位 2 块、p90 9 块） | 紧跟自己的 prefill，下一条开始加载前约 50 ms 空档 |
| req unblocking | 12980 块，304 GiB | 656 个，中位 174 ms、p90 1.1 s | 判据到达时（上一轮 decode 结束），和别的请求的加载不错开 |
| blockwise | 22294 块，523 GiB（write-through 的 1.4 倍） | 2248 个，中位 110 ms、p90 0.5 s | 同上；这些 job 在跑的时间占运行的 48%（req unblocking 28%） |

写 SSD 的块数取自 P 日志 `KVPilot placement applied` 的 `write_ssd`；job 行的 `chunks written` 含 DRAM 写，不能当 SSD 写量。

**exclusive 为什么写得比 write-through 多**：每块只存一份，判据换层时旧层的副本当场删掉，SSD 也一样；块后来再回到
SSD 就得重写。按判据日志重放每个块的档位：

| | 写 SSD 的块 | 第一次上 SSD | 离开 SSD 后又写回 | 中间去了哪 |
|---|---|---|---|---|
| req unblocking | 13268 | 9054 | 4214（32%） | S→D→S 44%，S→H→S 27% |
| blockwise | 22583 | 11299 | 11284（50%） | S→D→S 82%，S→H→S 9% |

blockwise 里同一 session 相邻两轮的 DRAM / SSD 分界来回移动（向后 41%、向前 49%，中位 10 块），分界前后的块在 DRAM
和 SSD 之间来回搬，每搬一次两层各写一遍；2971 个块写过 SSD 3 次以上。

blockwise 单次搬得少（有搬动的轮中位 11 块，req unblocking 17 块），但总量更多：

| 相邻两轮之间 | 有搬动的轮 | 搬动总块数 | 主要的搬动 | 新块直接放 SSD | 块放在 HBM / DRAM / SSD |
|---|---|---|---|---|---|
| req unblocking | 58% | 37329 | D→H 10272、H→D 8672、S→D 6103、D→S 5595、H→S 5154 | 2519 | 14 / 73 / 13% |
| blockwise | 93% | 40317 | **S→D 16012、D→S 13832**、D→H 3589、H→D 3216 | 6441 | 5 / 61 / 33% |

- 一个请求实际只有一到两段（`place`：先取最少的 HBM 块，再在其余块里取最多的 SSD 块）：下一轮 HBM-safe 的 382 轮是
  HBM + DRAM（96%）或只有 HBM，DRAM-safe 的 1890 轮是 DRAM + SSD（84%）或只有 DRAM，SSD-safe 的 418 轮全放 SSD，
  没有三段都有的。
- 按相邻两轮的档位变化拆：换档时 blockwise 搬得比 req unblocking 少（DRAM→SSD 的 D→S 4184 对 5476，SSD→DRAM 的
  S→D 4670 对 5887）；多出来的主要在 DRAM→DRAM 的 1272 轮：req unblocking 只把新块写进 DRAM，blockwise 搬了
  S→D 7292、D→S 6850，写 SSD 7865 块（它全部 SSD 写的约 35%）。DRAM + SSD 里 SSD 的块数由下一轮 suffix 的 95% 分位
  上计算能盖住多少 SSD 加载决定，每轮都变，也不随 prompt 变长（prompt 平均每轮长 2.1 块，SSD 部分平均只变 +0.1 块）。
  HBM↔DRAM 换档时，DRAM-safe 的布局带 SSD 尾巴、HBM-safe 的没有，所以 HBM→DRAM 要写 2092 块到 SSD，DRAM→HBM
  要把 3312 块从 SSD 提回 DRAM。
- req unblocking 只在请求换档时搬，搬动多在 DRAM 和 HBM 之间：D→H 是加载后删 DRAM 副本，不写；H→D 写 DRAM。
- SSD 写 = 新块直接放 SSD + D→S + H→S：blockwise 6441 + 13832 + 2310 = 22583，req unblocking 2519 + 5595 + 5154 = 13268。

每次判据写 SSD 的量（`write_ssd`，每块每 rank 24 MiB）：req unblocking 中位 8 块、p90 55 块，其中换档占 49%（中位 23、
p90 68 块，整段 prompt）、没有下一轮占 37%（session 最后一轮把 HBM 里的块全写到 SSD）；blockwise 中位 5 块、p90 25 块，
档位不变（分界移动）占 39%、换档 38%。write-through 每条请求中位 2 块、p90 9 块。

**反事实重放**（判据日志，每轮 blockwise 的 (HBM, DRAM, SSD) 块数不变；时间模型只看各层块数，不看是哪几块）：

| blockwise 的做法 | SSD 写 | DRAM 写 |
|---|---|---|
| 实际（SSD 放尾部，换层删旧副本） | 22583 | 25417 |
| 块数不变，已在 SSD / DRAM 的块尽量不动 | 20536 | 21642 |
| SSD 放尾部，搬离 SSD 时保留盘上副本 | 11299 | 25417 |
| 两者都做 | 9462 | 21875 |

req unblocking 保留 SSD 副本时 SSD 写 13268 → 9054。write-through 是 16201。第一次上 SSD 的块（11299）反而比 write-through
（16201）少，多出来的全是重写。SSD 不缺容量（12 TB，每组用 300 GB 左右），删 SSD 副本省不下有用的空间。

两块 NVMe 条带的 dm-0 上，几百 MiB 的写加 `fdatasync` 让同时进行的小块 O_DIRECT 读排队。`utils/ssd_load_check` 的混合
测试一次只发一条、后台没有写，所以没测到；inclusive 的写量小、时间错开，影响很小；sweblock160 的 exclusive A/B 已经
看到过（撞上写盘的请求慢 6–8 ms，blockwise 撞得更多，62% 对 31%），当时记成模拟器没建模的剩余误差。
两个混合模型每 token 的 KV 小 4–5 倍，写得少，blockwise 的 DRAM + SSD 组合在 ±3% 内。这次模拟的余量只有 5.3 ms，
被吃掉并反过来。

**Qwen3.8 收益小**：prefill 慢，加载大多被计算盖住。blockwise 在两边都读 SSD 的 1208 条上快 10.9 ms（空等 14.4 → 4.7 ms），
但 belady 下从 HBM / DRAM 读的 1421 条被它部分放到 SSD 后每条慢 7–9 ms，相抵只剩 2.5 ms。

**混合模型「状态在 HBM、KV 还在写」的等待**：exclusive 下同一 session 的下一轮紧跟在上一轮判据之后到达时，上一轮最后
一块的 KV 还在写往 SSD，请求先不调度、下一步再查（`docs/kv-placement.md`「混合模型」）。SWE 3000 条里 Qwen3.8 3 次
（9–260 ms），Qwen3.6 4 次（37–59 ms），都是第 670、1692、1810 条这几处，时间算在 KVPilot 的 TTFT 里。第一次跑时这里
有个 bug（lookup 结果按请求缓存、重查拿到的还是旧值，等满 30 s 后 P 报错退出），修了以后重跑。

**NUMA**：最初几组 worker 没绑 NUMA，TP 4 上每次起服务都有随机几个 rank 的 SSD 第 0 层读慢 3–4 倍（pinned 读缓冲落在
另一个 socket，`docs/pitfalls.md`「跨进程/跨机」）。Qwen3.8 第一次的 belady 因此只比 LRU 快 1.5 ms（`belady`），
重跑一次正常（`belady-r2`，快 6.9 ms）。之后 `disagg_lmcache.sh` 默认按 GPU 绑 NUMA，两个混合模型的三组整套重跑
（`-numa`），上表用的就是这一套；绑了以后 24 个 rank 都正常，各组 TTFT 普遍快 4–11 ms。Qwen3-30B（TP 2）的五组
在绑之前跑，两个 rank 都没有慢的，没有重跑。没绑时的那几组留在 `raw/` 里，结论方向相同：

| 模型（未绑 NUMA） | LRU | belady | excl blockwise |
|---|---|---|---|
| Qwen3.6 | 131.5 | 114.0（−13.3%） | 109.6（−16.7%） |
| Qwen3.8 | 273.9 | 272.4 / 267.0（重跑） | 264.5（−3.5%） |

## TraceLab：没有结果

`run_all.sh` 的 TraceLab 九组（`EXP1_TL_K=40`，绑 NUMA）2026-11-24 00:27 UTC 起在 GPU 上跑，结果拉回本地前机器被释放，
日志全丢；本地 `raw/` 里只有 SWE。重跑：恢复环境（`docs/gpu-restore.md`）后执行 `run_all.sh` 的 TraceLab 三行。

## trace 怎么来的

`datasets/block-priority-trace/search_trace_models.py` 在三个模型的模拟器（`sim_placement.py`，各自的时间模型、
块大小、容量、状态块）上搜参数，`make_traces.py` 写出选中的那个。

**两类 session（`h<H>f<pct>`）**：K 个「慢」session 轮转发送，制造容量压力；另有 H 个「快」session 也轮转，
占 pct% 的请求，复用距离只有几十个请求，LRU 也能在 HBM / DRAM 命中它们。快 session 的下一轮仍晚于它上一轮
decode 完（复用距离 × 每条约 0.2 s > decode 时间），不会堵 `-p 1` 的回放。

试过而放弃的发送方式（模拟，Qwen3-30B 为主）：

- **完全随机挑 session**（K 个活跃 session，每条请求随机挑一个）：LRU 的 SSD 比例降到 40–60%，但 inclusive req
  比 belady 慢 1–2 个百分点。随机复用距离正是 belady 的未来信息（下次什么时候用）最有用的情形，inclusive req
  只知道放低一层慢多少；快慢两类 session 把同类的复用距离拉齐，等于压掉了 belady 的这项优势，不通用。
  **2026-11-24 起不再要求 inclusive req 快于 belady**，随机发送就够了，见下「随机发送的候选」。
- **轮转加随机窗口**（从队头 w 个里随机挑）：窗口大时同上，窗口小时 LRU 仍约 98% SSD。
- **按一定概率紧跟上一轮**：LRU 有了 HBM 命中，但紧跟的那一轮要等上一轮 decode 完，回放时间涨到 35–53 分钟。
- **放大 DRAM**：只有 LRU 约 98% SSD 时 blockwise 才赢 req unblocking。

结论：在当前设计下，Qwen3-30B 的五级顺序要求容量足够紧，容量越紧 LRU 越接近全读 SSD；两类 session 是在
顺序仍成立时让 LRU 有一部分非 SSD 命中的折中。

**SWE 的选择**（模拟，各模型自己的时间模型，对齐后的容量，第 176 条以后）：

| 模型 | LRU TTFT | belady | inclusive req | excl req unblocking | excl blockwise | LRU 前缀 HBM/DRAM/SSD | 回放 |
|---|---|---|---|---|---|---|---|
| Qwen3-30B | 251.8 ms | −12.1% | −13.8% | −19.0% | −21.1% | 3 / 11 / 86% | 21 min |
| Qwen3.6 TP4 | 148.8 ms | −11.7% | | | −17.3% | 0 / 12 / 88% | 13 min |
| Qwen3.8 TP4 | 290.0 ms | −2.9% | | | −5.0% | 0 / 12 / 88% | 23 min |

相邻两组之差（ms，±95% 区间）：Qwen3-30B +30.4、+4.2±4.7、+13.3±1.0、+5.3±0.9；Qwen3.6 +17.4±2.2、+8.3±2.5；Qwen3.8
+8.4±1.2、+6.2±1.8。只有 Qwen3-30B 的 belady → inclusive req 在噪声以内（所有候选和种子里均值都为正，+2 到 +5 ms），
真机 sweblock160 时这一对也只差 7.8±10.7 ms。Qwen3.8 收益小，是因为它 prefill 慢，加载大多被盖住（request-safe
上 prefix 全从 SSD 只比全在 HBM 慢 12%）。

**随机发送的候选**（2026-11-24，模拟，按改后的要求：Qwen3-30B 4 对、混合模型 2 对；容量同下文；3000 条，
第 K 条以后）。`search_trace_models.py --orders rand --pools 10,20,30,50 --ks 128..320 --seeds 0,1,2`，84 个候选里
50 个全部成立。余量 = 配对差减 95% 区间（ms），各对依次为 LRU→belady / belady→blockwise / inclusive req→unblocking /
unblocking→blockwise：

| 候选（种子 0 / 1 / 2） | Qwen3-30B 最小余量 | Qwen3.6 最小余量 | Qwen3.8 最小余量 | LRU 前缀从 SSD | 回放 |
|---|---|---|---|---|---|
| 池 10、K 160 | +5.1 / +3.9 / +4.2 | +4.8 / +4.1 / +4.6 | +3.6 / +3.7 / +4.0 | 47–50% | 20–32 min |
| 池 20、K 192 | +2.0 / +3.1 / +4.5 | +3.9 / +3.7 / +3.3 | +2.2 / +2.2 / +2.7 | 50–52% | 17–29 min |

最小余量都落在 belady→blockwise（混合模型）或 unblocking→blockwise（Qwen3-30B，均值 +3.0–6.2 ms）。其余各对：
Qwen3-30B LRU→belady +18–23、belady→blockwise +9–12、inclusive req→unblocking +10–13；Qwen3.6 LRU→belady +12–14；
Qwen3.8 LRU→belady +6–8。

K 越大 Qwen3-30B 的 unblocking→blockwise 越大（池 10、K 320 时 +14.6 ms），但 Qwen3.6 的 belady→blockwise 变负；
三个模型都成立时这一对最多约 8 ms。前两次真机这一对都比模拟少 8–10 ms（sweblock160 模拟 11.3、真机 3.7；本次 SWE
模拟 5.3、真机 −4.7），原因是 exclusive writer 写 SSD 时拖慢同时读 SSD 的加载（上文），模拟里没有这一项。不解决写盘
干扰，换成随机发送后这一对在真机上大概率仍不成立。池 10 / 20 是按每轮 blockwise HBM 需求排前 10% / 20% 的 session，
即上下文长的 session（prompt 均值约 21–25K）。

**TraceLab**（`tracelab_all.jsonl`，去掉单轮和 prompt+输出超过 200K 的 session，1698 个）：中位 prompt 70K token
（SWE 16K），中位新算 1K，中位输出 248 token（SWE 64），几乎全是加载受限。prompt 长，每条要几百毫秒，所以只取 1250
条（30 分钟内）。模拟：

| 模型 | LRU TTFT | belady | inclusive req | excl req unblocking | excl blockwise | LRU 前缀 HBM/DRAM/SSD | 回放 |
|---|---|---|---|---|---|---|---|
| Qwen3-30B | 661.1 ms | −16.0% | −14.1% | −19.2% | −18.7% | 12 / 11 / 77% | 27 min |
| Qwen3.6 TP4 | 319.6 ms | −22.9% | | | −27.9% | 10 / 13 / 77% | 18 min |
| Qwen3.8 TP4 | 535.0 ms | −11.8% | | | −14.0% | 9 / 14 / 77% | 23 min |

三个模型上 LRU > belady > blockwise 都成立，但 Qwen3-30B 的完整五级顺序在 TraceLab 上不成立：inclusive req 慢于
belady，blockwise 与 req unblocking 持平。所有 TraceLab 候选（池 0/20/50、K 32–80、各种发送方式）都是这样：请求几乎
都要 HBM 才 safe，penalty 区分不出请求；blockwise 每条允许的 5% 露出也换不回容量。所以 TraceLab 上每个模型只跑三组。

## 容量：三个模型按 token 对齐

默认容量下三个模型能装的 token 数差 2–5 倍（Qwen3.6 TP4 的 HBM 708 万 token，Qwen3-30B 121 万），同一个
trace 不可能同时给它们相近的压力（Qwen3.6 在 K=128 时 LRU 全部命中 HBM）。所以三个模型的 KV 容量按 token
对齐到 Qwen3-30B 的默认值，同一模型的各组完全相同（`run_exp1.sh`，`run_ab.sh` 的 `HBM_BLOCKS` / `DRAM_GIB`）：

| 模型 | HBM（块） | DRAM（GiB/rank，0.9 水位） | KV token：HBM / DRAM |
|---|---|---|---|
| Qwen3-30B，TP 2，块 512 | 2361 | 100 | 121 万 / 197 万 |
| Qwen3.6，TP 4，块 1024 | 1180 + 3 × K（状态） | 21 | 121 万 / 198 万 |
| Qwen3.8，TP 4，块 1024 | 1180 + 4 × K（状态） | 27 | 121 万 / 200 万 |

混合模型每个活跃 session 在 HBM 里留一份状态（Qwen3.6：3 个 GDN 组；Qwen3.8：3 个 GDN 组和 PLE），所以 HBM
多给 K 个 session 的状态块，KV 能用的 HBM 与 Qwen3-30B 相同。

## 各组之间只差 placement

- 同一模型各组：同一 trace、同一服务配置（HBM 块数显式指定，避开 Qwen3.8 首次编译时 KV 被压小）、同一 DRAM、
  同一 SSD offload、同一回放参数（`-p 1 --release-at-first-token --forced-output`）。
- 混合模型的线性注意力状态在所有组里都只留在 HBM，按同一规则持有（LRU 也按回放的 trace 持有下一轮还会用的
  状态，见 `docs/kv-placement.md`「混合模型」），只差 KV 放在哪一层。
- 判据表是 oracle（下一轮的真实 suffix），时间模型来自各模型自己的 request-safe（`paper-figure/Figure3-request-safe`）。
