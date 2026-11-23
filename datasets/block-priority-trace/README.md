# block-priority-trace

给按块放置（`exclusive-penalty-blockwise-unblocking`）和 exclusive 造的 trace：容量要紧到连 blockwise（每块只占它
需要的最快层）都放不下，按请求放、多副本才会因为占了不必要的快层而吃亏。

| 文件 | 内容 |
|---|---|
| `gen_trace_capacity.py` | 生成脚本（读取、回放时间估计、写 trace 复用 `../req-priority-trace/gen_trace.py`） |
| `sim_placement.py` | 块级模拟：七个策略在一条 trace、一组容量下的命中层、prefill e2e 和 TTFT，见「sim_placement.py」一节 |
| `search_trace.py` | 用模拟器在 `swe_full` 里搜「最少请求就能分开各策略」的 trace，见末节 |
| `swe_blockprio.jsonl` | 从 `agent-traces/swe-smith-trajectories/swe_full.jsonl` 生成，判据表 `utils/placement_ab/signal_swe_blockprio.json` |
| `swe_blockprio_k160.jsonl` | `search_trace.py` 给默认 HBM 挑的短 trace（1300 条，前 160 条预热），判据表 `signal_swe_blockprio_k160.json` |

## swe_blockprio：blockwise 下 HBM 和 DRAM 也满

- **每轮的放置**：和 P 上用 oracle 判据时完全一样，用时间模型（`fit_times2_swe.json`）和下一轮的实测 suffix 算
  blockwise 的 (HBM, DRAM, SSD) 块数和 reqwise 的整段 tier（`kvpilot_blockwise.place` / `reqwise`，这里用二分实现）。
  一个 session 在两轮之间占着它最近一轮的放置，轮转回放时一层的需求 = 活跃 session 之和。
- **为什么要缩 HBM**：全量 `swe_full` 里 blockwise 只把 2.6% 的块放进 HBM（每个 session 平均 0.7 块），默认 2361 块
  的 HBM 要约 480 个 HBM 需求最高的 session 同时进行才填得满，那时 DRAM 超出 5 倍、回放一轮就要 2 万多条请求。
  所以取 blockwise HBM 需求前 5% 的 session（每个平均 HBM 3.2 块、DRAM 31.7 块，约 1:10），HBM 缩到 600 块
  （去掉一个 chunk 的预留和最长的 prompt 290 块，能放 held 块的 294 块），DRAM 不变（每 rank 100 GiB，0.9 水位
  3840 块），两层需求比例正好对上。
- **进场**：候选随机排序（`--seed 0`），轮转发送；blockwise 平均需求低于 1.2 倍容量（`--over`）时补新 session；回放
  估到 45 分钟截断。
- **结果**：6897 条请求、249 个 session。去掉前 1/4 预热后，blockwise 需求中位 HBM 412 块（1.40 倍）、DRAM 4506 块
  （1.17 倍），两层同时超出的时间占 89%；reqwise HBM 1105 块（3.76 倍）、DRAM 5341 块（1.39 倍）；活跃 prompt KV
  11189 块，是 DRAM 的 2.9 倍（LRU / inclusive 大量从 SSD 读）。
- **判据表** `utils/placement_ab/signal_swe_blockprio.json`：`next_suffix` 来自 trace；`next_tier` / `next_time` /
  `own_tier` 用同一个时间模型算（放置策略只看有没有下一轮和 suffix），不需要 request-safe 实测。
- A/B：`run_ab.sh <arm> sweblock`，`HBM_BLOCKS` 默认 600。

```bash
cd datasets/block-priority-trace
python3 gen_trace_capacity.py   # -> swe_blockprio.jsonl 和判据表；参数见 -h
```

## sim_placement.py：不上 GPU 预测各策略

```bash
python3 sim_placement.py --hbm-blocks 600      # swe_blockprio，七个策略，稳态从第 1970 条起；默认 HBM 2361 块
python3 sim_placement.py --strategies lru,incp --logs lru=<dynamo-log 目录> incp=<目录>   # 和真机对比
```

按 trace 顺序逐条请求模拟（模型和不模拟的部分写在脚本开头）：块按「内容 + 前缀链」识别；请求先读 HBM 里连续的前缀，
再在 DRAM、SSD 里接着读；prefill 时间用时间模型（`fit_times2_swe.json`）。HBM / DRAM 按各策略的键淘汰，DRAM 超水位时
inclusive 一次删一成；exclusive 和 master 一样每轮最多下沉 85 块（`kKvpExclMaxInflight` 8192 个 value），直到回到水位以下，
所以 DRAM 一直贴着水位。时钟按回放推进（首 token 到了才发下一条），判据在这一轮 decode 结束（输出 token × `--tpot`）
后到达；exclusive 往下写走一个串行 writer（每个 job 4.9 ms + 每块 DRAM 1.0 ms / SSD 25 ms，由真机 job 日志拟合），
写完块才离开 HBM，放不下时等写完或等判据。

**TTFT**（`done − sent`）= 准入前开销 + P 上排队 + 分配阻塞 + prefill e2e：
- 发送：上一条的首 token 到客户端后 0.6 ms（`-p 1 --release-at-first-token`），或同 session 上一轮 decode 完；
- 准入前开销（前端、查找）：约 6 ms + 每千 token（新算、HBM、DRAM、SSD 分别计价）约 1 ms；HBM、DRAM 命中的 token
  inclusive 要 1.3 / 1.4 ms，exclusive 只要 0.9 / 1.0 ms（HBM 600 和 2361 两种容量下一致），按两族分别拟合（8 组真机，残差 sd 5–7 ms）；
- P 上排队：P 一次只做一个 prefill，到达时上一条还没算完才排；`-p 1` 下首 token 要等 P 算完，所以恒为 0，真机也是 0；
- 分配阻塞：exclusive 系列 HBM 不够时等写回或判据；
- P 算完到客户端首 token：52.9 ms + 每千 token 0.97 ms + 一个 `--tpot`（D 拉 KV、第一步 decode）。

**和真机对照**（2026-11-22，`swe_blockprio`、HBM 600 块，第 1970 条以后，`--tpot 14.7`）：

| | TTFT 模拟 / 真机 | prefix 从 DRAM 读 模拟 / 真机 |
|---|---|---|
| LRU | 316.0 / 321.0 ms | 0.2 / 0.3% |
| inclusive-penalty | 263.3 / 263.6 ms | 40.6 / 42.6% |
| unblocking（真机是旧借块规则，模拟是新的） | 298.1 / 297.7 ms，阻塞 217 / 196 s | 51.8 / 48.5% |

默认 HBM（`swe_blockprio_k160`，第 160 条以后）：LRU 259.2 / 259.8，belady 215.0 / 216.6，inclusive-penalty 209.5 / 208.8，
unblocking 193.8 / 190.8，blockwise 183.9 / 187.2 ms。

独立性：时间模型系数来自别的 trace；准入前开销和 writer 耗时取自这些真机本身，是样本内的。阻塞对 `--tpot`（判据多晚到）
很敏感：各臂实测 decode 每 token 12.5（LRU）、17.5（inclusive-penalty）、14.7 ms（unblocking），原因不明。

**还没模拟的**：exclusive 的 writer 往 SSD 写时，同时从 SSD 加载的请求比没有写时慢 6–8 ms（e2e 减去同命中层下的时间
模型）；blockwise 直接写 SSD 多，62% 的 prefill 撞上写盘，unblocking 31%。所以模拟偏向 blockwise，blockwise 对 unblocking
模拟 9.9 ms、真机 3.7 ± 2.4 ms。真机 blockwise 在第 445–730 条还有一段比时间模型慢 6 ms，与写盘无关，单次运行分不清是
模型还是波动。模拟器发现过两处和真实代码不一致的地方并已改掉：同 session
下一轮只取代已收到判据的 head；blockwise 借不够块时不借，转为下沉并等待。

## search_trace.py：最少请求分开各策略

```bash
python3 search_trace.py --ks 128,160,192,224,256,320,384 --mids 1 --seeds 0,1,2   # 网格，默认 HBM 2361 块
python3 search_trace.py --write pool=10,k=160,mid=1,seed=1 --n 1300 --out swe_blockprio_k160.jsonl
```

候选：`swe_full` 里按 blockwise 每轮 HBM 需求排前 `pool`% 的 session（至少两轮），随机排序，K 个同时轮转，完成一个补
下一个。`mid=1` 时开头 K 个 session 从随机一轮加入：它们第一轮全是冷的长 prompt，一轮下来 HBM 和 DRAM 就满，不用
几千条短的首轮慢慢填；之后补进来的 session 从第 0 轮开始。`mid=0` 时 K 个 session 同时从第 0 轮起步、一起变长，
稳态要等很久，且不同截断点压力差很多。

对每个候选模拟 LRU、belady、inclusive-penalty、unblocking、blockwise 的逐条 TTFT。一对策略 (a, b) 在去掉前 W 条后要的
稳态条数 = (3 × sd / 均值差)²，sd 用每 50 条一批的批均值折算（缓存状态有惯性，逐条不独立）。要分开的对：blockwise 对
其余四个、unblocking 对 inclusive-penalty、各策略对 LRU；均值差不到 LRU TTFT 的 2.5%（模型误差量级）算分不开。
所需条数 = min over W（W + 最难那对的稳态条数）。

**默认 HBM（2361 块）、DRAM 每 rank 100 GiB 的结论**（`--tpot 14`，12.5 / 17.5 结论不变）：

- K ≤ 96 装得下，各策略几乎一样；中途加入、K ≥ 128 后排序稳定：blockwise > unblocking > inclusive-penalty ≈ belady > LRU。
- inclusive-penalty 和 belady 在所有候选里只差 −1 ~ +2%，分不开；unblocking 对 belady 差 3–5%，要 1500–3000 条稳态。
- 最优预热就是 K（第一轮）；最紧的一对是 blockwise 对 belady（差 7–9%，sd 大）。
- 选 `pool=10, K=160, seed=1`：需要约 700 条（预热 160 条），三个种子一致；写出 1300 条（约 2 倍余量，模拟回放约 12 分钟，
  其中第一轮冷启动约 5.4 分钟）。

| 模拟（第 160 条以后，1140 条） | TTFT 均值 | 相对 LRU | prefix 从 HBM / DRAM / SSD |
|---|---|---|---|
| LRU | 259.8 ms | — | 1.5 / 0.0 / 98.5% |
| belady | 212.9 ms | −18.0% | 35.2 / 21.9 / 42.9% |
| inclusive-penalty | 207.7 ms | −20.0% | 24.3 / 22.4 / 53.4% |
| unblocking | 199.2 ms | −23.3% | 14.1 / 46.4 / 39.5% |
| blockwise | 188.0 ms | −27.7% | 6.4 / 46.5 / 47.0% |

A/B：`run_ab.sh <arm> sweblock160`，HBM 用默认。真机结果（第 160 条以后）：LRU 259.8 ms，belady −16.6%，
inclusive-penalty −19.6%，unblocking −26.5%，blockwise −28.0%；LRU、belady、inclusive-penalty、blockwise 的 TTFT 和模拟差
不到 4 ms，unblocking 真机比模拟快 8.4 ms，所以 blockwise 对 unblocking 只差 3.7 ± 2.4 ms（模拟 11.3 ms）。逐对结果见
`docs/kv-placement.md`「A/B」。
