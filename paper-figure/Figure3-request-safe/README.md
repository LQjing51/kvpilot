# SWE：HBM / DRAM / SSD safe（仅 prefix>0）

- `request_safe.pdf`：每个数据集三柱（HBM / DRAM / SSD safe）。目前只有 SWE 的数据；ToolAgent、TraceLab
  按同样测法跑出 `raw/toolagent/`、`raw/tracelab/` 后 `plot.py` 会自动加上。**prefix=0 不跑、不进分母**。
- `model_safe_boundary.pdf`：在 `raw/swe/` 上拟合的时间模型给出的 DRAM / SSD safe 边界，以及它们在
  (prefix, suffix) 平面上分出的 HBM / DRAM / SSD safe 三个区域（`plot_boundary.py`，见末节）。

## 配置

- 2026-11-08（part0）/ 11-09（part1），Taiji-GPU（H20 × 8），1P1D，Qwen3-30B-A3B-Instruct-2507，TP=2，block 512，prefill
  `--max-num-seqs 1` + cudagraph `[64,128,256,512,1024]`，`--gpu-memory-utilization 0.9`
- `KVPILOT_PRELOOKUP_VIEWS=true`、`LMCACHE_FIRST_LAYER_LOOKUP=false`、`KVPILOT_LOAD_STALL=true`；C++ DRAM
  loader 在 `start` 时排完全部层；Mooncake SSD offload 开，heartbeat 100 ms，SSD 配额每 TP rank 4 TiB
- 源：`swe_mooncake.jsonl` 全部 5140 行，分两段各起一次服务：`raw/swe/part0/` = 第 0–1999 行，`part1/` =
  第 2000–5139 行（prefix 仍按从第 0 行起出现过的块算）。和 `swe_mooncake_interleaved.jsonl` 是同一批请求
- LMCache 用上游的 save 逻辑：只在 SSD 上的命中不会被写回 DRAM（见 `docs/pitfalls.md`）

## 复现

```bash
cd paper-figure/Figure3-request-safe
python3 make_trace.py swe --start 0 --stop 2000 --part part0       # -> raw/swe/part0/trace.jsonl
python3 make_trace.py swe --start 2000 --stop 5140 --part part1    # -> raw/swe/part1/trace.jsonl
bash /root/kvpilot/paper-figure/Figure3-request-safe/raw/run_one.sh swe   # GPU 上，逐段起新服务
python3 plot.py
```

整条 SWE 约 8.8 TB KV，超过一次服务的 SSD 配额（每 TP rank 4 TiB），所以按行号切两段；不带 `--part`
时只生成前 2000 行（`LINES`）。

`make_trace.py` 拒绝往已有日志的目录写 trace：重跑前先把旧日志挪走。`run_one.sh` 跳过已有日志的 run
目录，中断后可以直接重跑。`raw/run_all.sh` 依次跑 ToolAgent、SWE、TraceLab。

## 请求构造

- **只取 trace 的长度**：prefix 长度 = 该行开头已在更早行出现过的整 block 数（无限池），至少留 1 个 block
  计算；suffix = 余下部分
- **每行一组全新 hash**：warm 只发 prefix，从头算完并写进 HBM / DRAM，随后 offload 到 SSD；`[1]` HBM、
  `[2]` DRAM、`[4]` SSD 共用这段 prefix，**各自再接一段不同的、长度相同的 suffix**，所以三个测试都会算并
  写自己的 suffix。测量不依赖更早的行在 HBM / DRAM / SSD 里留下了什么
- **每行只有 warm、`[1]`、`[2]`、`[4]` 四个请求**，不等待、不预热。`[4]` 排在下一行的 `[2]` 之后：下一行
  的 warm 给 prefix 落盘的时间，`[4]` 也紧跟在一个请求之后而不是空闲之后
- **prefix 第一块的 id 按行单独分配**：回放把 id 按 vocab 取模映射成 token，id 用到约 15 万后会回绕；只有
  第一块的 cache key 只取决于它自己的 token，所以只需保证第一块互不相同
- **来源核对没过的行重测**（`retry.py`）：同一服务上换一段新 prefix，warm 后只测没过的 tier，`[4]` 前等
  4 s。不复用主轮的 prefix：那时它可能只剩 SSD，warm 从 SSD 读不会写回 DRAM
- **TraceLab 要拆成两段**：整段约 10 TB KV，超过一次服务的 SSD 配额。`make_trace.py` 按 KV 量切成
  `raw/tracelab/part0/`、`part1/`，`run_one.sh` 逐段起新服务跑，`cleanforce` 会清空 SSD

## 口径

- t1 / t2 / t4 = `[1]` / `[2]` / `[4]` 的 `KVPilot prefill ... e2e`（调度到 prefill 算完，不含排队）
- SSD safe：(t4 − t1) / t1 < 5%；否则 DRAM safe：(t2 − t1) / t1 < 5%；否则 HBM safe
- 来源核对（`tier_logs.check`）：`[1]` 要 `hit ≥ prefix` 且 `dram = 0`；`[2]` / `[4]` 要 `hit = 0`、`dram ≥ prefix`，
  两个 TP rank 都读满 prefix 的 chunk，`[2]` 的 `ssd_values = 0`、`[4]` 的 `ssd_values = 48 × chunks`
- `tier_logs.dataset()` 合并各 part 与重测，`plot.py`、`fit_times.py` 都走它

## 结果（`raw/swe/`）

4900 行（prefix>0），19600 个请求：part0 1910 行 47 分钟，part1 2990 行 74 分钟，全部通过来源核对、无重测。

| 数据集 | 行数 | HBM safe | DRAM safe | SSD safe |
|---|---|---|---|---|
| SWE | 4900 | 14.3% | 69.2% | 16.4% |

两段分开看：part0 16.6% / 67.9% / 15.5%，part1 12.9% / 70.1% / 17.0%。

- 14 行的 `[1]` 异常慢（如 line 79：791 ms，同形状的 `[2]` / `[4]` 约 235 ms），多在一次服务开头、eager 形状
  第一次出现时，`[2]` / `[4]` 都比 `[1]` 快，被判成 SSD safe
- 在这批真实形状上 gpu_stall 和 delay 逐请求对比：`[2]` 中位 +0.40（suffix ≤ 1K）/ +0.61 ms（> 1K），`[4]`
  −0.64 / −0.04 ms，|err| p90 1.4–3.2 ms；更大的误差都来自上面那几行 `[1]` 自己慢

## SWE 1:2 子集（`raw/swe_hbm1to2/`）

`swe_hbm1to2.jsonl`（`utils/trace_select/`，按期望 HBM : DRAM ≈ 1:2 挑的 368 个 session）用同一套测法：
2026-11-10，配置同上，约 7.7 TB KV，`make_trace.py` 按 KV 切成两段（part0 2895 行、part1 2938 行），
各起一次服务，共 99 分钟。

```bash
python3 make_trace.py swe_hbm1to2 --stop 6026    # -> raw/swe_hbm1to2/part{0,1}/trace.jsonl
bash /root/kvpilot/paper-figure/Figure3-request-safe/raw/run_one.sh swe_hbm1to2   # GPU 上
```

5833 行（prefix>0），23332 个请求，全部有日志；一行的 `[4]` 来源核对没过，重测后通过。

| | 行数 | HBM safe | DRAM safe | SSD safe | HBM : DRAM |
|---|---|---|---|---|---|
| 实测 | 5833 | 27.5% | 57.8% | 14.7% | 1:2.10 |
| `select_safe_mix.py` 期望 | 5833 | 27.9% | 56.6% | 15.4% | 1:2.03 |

两段分开看：part0 27.2% / 56.9% / 15.9%，part1 27.8% / 58.6% / 13.6%。`plot.py` 没有加这组。

## req-priority trace（`raw/swe_reqprio/`、`raw/tracelab_reqprio/`）

`datasets/req-priority-trace/` 的两条 trace，给请求粒度优先级 A/B 生成分档表（`paper-figure/Eval-three-req-priority/`）。
测法同上，和上面几组的差别：

- 2026-11-11 / 12，11-11 重新分配后的 Taiji-GPU（同样 H20 × 8、2 块 NVMe）；模型从本地拷贝加载
  （`MODEL=/root/models/Qwen/Qwen3-30B-A3B-Instruct-2507`，同一份权重）
- SSD 命中写回 DRAM 是默认开的（上面几组都不写回）。`[4]` 读的 value DRAM 里通常还在，写回被 master 拒绝，每个
  value 在 `mooncake_master.log` 里打一行 `object_already_exists`，一段日志约 1.5 GB，本地存 `.gz`

```bash
python3 make_trace.py swe_reqprio --stop 5252        # -> raw/swe_reqprio/part{0,1}/（各约 6.1 TB KV）
python3 make_trace.py tracelab_reqprio --stop 2267   # -> raw/tracelab_reqprio/part{0,1}/（各约 5.3 TB KV）
bash /root/kvpilot/paper-figure/Figure3-request-safe/raw/run_one.sh swe_reqprio   # GPU 上
```

| | 行数（prefix>0） | HBM safe | DRAM safe | SSD safe | 与 `gen_trace.py` 估计逐行一致 |
|---|---|---|---|---|---|
| SWE 实测 | 5198 | 14.4% | 74.1% | 11.4% | 87.1% |
| SWE 估计 | 5198 | 11.6% | 75.8% | 12.6% | |

SWE：part0 3062 行 83 分钟、part1 2136 行 99 分钟，全部通过来源核对、无重测；两段分开看 part0 15.6% / 71.8% / 12.6%，
part1 12.7% / 77.5% / 9.7%。

## 时间模型拟合

**现在用的是 `fit_times2.py`**（2026-11-22），模型在 `vllm-dev/vllm/v1/core/kvpilot_time_model.py`（在线预测、
blockwise 放置、这里的拟合共用），系数 `fit_times2_swe.json`：

```bash
python3 collect_points.py swe swe_reqprio swe_hbm1to2 --out points.json   # 每条请求：三层 e2e + 加载 / 首层时间
python3 fit_times2.py --points points.json --out fit_times2_swe.json      # 不给 --points 就现场读 raw/
```

和下面的旧模型相比：计算按 cudagraph 补齐和 chunk 切分算、多一个正比于上下文的 attention 项；加载和首层直接用
loader 日志里的 `load=` / `first=` 拟合（DRAM 只用 suffix ≤ 256 的点，计算重时拷贝会被拉长）；加载和计算的组合
从 max 换成平滑过渡，四个参数在 [2] / [4] 的 e2e 上拟合。公式和验证结果见
[docs/kv-placement.md](../../docs/kv-placement.md)「exclusive-penalty-blockwise-unblocking」。suffix 超过一个
chunk 的点不用：这批日志里它们的 e2e 少了最后一个 chunk（见 [docs/observability.md](../../docs/observability.md)）。

以下是旧模型（`fit_times.py`，`plot_boundary.py` 和几个离线估计脚本还在用）：

`python3 fit_times.py --datasets swe --out fit_times_swe.json` 从这批 e2e 拟合 (prefix, suffix) → (t_hbm,
t_dram, t_ssd)，是 penalty placement 用的时间模型，A/B 时经 `KVPILOT_TIME_COEFFS_PATH` 传给 P / D。
`utils/placement_ab/dram_miss_sim.py` 估 e2e 也用它。

`python3 plot_boundary.py` 现场在 `raw/swe/` 上重新拟合（并打印是否与 `fit_times_swe.json` 一致），画出
`model_safe_boundary.pdf`。边界 = 每个 prefix 下，预测的 t_hbm 把该介质的加载盖到 5% 以内的最短 suffix
（`fit_times.bounds`，SSD 边界不低于 DRAM 边界）；prefix 画到 72K，不超出 `raw/swe/` 最大的 prefix（74240）。
