# Agent trace 数据处理

`datasets/agent-traces/` 下两条 agent trace，各占一个目录，**每个目录只放这几样东西**：
原始数据、一个预处理脚本、脚本产出的 Mooncake 格式 trace，以及它的跨 session 重排版
（`utils/trace_order/make_order.py` 产出，见下面「发送顺序」）。两条都另有全量版，SWE 还有按
session 挑出的子集，都是各自预处理脚本的产物。

| 目录 | 原始数据（gitignore） | 预处理脚本 | 产物 | 重排版 |
|---|---|---|---|---|
| `tracelab/` | `syfi_coding_trace.jsonl.gz` | `prepare_tracelab_trace.py` | `tracelab_mooncake.jsonl`；`tracelab_all.jsonl`（全量，gitignore） | `tracelab_mooncake_interleaved.jsonl`、`tracelab_mooncake_poisson.jsonl` |
| `swe-smith-trajectories/` | `swe_smith_all.jsonl` | `prepare_swe_trace.py` | `swe_full.jsonl`（全量，gitignore）。`swe_mooncake.jsonl`、`swe_hbm1to2.jsonl`（HBM : DRAM safe ≈ 1:2 的子集）、`swe_hbm1to2_ssdmax.jsonl`（1:2 且 SSD safe 尽量多，303 GiB）2026-11-22 本地已删，按下文命令可重新生成；GPU 上仍有 | `swe_mooncake_interleaved.jsonl`、`swe_hbm1to2_budget220.jsonl` 本地同样已删（`utils/trace_order` 可重新生成）；`swe_mooncake_poisson.jsonl`；`swe_hbm1to2_budget260.jsonl`（子集按 KV 预算并发，模拟 SSD 10.3% / 15.5%） |

在各自目录下直接 `python3 prepare_*.py`，参数都有默认值。

## Mooncake 格式

一行一个请求：

| 字段 | 含义 |
|---|---|
| `timestamp` | 到达时间（ms）。TraceLab 是相对文件第一条的偏移；SWE-smith 没有时间，写 `null` |
| `session_id` / `round_index` | 会话与轮次，轮次沿用原始编号（TraceLab 删掉的请求会留下空号） |
| `input_length` | 整个 prompt 的 token 数 |
| `prefix_length` | 能从缓存复用的 token 数 |
| `output_length` | 本轮生成的 token 数 |
| `hash_ids` | 见下 |
| `tool_calls` | **本轮输出**发起的调用类型，按顺序；没有调用就是 `[]` |

`timestamp` / `input_length` / `output_length` / `hash_ids` 沿用 Mooncake 原 trace 的字段名，
`mooncake_trace_replay.py` 可以直接读。

**hash_ids 语义**：block 512 token（`mooncake_trace_replay.py` 把每个 id 展开成 512 个
token），`hash_ids[i]` 代表第 0..i 块的整段内容，即前缀树上的一个节点，末块可以不满。
两个请求前 k 个 id 相同 ⇔ 前 k×512 token 相同，和 vLLM 链式 block hash 同一语义。id 按
在文件里首次出现的顺序从 0 编号。

两个脚本最后都做同一个**自校验**：按文件顺序用无限大的缓存回放，每条请求命中的块数必须
等于 `floor(prefix_length / 512)`，有一条不符就以非零码退出。

两个脚本都会在 session 里第一次 `input + output > --max-model-len` 的那一轮截断，这一轮
和之后的全部丢掉。默认 262,144，即服务端未显式设置时的推导值。不能只跳过超长那一轮：
它的下一轮 prefix 会变冷，读起来像 placement 出了问题。

## TraceLab

Claude Code / Codex 的真实会话，只有长度没有明文，一行一个 agent round。

**筛选**，每条都整个 session 丢弃：

- **只保留 claude**：codex 的 prefix 按 128 量化，描述的是它的分桶而不是负载。
- **round 0 就有 `prefix > 0` 的丢掉**：这部分命中来自别的 session。
- **input 没有严格递增的丢掉**：中途压缩或重置过上下文。

去掉压缩之后 session 会一路涨，长度上限截断不是个别现象。

**删掉 `output_tokens = 0` 的请求**：回放会把它变成 `max_tokens = 0`，服务端直接拒绝。只删
这一条，不丢整个 session；它的下一轮接在再上一轮后面，prefix 和 hash_ids 都从那一轮继承。

**转换**：`prefix` 取上一条保留下来的请求的 input，session 第一条为 0。比上一轮 input 多，说明
Anthropic 连上一轮 decode 输出也缓存了，而我们的 decode 实例不存 KV；比它少，是缓存断点
或过期，prefix cache 不会因此 miss。

**hash_ids**：第 N 轮沿用第 N-1 轮的前 `input[N-1] // 512` 个 id，其余位置发新 id，包括
上一轮末尾不满那块的位置（第 N 轮把它填上了不同的内容）。筛选之后不存在跨 session 命中，
id 不跨 session 共享。

**timestamp**：取 `timing_events` 第一条的时间（round 0 是 `user_message`，之后是
`tool_result`），再减去文件第一条请求的。间隔保持原样；整个文件横跨数千小时，按真实时间
回放不可行，要配 `--time-scale`，或之后重新生成时间轴。

**tool_calls**：工具名，Bash 按开头的可执行文件细分成 `Bash:cd` 之类。尖括号换成圆括号
（`<assign>` → `(assign)`）：label 会被渲染进 `<function=NAME>` 再用
`<function=([^>\s]+)>` 读回，`>` 会把它截断。

默认只写按时间最早的 2000 条（`--num-requests`，0 = 全部）。筛选和截断在全量上做完再取前
2000 条，所以落在边界上的 session 只保留前几轮，前缀关系不受影响。

**`tracelab_all.jsonl` 是 `--num-requests 0`（全量）**：读入 665,453 个 round / 5,319 个 claude
session，剔除 round 0 prefix>0 的 3,120 个、input 未严格递增的 190 个，63 个 session 因超长截掉尾部
7,178 轮，写出 47,270 条请求 / 2,009 个 session / 13,235 GiB（不同整块），时间跨度 7,035 小时。前
2000 行与 `tracelab_mooncake.jsonl` 逐字节相同。约 11 s，74 MB，gitignore。

## SWE-smith

HF `SWE-bench/SWE-smith-trajectories` 的 8 个 `tool-*.parquet` 合并成的 JSONL，一行一条
轨迹，有明文。

**规整**：每条轨迹先变成 user / assistant 严格交替、以 user 开头、以 assistant 结尾。

1. **统一成内联形态**。数据集混了两种：结构化（`tool_calls` 数组 + `role:"tool"` 消息）
   和内联（`<function=NAME>` 写在 assistant 正文里）。统一成内联，因为
   `kvpilot_signal.turn_labels`（在线判据）解析的是内联形态。两者过 Qwen3 模板的 token 差异仅 1.5%。
2. **把 list 形态的 `content` 拍平成字符串**。Qwen3 模板开头是
   `{%- if message.content is string %}...{%- else %}{%- set content = '' %}`，list 会被
   **静默替换成空串**——实测会丢掉 64% 的 token，且不报错。
3. `tool` 消息改成 `user`，开头的 system 并进第一条 user，相邻的同角色消息合并。
4. **剔除末条是 `Exit due to ...` 的轨迹**（harness 强制终止，没有 tool call，末条只有
   5-10 token，会让最后一次请求的输出预算失真）。

**每轮一个请求**：第 k 轮的 prompt 是 `messages[:2k+1]` 过 Qwen3 chat template
（`add_generation_prompt=True`）得到的 token，输出长度是第 2k+1 条 assistant 消息的 token
数。`hash_ids` 按真实 token 切块插进前缀树，跨轨迹共享有就自然出现；`prefix_length` 是和
所有更早 prompt 的最长公共 token 前缀，从同一棵树上读出（匹配上的整块，加第一个没匹配上的块
与其兄弟节点的最长公共前缀），不再逐个和更早的 prompt 比。

按文件顺序取。默认写满 2000 条请求为止（`--num-requests`，0 = 全部）；给 `--min-unique-kv-gib`
时改成按整条轨迹写，直到不同的整块合计 KV 达到这么多 GiB（`--kv-bytes-per-token` 默认 98,304，
Qwen3-30B-A3B BF16）；`--sessions FILE` 只处理列出的 traj_id，其余当作不存在。tokenize 由
`--workers` 个进程并行（默认核数 − 1），主进程按文件顺序建树。

- **库里的 `swe_mooncake.jsonl` 是 `--min-unique-kv-gib 400` 的产物**：5,140 条请求 / 198 条轨迹 /
  400.1 GiB，前 2000 行与之前的 2000 条版本逐字节相同（2026-11-06 改，为了让工作集超过 1P 的
  Mooncake DRAM，见下面「发送顺序」）。原始文件开头全是结构化轨迹，所以这些行里没有原本是内联
  形态的。2026-11-09 换成前缀树算 `prefix_length` 后重新生成，与库里的逐字节相同，约 20 s。
- **`swe_full.jsonl` 是 `--num-requests 0`（全量）**：701,515 条请求 / 23,144 条轨迹（剔除强制终止
  的 956 条，无截断）/ 55,735 GiB，round 0 跨轨迹命中 ≥ 1 块的 16,636 条（同一 instance 有多条
  轨迹）。10 核 Mac 约 74 min，397 MB，gitignore。
- **`swe_hbm1to2.jsonl` 是 `--sessions` 挑出的 368 条轨迹**：6,026 条请求 / 420 GiB，按
  request-safe 实测的经验概率估计 HBM safe : DRAM safe ≈ 1:2（全量是 1:5.5），挑法、校验和这批
  session 偏轻的程度见 [utils/trace_select/README.md](../utils/trace_select/README.md)。
- **`swe_hbm1to2_ssdmax.jsonl` 是 `--sessions` 挑出的 156 条轨迹**：2,136 条请求 / 302.6 GiB，期望 HBM 21.1% /
  DRAM 43.8% / SSD 35.1%（working set > 300 GiB 时 SSD 能到的最高），见同一份 README。

### tokenizer 用本地路径

默认 `Qwen/Qwen3-30B-A3B-Instruct-2507`（走 HF Hub），在集群上应传：

```
--tokenizer /cfs_cloud_code/kiry/model/Qwen/Qwen3-30B-A3B-Instruct-2507
```

既不联网，也保证和 vLLM worker 完全一致。

## 回放

`vllm-dev/benchmarks/multi_turn/mooncake_trace_replay.py -i <trace>`：按文件顺序、按时间戳发；
带 `session_id` 的行还要等同 session 上一轮**返回之后**才发，等的时候后面的行也不发（总是这样，
没有开关）。已知限制：SWE-smith 的 `timestamp` 是 `null`，`iter_trace` 里的 `int(None)` 会抛
TypeError 且没被捕获，**目前不能直接回放**。

两个和 KVPilot 实验相关的开关：

- `--forced-output`：把每行的 `tool_calls` 渲染成 `<function=NAME></function>` 放在输出开头，
  其余用一个 ASCII token 填到 `output_length`，经 `forced_output_tokens` 钉死 decode 输出，
  D 侧在线判据才认得出 tool call（不开时输出是随机文本，每轮都判成「没有调用」）。
  `--max-output-tokens N` 可再把输出截短。
- `--release-at-first-token`：client 槽在拿到首 token 时就放出，而不是等 decode 结束。配
  `-p 1` 时 P 上同一时刻只有一个 prefill，TTFT 里没有排队，decode 在 D 上与后面的 prefill
  重叠；同 session 的下一轮仍等上一轮**完整**返回。严格 `-p 1`（decode 也串行）
  下 SWE 光 decode 就要约 1.5 h。

`replay_tracelab.py` 也读 `tracelab_mooncake.jsonl`，但忽略 `hash_ids`，自己按 session 造
token（见 [request-paths.md](request-paths.md)「TraceLab 回放的 token 映射」），session 闭环
发请求，同样有 `--forced-output`。

### 发送顺序：让复用间隔够长

按文件顺序发，两条 trace 的复用几乎全在 HBM（离线模拟：SWE 92.4% HBM、TraceLab 87.3% HBM，
DRAM / SSD 都是 0）：同一 session 的相邻轮挨得太近。`*_mooncake_interleaved.jsonl` 是重排成
跨 session 轮转（session 内保序）的版本，所有 session 同时在最后一个周期结束；生成它的
`utils/trace_order/make_order.py`、离线三层命中模型和 1P1D 实测结果见那里的 README（实测
SWE DRAM 45.5% / SSD 27.5%，TraceLab DRAM 21.8% / SSD 53.4%）。

`*_mooncake_poisson.jsonl` 带真实时间戳：session 起点是泊松过程，session 内相邻两轮的间隔
服从指数分布（`make_order.py --align poisson`，seed 0）。顺序只取决于「到达率 × 间隔均值」，
两者同比例变化只是拉伸时间轴。都未实测：

| 文件 | 到达率 / 间隔均值 | 时间跨度（90% 请求在） | 模拟 HBM / DRAM / SSD |
|---|---|---|---|
| `swe_mooncake_poisson.jsonl` | 0.5/s / 60 s | 109 min（38 min） | 68.0% / 18.6% / 5.9% |
| `tracelab_mooncake_poisson.jsonl` | 0.2/s / 120 s | 128 min（51 min） | 44.4% / 18.6% / 24.4% |

SWE 的 SSD 到不了 10%：session 一起开始、结束时间相差很大，工作集峰值只有 DRAM 的 1.25 倍；
TraceLab 唯一 KV 是 DRAM 的 4.5 倍，所以够。到达率按各自实测吞吐（SWE 4.2、TraceLab
1.2 req/s）选，使峰值请求率能按时间戳原速回放。

- 重排后的 trace 时间戳全是 0，节奏只由 `-p` 和「同 session 等上一轮返回」决定。
- 请求粒度优先级的 A/B 另有专门构造的 trace：从 `swe_full.jsonl` / `tracelab_all.jsonl` 挑收益率高的 session，
  按活跃 KV 预算轮转、LRU 臂回放 30 分钟处截断，见 [`datasets/req-priority-trace/README.md`](../datasets/req-priority-trace/README.md)。
- **工作集必须大于 DRAM 才可能有 SSD 命中**，与顺序无关：DRAM 不发生淘汰就没有块只剩 SSD。
  1P × TP2、每 rank 100 GiB 时 DRAM 共 200 GiB，所以两条 trace 都要求不同的整块 ≥ 400 GiB
  （TraceLab 2000 条已有 899 GiB；SWE 改成按 400 GiB 生成，见上）。

## 已移除的离线验证与查表

整理成上面的结构时，下面这些从数据集目录删掉了，需要时从删除前的提交恢复：

| 删掉的 | 原来做什么 |
|---|---|
| `validate_premises.py` / `validate_tracelab.py` | 离线验证 tool call 类型能否预测 dram-safe，并写在线判据的查表 `tool_resp_len.json` |
| `tool_resp_len.json`（两份） | 旧在线判据的查表（开关 `KVPILOT_TOOL_RESP_LEN_PATH`，已删）；现在的在线判据用 `paper-figure/algorithm/resp_length_dist_<dataset>.json`，见 [kv-placement.md](kv-placement.md) |
| `req_ratio.json` | HBM 分布图的 y 轴（`utils/plot_hbm_pool_distribution.py --ratios`） |
| `build_measured_safe.py` / `score_measured.py` / `fit_constants.py` / `ceiling.py` | 实测 dram-safe 与理论判据对账 |
| `swe_trace.json` / `multiturn_schedule.jsonl` | `benchmark_serving_multi_turn.py` 的 SWE 输入与调度记录 |
| `tracelab_turns.jsonl` / `tracelab_schedule.jsonl` | TraceLab 旧的中间格式与 `replay_tracelab.py` 的调度记录 |

**开在线判据前得先把查表重新生成出来。** 两个 validator 读的是旧的中间格式，恢复后要改成
读 `*_mooncake.jsonl`。

## 原始数据不入库

两个原始下载太大（GitHub 单文件上限 100MB），已 gitignore，新 clone 后按需重新下载；
预处理脚本和产物在库里。

| 文件 | 来源 |
|---|---|
| `swe-smith-trajectories/swe_smith_all.jsonl`（3.5GB） | HF `SWE-bench/SWE-smith-trajectories` 的 `tool-*.parquet`，合并成 JSONL（`messages` 字段需 `json.loads`，parquet 里是字符串） |
| `tracelab/syfi_coding_trace.jsonl.gz`（96MB） | `https://github.com/uw-syfi/TraceLab/releases/download/v0.0.2/syfi_coding_trace.jsonl.gz` |
