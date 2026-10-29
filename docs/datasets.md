# Agent trace 数据处理

在线/离线判据共用的预测器实现见 [kv-placement.md](kv-placement.md)。

## SWE-smith 链路

`datasets/agent-traces/swe-smith-trajectories/`：

```
8 × tool-*.parquet        HF 原始下载（SWE-bench/SWE-smith-trajectories，MIT）
   ↓  合并成 JSONL（messages 字段需 json.loads，parquet 里是字符串）
swe_smith_all.jsonl       24,100 条轨迹
   ↓  prepare_swe_trace.py
swe_trace.json            23,144 条（剔除 956 条异常终止）
   ↓
├─ benchmark_serving_multi_turn.py   打请求（--forced-output 可钉死输出）
└─ validate_premises.py              离线验证两个前提，同时产出查表 JSON 与 ratio 表
```

## 预处理做四件事

`prepare_swe_trace.py`：

1. **统一成内联形态**。数据集混了两种：结构化（`tool_calls` 数组 + `role:"tool"` 消息）
   和内联（`<function=NAME>` 写在 assistant 正文里）。统一成**内联**，因为内联形态本来
   就满足 vLLM 脚本的断言，这样脚本零改动。两者过 Qwen3 模板的 token 差异仅 1.5%。
2. **把 list 形态的 `content` 拍平成字符串**。Qwen3 模板开头是
   `{%- if message.content is string %}...{%- else %}{%- set content = '' %}`，list 会被
   **静默替换成空串**——实测会丢掉 64% 的 token，且不报错。
3. **删掉模板不读的字段**：`agent` / `thought` / `action` / `message_type` /
   `cache_control` / `tool_call_id`。
4. **剔除末条是 `Exit due to ...` 的轨迹**（harness 强制终止，没有 tool call，末条只有
   5-10 token 会让最后一次请求的输出预算失真）。剔除后 assistant 轮的 tool call 覆盖率
   是 **100%**。

## 离线验证

`validate_premises.py` 在全量轨迹上验证两个前提（H20 稠密算力、Qwen3-30B-A3B），
**一次遍历产出三个文件**：

| 文件 | 用途 | 参数 |
|---|---|---|
| `validation.log` | 下面三张表 | — |
| `tool_resp_len.json` | 在线判据的查表 | `--resp-len-output` |
| `req_ratio.json` | HBM 分布图的 y 轴 | `--ratio-output` |

三张表：

- **表 1** tool call 类型 → tool resp 长度分布（p10/p50/p90/mean/p90÷p10），末行给出
  组间方差÷总方差，衡量类型对 append **数值**的解释力。
- **表 2** 按类型代入 p50 预测 dram-safe 的正确率，分列假阳（判 safe 但掩不住，伤
  TTFT）与假阴（判 unsafe 但其实能掩住，只是少赚）。
- **表 3** 对照组：用全体 tool_resp 的单一 p50 预测，不看类型。与表 2 的唯一差别就是
  这个估计量，**两表之差即为「知道类型」的价值**。

**具体数值不在文档里记录**——预测器持续迭代，抄下来的数会立刻过时且与代码矛盾。需要时
直接读 `validation.log`，或重跑脚本。

### 为什么 ratio 表和三张验证表出自同一次遍历

两者的 `collect()` 口径逐行相同（同一个 `stream_records`、同一套 prefix 累加规则、同一
个 `turn_labels`），分开写就是把 prefix 的定义写两遍——哪天「decode 输出算 append」这个
前提变了，改一处漏一处，`validation.log` 的准确率就不再描述 `req_ratio.json` 描述的那个
世界。

全量跑 2.5 GB trace 的代价由 tokenizer 主导，合并后从两遍降到一遍。

## 查表 JSON 随数据集走

`tool_resp_len.json` 由各自的 validator 写在自己目录（`swe-smith-trajectories/` 与
`tracelab/` 各一份），**不在 vLLM 仓库里**。跑服务时用 `KVPILOT_TOOL_RESP_LEN_PATH`
指定。

两个 trace 的 label 命名空间**实测零重叠**（SWE-smith 是 `bash:python` /
`str_replace_editor:view`，TraceLab 是 `Read` / `Bash:cd`），互换会让每次查询都落
`_default`、判据退化成常数**且日志看不出来**。所以那个环境变量是 fail-closed 的：
开在线判据时没设就启动报错。

表里的值取 validator 的 `--predict-percentile`（**未必是中位数**），所以键名是
`resp_len` 而非 `p50`，且文件里用 `_percentile` 记下当次用的分位点——否则不同分位点跑出
的表看起来一模一样，容易拿去互相比较。

**worker 启动时读表，所以换 trace 必须重启服务。** 该变量由 `disagg_lmcache.sh` 导出并
做存在性检查，benchmark 阶段再设已经来不及。

## tokenizer 用本地路径

脚本默认 `Qwen/Qwen3-30B-A3B-Instruct-2507`（走 HF Hub），在集群上应传：

```
--tokenizer /cfs_cloud_code/kiry/model/Qwen/Qwen3-30B-A3B-Instruct-2507
```

既不联网也保证与 vLLM worker 完全一致。

## TraceLab

另一条 trace，只有长度没有明文，回放方式见 [request-paths.md](request-paths.md) 的
「TraceLab 回放的 token 映射」。预处理与在线判据接入见飞书
`TZ9vdJ8KLo49OVx7cDoc6ClinKc`。

## 原始数据不入库

两个原始下载太大（GitHub 单文件上限 100MB），已 gitignore，新 clone 后按需重新下载；
预处理产物（`swe_trace.json`、`tracelab_turns.jsonl`、`tool_resp_len.json` 等）在库里。

| 文件 | 来源 |
|---|---|
| `swe-smith-trajectories/swe_smith_all.jsonl`（3.5GB） | HF `SWE-bench/SWE-smith-trajectories` 的 `tool-*.parquet`，合并成 JSONL |
| `tracelab/syfi_coding_trace.jsonl.gz`（96MB） | `https://github.com/uw-syfi/TraceLab/releases/download/v0.0.2/syfi_coding_trace.jsonl.gz` |
