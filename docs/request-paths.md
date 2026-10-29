# 三条打请求的路线

模板归属由**端点**决定，选错会导致重复套模板或数据被静默丢弃。

| 脚本 | 端点 | 发什么 | chat template |
|---|---|---|---|
| `benchmarks/multi_turn/mooncake_trace_replay.py` | `/v1/completions` | token id 数组（由 hash_ids 生成的假 token） | 不套 |
| `benchmarks/multi_turn/benchmark_serving_multi_turn.py` | `/v1/chat/completions` | `messages` 数组 | **服务端套** |
| `benchmarks/multi_turn/replay_tracelab.py` | `/v1/completions` | 预渲染 token | 调用方自己套 |
| `vllm bench serve --dataset-name custom` | `/v1/completions` | 预渲染字符串 | 调用方自己套 |

源码依据：`vllm/entrypoints/openai/completion/serving.py` 里 `chat_template` 出现
0 次，`chat_completion/serving.py` 里 23 次。

## Dynamo 的两个特殊约束

vLLM 原生没有这两条：

- **请求体 strict 校验**，未知字段直接 400。`benchmark_serving_multi_turn.py` 里
  `conversation_id` 已改为不发送（`:461-478`）。
- **request-id header 只认能解析为 UUID 的值**，且只读 `request-id` /
  `x-dynamo-request-id`。两个 replay 脚本都把**源文件行号**编码进 UUID 前 8 位，便于
  把服务端 block-pool 日志映射回具体会话和轮次。

那个行号就是各种观测数据的 join key，见 [observability.md](observability.md)。

## 输出长度怎么钉死

`benchmark_serving_multi_turn.py` 默认 `--limit-min/max-tokens 0`
（= `NUM_TOKENS_FROM_DATASET`），逐轮取 trace 里那一轮 assistant 回复的**真实 token
数**，且 `min == max` 精确钉死。

`--forced-output` 更进一步：直接钉死输出内容本身，走
`vllm/v1/sample/logits_processor/forced_tokens.py`。在线判据需要 D 侧 decode 出明文来
解析 tool call，所以在线模式下这个开关是必需的——否则模型自由生成，解析不到预期的
tool call 结构。

## chat template 因模型而异

不同模型家族的特殊 token 完全不重叠（`<|im_start|>` vs `<|start_header_id|>` vs
`<|turn>`）。换模型后必须重新确认「按轮截断出的 prompt 是否仍互为字节前缀」——有些
模板的 `loop.last` 分支会让末条消息渲染方式不同，从而破坏前缀复用（**实测某些模板下
共享前缀从 90% 掉到 0.4%**）。

Qwen3 的模板写成 `loop.last or (下一条不是 tool)`，两个分支等价，是安全的。

## TraceLab 回放的 token 映射

TraceLab trace 只有长度没有明文，回放时要自己造 token。**不能只按位置决定 token**：
agent 会压缩上下文，同一位置在不同轮里内容不同，naive 映射会让本该复用的前缀对不上。

正确做法是 token 由 **(块号, 谁写的)** 共同决定。细节见飞书
`TZ9vdJ8KLo49OVx7cDoc6ClinKc` 的「回放：如何保证 KV 复用」一节。

## 本机 Python 版本

`benchmark_serving_multi_turn.py` 与 `bench_dataset.py` 用了 `int | float` 注解，需要
**Python 3.10+**。macOS 自带的是 3.9。
