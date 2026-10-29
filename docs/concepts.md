# KVPilot 项目概念

这是最顶层的「为什么」文档。具体代码位置见 [kv-placement.md](kv-placement.md)，
跑起来见 [benchmark.md](benchmark.md)。

## 问题背景

Multi-turn agentic 场景下（长 context、短 append），做 prefix KV cache reuse 能省掉
大量 prefill 计算，但**把瓶颈从计算搬到了 KV cache loading**。

- Chatbot / agent 多轮对话与多轮 tool 调用很常见，请求内前缀重复率极高（理论复用率
  >95%）。
- 现有推理系统用分布式 memory pool 尽量多存 KV，并用 layerwise load/store 让读写与
  计算 overlap：算 layer i 时后台存 layer i-1、拉 layer i+1。要真 overlap 就要求
  **prefill 计算时间 ≥ prefix KV 加载时间**。
- 传统认知是 prefill 计算 bound、decode 显存带宽 bound。但高命中率下 **prefill 多了
  一个新瓶颈：跨机 KV loading 带宽**。计算量因复用而大幅减少，而跨机只有最多
  400 GB/s RNIC 带宽，于是 loading 时间反超计算时间。

现有缓解手段有两个，都是围绕「尽量从本地加载」：

1. **KV 亲和性调度**：调度请求时不只看节点负载，也看节点上已有的 KV，优先调度到存有
   其最长 prefix 的节点。
2. **热 KV 复制**：亲和性调度不可能 100% 命中，所以把热点 KV 复制到多个节点 / 更高层
   存储（HBM）。

## 核心论点

这两个手段都按**复用概率**决定放置，而复用概率没有回答「这个请求的 KV 加载，到底会不
会真的拖慢 GPU」。于是：

- **短请求**（prefill 计算量小）的瓶颈本来就在计算，把它的 KV 放进 HBM 或复制多份
  **没有性能收益，只浪费稀缺存储**。
- **长请求**的瓶颈才在 loading，这些才值得占 HBM。
- 而 multi-turn 下所有 prompt 都是**由短到长**增长的，短 prompt 的空间占比不可忽视。

一句话：**按 loading-compute ratio 而不是按 reuse probability 放置 KV。**

## Insight 1：按 loading-compute ratio 决定放置位置

```
T_comp = a · append · (prefix + append/2) + b · append
T_load = c · prefix
```

`T_load < T_comp` 的请求叫 **dram-safe**：其 KV 从 DRAM 加载能被计算完全掩盖，不必
占用 HBM。反之必须留在 HBM，否则 GPU 空等。

同一个判据还有一个分布式推论：`T_rnic-load < T_comp` 的请求，其 KV 可以在分布式 DRAM
池里**只存一个副本**——既然加载无论调度到哪个节点都能被掩盖，就不需要为它做冗余。

推导与各模型常数见飞书文档「KV Cache 加载与 Prefill 计算时间常数推导」
（`QC2Vdtmcqo2Cpzxt1Khc2j9inab`），代码实现是 `utils/plot_context_time.py`。

三个常数的含义：

| 常数 | 含义 | 依赖 |
|---|---|---|
| `a` | attention 的二次项系数 | 算力（稠密 BF16 FLOPS） |
| `b` | 每 token 的线性项（FFN 等） | 算力 |
| `c` | 每 token prefix 的加载耗时 | KV 字节数 / 加载带宽 |

## Insight 2：按排队行为自适应预取

队列里存在无需加载的请求时，优先调度它们并在后台加载；否则对队首请求走 layerwise
loading。

## Challenge 1：下一轮的 append 长度如何提前得知

判据要用 `append`，但放置决策发生在这一轮结束、下一轮到达之前。

agentic 场景下 append 主要是 tool 结果，长度可由 **tool call 类型**预测（`Read` 长、
`bash ls` 短），而 tool call 在**本轮 decode 的输出里就能看到**。所以 D 侧在 turn 结束
时解析出 tool 类型、查表估 append，再把判据回传给 P 侧。

离线验证这个前提成不成立，见 [datasets.md](datasets.md) 的「离线验证」。

## Challenge 2：reuse-probability 与 overlap-ratio 如何权衡

不用带权重的加权和（`reuse_prob + w·(1-overlap)` 引入难调参数），而是**两级过滤**：

- `overlap_ratio = 1` → 直接放 DRAM
- `< 1` → 才在子集内按 reuse-prob 排序

## Challenge 3：判据与调度、存储层级的耦合

判据算出来之后要落到三个地方：HBM 内的淘汰顺序、本地 DRAM 与远端 DRAM 的选择、以及
副本数。目前实现覆盖第一个，后两个见飞书「管理 local DRAM 和 remote DRAM 的
placement」章节。

## 关键定义

**append 的定义**：`decode_len + tool_resp_len`。因为 decode 实例没接
LMCache/Mooncake，其输出的 KV 不会被下一轮 prefill 复用，所以 decode 输出算 append 而
非 prefix。`prefix` 只累加 prompt 历史。

**append = 0 要走独立分支**：会话终止时没有下一轮，代入公式会得 `T_comp = 0 < T_load`
而误判为「必须留 HBM」，与事实相反。这类块应判 dram-safe（放 DRAM 无损），实际上还应
最优先淘汰——那属于 reuse-prob = 0 的维度，不由 overlap-ratio 表达。

**判据与 TP 无关**：TP 同比例切分每卡算力与每卡 KV 字节，`a`/`b`/`c` 同步缩放，不等式
两边约掉。所以常数用整机聚合口径即可（与推导文档一致），不要引入 TP 参数。

更多容易搞错的地方见 [pitfalls.md](pitfalls.md)。

## 当前部署形态

H20 × 8（96 GB HBM/卡）、Qwen3-30B-A3B-Instruct-2507 bf16、TP=2、`--block-size 512`、
`gpu-memory-utilization 0.95`。`--max-model-len` 未显式设置，走默认推导 = 262,144
（取 `max_position_embeddings` 与 tokenizer `model_max_length` 的较小值）。

单卡 KV 开销：48 层 × 4 kv-head × 128 head-dim × 2（K/V）× 2 B = **96 KiB/token**
整机口径，TP=2 下每卡 48 KiB/token。block-size 512 → 24 MiB/block/卡。

**decode 侧未接 LMCache/Mooncake**，所以 decode 产生的 KV 不会被下一轮 prefill
复用——这一点在 append 长度的定义里很关键（见上）。
