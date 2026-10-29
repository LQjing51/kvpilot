# 易错点

按「读到哪个数会被误导」组织。每条都实测踩过。

## 判据与成本模型

**`append = 0` 要走独立分支**。会话终止时没有下一轮，代入公式得 `T_comp = 0 < T_load`，
误判为「必须留 HBM」，与事实相反。这类块应判 dram-safe，实际上还应最优先淘汰——那属于
reuse-prob = 0 的维度，不由 overlap-ratio 表达。

**别用 `c/a` 当阈值**。那是 `prefix → ∞` 的渐近值。H20 上 `b/a ≈ 6,943`，而实测 prefix
p50 只有 16,162，`b` 项不能忽略。精确临界 append 要解完整二次方程
（`append_threshold()`）——prefix=1,000 时精确值 **74**，而 `c/a` 给 **587**，差 8 倍。

**算力常数用稠密值**。datasheet 上的大数字通常带 "with sparsity" 脚注，而 LLM prefill
是稠密 GEMM 用不上 2:4 sparsity。官方稠密值（每卡）：H200 BF16 989、A100 BF16 312、
H20 BF16 148。历史上 `plot_context_time.py` 曾用稀疏值，且 H20 字段一度误填成 FP8 的
296，现已统一为稠密。

**判据与 TP 无关**。TP 同比例切分每卡算力与每卡 KV 字节，`a`/`b`/`c` 同步缩放，不等式
两边约掉。常数用整机聚合口径即可，不要引入 TP 参数。

**lookahead 已内建在成本模型里**。`dram_safe_times(prefix_j, append_j)` 就是请求 j+1
在请求 j 的 block 上的体验，再去链到 j+1 自己的时间就是把 lookahead 平移了两次。

## 读日志

**`total=NNNN` 是空闲块数，不是池子容量**。池子 2465 块时 `total=2464` 意味着全池
99.96% 空闲、**一次淘汰都没发生**。这个最像「出问题了」的大数字其实代表零压力。要看的
是 `blank` 的**最低值**，它归零才说明淘汰真的开始。

**没有 eviction 就没有 DRAM 加载**。实测两次跑（`test-logs`、`unlimit-logs`）
`grep -c evict` 都是 0，`GPU KV cache usage` 常年 0.0%，于是 `need to load: 0` 占
98.3%——不是加载路径有问题，是它压根没被触发。

**LMCache 的 hit tokens 是嵌套口径**：`hit` 包含了已在 HBM 的部分，命中率高不等于有
加载。要算真正从 DRAM 来的量用 `need to load`，不要用 `hit`。

**`Prefix cache hit rate: 0.0%` 全程不动**说明 `DISABLE_LOCAL_PREFIX_CACHE_LOOKUP`
开着，不是缓存失效。

## 跨进程/跨机

**非交互 ssh 会静默跑成 stock vLLM**。`PYTHONPATH` 只在 `/root/.bashrc` 里，
`.bashrc` 不对非交互 shell 生效。自查：
`grep -c 'Free blocks distribution' prefill-0.log`，0 就是没生效。详见
[benchmark.md](benchmark.md)。

**`setup <name>` 写到 `<name>-logs/`**，会自动加后缀。

**`scp` 到 Taiji-GPU 不通**（`Connection closed`），用 `base64 | ssh ... base64 -d`。

**本机没有 matplotlib**，作图要在 GPU 机器上跑。

**本机 Python 是 3.9**，而两个 benchmark 脚本需要 3.10+（用了 `int | float` 注解）。

**不要在 bash 循环里 `pgrep -f benchmark_serving_multi_turn`**：循环自己的命令行就包含
这个模式，永远等不完。

## nsys

**session 会卡在 `Collection` 状态不落盘**，`clean` 的 SIGTERM 不一定送得到。要
`nsys sessions list` 后逐个 `nsys stop --session=...`，否则 `.nsys-rep` 一个字节都没有。

**TTFT 分析的 `base_ns` 会被陈旧 profile 带偏**：换了跑之后如果没重新 export sqlite，
会读到上一次的时间基准，症状是 `load_total` 全 0。

**nsys 按 worker 全局摊会导致 hit=0 的请求也有 exposed load**：per-request 归因必须用
该请求**自己**的 load 区间（`loads_by_rid`），用 worker 全局集合会把每个并发请求的 load
重复计数（实测 8 路并发下 10 倍超计）。

## 数据处理

**Qwen3 模板会静默吞掉 list 形态的 `content`**，替换成空串，实测丢 64% 的 token 且不
报错。预处理必须先拍平成字符串。

**两个 trace 的 tool label 命名空间零重叠**，查表用错会 100% 落 `_default`、判据退化成
常数，日志里看不出来。所以 `KVPILOT_TOOL_RESP_LEN_PATH` 是 fail-closed 的。

**查表 JSON 的值未必是中位数**，取的是 `--predict-percentile`，所以键名叫 `resp_len`
不叫 `p50`，文件里还记了 `_percentile`——不同分位点跑出的表长得一模一样，不记就会拿去
互相比较。

**HBM 快照的 join key 是源文件行号**，建表和压测必须用同一个输入文件，否则全 miss
（脚本会报 `unmatched`，但不会报错退出）。

**换模型必须重新确认 chat template 的前缀性质**：有些模板的 `loop.last` 分支会让末条
消息渲染方式不同，破坏前缀复用（实测某些模板下共享前缀从 90% 掉到 0.4%）。

## 时间与统计口径

**`TZ` 必须是 `UTC0` 不能是 `UTC`**，见 [benchmark.md](benchmark.md)。方括号日志
是 CST 不是 UTC。

**per-request 的列不能相加**。N 路并发时同一个 kernel 落进 N 个 TTFT 窗口被重复计费，
求和后 compute 随并发增长而 load 不增长，这个不对称会把 `exposed_load/ttft` 压到远低于
成本模型的真值。要聚合就用 `ttft_share_aggregate` 那套并集口径。

**load 占比是陷阱，要看绝对值**：TTFT 膨胀 + compute 全局重复计数会压低占比。实测 `c`
比模型预测慢约 600 倍。

**排队占比两张图差 100 倍是设计使然**，不是 bug：一个分母是自己的 TTFT，一个是 GPU
墙上时间的并集，且后者只计 GPU 真正空转的部分。详见 [observability.md](observability.md)。

**aggregate 那几列里 `load_tot` 和 `store_tot` 不属于互斥划分**，它们与 compute 相交，
加起来超 100% 是正常的。

## 作图

**不要把恒为 0 的那组和有分布的那组画进同一个直方图**：前者堆成一根柱子，实测高度是
后者最高柱的 9 倍，把真正的分布压平。写成图例里的一个计数即可。

**过原点的关系要用约束拟合**：零输入必然零输出时，不受约束的 `polyfit` 斜率拿去画过
原点的线，线和标注的数字对不上。
