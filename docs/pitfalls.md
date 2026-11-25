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

**`Prefix cache hit rate: 0.0%` 全程不动**说明请求的 `cache_use_control` 不含 1（跳过本地
prefix 查找），不是缓存失效。

## 跨进程/跨机

**非交互 ssh 会静默跑成 stock vLLM**。`PYTHONPATH` 只在 `/root/.bashrc` 里，
`.bashrc` 不对非交互 shell 生效。自查：
`grep -c 'KVPilot placement=' prefill-0.log`，0 就是没生效。详见
[benchmark.md](benchmark.md)。

**`setup <name>` 写到 `<name>-logs/`**，会自动加后缀。

**Mooncake DRAM 被写满后，同机命中曾整次 lookup 失败**（已改成全映射，见下）。原来每个 TP worker 只
preload 另一个 prefill 引擎上同 rank 的 segment。本 worker 的 segment 分配不下时，Mooncake 会把 KV 放进
任何有空间的 segment（master 不区分 rank），落到别的 rank 的 segment 时 lookup 抛
`Mooncake shared-memory replica was not preloaded or rewritten`，整次按 0 命中，还常伴随
`Timeout occurred for rank N`（这一条请求多出 1–2 s）。64 GiB / TP worker 下写入集中的 trace 会触发
（`paper-figure/toolagent-turn5plus-tier-safe-per-request` 的 dram 轮 336 / 2105 行、
`prefix-suffix-compute-vs-load-dram-ssd` 第 2 遍 35 / 90 条，这两份数据都是改之前跑的），正常节奏的 turn5plus
回放一次都没出现。现在每个 worker preload 本机所有 prefill worker 的 segment（`_preload_shared_memory_peers`），
日志 `shared_memory_preload_ready ... peers=3`（2P × TP2）。刚写入就被淘汰、写入未完成时来读
（`REPLICA_IS_NOT_READY`）这两种部分命中仍会发生，测量类实验仍要先 grep 这几条。

**全映射 + 200 GiB segment 会让 prefill OOM**。`cudaHostRegister` 每 GiB host 内存占 2 MiB 显存（实测，
GPU 页表），而 Mooncake 注册自己和 peer 的 segment 是在 vLLM 测完可用显存、分完 KV cache 之后。每个进程注册
4 × 200 GiB = 1.6 GiB 显存，比只映射同 rank 时多 800 MiB，第一次 forward 申请激活就 `CUDA out of memory`，
两个 prefill 引擎随即 `EngineDeadError`，所有请求 500（2026-11-03 全量 toolagent 回放）。64 GiB segment
只占 512 MiB，没有问题；要用 200 GiB 得给显存留出这部分余量。

**单 KV 头的模型 LMCache 会读错 KV 布局，表现为 CUDA 非法访问**（Qwen3.6 TP2，2026-11-23 修）。vLLM 给 HND
提示，KV 张量却是逻辑 NHD 顺序 `(块, 2, 64, 1, 256)`（长度 1 的头那一维不影响 `is_contiguous`，不会被重排），LMCache
按 HND 读成 64 个头、每块 1 token。slot 小于 kernel 块数时不报错、只是 KV 存取到错的位置（输出错）；新分到的块号一大，
`illegal memory access` 在下一个检查错误的地方冒出来：LMCache 存储的 `store_stream.synchronize`、前向结束的同步，
`CUDA_LAUNCH_BLOCKING=1` 下才停在 `LayerwiseLoader.start`。修法见 [kv-placement.md](kv-placement.md)「混合模型」。
查这类问题：纯 vLLM（`LLM(...)`，同样的 block / 前缀缓存 / cudagraph 配置）先跑一遍排除 vLLM，再对 LMCache 的
gather / scatter 打 slot 范围和张量形状。

**LMCache 逐层 connector 的 GPU buffer 不在 vLLM 的显存预算里**（Qwen3.6 TP4，2026-11-23 修）。它在第一次存数据时
懒分配，上游按一层整个 HBM 缓存的 token 数开：TP4 下 691 万 token × 1 KiB = 6.75 GiB。`GPU_MEM_UTIL=0.9` 时 vLLM 已
把 85.5 GiB 分给权重、激活和 KV，加上它 PyTorch 占 91.6 GiB，第 5 个请求申请 48 MiB 激活时 `CUDA out of memory`，
所有请求 500。TP2 下是 5.1 GiB，刚好放得下。现在按 `max_model_len` 开（prefill `--max-num-seqs 1`，一次只存一条
prompt）。日志 `Lazily initializing GPU buffer (max tokens=...)` 看实际大小。

**混合模型复用后的输出可能和冷算不同，不是加载错了**。vLLM 原生 HBM 前缀缓存也会这样（TP4 下 4 个种子里 2 个）：
冷算分几步算到边界，复用从缓存的状态接着算，数值不逐位相同。判断 LMCache 路径对不对，要拿 DRAM / SSD 复用和 HBM
命中比，`check_state_reuse.py` 两者都打印。

**Qwen3.8 第一次编译时可用 KV 显存会被压到几 GiB**（2026-11-23）。torch.compile 第一次编译这个模型时，编译的峰值
（约 35 GiB）落在 vLLM 测显存的 profile run 里，可用 KV 只剩 4.46 GiB（P）/ 2.96 GiB（D，再小于 262K 上下文所需时
起不来）；编译缓存热了以后是 38.3 / 34.5 GiB。改了 vLLM 代码或启动参数（包括关异步调度）都会让缓存失效。A/B 里
显式给 `HBM_BLOCKS`（`--num-gpu-blocks-override`），各组 HBM 不受影响；D 第一次起不来时 `run_ab.sh` 会重试 setup。

**Qwen3.8 不能开异步调度**：V1 runner 的 PLE n-gram 上下文从 CPU 端的 token 读，异步调度下 decode 时它落后一步，
同一 prompt 两次输出不同。`Qwen4ExpForConditionalGenerationConfig` 默认关掉，显式打开直接报错。

**FP8 模型开专家并行要 nvcc 在 PATH 里**（2026-11-25；现在所有模型都开 EP）。Hopper 上 block FP8 的 MoE 在 EP 下
选 FlashInfer CUTLASS（`fused_moe/oracle/fp8.py`：TP 优先 Triton、EP 优先 FI CUTLASS），它的分组 GEMM 由 TensorRT-LLM
第一次用到时用 nvcc 现编（缓存在 `~/.tensorrt_llm/cache`）。镜像的 PATH 里没有 nvcc 时日志是 `sh: 1: nvcc: not found`、
`Failed to copy kernel files to cache ... nvcc_kernel.cubin`，随后 `Assertion failed: !cubin.empty() || isPathValid(path_)`。
11-23 接 Qwen3.8 时把它当成「缺 cubin」、对 Qwen3.8 关了 EP（之前 Qwen3.8 的时间模型和 Eval-EXP1 都是不开 EP 测的）；
现在 `disagg_lmcache.sh` 设 `CUDA_HOME=/usr/local/cuda`、把它的 `bin` 放进 PATH，所有模型 `--enable-expert-parallel`。
同时装了 `flashinfer-cubin==0.6.12`（离线 cubin 包，见 [gpu-restore.md](gpu-restore.md)）。

**慢的是 FlashInfer CUTLASS，不是 EP**（2026-11-25，`utils/moe_bench/`，离线单条 prefill，chunk 8192、CUDA graph 64–1024）。
`--moe-backend triton` 在 EP 下也能用：

| 单条 prefill（ms），256 / 1K / 2K / 4K / 8K token | 不开 EP + Triton | 开 EP + Triton | 开 EP + FI CUTLASS | 不开 EP + FI CUTLASS |
|---|---|---|---|---|
| Qwen3.6-FP8 TP2 | 19.4 / 37.9 / 73.1 / 113.0 / 218.0 | 19.8 / 38.9 / 74.2 / 115.9 / 224.4 | 24.8 / 45.8 / 75.8 / 131.1 / 248.0 | 27.8 / 53.1 / 85.5 / 147.8 / 278.5 |
| Qwen3.8 TP4 | 47.7 / 108.9 / 190.3 / 347.0 / 650.1 | 37.6 / 90.4 / 163.0 / 302.0 / 573.0 | 42.2 / 95.3 / 168.3 / 309.0 / 580.9 | — |

Qwen3.8 不开 EP 时每卡中间维 640/4 = 160 不是 128 的倍数，FP8 块缩放只能细化成 32×32、只有 Triton 能跑，所以必须开 EP；
FI CUTLASS 下它的 MoE GEMM 其实已经是 TensorRT-LLM 里的 `deep_gemm::fp8_gemm_kernel`，不用另装 DeepGEMM。给 EP 的形状
（E=128,N=512 / N=640）用 `benchmarks/kernels/benchmark_moe.py --tune` 调了 Triton 配置表（要 ray，从 CFS 装），按 token 数直接用、
或按每个专家的 token 数错位用（脚本在 EP 下让每个专家分到真机 EP 倍的 token，`shift_tuned.py`），都和默认配置差不到 1%。
MoE 只占 GPU 时间的一部分（torch profiler，8K token：Qwen3.6 27%、Qwen3.8 19%；稠密 FP8 GEMM 25% / 33%），换 kernel 最多
让 Qwen3.6 快 10–20%（小 chunk）、Qwen3.8 快 1–11%。Eval-EXP1 第二轮用的是 FI CUTLASS。Qwen3.8（512 个专家）反过来，开 EP 后 request-safe 拟合的每千 token 计算从 86.6 ms
降到 74.5 ms。开 EP 后第一次启动同样要重新编译，D 端会因为可用 KV 不够起不来（见「Qwen3.8 第一次编译时可用 KV
显存会被压到几 GiB」），热缓存后 KV 回到
38.4 / 35.5 GiB（P / D），和不开 EP 一样。

**很小块的 KV 组会把全局块大小拖下来**：0.23 的 `EngineCore` 把 `cache_config.block_size` 设成所有 KV 组里最小的
块，Qwen3.8 的 indexer 环形缓冲组是 4，于是 mamba 对齐切分、LMCache 都按 4 个 token 一块（LMCache 报
`Num tokens: 3072, num blocks: 4, block size: 4`）。现在只取参与前缀缓存的组（同上游 0.30）。

**mooncake_master 起不来时客户端只报 `Client not available`**：快速重启时上一个 master 还占着 50052，新 master
`bind port 50052 error: Address already in use` 后退出，所有 worker 重试 20 次后 `store.setup() failed`。清理后
确认没有 `mooncake_master` 进程、端口空闲再起。

**worker 不绑 NUMA 时，TP 4 随机几个 rank 的 SSD 加载慢 3–4 倍**（2026-11-24 Eval-EXP1，现在默认绑，
`KVPILOT_NUMA_BIND`）。这台机器 GPU 0–3 和 nvme0 在 NUMA 0、GPU 4–7 和 nvme1 在 NUMA 1，worker 进程原来不绑 CPU。
LMCache SSD loader 的 256 MiB pinned 读缓冲在 worker 初始化时分配，落在当时线程所在的节点；落到另一侧时第 0 层的
SSD 读每块 1.0–1.4 ms（正常 0.2–0.4 ms）。哪几个 rank 慢每次起服务都不一样，与策略无关：同一 trace 上 Qwen3.8 的
belady 第一次 TP0/2/3 慢、TTFT 比 LRU 只快 1.5 ms，重跑四个 rank 都正常、快 6.9 ms。不绑时两个 TP 4 混合模型的 7 次
服务里 4 次有慢 rank；绑了以后 6 次、24 个 rank 一个都没有，各组 TTFT 也普遍快 4–11 ms（DRAM 加载也少了跨 socket）。
TP 2（GPU 0–3）上没出现过。
查法：`KVPilot load req=... ssd=N first=...ms` 按 rank（日志前缀 `Worker_TPx`）算 first / (N / KV 层数)，
`paper-figure/Eval-EXP1/summarize.py` 的「SSD first layer p90」一列也会跟着变大。

**`pkill -f <pattern>` 会杀掉执行它的 ssh 自己**：`ssh host 'pkill -f run_check.sh; ...'` 的命令行里就有这个模式，
后面的命令不会执行。按 PID 杀。

**`scp` 到 Taiji-GPU 不通**（`Connection closed`），用 `base64 | ssh ... base64 -d`。

**本机没有 matplotlib**，作图要在 GPU 机器上跑。

**本机 Python 是 3.9**，而两个 benchmark 脚本需要 3.10+（用了 `int | float` 注解）。

**不要在 bash 循环里 `pgrep -f benchmark_serving_multi_turn`**：循环自己的命令行就包含
这个模式，永远等不完。

## nsys

**session 会卡在 `Collection` 状态不落盘**，`clean` 的 SIGTERM 不一定送得到。要
`nsys sessions list` 后逐个 `nsys stop --session=...`，否则 `.nsys-rep` 一个字节都没有。

## 数据处理

**Qwen3 模板会静默吞掉 list 形态的 `content`**，替换成空串，实测丢 64% 的 token 且不
报错。预处理必须先拍平成字符串。

**两个 trace 的 tool label 命名空间零重叠**，在线判据的分布文件用错数据集时几乎每轮都落到混合分布，
判据退化成近似常数。所以 `KVPILOT_TOOL_RESP_DISTRIBUTION_PATH` 必须显式给，D 侧日志带 `label_misses=未命中/总数`，
接近总数就是用错了。

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

**`save_kv_layer` 每层都会被调用，里面的 RPC 会乘以 48**。2026-11-04 为了让 SSD-only
的命中能写回 DRAM，`save_kv_layer` 在「没有新 token 要存」时也去查一遍前缀是否在 DRAM
（`leading_dram_tokens`，当时每个 chunk 一次 master RPC），但没把结论缓存，48 层每层重查：
suffix < 512（不足一个 chunk）的请求 prefill 慢到每前缀 token 约 77 µs（20K 前缀 1.5 s，
正常约 60 ms），`[1]` 与 `[2]`/`[4]` 一起被拖慢、delay / compute 被稀释。症状是 e2e 在
suffix = 512 处断崖、`stall=0`、两行日志之间一片空白。期间的 1P1D boundary、SWE
request-safe、ToolAgent 的补测都作废。

**空闲约 1 s 后的第一个请求 prefill 慢约 2 ms**（2026-11-05 对照实验，1P1D：同一
(prefix, suffix) 上「前面紧跟一个请求」vs「先空闲 1 s」，t1 稳定差 1.2–2.3 ms）。对比两个
请求的 e2e 时它们必须处于同样的冷热状态：boundary 曾在每个请求前空闲 1 s（compute 被抬高、
比值偏 safe），request-safe 只在 `[4]` 前等 1 s（d4 多出约 2 ms、SSD 偏 unsafe），compute
约 50 ms 时这 2 ms 就是 4 个百分点。现在被计时的请求前都紧跟一个请求：boundary 在每对前发同形状预热，
request-safe 不预热，把每行的 `[4]` 排在下一行的 `[2]` 之后。

**safe 与否对 suffix 不单调**：加载逐层流水，同一 prefix 下加载总时长 L 基本恒定，compute ≥ L
就藏得住；但 compute 随 suffix 跳变（attention kernel 选择；suffix 1024 以上走 eager），
如 60K 时 suffix 144 / 148 / 180 / 212 的 compute 为 58 / 84 / 67 / 100 ms，safe / unsafe 来回翻。
eager 段（suffix 1024–2048）DRAM 还多 2–4 ms 固定开销，比值贴着 5%。「每个 prefix 一个 suffix
分界点」只是近似，比对时要知道不一致集中在这些地方。

**单请求测试不要依赖跨行的缓存历史，也不要为它改端到端的 save 路径**。request-safe 曾让
各行共用 trace 里的 prefix hash：前缀几分钟前写入、早被挤出 DRAM 只剩 SSD 时，warm 从
SSD 直接读进 GPU、不写回 DRAM，紧接着的 `[2]` 只拿到一个 chunk。为此加的 SSD-only 写回
（上一条的 `leading_dram_tokens`，以及 store 判重要求 DRAM 副本、put 前删 SSD 元数据、
last prefill 不再 `skip_save`）改的是所有请求的 save 路径，被计时的请求也背上了关键路径
上的查询，2026-11-05 全部撤回。现在每行用全新 hash 的 prefix，只取 trace 的长度
（`paper-figure/Figure3-request-safe/make_trace.py`）。

**比几毫秒的开销，要在同一个服务里对照**。2026-11-10 写回诊断：同一个 trace（32K prefix 只从 SSD 读）
起两次服务、都不写回，这个请求的 e2e 中位数 263.1 对 269.5 ms、`gpu_stall` 也差约 6 ms（e2e − `gpu_stall` 只差
1 ms）；而同一个服务里写回与不写回只差 1.5 ms。跨服务比会把 6 ms 的实例差异当成改动的效果。做法：在同一个服务里
按请求交替开关（当时用 `KVPILOT_SSD_WRITEBACK` 给请求号列表、`run_wb_diag.sh half`，两者已随写回开关删掉；
要再做这种对照，得临时加一个按请求的开关）。

## 作图

**不要把恒为 0 的那组和有分布的那组画进同一个直方图**：前者堆成一根柱子，实测高度是
后者最高柱的 9 倍，把真正的分布压平。写成图例里的一个计数即可。

**过原点的关系要用约束拟合**：零输入必然零输出时，不受约束的 `polyfit` 斜率拿去画过
原点的线，线和标注的数字对不上。
