# 观测：HBM 分布、load 阻塞

两套观测互相独立，也与 placement 策略**完全无关**，LRU 基线也能开——A/B 的意义就在
于两边各出一份来对比。

| 变量 | 默认 | 开启什么 |
|---|---|---|
| `GET_HBM_POOL_DISTRIBUTION` | `false` | HBM block 分布采样，写 `LOG_DIR/hbm-pool-<role>-<pid>.jsonl` |
| `KVPILOT_LOAD_STALL` | `false` | 每次 prefill 的 load 阻塞计量，**只打日志不写文件** |

**exposed load time / GPU stall 只有一种测法**：`KVPILOT_LOAD_STALL` 打出的 `gpu_stall`，它对照
的真值是「warm 只发 prefix、HBM / DRAM / SSD 三个测试各用一份不同的、长度相同的 suffix hash」
测出的 e2e delay，见下面「C++ loader 下 gpu_stall 就是 e2e delay」。不用别的口径。

## HBM block 分布图

回答「此刻 HBM 里躺着的 block，LRU 想淘汰哪些、判据想淘汰哪些」，**两者的分歧就是
KVPilot 的收益空间**。

二维直方图：x = block 空闲时长（对数分桶，**LRU 的唯一依据**），y =
`log2(load/comp ratio)`（**判据的依据**），格子的值 = 落在该桶的 block 数。四个象限里
**两个非对角的才是论点**：

- **左上**（老且必须留 HBM）→ LRU 先淘汰它，GPU 空等
- **右下**（新且放 DRAM 无损）→ LRU 留着它，白占 HBM

```
vllm/v1/core/kvpilot_hbm_snapshot.py   采样：每秒一次全池普查，写 LOG_DIR/hbm-pool-<role>-<pid>.jsonl
   ↓  （join on req_line）
datasets/.../validate_premises.py      离线：req_ratio.json（与三张表同一次遍历）
   ↓
utils/plot_hbm_pool_distribution.py    出图
```

开关 `GET_HBM_POOL_DISTRIBUTION`。`validate_premises.py` 和 `req_ratio.json` 已从数据集目录
移除，出这张图前要先恢复，见 [datasets.md](datasets.md)「已移除的离线验证与查表」。

```bash
python3 utils/plot_hbm_pool_distribution.py \
    --census '<log_dir>/hbm-pool-*.jsonl' \
    --ratios datasets/agent-traces/swe-smith-trajectories/req_ratio.json \
    --pdf-output <log_dir>/pool_distribution.pdf
```

`--pdf-output` 才出 PDF。不给它就只有 `--output-file` 的文本热力图和 `--csv-output` 的
网格——**而 argparse 的前缀匹配会把 `--out` 当成 `--output-file`**，于是文本被写进你以为
是 PDF 的文件里，退出码照样是 0。出图后确认一下 `head -c 4` 是 `%PDF`。

### 几条设计约束

**采样与 placement 完全解耦**：`kvpilot_hbm_snapshot.py` 不 import 判据也不 import
队列子类，block 上只多存 `kvpilot_last_access_ns` / `kvpilot_last_req_line` /
`kvpilot_last_req_block_idx` 几个观测字段。每块一行 `[block_id, state, idle_ns, req_line, req_block_idx]`
（2026-11-19 之前多两列 `queue_id` / `next_dram_safe`，`req_block_idx` 都在最后一列）。这是**功能需要**不是洁癖——主图的价值在于 baseline（原版
`FreeKVCacheBlockQueue`）和 KVPilot 各出一张对比，所以采样必须在所有 KVPilot 开关关掉
时照样工作。

**ratio 不进快照，只存 req_line 事后 join**。ratio 是**请求**的属性不是 block 的属性，
而且要等这一轮结束才算得出——对会话最后一轮来说那已经是 block 早被释放之后了。所以采样
只记「这块最后属于哪个 req」，y 值画图时查表。

**join key 是源文件行号**（replay 脚本编进 request_id 前 8 位，
`benchmark_serving_multi_turn.py:471-485`），所以**建表和压测必须用同一个输入文件**，
否则全部 join 不上（脚本会报 `unmatched`）。

**lookahead 已经在成本模型里，不要再 shift 一次**：`dram_safe_times(prefix_j, append_j)`
比较的是「请求 j 留下的 KV 的载入代价」与「紧随其后那一轮的计算」，这正是请求 j+1 在
请求 j 的 block 上的体验。所以 line_j 的 block 取的就是**在 j 处算出的**那组时间；再去
链到 j+1 自己的时间就是把 lookahead 平移了两次。`next_line` 只用来验证 join，不是 y 的
来源。

**采样点在 `schedule()` 末尾**而不是 `free_blocks()` 里：后者每次释放都触发且呈突发状，
step 末尾是池子状态一致的静止点。间隔**硬编码 1 s**（`INTERVAL_NS`），不给环境变量——
两次跑用不同采样率就没法比。全池扫描 O(num_gpu_blocks) ≈ 2,400 块约 1 ms，1 Hz 下开销
~0.1%；序列化和落盘在后台线程，engine 线程只做扫描。

**默认只画 cached-free 块**（真正的淘汰候选）。in-use 块此刻不参与淘汰，混进去会让
「LRU 会不会淘汰它」失去意义；blank 块无 KV 无请求，只计数不逐条列（启动阶段几乎是
整个池子）。

**终止轮会整批落在图上方**：`append = decode_len` 且 prefix 最长，算出的 ratio 很高
（p50 ≈ 22），**100% 判 unsafe**。占请求总数约 4%（实测 200/5,173）。这是
`append = decode_len + tool_resp` 这个统一定义的直接推论，不是 bug；但读图时要知道上方
有这么一撮是会话末轮，它们实际上再也不会被复用。

## load 阻塞计量

回答「这次 prefill 有多少时间是在等 KV 而不是在算」。layerwise loading 下每层 attention
都要等自己那层的 KV 就位，一次 forward 里所有等待之和，就是加载**没能**被计算掩盖的
部分。**KVPilot 的目标就是把它压到 0。**

开关 `KVPILOT_LOAD_STALL`。**只打日志，不写数据文件**：scheduler 对每条完成的 prefill
打一行 `KVPilot prefill ...`（`logger.info`），其中 `gpu_stall` 就是这个量。

### 为什么不写 JSONL

早期版本同时写 `LOG_DIR/load-stall-<role>-<pid>.jsonl`，实测 4004 条记录跑完之后文件
**0 字节**。原因：文件用 `buffering=1024*1024` 打开、写盘在 `daemon=True` 线程上，模块
里**没有 flush、没有 close、没有 atexit**，worker 退出时缓冲区直接丢弃。约 1 MB 的记录
刚好卡在 1 MiB 门槛下面，一次都没触发自动落盘，`clean` 优雅停止也救不回来。

**删掉而不是加 `flush()`**：数据量刚好跨过 1 MiB 时会落盘**一部分**，那比整个空文件更
糟——拿到一个被截断的文件却看不出来。日志行本来就是持久的、按 worker 分好的、还自带
时间戳，留着文件只是同一份数据的第二种编码，而它的失败是静默的。

### 一次 forward 有两类计时窗口，都要计

- `start_load_kv` **序幕**：C++ loader 下是建加载——DRAM 把全部层的 H2D 发到 load
  stream；SSD 建读计划、等上一个 SSD 请求的读结束等。此时 forward 还没开始，GPU 通常
  没活，这段 CPU 时间基本全是计算流空闲。
- 每层的 `wait_for_layer_load`：DRAM 只发 `WaitEvent(done_[i])`；SSD 先在 CPU 上阻塞到
  第 i 层 H2D 入队，再 `WaitEvent`。

漏掉序幕会**系统性低估**：第 0 层之前没有任何计算可以掩盖加载。

### 怎么量：计算流上的 event 对

每个窗口前后在当前（计算）stream 上各记一个 CUDA event，`gpu_stall` = 两个 event 在 GPU
上完成时刻之差，即计算流在窗口里空等的时间。记进入窗口的 CPU 时刻 `t_enter`、离开
`t_exit`，计算流把之前排队的活做完的时刻 `t_gpu_free`，第 i 层加载完成 `t_done`（忽略
发射延迟）：

```
gpu_stall = max(t_exit, t_done) − max(t_enter, t_gpu_free)     （< 0 记 0）
```

- 起点取 `t_gpu_free`：CPU 跑在 GPU 前面时，前面的层还在算，这段等待被计算掩盖，不计。
- 终点取 `t_done`：结束 event 排在 `WaitEvent` 之后，DRAM 路径 CPU 早已返回，GPU 侧的
  等待仍被覆盖。

以前还有一个 CPU 墙钟的 `stall`（`perf_counter_ns`），已删除（2026-11-08）：DRAM 路径
CPU 不阻塞，它看不到 GPU 侧的等待；SSD 路径它把被计算掩盖的 CPU 阻塞也算进去，中位偏高
17–75 ms。

### 归因：只考虑 bs=1

- **worker** 按顺序报这一步每个窗口的空闲（`KVConnectorOutput.kvpilot_wait_ns`，一个窗口
  一项，图捕获时记 0）。窗口数就是日志里的 `waits`：每次 forward 1 个序幕 + 每个 attention
  层 1 个，48 层模型单步 49。
- **scheduler** 把整步的量全部记给这一步里正在 prefill 的请求（`_kvpilot_record_step`），
  不做拆分。`dram=0` 的请求记 0：没有加载时计时窗口里只有 hook 自身的开销。
- **前提是一次 forward 只有一条 prefill。** 同一 forward 里有多条在加载的 prefill 时，
  每条都会记整步，按请求求和会重复计。

**TP 之间逐窗口取 max，再求和**（`KVOutputAggregator`）：各 rank 持有每层的一个分片，每层
经 all-reduce 锁步，一个窗口让整步多花的是最慢那个 rank 的空闲；这时别的 rank 在 NCCL
里等它，计算流不空闲，不计。所以 `gpu_stall = Σ_窗口 max_rank`。「每个 rank 先求和再取 max」
在 ranks 卡在不同层时会少算；求和会把一次停顿放大 TP 倍；只取 rank 0 会漏掉别的 rank 上的拖后腿者。

**chunked prefill 会跨多个 forward**，所以耗时和 step 数是累加的，记录在**算完整个
prompt 的那一步**写出，且只写一次（`kvpilot_scheduled_ns` 归零作哨兵）。「算完」看这一步有没有给这个
请求采样出 token：async scheduling 下处理第 k 个 chunk 的输出时，下一个 chunk 已经调度出去，
`num_computed_tokens` 已经到了整条 prompt。2026-11-22 之前用的是 `num_computed_tokens`，新算超过一个
chunk（8192 token）的请求提前一个 chunk 写出，`e2e`、`steps`、`done=` 都少了最后一个 chunk（`steps` 恰好比
chunk 数少 1），这之前的 TTFT 对这类请求偏小。后续因抢占重新
prefill 是另一次事件、有它自己的 load 行为，不算这一次的延续。

### 日志行字段

```
KVPilot prefill req=<id> line=<req_line> prompt=N new=N hit=N dram=N
  e2e=X.XXXms waits=N steps=N gpu_stall=X.XXXms done=T
```

`new`（本轮真算的）+ `hit`（本地 HBM 前缀命中，无需加载）+ `dram`（connector 提供，
**只有这部分会产生 stall**）= `prompt`。`line` 是源文件行号，与 HBM 快照可对齐。`done`
是 prefill 结束的墙钟时间，减去客户端发送时间就是 prefill 结束时的 TTFT。

2026-11-08 之前的日志在 `e2e` 后面还有 `stall=X.XXXms (P%)`；`utils/placement_ab/analyze.py`
两种格式都认。`utils/` 和 `paper-figure/` 下多个脚本按正则解析这一行，改字段名或顺序要同步改
它们。

## C++ loader 下 gpu_stall 就是 e2e delay（2026-11-08）

前提是 C++ loader（prelookup 现在固定开，`LayerwiseLoader` / `SsdLayerwiseLoader` 每层一次
调用、释放 GIL），加载的代价都落在等待窗口里，**`gpu_stall` 可以直接当 exposed load 用**。Python
逐层 fetch 路径下 fetch 在 LMCache 事件循环线程上和 forward 线程抢 GIL，约一半代价落在窗口外，
gpu_stall 不成立。

**测法**（`utils/ssd_load_check`，`make_trace.py 64,256,1024,4096`）：prefix 4K–64K × suffix
64/256/1K/4K × 3 次，1P2D、TP=2、`-p 1`、start 全 enqueue。同一 prefix 的 warm 只发 prefix；`[1]` HBM /
`[2]` DRAM / `[4]` SSD 三个测试各用一份**不同的、长度相同的 suffix hash**，`cache_use_max_tokens =
prefix`，三个测试都算并写自己的 suffix（日志里各有一行 `Stored`）。delay = `[2]` / `[4]` 的 e2e −
同一次重复 `[1]` 的 e2e，逐请求和 gpu_stall 比：

| 加载 | 档位 | gpu_stall − delay（中位） | 绝对误差 p50 / p90 / max | 同格 3 次 delay 极差（中位 / max） |
|---|---|---|---|---|
| DRAM | suffix ≤ 1K（cudagraph） | 0.0 ms | 0.3 / 1.6 / 3.1 ms | 1.1 / 3.7 ms |
| DRAM | suffix 4K（eager） | 0.0 ms | 0.7 / 3.2 / 4.9 ms | 2.0 / 4.4 ms |
| SSD | suffix ≤ 1K | −0.1 ms | 0.4 / 2.0 / 2.9 ms | 8.5 / 32.6 ms |
| SSD | suffix 4K | −0.1 ms | 0.8 / 1.3 / 2.0 ms | 2.2 / 5.8 ms |

**DRAM + SSD 混合命中**同样成立（`make_trace_mixed.py` / `analyze_mixed.py`，同一网格，每组在 `[1]`
之后按 SSD 占比 0 / ¼ / ½ / ¾ / 1 各发一个测试，混合档用 `cache_use_control=[2, 4]`、
`cache_use_max_tokens=[prefix·(1−f), prefix·f]`，前段从 DRAM、后段从 SSD；每个 rank 的 `ssd_values`
逐条核对等于计划的 SSD chunk × 48）：

| SSD 占比 | suffix ≤ 1K：中位 / p90 / max | suffix 4K：中位 / p90 / max |
|---|---|---|
| ¼ | −0.3 / 1.8 / 3.5 ms | −0.9 / 2.3 / 4.9 ms |
| ½ | −0.6 / 2.0 / 4.1 ms | +0.3 / 1.3 / 3.0 ms |
| ¾ | −0.3 / 1.9 / 6.2 ms | +0.1 / 1.8 / 3.3 ms |

（中位是 gpu_stall − delay，p90 / max 是绝对误差；同一轮的 0 和 1 两档与上表一致。）混合命中和纯 SSD
走同一个 `SsdLayerwiseLoader`，每层的 DRAM chunk 和 SSD chunk 在同一次 batch 拷贝里，窗口和 event
语义不变。

- 误差比同一格重复之间的波动还小：gpu_stall 跟的是**这一次**请求的 delay，不只是平均值。
- **TP 必须逐窗口取 max**。每个 rank 先求和再取 max 在 SSD 短 suffix 上 p90 少算 45 ms、最多
  67 ms（例如 64K / 256 中位 350.7 对 delay 395.5）：两 rank 读同一组盘，每层谁先到不固定，慢的
  rank 在等数据，快的在 all-reduce 里等它，后者不是空闲。DRAM 两 rank 对称，两种取法相同。
- **三个测试必须各写自己的 suffix**。之前的网格三个测试共用一份 suffix hash，只有排在最前的 `[1]`
  写 suffix，`[2]` / `[4]` 发现已存在就不写；suffix ≥ 1K 时 delay 被低估 1–3 ms（DRAM 甚至为负），
  看起来像 gpu_stall 在 eager 档多算约 5 ms。那批数据已删除。
- **不是 `step()` 阻塞后重新发射 attention 的空隙**：窗口的结束 event 排在 `WaitEvent(done_[i])`
  之后，要等第 i 层 scatter 完才完成；`submitted_ > i` 只表示 H2D 已入队，CPU 醒来时这一层的 H2D
  还在跑，attention 早已排在后面。

## load time（`KVPilot load` 行）

```
KVPilot load req=<id> tokens=N layers=48 fetch=X.XXXms h2d=X.XXXms load=X.XXXms ssd=N first=X.XXXms start=X.XXXms
```

每个 TP rank 一行（`lmcache/v1/kvpilot_load_time.py`），和 `gpu_stall` 同一个开关。这是加载本身花了
多久，不论有没有被计算盖住；露出了多少只看 `gpu_stall`。

- `load`：load stream 上从加载开始到最后一层 scatter 完（SSD 从第一次读盘前算起），两 rank 取大。
  C++ loader 在 `start` 时就把 DRAM 全部层排进 load stream，不等计算，所以它就是加载本身。
- `h2d`：各层 copy + scatter 的 CUDA event 时长之和。
- `first`：第 0 层在 GPU 上就绪的时刻，和 `load` 同一时钟；`start`：发起加载的 CPU 时间，在这个
  时钟之前。
- `ssd`：从 SSD 读的 value 数（层 × chunk）。
- `fetch`：只有 Python 逐层取数路径（以前关掉 prelookup 时）才有值，现在总是 0。
