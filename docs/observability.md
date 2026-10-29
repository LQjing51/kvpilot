# 观测：HBM 分布、load 阻塞、TTFT 拆解

三套观测互相独立，也与 dram-safe 各开关**完全无关**，baseline 也能开——A/B 的意义就在
于两边各出一份来对比。

| 变量 | 默认 | 开启什么 |
|---|---|---|
| `GET_HBM_POOL_DISTRIBUTION` | `false` | HBM block 分布采样，写 `LOG_DIR/hbm-pool-<role>-<pid>.jsonl` |
| `KVPILOT_LOAD_STALL` | `false` | 每次 prefill 的 load 阻塞计量，**只打日志不写文件** |

TTFT 拆解不需要开关，但需要 nsys profile（`NSYS_PROFILE`，见
[benchmark.md](benchmark.md)）。

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

开关 `GET_HBM_POOL_DISTRIBUTION`。

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

**采样与 dram-safe 功能完全解耦**：`kvpilot_hbm_snapshot.py` 不 import 判据也不 import
队列子类，block 上只多存 `kvpilot_last_access_ns` / `kvpilot_last_req_line` 两个观测
字段。这是**功能需要**不是洁癖——主图的价值在于 baseline（原版
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
打一行 `KVPilot prefill ...`（`logger.info`）。

```bash
python3 utils/plot_load_stall.py <log_dir> [out.pdf]     # 自己 glob prefill-*.log
```

### 为什么不写 JSONL

早期版本同时写 `LOG_DIR/load-stall-<role>-<pid>.jsonl`，实测 4004 条记录跑完之后文件
**0 字节**。原因：文件用 `buffering=1024*1024` 打开、写盘在 `daemon=True` 线程上，模块
里**没有 flush、没有 close、没有 atexit**，worker 退出时缓冲区直接丢弃。约 1 MB 的记录
刚好卡在 1 MiB 门槛下面，一次都没触发自动落盘，`clean` 优雅停止也救不回来。

**删掉而不是加 `flush()`**：数据量刚好跨过 1 MiB 时会落盘**一部分**，那比整个空文件更
糟——拿到一个被截断的文件却看不出来。日志行本来就是持久的、按 worker 分好的、还自带
时间戳，留着文件只是同一份数据的第二种编码，而它的失败是静默的。

### 一次 forward 有两个阻塞点，都要计

- `start_load_kv` 里的**序幕**——它提交所有 fetch 后会阻塞等第 0 层，此时 forward 还
  没开始
- 每层的 `wait_for_layer_load`——每个 attention 层一次

漏掉前者会**系统性低估**，因为第一层的等待通常是最长的（那时还没有任何计算可以掩盖它）。

### 为什么用 CPU 墙钟而不是 cuda event

LMCache 的 layerwise 路径靠 `task.result()`（asyncio future）和
`current_stream.wait_stream(load_stream)` 阻塞 **engine 线程本身**，所以
`perf_counter_ns` 量到的就是真实的线程停顿。这里不存在「CPU 跑在 GPU 前面」的问题——
正是因为它阻塞了，才叫 stall。

### 归因在两个进程里分开做

- **worker** 只报**每 step 总量**。一次 forward 是整个 batch 共享的，worker 无法诚实地
  说某个请求该占多少。经 `KVConnectorOutput.kvpilot_stall_ns` 回传（照
  `kv_connector_stats` 的形状）。
- **scheduler** 按各请求的 **DRAM token 数**加权拆分（`attribute()`）。理由：没有 load
  的请求（全 HBM 命中、或同 batch 里的 decode step）单独跑也一样快，不该分摊；拉了两倍
  KV 的请求就该担两倍。整除余数补给最大消费者，保证各份之和**精确等于**实测总量——否则
  长时间小 batch 会静默少报。

**TP 之间取 max 不是 sum**：各 rank 持有每层的一个分片且锁步执行，step 要等最慢的 rank
的 KV 落地。求和会把一次停顿放大 TP 倍；只取 rank 0 又会漏掉别的 rank 上的拖后腿者。

**chunked prefill 会跨多个 forward**，所以耗时和 step 数是累加的，记录在**算完整个
prompt 的那一步**写出，且只写一次（`kvpilot_scheduled_ns` 归零作哨兵）。后续因抢占重新
prefill 是另一次事件、有它自己的 load 行为，不算这一次的延续。

### 日志行字段

```
KVPilot prefill req=<id> line=<req_line> prompt=N new=N hit=N dram=N
  e2e=X.XXXms stall=X.XXXms (P%) waits=N steps=N
```

`new`（本轮真算的）+ `hit`（本地 HBM 前缀命中，无需加载）+ `dram`（connector 提供，
**只有这部分会产生 stall**）= `prompt`。`line` 是源文件行号，与 HBM 快照可对齐。

字段名和顺序是 `plot_load_stall.py` 正则的契约，改一个就要同步改那个正则
（`kvpilot_load_stall.py` 里 `StallLogger.PREFIX` 附近有注释提醒）。

### 实测标定（2026-09-11，session 400 / R=4.0 那组）

4004 条 prefill、4021 次 forward、2 个 prefill worker：

| 指标 | 值 |
|---|---|
| 聚合 stall 占比 | 23.2%（1974 s / 8494 s） |
| 每次 forward 平均 stall | 491 ms |
| 每次 forward 的等待次数 | 30（≈ 每个 attention 层一次） |
| 单次层等待中位数 | 14.4 ms |
| stall 随加载量的斜率 | 76.3 ms / 1k DRAM tokens |

**自洽检查**：`dram=0` 的请求 stall **恰好 0.000 ms**（1541/1541）。这条值得在每次新跑
之后重做一遍——它证明归因没有把别的请求的等待摊到不加载的请求头上。

### 读图注意

**聚合那一栏是 `sum(stall)/sum(e2e)`，不是墙上时间。** 多条 prefill 并发时窗口互相
重叠，分母超过实际墙上时长。正确读法是「每单位 prefill 工作里有多少在等 KV」。

**不要把无加载的请求和有加载的画进同一个直方图**：它们全部恰好是 0 ms，会堆成一根高
1541 的柱子，约为有加载请求最高柱（179）的 9 倍，把真正有形状的分布压平。当前脚本把
它们写成图例里的一个计数。

**拟合过原点**：零 token 必然零 stall，所以斜率要用约束拟合；不受约束的 `polyfit` 斜率
拿去画过原点的线，画出来的线和标注的数字对不上。

## TTFT 拆解（nsys）

`vllm-dev/examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/analyze_lp_ttft_breakdown.py`

```bash
python3 analyze_lp_ttft_breakdown.py --log-name <name>      # 不带 -logs 后缀
```

产出在 `<log_dir>/ttft_breakdown_static/`：`ttft_share_aggregate.{csv,pdf}`、
`ttft_breakdown.csv`、两张 by-hit-rate 图。前提是 `.nsys-rep` 已经生成，见
[benchmark.md](benchmark.md) 的「nsys 会卡在 Collection 不落盘」。

### 两张图的分母不同，不要直接对比

这是最容易困惑的地方（同一轮数据，排队占比一个 31.7% 一个 1.3%）：

| 图 | 分母 | 问的问题 |
|---|---|---|
| `ttft_breakdown`（每请求） | 该请求自己的 TTFT | 用户的 TTFT 被什么吃掉了 |
| `ttft_share_aggregate` | 所有请求窗口的**并集** | prefill GPU 因为什么而空转 |

差距分两步（实测 4,784,806 ms → 31,328 ms，共 153 倍）：

1. **取并集，2.5 倍**。并发下多个请求同时排队，它们的等待在墙上时间里是同一段，求和会
   重复计数（实测窗口重叠系数 6.0×）。
2. **只保留 GPU 真正空转的部分，砍掉 98.4%**。代码在 `analyze_lp_ttft_breakdown.py:969`
   附近：`idle_region = subtract_intervals(served, accounted)`，排队时间要先和
   `idle_region` 求交集才计入。而那一轮 GPU 空转只占 span 的 2.5%。

物理含义：**A 请求排队的时候 GPU 并没有闲着，它正在算 B 请求。** 从 A 的视角这是纯等待
（计入 A 的 TTFT 完全正确）；从「GPU 被什么阻塞」的视角，机器在满负荷工作，不该记成
stall。

要看用户体验/SLO 用 per-request；要看机器效率用 aggregate。

### aggregate 那几列不是一个饼图

`queue / compute / expLoad / expStore / p2d / postPF / inPFidle` 这七个是互斥划分（加起
来 ≈ 100%）。`load_tot` 和 `store_tot` 是**额外的诊断列**，与前面几列相交——layerwise
下加载和计算本来就重叠（这正是 KVPilot 要利用的），所以 `load_tot=90.4%` 和
`compute=71.4%` 加起来超过 100% 是正常的。

### 与 load-stall 互相印证

同一轮配置下 nsys 的 `kv_load_exposed_pct` = 25.9%，load-stall 的
`sum(stall)/sum(e2e)` = 23.2%。两条**完全独立的测量路径**（一个是 nsys 的 GPU kernel
区间求补集，一个是 engine 线程的 CPU 墙钟计时）得到接近的数，是真正的交叉验证。
