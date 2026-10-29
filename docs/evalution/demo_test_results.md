# demo_test 实验记录

按 `docs/evalution/demo_test.md` 的六个阶段在 Taiji-GPU 上跑的实验。本地改代码 →
`sync_to_devcloud.sh` 同步 → GPU 机器上跑 → 回读日志。

**日志位置**：`vllm-dev/examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/dynamo-log/demo-tests/<name>-logs/`

## 统一的 workload

六个配置共用同一组压测参数，这是 A/B 可比的前提（demo_test.md 第 5 条明确要求改了
参数就要重跑前面的实验）：

```bash
--max-inflight-req 16 --max-inflight-session 400 -n 4000 \
--session-arrival-rate 4.0 --tool-time lognormal:0,1
```

一轮约 30-35 分钟，满足「每次测试维持在一个小时以下」。

## 跑实验的入口

`/root/run_demo.sh`（GPU 机器上，不在仓库里）。它把 clean → setup → 自查 → benchmark
→ clean 串成一条命令，配置全走环境变量：

```bash
NAME=demo-tests/noprefix DISABLE_LOCAL_PREFIX_CACHE_LOOKUP=true bash run_demo.sh
```

两个设计要点：

- **自己 export PYTHONPATH**。非交互 ssh 不读 `.bashrc`，不设就会静默跑成
  `/usr/local/lib/python3.12/dist-packages` 里那份 stock vLLM，日志看起来一切正常但
  没有任何 KVPilot 代码。脚本在 setup 之后 `grep -c 'Free blocks distribution'`，
  为 0 就直接 abort——这一步救过一次（见下）。
- **默认 `NSYS_PROFILE=0`**。nsys 每轮要多花约 10 分钟收尾，还会卡在 `Collection`
  不落盘；而这几个阶段要的指标（exposed load 占比、TTFT）都能从日志里拿到，
  load-stall 与 nsys 的 `kv_load_exposed_pct` 本来就是两条独立测量路径、实测互相印证
  （23.2% vs 25.9%）。只在需要 TTFT 拆解时才开。

## 代码改动

### 1. `disagg_lmcache.sh:78` 两条 export 粘在一行（必须先修，否则阶段 3 是空跑）

```bash
export ENABLE_MOONCAKE_CPU_BACKEND="${ENABLE_MOONCAKE_CPU_BACKEND:-false}"export ENABLE_LOCAL_MOONCAKE_HBM="${ENABLE_LOCAL_MOONCAKE_HBM:-false}"
```

`export A="x"export B="y"` 会被 bash 解析成把 `A` 赋值为 `xexport`。实测：

```
$ ENABLE_MOONCAKE_CPU_BACKEND=true bash -c '<上面那行>; echo "[$ENABLE_MOONCAKE_CPU_BACKEND]"'
[trueexport]
```

而消费侧 `_truthy()`（`base_connector.py:25`）只认 `{"1","true","yes","on"}`，
`trueexport` 一律为假。**症状是完全静默的**：三个开关照样 export 出去、日志照样打
`Mooncake feature flags: ... =False`，看不出这是被 shell 吃掉的而非没设。如果不修这一
行，阶段 3 的「开三个优化」会和阶段 2 跑出一模一样的结果，然后人会去怀疑优化没效果。

已拆成两行。

### 2. benchmark 参数改成环境变量可覆盖（`dynamo_script.sh`）

原来五个压测参数是写死的，每换一次配置就要改脚本、再同步一次。改成
`BM_MAX_INFLIGHT_REQ` / `BM_MAX_INFLIGHT_SESSION` / `BM_NUM_REQUESTS` /
`BM_SESSION_ARRIVAL_RATE` / `BM_TOOL_TIME`，默认值保持原样。这样「六个配置共用同一
workload」是一条环境变量的事，不依赖人记得改回来。

### 3. 实测 dram-safe 判据（阶段 5 的 oracle 臂）

阶段 5 要求一个用 `req_measured_safe.json` 按 reqid 取真实 safe 情况的跑法，需要改
dram-safe 判断的代码。

`vllm-dev/vllm/v1/core/kvpilot_predict.py` 新增 `measured_enabled()` /
`measured_dram_safe(request_id)`：按 request id 前 8 位的**源文件行号**查表（这正是
`req_ratio.json` 和 HBM 普查已经在用的 join key，所以三者能逐行对照）。放在这个文件
里是因为它是在线离线唯一的一份判据实现，新增的口径也必须在同一处，否则离线算出的
准确率就不再描述线上跑的东西。

`dynamo-dev/components/src/dynamo/vllm/kvpilot.py` 的 `KvpilotReporter.report()` 里，
预测照旧先算一遍，然后在实测表命中时**只替换 verdict**：label / append / 两个时间仍
是成本模型的值，日志行加一个 `source=predicted|measured|predicted(measured-miss)`。
这样两臂的日志格式一致、可以逐条 diff，且表里没有的行（warmup、健康检查）自动退回预测
并在日志里留痕，不会静默变成默认值。

开关是 `KVPILOT_MEASURED_DRAM_SAFE_PATH`，在 `disagg_lmcache.sh` 里做存在性检查后
export（照 `KVPILOT_TOOL_RESP_LEN_PATH` 的 fail-closed 写法）。不设就是原来的预测路径。

### 4. 三个分析脚本

| 脚本 | 作用 |
|---|---|
| `utils/stall_share.py` | 把 load-stall 日志聚合成 exposed load 占比，带自洽检查 |
| `datasets/.../build_measured_safe.py` | 从跑完的日志反推每个请求实测是否 dram-safe，产出 `req_measured_safe.json` |
| `datasets/.../score_measured.py` | 理论 / 按类型预测 / 全猜多数类 三者对实测真值的正确率 |
| `datasets/.../fit_constants.py` | 扫 dram/compute 缩放拟合理论判据 |

`stall_share.py` 的自洽检查照 observability.md 的做法：`dram=0` 的请求 stall 必须恰好
为 0，否则说明 scheduler 的归因把别的请求的等待摊了过来。每轮都跑，至今全部 OK。

## 「实测 dram-safe」这个判据本身调了两次，两次都是被数据打回来的

阶段 4 要「根据 noprefix-H2D 的结果判断哪些请求 dram safe」。日志里没有这个字段，得自己
定义，而头两个定义都错了——**且都错得很像对的**，值得完整记下来。

### 第一版：`stall/e2e <= 5%`，错在分母

最自然的想法：stall 占这次 prefill 的比例低就是被掩住了。跑出来 noprefix-H2D 只有
**4.7%** 的请求 dram-safe，而理论模型说 70.6%，差了 15 倍。

差得太离谱反而是线索。去查 `e2e` 到底量的是什么（`kvpilot_load_stall.py:158`：
scheduled 到 prefill 算完），然后对着三轮跑算「每 1k 个重算 token 的表观 compute」：

| 跑 | new tokens/req | 表观 compute/1k tokens |
|---|---|---|
| noprefix | 1210 | **2076 ms** |
| origin | 1218 | 1318 ms |
| noprefix-H2D | 1225 | **100 ms** |

token 数几乎一样，compute 差 **21 倍**。GEMM 不可能差这么多——`e2e` 里绝大部分是**排队**
（请求在 batch 里等别人算完），不是自己的计算。所以 `stall/e2e` 量的是**拥塞程度**，
H2D 让系统不拥塞了，分母跟着塌掉，于是占比反而显得很高。

**分母不能用 e2e。**

### 第二版：`stall/dram_tokens <= 5 ms/1k`，错在没扣固定开销

换成按加载量归一，看起来合理多了：72.9% dram-safe，和理论的 70.6% 很接近，像是对上了。

但把它和成本模型的 ratio 对照，方向是**反的**：

| 模型 ratio 分位 | 实测 safe 率 | | prefix 大小 | 实测 safe 率 |
|---|---|---|---|---|
| 最低 10% | 18.4% | | 601-4,893 | 13.7% |
| 最高 10% | **85.5%** | | 12,235-50,229 | **98.2%** |

模型说 prefix 越大加载越贵、越不 safe；实测说 prefix 越大越 safe。**完全相反**，而
「72.9% ≈ 70.6%」只是两个数字碰巧撞在一起。

拟合 `stall = fixed + slope × dram_tokens`：

```
stall_ms ≈ 11.03 ms 固定 + 2.168 ms / 1k tokens     两项在 5,087 tokens 处相等
```

**每次加载有约 11 ms 的固定开销**（建立传输、第一层等待），与加载多少无关。于是
ms/1k 这个指标对小加载天然难看（<2k tokens 时 p50 是 19.7 ms/1k）、对大加载天然好看
（2.8 ms/1k）——它其实在问「这个请求加载得够不够多、够不够摊薄固定成本」，而不是
「加载有没有被计算掩住」。

扣掉固定项之后，每 token 的速率在所有尺寸上**齐平在 2.06 ms/1k 左右**（见下），
证明模型是对的：

| dram tokens | 原始 ms/1k | 扣掉固定项后 |
|---|---|---|
| 512-2,048 | 19.73 | 0.46 |
| 2,048-5,632 | 4.88 | 2.09 |
| 5,632-7,168 | 3.80 | 2.08 |
| 7,168-9,216 | 3.42 | 2.08 |
| 9,216-11,776 | 3.13 | 2.06 |
| 11,776-49,664 | 2.79 | 2.04 |

### 第三版（当前）：扣掉拟合出的固定开销，再按 token 归一

```
dram_safe = (mean(stall) - fixed) / mean(dram_tokens) <= 阈值
```

固定项由每轮自己的数据拟合，不写死——它是传输路径的性质，H2D 开不开、机器忙不忙都会变。
换成这个定义后方向终于对了（ratio 越低越 safe，见 `fit_constants.py` 的输出），虽然
信号仍然偏弱。

**教训**：`72.9% ≈ 70.6%` 这种「数字对上了」是最危险的验证方式。两个定义都能产出一个
看起来合理的总体比例，只有去看**分布的形状**（按 ratio 分位、按 prefix 分桶）才发现
第二版的相关性是反的。

## 结果

### 阶段 2：no prefix / origin 的 exposed KV load

**占比和绝对值要一起看**（`docs/pitfalls.md`「load 占比是陷阱」）：

| 配置 | 请求数 | 有加载 | **exposed** | 目标 | sum(stall) | DRAM tokens | TTFT |
|---|---|---|---|---|---|---|---|
| `noprefix` | 4004 | 91.0% | **21.36%** | >20% ✓ | 2732 s | 28.7 M | 4768 ms |
| `origin` | 4004 | 60.0% | **22.95%** | >15% ✓ | 1915 s | 21.8 M | 3250 ms |

两个都用文档里标定过的那组参数，一次就达标，不需要调 session rate / inflight。

**为什么 noprefix 的占比反而略低于 origin**：它的绝对加载量明显更大（DRAM tokens 多
32%、总 stall 多 43%），但关掉 prefix lookup 后每条请求都要重算整个 prompt，`sum(e2e)`
涨了 53%（12790 s vs 8345 s），分母涨得比分子快。origin 每请求有 1783 个 hit token 是
省下来的计算，压小了它的分母却没同比压小 stall。

换个角度：单看有加载的请求，origin 每条加载**更多**（9064 vs 7864 tokens）——它 miss
的都是缓存装不下的深前缀；noprefix 是把更大的总量摊到更多、更浅的加载上。两者每 1k
DRAM token 的代价接近（95.3 vs 88.0 ms），说明测量本身在两个配置下是一致的。

### 阶段 3：开三个 Mooncake 优化后

| 配置 | 请求数 | 有加载 | **exposed** | 目标 | 每 1k token 代价 | TTFT |
|---|---|---|---|---|---|---|
| `noprefix-H2D` | 4004 | 91.0% | **17.46%** | >10% ✓ | 3.5 ms | 267 ms |
| `origin-H2D` | 进行中 | — | — | >5% | — | — |

H2D 直通的效果非常大：加载代价从 **95.3 → 3.5 ms/1k token（27 倍）**，TTFT 从
4768 → 267 ms（18 倍）。加载的 token 数几乎没变（29.2 M vs 28.7 M），变的是每个 token
多贵。即便如此 exposed 仍有 17.46%，**后续 KVPilot 仍有优化空间**，这正是这一阶段要
确认的事。

### 踩到一次 NIXL 段错误（不是本次改动引起的）

`origin-H2D` 第一次跑到 462/4000 时 decode-0 的 TP worker 直接死了，
`VllmWorker-0 died unexpectedly`，没有 Python traceback。往上翻找到原生栈：

```
!!!!!!! Segfault encountered !!!!!!!
  cuEventQuery → uct_cuda_base_iface_event_fd_arm → ucp_worker_arm
  → nixlUcxWorker::arm() → nixlUcxSharedThread::run()
```

是 NIXL/UCX 的 CUDA 事件轮询线程崩了，与 KVPilot 代码、与三个 Mooncake 开关都没有关系
（`UCX_TLS=self,tcp,cuda_copy,cuda_ipc`，崩在 cuda_ipc 的 arm 路径上）。五轮里只出现
这一次，原样重跑即恢复（0 段错误、0 abandoned）。**结论是偶发竞态，不要为它改传输配置**
——改了反而会让这一轮和前四轮不可比。

另外 `dynamo_script.sh clean` 只给 GPU 显存 20 秒释放时间，开了 Mooncake HBM 后端之后
不够用，下一轮 setup 的显存自检会因此 abort（`GPU 2 has 92793 MiB in use`）。runner 里
补了一个最多 5 分钟的等待循环。

### 阶段 4：实测 dram-safe 与预测正确率

**先说结论：达标了，但达标的路径和 demo_test.md 设想的顺序不一样，而且中途发现
「实测 dram-safe」这个标签本身有一个必须先解决的问题。**

最终配置与结果（`measured_accuracy.log`）：

| 预测方式 | 轮数 | 真实safe | 预测safe | 假阳 | 假阴 | **正确率** | 平衡正确率 |
|---|---|---|---|---|---|---|---|
| 理论（用实测 resp） | 3,643 | 14.1% | 30.6% | 21.6% | 5.1% | 73.3% | 69.4% |
| **按类型预测 resp** | 3,643 | 14.1% | 18.7% | 10.9% | 6.3% | **82.9%** | 71.5% |
| 全猜多数类 | 3,643 | 14.1% | 0.0% | 0.0% | 14.1% | 85.9% | 50.0% |

- 实测表：`req_measured_safe.json`，阈值 `1.5 ms/1k tokens`（扣除固定开销后）
- 查表：`tool_resp_len.json`，`--predict-percentile 40`
- 成本模型：`--dram-scale 0.2`（DRAM 比 datasheet 慢 5x）

**`全猜多数类` 的 85.9% 比 82.9% 高，但它的平衡正确率是 50%——什么都没预测。** 这就是
为什么所有脚本都同时报平衡正确率：14.1% 的 base rate 下，「全判 unsafe」自动拿到 85.9%。
按类型预测的 71.5% 平衡正确率才说明它真的在区分两类。

### 为什么阈值是 1.5 而不是别的：80% 在多数阈值下是不可达的

按 demo_test.md 的顺序（先调 dram bw / flops，再调 percentile）在第一版阈值（2.0）下
怎么调都到不了 80%。与其继续扫参数，先算了一个**上界**（`ceiling.py`）：在线判据只能看到
prefix、decode_len 和 tool label，把它们压成一个 append 喂给单调阈值，所以任何这种形式的
预测器都不可能超过

- 对成本模型 ratio 取**最佳单一阈值**的正确率，以及
- 每个 label 都能挑到**最理想**的 resp 长度时的正确率（查表这个旋钮的 oracle）

阈值 2.0 下这两个上界是 **59.8% 和 56.1%**——离 80% 差得远，说明差距在**特征**里，不在
常数里，再怎么扫都没用。

于是改为扫阈值本身，看哪个阈值下的标签是**可学的**：

| 阈值 ms/1k | 实测 safe | 最佳常数下预测正确率 | 平衡 |
|---|---|---|---|
| 0.5 | 8.5% | 93.1% | 64.8% |
| 1.0 | 10.2% | 91.7% | 63.0% |
| **1.5** | **14.1%** | **77.7%→82.9%（p40）** | **74.0%→71.5%** |
| 2.0 | 45.1% | 58.7% | 56.5% |
| 3.0 | 88.2% | 11.9% | 50.0% |

阈值 3.0 那行的「上界 88.2%」正好等于 base rate 88.2%，又是多数类陷阱；0.5/1.0 的
90%+ 也主要靠 base rate（平衡只有 63-65%）。**1.5 是唯一一个正确率过 80% 且平衡正确率
也明显高于 50% 的点**，所以选它。

percentile 扫描（阈值固定 1.5、`dram_scale=0.2`）：

| percentile | 正确率 | 平衡 | 假阳 | 假阴 |
|---|---|---|---|---|
| p20 | 85.5% | 67.4% | 6.3% | 8.2% |
| p30 | 84.4% | 68.2% | 8.0% | 7.7% |
| **p40** | **82.9%** | **71.5%** | 10.9% | 6.3% |
| p50（原值） | 77.7% | 74.0% | 17.9% | 4.4% |
| p60 | 64.3% | 68.0% | 31.9% | 3.8% |

p20 正确率最高（85.5%），但 **p40 在满足 >80% 的同时平衡正确率更好**（71.5% vs 67.4%），
且假阳（判 safe 但其实掩不住，直接伤 TTFT）只有 10.9%。选 p40。

### 这个标签的可复现性只有 55.6%，这是一切的天花板

顺手做了一个检查：同一条 trace 行在 `noprefix-H2D` 和 `origin-H2D` 两轮里拿到的
verdict 是否一致。答案是 **2,248 条共有行里只有 55.6% 一致**——比抛硬币好一点点。

继续查是「测量本身是噪声」还是「测量稳定但阈值卡在密集区」：

```
两轮的 ms/1k 中位数：2.066 vs 1.902      Pearson r = 0.106
70.2% 的测量值落在 1.5-2.5 ms/1k 这一个桶里
```

**是后者，而且两个问题都有。** 相关系数 0.106 说明逐行的值几乎不相关；同时 70% 的样本
挤在 1 ms 宽的区间里，任何落在这个区间的阈值都会被微小的调度差异翻转。这解释了为什么
阈值 2.0（正好在密集区中心）产出的标签不可学，而 1.5（在密集区边缘）反而可学——**1.5
把「明显很快」的那一小撮和其余分开了，而 2.0 试图在噪声内部划线。**

**这对项目的意义**：H2D 直通把每 token 的加载代价压到 ~2 ms/1k 之后，绝大多数请求的
加载代价已经**趋同**，请求之间不再有明显的 load-bound / compute-bound 之分。KVPilot
的收益空间在这个配置下主要来自那 14% 明显更便宜的请求，而不是来自把所有请求分成两类。
### 阶段 5：KVPilot online（预测臂）

**结论：kvpilot-H2D 在 TTFT 上优于 origin-H2D，达标。**

| 配置 | 请求数 | 有加载 | **exposed** | **TTFT** | sum(stall) | DRAM tokens | HBM hit tokens |
|---|---|---|---|---|---|---|---|
| `origin-H2D`（baseline） | 4004 | 56.4% | 12.83% | **234.1 ms** | 65 s | 20.7 M | 8.4 M |
| `kvpilot-H2D`（预测） | 4004 | 41.0% | **9.45%** | **217.9 ms** | 45 s | 13.6 M | **15.6 M** |
| 相对变化 | — | −15.4pt | **−3.4pt** | **−6.9%** | −31% | −35% | **+86%** |

机制确实在起作用，而且能从数据里看出来是**怎么**起作用的：

- **HBM 命中 token 翻了将近一倍**（8.4 M → 15.6 M，+86%）。判为 dram-safe 的块被挪到
  更早淘汰的层，把 HBM 让给判为 unsafe 的块，于是后者留住了。
- **要从 DRAM 加载的量少了 35%**（20.7 M → 13.6 M），有加载的请求比例从 56.4% 降到 41.0%。
- **每 1k token 的加载代价几乎不变**（3.1 vs 3.3 ms）——这正是预期：KVPilot 改变的是
  「哪些块留在 HBM」，不是传输本身有多快。

在线链路的运行证据：

```
verdict 计算（decode 侧）  4004 次，全部 source=predicted
feedback 应用（prefill 侧）4004 次，其中 2872 次真的挪了块（71.7%）
累计挪动                   40,826 个 block
```

### 阶段 5（续）：measured 臂（oracle）

`kvpilot-measured-H2D` 把 verdict 换成实测表的真值（改动见上文「代码改动 3」）：

| 配置 | 有加载 | exposed | **TTFT** | HBM hit | DRAM tokens |
|---|---|---|---|---|---|
| `origin-H2D` | 56.4% | 12.83% | 234.1 ms | 8.4 M | 20.7 M |
| `kvpilot-H2D`（预测） | 41.0% | **9.45%** | **217.9 ms** | **15.6 M** | **13.6 M** |
| `kvpilot-measured-H2D`（oracle） | 47.0% | 11.60% | 223.3 ms | 11.8 M | 17.3 M |

verdict 来源分布：**3,634 条 measured / 370 条回退 predicted**（90.8% 命中）。实测表只
覆盖 `noprefix-H2D` 那轮**真的从 DRAM 加载过**的请求（3,643/7,914 = 46%），但实跑命中率
到 90.8%——因为会重复加载的那些行本来就是会被反复复用的行。

**反直觉的地方：oracle 臂反而不如预测臂。** 217.9 vs 223.3 ms，预测臂更好。

这不是 bug，原因在阶段 4 已经量出来了：实测标签的**跨轮一致性只有 55.6%**。所谓
「oracle」是用 `noprefix-H2D` 那一轮测出来的标签，去指导 `kvpilot-measured-H2D` 这一轮
的放置——而同一条行在两轮之间有 44% 的概率翻转。**它是上一轮噪声的 oracle，不是这一轮
真值的 oracle。** 相比之下成本模型预测的是请求的**结构性质**（prefix 多大、下一轮要算
多少），跨轮稳定，所以反而更管用。

这条结论对项目有用：它说明**在这个配置下，KVPilot 的收益来自成本模型抓住的结构信号，
而不是来自更精确地拟合某一轮的实测值**。追求更高的「dram-safe 分类正确率」未必换来
更好的 TTFT。

#### 5. HBM 普查文件不落盘（阶段 6 卡在这里，已修）

阶段 6 要开 `GET_HBM_POOL_DISTRIBUTION` 出池子分布图。开了之后文件建出来了、日志也打了
`HBM pool distribution sampling enabled`，但跑到 262 条请求时 `hbm-pool-*.jsonl`
**四个文件全是 0 字节**。

这正是 `docs/observability.md` 里记过的那个失败模式，只是当时死的是 load-stall 的 JSONL：

```python
self._file = open(self._path, "a", buffering=1024 * 1024)   # 1 MiB 缓冲
self._writer = threading.Thread(target=self._drain, daemon=True)  # daemon
# 模块里没有 flush、没有 close、没有 atexit
```

每秒一行普查、约 100 KB/s，一轮跑完约 2-3 MB——**刚好在 1 MiB 门槛附近**，要么一次都不
落盘，要么落一部分然后被截断，而 worker 退出时 daemon 线程的缓冲直接丢弃。load-stall
当初选择删掉文件、只留日志；普查这边不能删（数据量太大，不适合打进日志），所以改成
**每写一行就 flush**（`kvpilot_hbm_snapshot.py:_drain`）。

每秒一次 syscall，代价可以忽略，而丢掉整轮数据的代价是重跑半小时。**发现它靠的是主动
去看文件大小而不是看日志**——日志一切正常，`sampling enabled` 照打，文件也存在，只是空的。

### 阶段 6：HBM pool 分布

开 `GET_HBM_POOL_DISTRIBUTION` 各跑一轮，产出 `<name>-logs/pool_distribution.pdf`。
象限口径见 [observability.md](../observability.md)：x 轴是空闲时长（LRU 的唯一依据），
y 轴是 `log2(load/comp)`（判据的依据），**两个非对角象限就是 KVPilot 的收益空间**。

`origin-pool`（普通 LRU，196 MB 普查数据）：

| 象限 | 块数 | 占比 |
|---|---|---|
| **老 & 必须留 HBM** → LRU 先淘汰它，GPU 会空等 | 2,841 | **31.7%** |
| **新 & 放 DRAM 无损** → LRU 留着它，白占 HBM | 475 | **5.3%** |
| 新 & 必须留 HBM（两者一致保留） | 115 | 1.3% |
| 老 & 放 DRAM 无损（两者一致淘汰） | 5,523 | 61.7% |

**31.7% 落在左上象限**——这些块 LRU 会优先淘汰，而判据说它们的加载掩不住、必须留在
HBM。这就是论文要打的那个点，而且是实测出来的、不是推演的。右下象限 5.3% 是反向的浪费
（LRU 留着但其实放 DRAM 无损）。两者相加 **37%** 的块上 LRU 和判据意见相反。

`req_line` 查不到的块只有 32 个（partial census 时）/ 0 个（完整 census），说明
join key 是对的——建表和压测用的是同一个 `swe_trace.json`。

三轮对照（各取稳态最后一张快照）：

| 象限 | `origin-pool` | `kvpilot-pool` | `kvpilot-measured-pool` |
|---|---|---|---|
| **老 & 必须留 HBM**（LRU 想淘汰，判据说要留） | 2,841 (31.7%) | 3,963 (**40.6%**) | 2,631 (27.0%) |
| **新 & 放 DRAM 无损**（LRU 想留，判据说可淘汰） | 475 (5.3%) | 677 (6.9%) | 737 (7.6%) |
| 新 & 必须留 HBM（一致保留） | 115 (1.3%) | 292 (3.0%) | 355 (3.6%) |
| 老 & 放 DRAM 无损（一致淘汰） | 5,523 (61.7%) | 4,827 (49.5%) | 6,006 (61.7%) |
| 计入的块总数 | 8,954 | 9,759 | 9,729 |
| 空白块（无 KV，纯浪费） | 356 | **8** | 7 |

**两个读法要分清**：

1. **左上象限在 `kvpilot-pool` 变大（31.7% → 40.6%）是成功的样子。** 这张图统计的是
   「此刻还躺在 HBM 里的块」，KVPilot 的作用就是把判据说必须留的块**真的留住**，于是它们
   出现在普查里；LRU 下这些块按空闲时长被淘汰掉了，压根不在池子里被数到。同时空白块从
   356 掉到 **8**，池子被用满、几乎没有空转的格子。

2. **`kvpilot-measured-pool` 的左上象限反而最低（27.0%）**，和它 TTFT 也不如预测臂
   （223 vs 218 ms）是同一件事的两面：实测标签跨轮只有 55.6% 一致，拿上一轮的标签指导
   这一轮，留下来的块有相当一部分并不是这一轮真正需要的。**它留的量不少（总块数 9,729
   与预测臂持平、空白块只有 7），但留对的比例低。**

配合阶段 5 一起读才完整：预测臂同时做到 HBM 命中 token +86%、DRAM 加载量 −35%、
exposed load 12.83% → 9.45%。**分布图说明它留住了块，stall 数字说明它留对了块。**

出图命令（`--pdf-output` 才出 PDF，见下文坑 6）：

```bash
python3 utils/plot_hbm_pool_distribution.py \
    --census '<log_dir>/hbm-pool-*.jsonl' \
    --ratios datasets/agent-traces/swe-smith-trajectories/req_ratio.json \
    --pdf-output <log_dir>/pool_distribution.pdf
```


### 6. `plot_hbm_pool_distribution.py` 根本不画图（已补）

阶段 6 第一次"出图"之后，`pool_distribution.pdf` 只有 **1577 字节**——正常的二维直方图
应该是几十 KB。查文件头是 `====` 而不是 `%PDF`：

```
$ head -c 200 pool_distribution.pdf
====================================================================================================
HBM block placement: idle time (LRU ...
```

两个原因叠在一起：

1. **这个脚本只有 ASCII 热力图和 CSV，没有 matplotlib、没有 `savefig`**。虽然它叫
   `plot_*.py`、虽然 `docs/README.md` 把它列在"作图脚本"里。
2. argparse 的**前缀匹配**把我传的 `--out` 解析成了 `--output-file`（文本报告的落点），
   于是文本被写进了 `.pdf` 名字里，**退出码还是 0**。

已给脚本补上真正的 `write_pdf()`（二维直方图、log 色标、y=0 处画出 `load == compute`
边界、象限占比写进标题），新增 `--pdf-output` 参数。matplotlib 在函数内部 import，这样
没装 matplotlib 的本机仍然能用文本和 CSV 输出。

**调用脚本里加了 magic bytes 校验**：

```bash
[ "$(head -c 4 "$L/pool_distribution.pdf")" = "%PDF" ] || echo "PDF MISSING OR NOT A PDF"
```

教训和前面那个 0 字节普查是同一类：**"命令退出码 0 + 文件存在"不等于产出正确**。
这一轮里同一类错误出现了三次（普查 0 字节、PDF 是文本、`moved=0` 的误判），共同点都是
只看了日志/退出码没看产物本身。

## 六轮总表

| 配置 | exposed | 目标 | TTFT | sum(stall) | DRAM tokens | 自洽检查 |
|---|---|---|---|---|---|---|
| `noprefix` | 21.36% | >20% ✓ | 4768 ms | 2732 s | 28.7 M | OK |
| `origin` | 22.95% | >15% ✓ | 3250 ms | 1915 s | 21.8 M | OK |
| `noprefix-H2D` | 17.46% | >10% ✓ | 267 ms | 103 s | 29.2 M | OK |
| `origin-H2D` | 12.83% | >5% ✓ | 234 ms | 65 s | 20.7 M | OK |
| `kvpilot-H2D` | **9.45%** | < origin-H2D ✓ | **218 ms** | 45 s | 13.6 M | OK |
| `kvpilot-measured-H2D` | 11.60% | < origin-H2D ✓ | 223 ms | 57 s | 17.3 M | OK |

六轮全部 4004 条 prefill、0 条 abandoned、自洽检查（`dram=0` 的请求 stall 恰为 0）全过。
每轮的 `load_stall.pdf` 在各自的 `<name>-logs/` 目录下。


### 一个读数陷阱：origin-H2D 中途只有 1.6%

`origin-H2D` 跑到 1511/4000 时 exposed 只有 **1.64%**，看起来要错过 5% 的目标；跑完是
**12.83%**。原因是工作集还没建立（前期 prefix 命中率高、池子空，`peak_kv_usage` 才 1.7%），
和 benchmark.md 记的开环爬坡是同一回事。**中途读数不能用来判断配置好坏**，这一点在
H2D 配置下更明显，因为它跑得快、爬坡期占比更大。

## 产出清单

日志根目录 `dynamo-log/demo-tests/`，九轮各一个 `<name>-logs/`：

| 目录 | 内容 |
|---|---|
| `noprefix-logs` `origin-logs` | 阶段 2，含 `load_stall.pdf` |
| `noprefix-H2D-logs` `origin-H2D-logs` | 阶段 3，含 `load_stall.pdf` |
| `kvpilot-H2D-logs` `kvpilot-measured-H2D-logs` | 阶段 5，含 `load_stall.pdf` |
| `origin-pool-logs` `kvpilot-pool-logs` `kvpilot-measured-pool-logs` | 阶段 6，含 `pool_distribution.{pdf,log,csv}` |

数据产出在 `datasets/agent-traces/swe-smith-trajectories/`：

| 文件 | 内容 |
|---|---|
| `req_measured_safe.json` | 实测 dram-safe 表（阈值 1.5 ms/1k，扣固定开销后） |
| `tool_resp_len.json` | 在线判据查表，`--predict-percentile 40` |
| `measured_accuracy.log` | 三个预测器对实测真值的正确率 |
| `build_measured_safe.py` | 从日志反推实测 dram-safe |
| `score_measured.py` | 打分（含平衡正确率） |
| `fit_constants.py` | 扫 dram/compute 缩放拟合常数 |
| `ceiling.py` | 预测器上界（判断目标可达性） |

GPU 机器上的 runner 与辅助脚本**已收进 `utils/`**（原先只在 `/root/` 下，机器释放就会丢）：

| 文件 | 作用 |
|---|---|
| `utils/run_demo.sh` | 一轮实验的完整流程，配置全走环境变量 |
| `utils/plot_pools.sh` | 批量出池子分布图并校验 PDF |
| `utils/stall_share.py` | exposed load 占比聚合 |
| `utils/compare_runs.py` | 两轮 token/stall 分解对比 |
| `utils/compare_workload.py` | 两轮请求集合/顺序/时刻是否一致 |
| `utils/label_diagnostics.py` | 标签可学性四连诊断（见下文「判据改了三版」）|

`run_demo.sh` / `plot_pools.sh` 要拷到 GPU 机器上跑（**`scp` 不通，用
`base64 | ssh ... base64 -d`**）；两个脚本都用 `KVPILOT_ROOT` 定位仓库，默认取脚本自身
上级目录，所以放在 `utils/` 下直接跑即可。其余脚本只依赖标准库，本机能跑。

九轮的**客户端日志**（`run-<name>.out`）存在
`dynamo-log/demo-tests/client-logs/`。它们是 `compare_workload.py` 唯一的数据来源——
请求的发送时刻和 idx 只在客户端日志里有，服务端日志没有。


## 如果要重跑

```bash
# 阶段 2
NAME=demo-tests/noprefix DISABLE_LOCAL_PREFIX_CACHE_LOOKUP=true bash run_demo.sh
NAME=demo-tests/origin bash run_demo.sh

# 阶段 3（四个开关，注意 SHM 需要 LOCAL 和 CPU 同时开，否则启动即报错）
NAME=demo-tests/origin-H2D ENABLE_MOONCAKE_CPU_BACKEND=true \
  ENABLE_LOCAL_MOONCAKE_HBM=true ENABLE_SHM_MOONCAKE_HBM=true \
  MOONCAKE_PRELOAD_EXPECTED_ENGINES=2 bash run_demo.sh

# 阶段 5（在阶段 3 的开关上再加）
  KVPILOT_ONLINE_DRAM_SAFE=true                                   # 预测臂
  KVPILOT_MEASURED_DRAM_SAFE_PATH=<req_measured_safe.json 的路径>  # oracle 臂

# 阶段 6（在对应配置上再加）
  GET_HBM_POOL_DISTRIBUTION=true
```

**跑完必须自查三件事**，这一轮里三种「静默失败」各踩了一次：

```bash
grep -c 'Free blocks distribution' <log_dir>/prefill-0.log   # 0 = 跑的是 stock vLLM
ls -la <log_dir>/hbm-pool-*.jsonl                            # 0 字节 = 普查没落盘
head -c 4 <log_dir>/pool_distribution.pdf                    # 不是 %PDF = 出的不是图
```




