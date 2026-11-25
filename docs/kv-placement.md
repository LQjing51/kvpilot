# KV placement 的实现

概念背景见 [concepts.md](concepts.md)。

## 策略

`KVPILOT_PLACEMENT` 选一个。prefill engine、LMCache、mooncake_master 都读它，取值不在下表里就启动报错：

| `KVPILOT_PLACEMENT` | 写 | HBM 淘汰 | DRAM 淘汰（Mooncake master） | SSD 命中 |
|---|---|---|---|---|
| 空（LRU 基线） | write-through：prefill 逐层写 DRAM，DRAM 全量 offload 到 SSD | 上游空闲队列（FIFO） | 按 chunk 组整组删内存副本，最后释放它的请求旧的先走 | load-through 进 DRAM |
| `belady` | 同 LRU | 下次被用到最晚的块先走（不再用到的最先），下次用到的行号从回放的 trace 离线算 | 同样按下次用到的行号，整组删内存副本 | 同 LRU |
| `inclusive-safe-reqwise` | 同 LRU | 按下一轮的 safe tier 分档：没 hash 的空块最先，没有下一轮 1 < SSD 2 < DRAM 3 < HBM 4 = 判据未到 4；档内按释放顺序（FIFO） | 同样分档（判据未到算 HBM），档内按最后释放它的请求（旧的先），整组删内存副本 | 同 LRU |
| `inclusive-penalty-reqwise` | 同 LRU | 按请求的 HBM → DRAM penalty 从小到大：没有下一轮 0，判据未到排最后；同值按释放顺序 | 按请求的 DRAM → SSD penalty 从小到大（没收到判据的算 ∞），整组删内存副本 | 同 LRU |
| `exclusive-penalty-reqwise` | write-back + direct move：HBM / DRAM 里每块只存一份，放在下一轮需要的那一层；SSD 副本写过就留着（块提到上层也不删）；没有下一轮不写 | 判据未到的块 pin 住、不进空闲队列；留在 HBM 的块排在空闲队列最后，新请求放不下时按 penalty 写到 DRAM 再复用 | 只放判据 DRAM 的组；超水位时按 penalty 下沉，已有 SSD 副本的只删内存副本，其余写到 SSD | 不 load-through（会多一份） |
| `exclusive-penalty-reqwise-unblocking` | 同 exclusive | 同 exclusive，但新请求不等下沉：空闲块（含一个 prefill chunk 的预留，每次判据和分配后在后台补）不够时，从 held 块里借给要加载的 prefix，旧 KV 在这次加载里先写到 DRAM | 同 exclusive（水位 0.9 留的约 218K token 空闲够借块写回） | 同 exclusive |
| `exclusive-penalty-blockwise-unblocking` | 同 exclusive，但按块放：一轮的 prompt 头部 HBM、中间 DRAM、尾部 SSD，块数由时间模型定 | 同 unblocking，按块 penalty（下沉一层使下一轮 TTFT 期望增加多少），每个请求只能踢它最后一个 HBM 块 | 同 exclusive，组 penalty 换成按块的 | 同 exclusive |

**所有策略在 HBM 和 DRAM 都从请求尾部往前踢**，所以一轮的 prompt 留下的总是 HBM 一段、DRAM 一段、SSD 一段，命中按
HBM → DRAM → SSD 连续读，不会出现 DRAM 中间夹着 SSD 的情况。见下文「从尾部往前踢」。

**判据（signal）**：每轮 decode 结束时，D 侧算出同一 session 下一轮的 `{tier, time: [t_hbm, t_dram, t_ssd], suffix}`，
回传给服务这一轮的 P。`tier` 是下一轮的 request-safe 层（`None` = 没有下一轮），`time` 是下一轮的 prefix
整段从 HBM / DRAM / SSD 读时的 prefill e2e（ms），`suffix` 是下一轮 suffix 长度的分布 `[[概率, token 数], ...]`
（查表时是一个点 `[[1.0, next_suffix]]`，表里没有就是 None）。

- **除 LRU 外的五个策略**（`kvpilot_placement.MODEL`）都只从 `tier` 看有没有下一轮，tier 和 penalty 都在 P 上由
  `suffix` 和时间模型（`KVPILOT_TIME_COEFFS_PATH`）重算，按请求和按块用同一套；各策略只是拿它们做的事不同
  （inclusive-safe 用 tier 定档，inclusive-penalty 用 penalty 排序，exclusive 用 tier 放置、用 penalty 下沉）：
  - tier：在 suffix 分布的 95% 分位点上（满足 P(S ≥ s) ≥ 95% 的最大 s，不插值）safe 的最低层，safe 即露出 / 计算 < 5%；
  - penalty：下沉一层使下一轮 prefill 增加的期望（对整个 suffix 分布取期望）。按请求时（`kvpilot_blockwise.reqwise`）
    就是整段 prefix 在 HBM / DRAM / SSD 的期望时间 [E t_hbm, E t_dram, E t_ssd] 之差（`kvpilot_placement.penalties`）：
    HBM → DRAM = `max(0, E t_dram − E t_hbm)`，DRAM → SSD = `max(0, E t_ssd − E t_dram)`。按块见下文 blockwise。
  - 2026-11-22 之前 inclusive-safe 和 reqwise exclusive 直接用 D 回传的 `tier` 和 `time`：oracle 时是下一轮实测的 tier
    和三层时间，在线预测时 tier 按期望时间（不是 95% 分位）判。exclusive 的 P stats `model_<D 的 tier>_to_<模型的 tier>`
    记两者不一致的次数。

## 开关

都在 `dynamo-dev/examples/backends/vllm/launch/disagg_lmcache.sh` 里 export，启动时检查组合：

| 变量 | 默认 | 作用 |
|---|---|---|
| `KVPILOT_PLACEMENT` | 空 | 见上表 |
| `KVPILOT_ONLINE_PREDICT` | `0` | 判据来源：`0` 查表，`1` 在线预测。只在设了策略时用 |
| `KVPILOT_OFFLINE_TABLE` | 无 | `0` 时必填：判据表 `utils/placement_ab/signal_<trace>.json`（格式见下文「A/B」），按请求 id 开头 8 位的回放行号取 `next_tier` / `next_time`。查不到的行直接报错 |
| `KVPILOT_TOOL_RESP_DISTRIBUTION_PATH` | 无 | `1` 时必填：`paper-figure/algorithm/resp_length_dist_<dataset>.json`。按本轮的 tool call 组合取 tool 响应长度分布；没见过的组合用全部 label 的混合分布。两个数据集的 label 不重叠，用错文件时几乎每轮都落到混合分布，D 侧日志的 `label_misses` 会接近 `labels_seen` |
| `KVPILOT_TIME_COEFFS_PATH` | 无 | 除 LRU 外都必填（`1` 时 D 侧也用）：时间模型（`kvpilot_time_model.py`）的系数，`paper-figure/Figure3-request-safe/fit_times2_swe.json`（`fit_times2.py`）。旧格式（`fit_times_swe.json`、`safe_boundary_coeffs.json`）启动报错。在线预测对响应长度分布取期望得三档 time，按系数里的 `thresh` 定 tier；P 用它算 tier 和 penalty，并核对系数里的 `chunk` 等于 `max_num_batched_tokens` |
| `KVPILOT_SSD_OFFLOAD_PATH` | 空 | Mooncake SSD offload 目录（空 = 关），见下文「L4」 |
| `KVPILOT_SSD_READ_GIBPS` | 空 | SSD loader 读带宽限速（模拟慢盘） |
| `KVPILOT_LOAD_STALL` / `GET_HBM_POOL_DISTRIBUTION` | `true` / `false` | 观测，见 [observability.md](observability.md) |
| `MOONCAKE_EVICTION_HIGH_WATERMARK` | `0.9` | `dynamo_script.sh` 传给 mooncake_master，所有策略同一个默认值（每 rank 100 GiB、TP2 时 0.9 留约 218K token 空闲） |

- `MOONCAKE_PRELOAD_EXPECTED_ENGINES` 由 `NUM_PREFILL_WORKERS` 推出。
- 以前是开关、现在固定的：prelookup（`_lookup_views`）、Mooncake 零拷贝三件套（CPU backend / local HBM / shm
  segment）、SSD 命中 load-through（exclusive 下代码里绕开）、Mooncake offload 心跳 100 ms、SSD loader 64 个 I/O
  线程 / 256 MiB 在途槽位 / 8 个拷贝线程，以及 exclusive 下沉时 offload 的超时 120 s 和在途上限 8192（`master_service.h`
  里的常数，两个都来自真实卡死，见下文）。
- 判据表和分布表在 worker 启动时读，换 trace 要重启服务。

## 文件分工

| 位置 | 文件 | 作用 |
|---|---|---|
| vLLM `vllm/v1/core/` | `kvpilot_placement.py` | 策略名、档位 `CLASS`、`penalties(time)` |
| | `kvpilot_claims.py` | inclusive 和各 exclusive 共用的 session head 与认领（blockwise 按块记层） |
| | `kvpilot_inclusive.py` | 两个 inclusive：判据到达后 rekey HBM 空闲队列（档位或 penalty）、设 Mooncake 组档位和 penalty |
| | `kvpilot_exclusive.py` | exclusive：pin、判据到达后发 job、准入失败时下沉；unblocking 的预留（`reserve`）和借块（`lend`）；blockwise 的按块放置、HBM 尾块候选（`_plan_held` / `_commit_held`） |
| | `kvpilot_time_model.py` | 时间模型：prefix 拆在三层时的 prefill 时间（在线预测、blockwise、拟合脚本共用这一份） |
| | `kvpilot_states.py` | 混合模型：每个 session 最新一轮结尾的线性注意力状态留在 HBM（`StatePins`），见下文「混合模型」 |
| | `kvpilot_blockwise.py` | 由时间模型和 suffix 分布定层和 penalty：按块（`place`、`hbm_penalty` / `dram_penalties`），按请求（`reqwise`） |
| | `kvpilot_lru.py` | LRU：不放置，只把释放顺序发给 master |
| | `kvpilot_signal.py` | D 侧判据：查表 / 在线预测、请求 id 解析 |
| | `kv_cache_utils.py` | `FreeKVCacheClassQueue`：按 (档位, 释放序号) 的索引堆（`csrc/kvpilot_heap.cpp`，首次使用时 JIT 编译，失败退回 Python）；exclusive 的 held 块（`hold`，块上 `kvp_held`）排在最后、不计入空闲；`lend` 把借出的 held 块放到最前 |
| | `single_type_kv_cache_manager.py` | 准入时 prefix 命中 held 块不占空闲额度（`_get_num_evictable_blocks`） |
| | `block_pool.py` | 设了策略时换成档位队列 |
| | `kv_cache_manager.py` | 建策略对象；`free()` 之后 `track`；`allocate_slots` 失败时记下差多少块（`kvpilot_shortfall`）；`kvpilot_admission_blocks`（准入要的块数，unblocking 据此算借多少） |
| | `sched/scheduler.py` | unblocking 分配前先 `lend`、准入后 `keep_reserve`；分配失败时 `demote`；`kvpilot_apply_placement`；exclusive job 完成（`kvx-N`）；记每个请求等 HBM 的时间（`blocked=`） |
| | `kvpilot_hbm_snapshot.py` / `kvpilot_load_stall.py` | 观测 |
| dynamo | `components/src/dynamo/vllm/kvpilot.py` | D 侧 reporter |
| | `components/src/dynamo/vllm/handlers.py` | P 侧 `kvpilot_feedback` endpoint |
| LMCache | `lmcache/v1/kvpilot_placement.py` | 策略、下发组档位（`set_groups`）、释放顺序（`set_order`） |
| | `lmcache/integration/vllm/vllm_v1_adapter.py` | 释放顺序经 connector metadata（`kvx_order`）带到 rank 0 worker，后台线程发给 master |
| | `lmcache/v1/kvpilot_exclusive.py` | exclusive 的 worker 侧写 / 删；unblocking 的 `Evictor`（借出块的 PutStart、PutEnd） |
| | `lmcache/v1/gpu_connector/gpu_connectors.py` | `start_evict`：在 evict stream 上把借出块的旧 KV 拷到 DRAM；`start_layers(_ssd)` 的 `evict_events` |
| | `csrc/mem_kernels.cu`、`ssd_loader.cu` | C++ `LayerEvictor`（逐层 gather + 批量 D2H + event）；两个 loader 每层 scatter 前等对应 event |
| | `lmcache/v1/lookup_client/lmcache_lookup_client.py` | 记每个请求的 chunk hash；把组档位转发到 rank 0 lookup server |
| | `lmcache/v1/cache_engine.py` | prelookup、load-through（代码和日志里仍叫 writeback） |
| Mooncake | `mooncake-store/src/master_service.cpp` | 读 `KVPILOT_PLACEMENT`；组状态、`SetGroupPlacement`、`KvpBatchEvict`、exclusive 的 offload 与删除 |
| | `storage_backend.cpp`、`file_storage.cpp`、`real_client.cpp` | `batch_remove_local_disk`：按 key 删本机 SSD 上的数据；`batch_offload_local`：把调用方主机内存里的值直接写成一个 bucket 并登记磁盘副本 |

## 判据链路

```
P 释放请求的块  KVCacheManager.free → {Lru,Inclusive,Exclusive}Placement.track
                （整块 prompt；LMCache chunk hash 用 lookup 时算过的）
                → connector.kvpilot_group_order：释放顺序，见「从尾部往前踢」
decode 结束    dynamo kvpilot.py KvpilotReporter.report → kvpilot_signal 查表 / 预测
  → client.direct → P handler kvpilot_feedback → AsyncLLM → EngineCore
  → Scheduler.kvpilot_apply_placement(request_id, {tier, time}) → placement.apply
     inclusive: FreeKVCacheClassQueue.rekey
                + connector.kvpilot_set_group_placement（PdConnector → LMCacheConnectorV1
                → LMCacheLookupClient → lookup 通道 → rank 0 lookup server → RemoteBackend
                → MooncakestoreConnector → store.batch_set_group_placement → master SetGroupPlacement）
     exclusive: 发 job，见下文
```

- 只管**整块**的 prompt 块（block = LMCache chunk，512；混合模型 1536，见下文「混合模型」）：prompt 不足一块的请求
  P 不 track，D 也不回传。
- 回放保证同一 session 的下一轮在上一轮返回后才发（`mooncake_trace_replay.py` 总是这样），所以判据一定在释放之后、
  下一轮之前到达。到达时找不到等判据的 head 就直接报错。
- NIXL 预热请求（`discover_dynamo_workers.py`）的 id 第一段是 `ffffffff`，判据 tier 为 None。其余请求 id 第一段
  必须是 8 位行号，否则报错。
- 日志：D `KVPilot signal remote_request_id=... tier=... time=[...]`（在线模式带 `label_misses=a/b`）；
  P `KVPilot placement applied remote_request_id=... {...}`；启动时 P / D 各一行 `KVPilot placement=<策略|lru>`。

## Session head 与认领（inclusive、exclusive 共用）

`kvpilot_claims.py`。一个请求释放块时成为它 session 的 head，按下一轮需要的层认领它的整块 prompt（判据未到算
HBM）。下一轮的 prompt 包含上一轮，所以新释放的请求里出现了某个 head 的最后一块，这个 head 就被取代、放弃认领。
多个 session 共享的块（SWE 98 个 session 共用的 system prompt 块、43 个两两共享的块）放在认领它的 head 里最快的
那一层（`required`）。

## inclusive-safe-reqwise

写和 LRU 一样，三层各一份；判据只改淘汰顺序（tier 由时间模型算，见上文「判据」）。

- HBM：块释放时进 `FreeKVCacheClassQueue` 的 pending 档（4），没 hash 的块 −∞。判据到了，这个 head 还在空闲队列
  里的块 rekey 到 `CLASS[required]`。同档按入队序号出队（释放顺序，即上游 FIFO；改档不改序号）。
- DRAM：同一次判据把这些 chunk 组的 tier 发给 master（penalty 全 0）。master 按 (组 tier, 释放顺序) 淘汰，
  没收到过判据的组算 HBM。读（`ExistKey` / `GetReplicaList`）只刷新组的访问时间，不清判据：读时清判据会让刚读过的
  整条 prompt 在新判据到达前（r5 实测平均晚 3 s、约 10 个 prefill）都落进最高档，SWE 上 DRAM-safe 请求的 DRAM
  命中因此从 71% 掉到 57%。

## belady

reuse-probability 类淘汰（LRU 等）要逼近的上界 Belady MIN：和 LRU 只差淘汰算法，不用判据（D 侧不发）。

- 下次用到的行号（`kvpilot_belady.next_uses`）：回放按 trace 文件顺序发，请求 id 带行号；第 i 行的第 k 个整块，下次被
  后面第一个整块里有同一内容、同一前缀（hash id 的链）的行用到。P 启动时从 `KVPILOT_BELADY_TRACE`（`run_ab.sh` 设成
  回放的 trace）算出。
- 请求释放时给它的整块 prompt 设键：HBM 空闲队列 −行号（小的先走，不再用到的 −FAR）；master 组 penalty FAR − 行号，
  用 inclusive-penalty 的排序（小的先删，`FAR` = 1e9）。被下一次使用的请求释放时再按之后的使用重设。
- 两层都是请求内从尾部往前（同一请求的块下次用到的行号一般相同，共享的头部块用得更早，本来就后走）。

## inclusive-penalty-reqwise

写和 inclusive-safe 一样，三层各一份；淘汰顺序从档位换成 penalty（tier 和 penalty 怎么算见上文「判据」）：

- HBM：块释放时进空闲队列，键是「判据未到」（`PENDING_PENALTY` = 1e8，高于任何 penalty）。判据到了，还在空闲队列里的
  块 rekey 到这条请求的 HBM → DRAM penalty（ms）；没有下一轮的 0。同值按释放顺序。
- DRAM：同一次判据把这些 chunk 组的 DRAM → SSD penalty 发给 master，master 按 (penalty, 释放顺序) 整组删内存副本，
  没收到过判据的组算 ∞。
- 共享块取认领它的 session 里最大的 penalty；只要还有一个认领者的判据没到，就保持「判据未到」（HBM 键和组 penalty
  都是 1e8）。

## 从尾部往前踢（所有策略）

一轮的 prompt 在每一层都从尾部往前被踢，留下的是 HBM 一段前缀、DRAM 一段、SSD 一段，命中连续。

- **HBM**：块按尾部先的顺序释放进空闲队列，同档 / 同 penalty 时按释放序号出队，所以 LRU、inclusive、exclusive 本来就
  尾部先走。blockwise 的 HBM 块 penalty 头大尾小，但不靠这个保证顺序，而是每个请求只把它最后一个 HBM 块当候选（见下文）。
- **DRAM**：P 每释放一个请求，把它整块 prompt 的 chunk 按顺序发给 master（`kvpilot_group_order`：组 → (释放序号,
  在 prompt 里的位置)，经 `SetGroupPlacement` 的 tier 码 4，penalty 位放 `序号 × 65536 + 位置`）。被后面的请求再次
  释放的 chunk（同 session 的下一轮、跨 session 共享的块）归到最新的那个请求。master 淘汰时按请求分桶，每个请求只有
  它在内存里的最后一个组参与比较，比较键 `(档位, penalty, 释放序号)`：踢走一组后同一请求的前一组顶上，直到这一轮的
  目标量。还没收到顺序的组自成一个请求，排在所有有顺序的组之后（按访问时间）。
- 释放顺序在 P 的 scheduler 里攒着，随下一步的 connector metadata 带到 rank 0 worker，在后台线程发给 master，不占 forward。
- 单测：`master_service_test.cpp` 的 `KvpEvictionTakesRequestTailFirst`。

## exclusive-penalty-reqwise

HBM / DRAM 里每个 block 只存一份，放在下一轮需要的那一层（tier 和 penalty 由时间模型算，见上文「判据」）。SSD
副本写过就一直留着：块提到 DRAM / HBM 时不删，再回到 SSD 时不用重写（SSD 不缺空间）。

| 事件 | 动作 |
|---|---|
| prefill 结束 | 什么都不往下写（LMCache 跳过 save），块 pin 在 HBM、移出空闲队列，等判据 |
| 判据 HBM | 留在 HBM；之前某轮留在 DRAM 的副本删掉（SSD 副本留着）；块回到空闲队列，作为 held 块排在最后（见下） |
| 判据 DRAM | 不在 DRAM 的块从 HBM 写过去（有 SSD 副本也照写，SSD 副本留着），写完删 HBM 副本；已在 DRAM 的直接删 HBM 副本 |
| 判据 SSD | 有 SSD 副本的块删掉 DRAM / HBM 副本，不写；没有的从 HBM 写到 SSD（DRAM 副本先删），写完删 HBM 副本 |
| 没有下一轮 | 什么都不写。DRAM 副本留着（组 penalty 置 0，DRAM 满时最先下沉），HBM 副本直接释放 |

复用后整段 prompt 都在 HBM（加载进来的 + 新算的），新判据按上表处理：同层只写新增的块；换层时 DRAM 和 HBM 之间
整段搬，回到 SSD 的块只有没写过 SSD 的要写。2026-11-25 之前换层会删 SSD 副本、没有下一轮会把只在 HBM 的块写到
SSD：Eval-EXP1 的 SWE 上 Qwen3-30B blockwise 写 SSD 22294 块，一半是离开 SSD 后又写回来的，比 write-through
（16201 块）还多；req unblocking 写 SSD 的块 37% 是没有下一轮时写的。这些写和别的请求的 SSD 读抢盘，撞上的混合
请求每条慢 30–38 ms（`paper-figure/Eval-EXP1/README.md`）。改后验证（2026-11-25）：vLLM `test_kvpilot_exclusive.py`、
Mooncake `KvpExclusive*` 单测、`check_excl_writer.py` 都过；Qwen3-30B blockwise 冒烟（Eval-EXP1 SWE trace 前 600 条）
601 次判据写 DRAM 7728 块、SSD 4751 块、删 DRAM 副本 2627 块、没有下一轮直接释放 251 块，594 个 job 无 skipped /
failed / no_room，留着 SSD 副本的块写 DRAM 全部分到了内存副本。

- **direct move**：判据一到就写，写完就从 HBM 删，把空闲块留给后面的请求，而不是等被淘汰时才写。写都在后台：job
  经 connector metadata 带给每个 TP rank，worker 的 `ExclusiveWriter` 在后台线程和独立 stream 上做。写的块被 job
  pin 住，两个 rank 都回报 `kvx-N` 完成后才 unpin、删 HBM 副本。没有请求时，scheduler 因为有 job 未完成会继续空跑 step。
- **写 SSD 不经 Mooncake 的 DRAM**：写 DRAM 是在本 rank 的 Mooncake segment 里分配内存副本、逐层批量 D2H 写进去（见
  下文「Mooncake master」的 HBM→DRAM 写）。写 SSD
  则用 writer 自己的 pinned 暂存区（每次最多 8 个 chunk 的全部层，Qwen3-30B-A3B TP2 下每 rank 192 MiB）：从 HBM
  gather 进来，每个 chunk 写成一个 bucket（`batch_offload_local`），写完 master 直接登记磁盘副本，不建内存副本。
  去 SSD 的数据如果先放进 DRAM 中转，会占着 DRAM、把同样多的 DRAM 层数据挤到 SSD。代价是写盘在 writer 线程里同步
  完成，job 里有 SSD 写时完成得更慢，块 pin 得更久。
- **一个空闲队列**：分配器复用和下沉都操作 `FreeKVCacheClassQueue`。判据未到（pending）、正被请求使用、正被写到
  下层的块都不在队列里，所以 pending 的块不会被淘汰。队列里依次是：空块（−∞）、没被追踪的块（可直接覆盖）、
  held 块（判据已到、要留在 HBM 的块，是这份 KV 唯一的副本，按 HBM → DRAM penalty 从小到大排在最后）。held 块
  不计入空闲块数，分配器不会直接拿走它。一个块还有 pending 的 head 认领时一直 pin 着，最后一个认领它的 head
  的判据到了才放回队列。
- **HBM 满（阻塞下沉）**：等待队列里的请求 `allocate_slots` 失败时，scheduler 按差的块数（`kvpilot_shortfall`，减去
  已经在下沉的块）从 held 块里取 penalty 最小的（同 penalty 尾块先，`kvp_free_seq`），写到 DRAM（组 penalty 设成
  DRAM → SSD penalty），写完变成空块再分配。这期间请求留在等待队列里；held 块写完还不够，就等后面的判据到达。
  请求自己 prefix 命中的 held 块不下沉（它马上要用）。
- **准入不多算 held 命中**：上游准入把 prefix 命中里 `ref_cnt == 0` 的块算作要占的空闲块（touch 会把它移出空闲
  队列），但 held 块本来就不计入空闲，所以命中 held 块不占额度（`kvp_held`）。2026-11-21 之前没有这条：HBM-safe 的
  请求回来时，自己 held 在 HBM 的整段 prefix 都被算进缺口，HBM 一满就被多挡、还会把自己的 prefix 下沉掉；那之前的
  exclusive 实验 HBM 都没满过，不受影响。
- **写到一半下一轮来了**：一个块正被写到下层时，同 session 的下一轮命中它、释放后重新认领（pending）。写完时它
  保持 pin，不删 HBM 副本，等新判据到了按上表处理（它在下层已经有一份）。
- **下一轮等还在写的尾段**（2026-11-25）：判据一到，下层已有的中段马上删 HBM 副本，尾段排队写下层。下一轮在写完前
  来时，HBM 的前缀匹配停在中段之前，LMCache 接着读到中段为止，尾段 HBM 里有、下层还没有，以前只能重算：Eval-EXP1
  SWE 上 Qwen3-30B blockwise 56 条（2%）平均重算 6318 token，每条比 belady 慢 863 ms，把其余请求快的 18 ms 抵光。
  现在 `KVCacheManager._kvpilot_pending_hit` 找出 HBM 前缀之后、有 job 正在往下写的最远一块（`ExclusivePlacement.writing`，
  按 job 的 `stored` 计数），记到 `request.kvpilot_pending_hit`；adapter 的 lookup 没够到它就和混合模型一样先不调度、
  下一步再查，最多 30 s 报错。第一版只看「HBM 前缀之后 HBM 里还有」，req unblocking 第 2354 条等满 30 s 报错：held 块
  前面的块被借走或下沉后，held 块隔着缺口留在 HBM，它不会被写下去，等不到。
  日志 `KVPilot exclusive wait req=... KV ends at ..., HBM holds a block at ...` / `KVPilot exclusive waited req=... ms`。
  混合模型本来就靠 `kvpilot_state_hit` 等（见「混合模型（Qwen3.6-35B-A3B）」的「命中」），没有重算过。单测
  `test_pending_hit_reaches_a_block_being_written`。修后 Qwen3-30B blockwise（2800 条）等了 51 次，中位 109 ms、最长
  1.2 s，重算的请求从 56 条降到 5 条，TTFT 321.8 → 309.4 ms，从和 belady 打平变成快 11.8 ± 4.0 ms。
- **没人需要的 held 块 penalty 是 0**（2026-11-25）：`_hbm_penalty` / `_dram_penalty` 是需要它在更快层的 head 的最大
  penalty；SWE 各 session 共用开头的 system prompt 块，它的 HBM head 被替换后可能一个这样的 head 都没有，reqwise 分支
  原来对空序列取 `max` 直接抛异常，P 挂掉（Eval-EXP1 SWE Qwen3-30B req unblocking 第 1222 条，`lend` 里）。现在按定义取 0
  （挪走它不暴露任何加载），blockwise 分支本来就是这样。
- **DRAM 满（master 下沉）**：master 不删只有内存副本的对象。超过淘汰水位时按组 penalty 从小到大（没收到过判据算 ∞）
  下沉：已有 SSD 副本的直接删内存副本，其余 offload 到 SSD、落盘后删内存副本；组都改成 SSD。worker 写 DRAM 遇到没
  空间就一直等着重试，所以 DRAM 也满时，HBM 的下沉要等 DRAM 先腾出来才能完成。
  - 在途的 offload 计入本轮目标；在途已经够了就跳过全表扫描。扫一遍 50 万个对象要轮流拿所有 shard 锁，耗时秒级，
    会饿死 client 的 offload RPC（2026-11-13 x3 卡死）。在途上限 8192 个 value（约 4 GiB）。
  - 在途 offload 超过 120 s 没回报就当作丢了：放掉 pin，对象下一轮重新下沉。client 遇到磁盘索引里已有的 key、或
    一个 bucket 查询失败时，既不写也不通知 master；上游的超时清理只在全表扫描里跑，扫描被跳过就永远不清，这些任务
    pin 住的内存副本占满 DRAM 后整个服务卡死（2026-11-13 x4：在途 10 万个、DRAM 100%、P 停在第 4315 条）。
- **离开 DRAM 只删内存副本**：job 的 `drop` 是要删 DRAM 副本的块，worker 调 `batch_remove_memory_replicas`（master
  `EvictDiskReplica` 的 `ReplicaType::MEMORY`，复用 `BatchEvictDiskReplica` RPC），只删完成且 `refcnt == 0` 的内存
  副本，磁盘副本留着。读者或在途 offload pin 着的内存副本不删，多留一份，直到 DRAM 下沉把它删掉。SSD 副本 placement
  不再删（之前换层时 `Remove` 对象、再删本机 bucket）。
- **LMCache lookup 要知道 HBM 已覆盖多少**：prelookup 要求从第 0 个 chunk 起每个 chunk 在 DRAM 或 SSD 有副本
  （write-through 下天然成立）。单副本下 HBM 里的块下层没有，所以 exclusive 时 adapter 把 vLLM 的 HBM 命中长度放进
  request_configs（`lmcache.kvpilot.hbm_tokens`，不进 cache key），`_lookup_views` 对 HBM 覆盖的 chunk 不找副本。
- **只在 prefill 上生效**：`KVPILOT_PLACEMENT` 传给所有进程，decode 的 scheduler 也会建策略对象。decode 的
  connector 是裸 NixlConnector，没有 `kvpilot_chunk_hashes`，scheduler 就关掉 placement，decode 日志一行
  `KVPilot placement off: connector has no LMCache`。以前 decode 也 pin 块、永远等不到判据，KV 漏满、请求卡在
  Waiting（2026-11-13 toolagent 第 170 条）。
- 日志：P `KVPilot exclusive heads=... jobs_in_flight=... pinned=... held=... demoting=... lower DRAM/SSD=... ssd_copies=... free=... stats={...}`
  （随判据打，最多 30 s 一行；`stats` 里 `hbm_freed_no_next` 是没有下一轮直接释放的块）；每个判据一行
  `KVPilot placement applied ... {'tier', 'blocks', 'write_dram', 'write_ssd', 'drop', 'hbm_released', 'hbm_freed', 'job'}`；
  每个 job 每个 rank 一行
  `KVPilot exclusive job=... drop=ok/总数 dram=N ssd=N chunks written=... skipped=... retries=... no_room=...`
  （`chunks written` 是 DRAM 和 SSD 加起来的块数，SSD 写量看 `write_ssd`）；
  master `[KVPILOT-EXCL] offloaded=... memory_replicas_dropped=... spilled=...`。

## exclusive-penalty-reqwise-unblocking

和 exclusive 一样放置、搬迁，只是新请求不再等 HBM 下沉（上面的「阻塞下沉」）。每一步给要算的 token 最多分配
一个 prefill chunk（`max_num_batched_tokens` / block size，现在 8192 / 512 = 16 块），要加载的 prefix 在准入那一步
全部分配。vLLM 默认准入时要求整条 prompt 放得下（`scheduler_reserve_full_isl`），unblocking 关掉这一条：准入只要求
「要加载的 prefix + 第一个 chunk」放得下。块从后往前分：

- **先用空闲块，不够的才借**：分配前（`lend`，所需块数按「prefix + 第一个 chunk」算，和准入检查同口径）空闲块
  （含预留）够就直接分配；缺的块从 held 块里按 demote 的顺序借，跳过请求自己的 HBM 命中、和下层旧副本还在被删的块
  （删 job 在后台线程上，晚于这次写会把新副本删掉）。借出的块立即 uncache、记成 DRAM，在空闲队列里排最前
  （`FreeKVCacheClassQueue.lend`，seq 取极小值），所以先分配的 external 块（要加载的 prefix）正好拿到它们。
- **借来的块只能落在 prefix 上**：prefix 的加载每层 scatter 前等这层的写回，suffix 的 KV 由 forward 直接写、不等写回。
  所以缺的块数超过要加载的块数（预留还没补回来、suffix 分不到），或 held 块不够，就一块不借，退回阻塞下沉（stats
  `lend_no_fit`）。2026-11-22 之前另有一条「prefix 只用预留以上的空闲块」，没借到块时会发出 0 块的借块记录，
  worker 的 `LayerEvictor` 报错（sweblock 上 HBM 600 块时出现）；预留本来就在后台补，这条限制没有必要，已去掉。
- **预留**：一个 prefill chunk 的空闲块，保证 suffix 不用等。每个判据处理完、每次分配（准入和后续 chunk）之后，空闲块
  （加上在下沉的）不足 16 块时，按 demote 的顺序把差额 held 块在后台写到 DRAM（`keep_reserve`，stats `reserve_demote`），
  和当前 chunk 的 forward 并行。
- **长 prompt 的后续 chunk** 每一步再分配，用后台补回来的空闲块。分不到时不走 vLLM 的抢占（只有这一个请求在跑，
  抢占会丢掉它算好的部分、从头重算；释放块时 exclusive 还会把它当成结束的请求 pin 成一个等不到判据的 head），
  而是这一步先不调度它、也不准入新请求，等补回来下一步再算（`chunk_short`，stats `chunk_wait`）。
- 第一版（2026-11-21 下面的真机结果）只在分配失败时才借：空闲块够时 prefix 直接从空闲池拿，平均每次准入加载 17 块，
  比预留还多，一次准入就把预留吃光；预留又只在判据到达时补，下一条请求连 2–5 块的 suffix 都分不到，被挡的 15 条里
  14 条是这样。
- **借出块的旧 KV 写到 DRAM**：scheduler 经 connector（`kvpilot_exclusive_evict`）把 {chunk hash, block id, DRAM penalty}
  记在这个请求上，`build_connector_meta` 挂到它的 `ReqMeta.kvpilot_evict`。worker 在 `start_load_kv` 里：
  1. `Evictor.prepare`（forward 线程）：`batch_put_start_local` 给每个（块, 层）在本 rank 的 segment 分配内存副本。
     已经存在的值跳过（master 不删正在 offload 的对象，会留下一份）；分不到空间直接报错，DRAM 留了缓冲。
  2. `start_evict`（加载开始前）：C++ `LayerEvictor` 在 GPU connector 的 evict stream 上把每层都排好：gather 借出块
     的 slot 到暂存区（两块，按层交替）、一次 `cudaMemcpyBatchAsync` 写进副本、记一个 event。
  3. 加载照常开始（DRAM `LayerwiseLoader` 或 `SsdLayerwiseLoader`），每层 H2D 进自己的暂存区后、scatter 进 paged KV
     之前等这一层的 evict event。D2H 和 H2D 走 PCIe 两个方向同时跑，只有 scatter 可能要等。forward 的第 l 层等
     加载的第 l 层，所以也排在 evict 之后。
  4. `Evictor.finish`（单独线程）：等最后一层拷完，先把组设成 DRAM 和它的 DRAM → SSD penalty（PutEnd 时组是 SSD
     会被推去 offload），再 `PutEnd`。
- **代价**（`utils/ssd_load_check/test_loader.py check-evict`，单卡、无计算，两个 loader 都逐字节核对过）：32K token
  的 DRAM 加载 31.3 ms；借出其中 16 / 32 / 64 块（1/4、1/2、全部）时 32.9 / 33.5 / 37.8 ms。串行的话全部借出要
  多一倍。prepare 的 PutStart 在 forward 线程上，日志里看 `put_start_ms`。
- **DRAM 空闲**：借块的写不能等 DRAM 往 SSD 下沉，要求 DRAM 留够空闲（约 100K token）。默认水位 0.9 留约 218K token，
  够用，所以和其他策略一样用 0.9。第一版曾单独留 1M token（水位 0.5422），能放 DRAM-safe 块的 DRAM 从约 3839 块降到
  约 2312 块，结果见下。
- 日志：P 启动一行 `KVPilot exclusive unblocking: reserve=16 blocks`；`KVPilot exclusive heads=...` 的 stats 多
  `lend_reqs` / `lent` / `reserve_demote` / `lend_no_fit` / `chunk_wait`；每次借块每个 rank 一行
  `KVPilot exclusive evict req=... blocks=N values=N skipped=N put_start_ms=... copied_ms=... failed=N`。

所有策略的 P 侧 `KVPilot prefill` 行末尾都有 `blocked=`：请求第一次分配失败到被准入的时间（不在 e2e 里，在 TTFT 里）。

**真机结果**（第一版：分配失败才借、DRAM 留 1M token；2026-11-21，`swe_hbm1to2_budget220.jsonl` 6026 条，HBM-safe token 占 28%；1P1D、**HBM 1500 块**（默认
2361 块时按离线估计 HBM 不会满）、DRAM 每 rank 100 GiB、`-p 1` 首 token 放行；两臂各一次，日志 GPU 上
`dynamo-log/pab-swe1to2b-{excl,exclu}-h1500-logs`，`analyze_excl.py signal_swe_hbm1to2_budget220.json excl=... exclu=...`）：

| | exclusive（水位 0.9） | unblocking（水位 0.542） |
|---|---|---|
| 等 HBM 的请求 / 合计 | 160 条 / 5.5 s | 15 条 / 0.8 s |
| 每条等待 均值 / p50 / p90 / max | 34 / 17 / 67 / 427 ms | 54 / 22 / 199 / 246 ms |
| prefill e2e 均值 / p50 / p99 | 77.8 / 52.0 / 372.5 ms | 79.4 / 53.9 / 372.1 ms |
| TTFT（prefill 结束 − 发出）均值 / p50 / p99 | 94.1 / 66.6 / 410.3 ms | 95.4 / 68.2 / 403.2 ms |
| 露出加载 `gpu_stall` 合计 | 14.9 s | 23.8 s |
| e2e − `gpu_stall` 均值 | 75.3 ms | 75.4 ms |
| prefix 从 HBM / DRAM / SSD 读 | 33.8 / 57.1 / 9.1% | 33.6 / 51.6 / 14.8% |

- 等 HBM 少了 86%：91 条请求靠借块直接准入（共借 1124 块），判据后补预留在后台下沉 1063 块。剩下 15 条里 14 条
  suffix 只有 2–5 块，是预留被前面准入的 prefix 吃光了（见上文「第一版」）；1 条 suffix 37 块，超过一个 chunk。
- exclusive 的 HBM 从第 3 分钟起基本占满（held 1000–1200 块、等判据 200–400 块、空闲多在 0–300 块），但只有 2.7% 的
  请求等了（按每 1000 条：0 / 15 / 77 / 61 / 7 / 0 / 0）。HBM 满不等于每条都要等：判据为 DRAM / SSD / 无下一轮的请求
  （这次 4449 条）写到下层后马上释放 HBM，整轮释放了约 8.1 万块，阻塞下沉只有 2212 块；HBM-safe 的请求回来时 prefix
  在 HBM，只要新算的 suffix（p50 1 块、p90 2 块）。只有准入那一刻空闲块少于它要的块（平均 14 块）才会等。
- 借块不拖慢 forward：e2e − `gpu_stall` 两臂只差 0.16 ms。借块的 PutStart 在 forward 线程上，中位 2.3 ms、p90 5.8 ms；
  拷完 DRAM 中位 5.9 ms；182 次 evict（两个 rank）没有失败，6 个值已在 DRAM 被跳过。
**真机结果：全 HBM-safe**（第二版：分配前从后往前借块、准入后补预留，准入仍要求整条 prompt；2026-11-21，数据集 `swereqhbm`：`swe_reqprio.jsonl`
5252 条、判据全标 HBM；1P1D、**默认 HBM 2361 块**、DRAM 每 rank 100 GiB、**两臂水位都 0.9**，两臂只差借块 + 预留；
日志 `dynamo-log/pab-swereqhbm-{excl,exclu}-r1-logs`）：

| | exclusive | unblocking |
|---|---|---|
| 等 HBM 的请求 / 合计 | 3373 条（64%）/ 130.8 s | 6 条 / 0.2 s |
| TTFT 均值 / p50 / p90 / p99 | 167.3 / 140.4 / 277.6 / 690.1 ms | 148.6 / 124.5 / 251.6 / 551.8 ms（均值 −11.2%） |
| prefill e2e 均值 / p50 / p99 | 115.3 / 89.0 / 504.0 ms | 123.4 / 98.5 / 504.4 ms（+7.1%） |
| 露出加载 `gpu_stall` 均值 | 1.6 ms | 7.0 ms |
| e2e − `gpu_stall` 均值 | 113.6 ms | 116.4 ms |
| prefix 从 HBM / DRAM / SSD 读 | 56.3 / 43.7 / 0% | 55.4 / 44.6 / 0% |
| HBM → DRAM | 阻塞下沉 89,484 块 | 借块 83,823 块（2299 次准入）+ 补预留 7,594 块 |

- 两臂 DRAM 都没到 0.9，没有往 SSD 下沉。
- 阻塞几乎全去掉：每条请求平均少等 24.9 ms，TTFT −18.7 ms。剩下 6 条要新算的 token 都超过一个 chunk（8.8K–30.8K），
  当时准入还要求整条 prompt 放得下；现在的版本准入只看第一个 chunk。
- 代价是 prefill e2e +8.2 ms（不在等待里，在 forward 里）：
  - 露出加载 +5.4 ms：一次借块中位 35 块（p90 59），写出 D2H 和加载 H2D 量相当，每层 scatter 要等写出，
    拷完中位 25 ms（p90 44 ms）。
  - 其余 +2.8 ms：借块的 PutStart（每块 48 个值）在 forward 线程上，中位 8.1 ms、p90 14.1 ms，只有借块的
    44% 请求付，摊到每条约 3.5 ms。
- 借块写 DRAM 4598 次（两个 rank），0 失败、0 跳过。

上面「第一版」的结果（swe1to2b）：

- 整体 TTFT 反而 +1.3 ms（+1.4%）：多出来的全在露出加载（`gpu_stall` +1.5 ms/请求）。1M token 的 DRAM 缓冲让
  DRAM 只用到约 54%（exclusive 到 90%），DRAM-safe 块放不下、被下沉到 SSD，从 SSD 读从 9.1% 涨到 14.8%。这条 trace
  上阻塞本身只占 5.5 s（平均每请求 0.9 ms），小于 DRAM 变小的代价。

## exclusive-penalty-blockwise-unblocking

在 unblocking 上把放置和 penalty 从「整个请求一层」改成按块：一轮的 n 个整块 prompt 是下一轮的 prefix，头部 h 块放
HBM，接着 d 块 DRAM，尾部 x 块 SSD（`kvpilot_blockwise.py`，时间模型系数 `KVPILOT_TIME_COEFFS_PATH`）。

- **时间模型**（`kvpilot_time_model.py`，系数由 `paper-figure/Figure3-request-safe/fit_times2.py` 拟合）：下一轮
  suffix s、prefix p，D / X 个 token 从 DRAM / SSD 读（其余命中 HBM）时：
  - 计算：suffix 按 chunk（8192）切；一个 n token 的 chunk 在 q token 上下文后耗时
    d + b·pad(n) + a·n·(q + n/2) + e·q，n ≤ 1024 时按 cudagraph 补齐后的大小 pad(n)、用 graph 的 d / b，否则用 eager 的。
  - 加载：只有 DRAM 时 L = c_dram·D + f_dram；有 SSD 时 L = max(c_dram·(D+X) + f_dram, c_ssd·X + f_ssd)（所有值都要拷，
    SSD 部分先读）。首层 first = g_dram·D + h_dram，有 SSD 时 g_dram·D + g_ssd·X + h_ssd。
  - 组合：加载和第一个 chunk 的计算 C₀ 重叠，e2e = C₀ + first + w·softplus((L + F − C₀ − first) / w) + 后续 chunk，
    w = ω·C₀ + w0。是 max(C₀ + first, L + F) 的平滑版：离折点远时就是 max；折点附近有几层的加载晚于计算到达，
    多露出最多约 0.7·w。过渡宽度随计算时间变宽（每层越长，抖动越大）。拟合值 F = 9.7 ms、ω = 0.158、w0 = 2.3 ms。
  - 验证（都没参与拟合）：只用 `swe` 拟合，在 `swe_reqprio`、`swe_hbm1to2` 上纯计算误差中位 5–7%（旧模型 8–9%），
    SSD 3–4%；`utils/ssd_load_check` 的两批网格（DRAM+SSD 混合、头部 HBM + DRAM/SSD）e2e 误差中位 2.2–2.5%、
    p90 8–10%（旧模型 7.6–7.9% / 13–21%，整体高估约 6%）。混合加载用 max 而不是相加：有 SSD 时实测加载时间和
    max 形式差中位约 ±1%，相加高估 17–30%。剩下的误差主要在 suffix 很短（≤ 64）、prefix 很长的小请求上。
- **放置**：在 suffix 分布的 95% 分位点上 safe（取分布里满足 P(S ≥ s) ≥ 95% 的最大的 s，不插值；查表时就是那一个值），
  safe 即 (e2e − C) / C < `thresh`（常数 0.05）。先取最少的 HBM 块（其余全放 DRAM 时 safe），再在这个 HBM 数下取
  最多的 SSD 块。stats `placed_hbm/dram/ssd`。
- **块 penalty** = 这一块下沉一层后下一轮 TTFT 的期望增加（对整个 suffix 分布取期望），算的时候它后面同层的块已经
  下沉了：HBM 的最后一块是 E(d+1, x) − E(d, x)；DRAM 第 j 个（从尾数）是 E(d−j, x+j) − E(d−j+1, x+j−1)。一个请求内
  从尾往头不减。
- **HBM**：held 块不再按 penalty 排在空闲队列里（penalty 记 0），下沉和借块从候选堆 `_cand` 里挑：每个 head 只有它
  最后一个 HBM 块是候选，键是它的 HBM penalty；挑走后这个 head 的布局变成 (h−1, d+1, x)，前一块以新键入堆。共享块
  要所有在 HBM 留着它的 head 都以它为最后一块才能挑（`_hbm_tail_everywhere`）。挑走时这个 head 的 DRAM penalty
  全部重算，已经在 DRAM 的块的新 penalty 发给 master（job 的 `repen`，stats `repen`）。
- **DRAM**：penalty 是放置时（和之后 P 每次挪这个请求的 HBM 块时）按上面的公式算的静态值。master 把块沉到 SSD
  时不回报 P，这个请求的 HBM penalty 暂时偏低（按 DRAM 还没沉算的），是接受的近似。
- **共享块**：层取认领它的 head 里最快的那层，penalty 取最大。
- 判据表要有 `next_suffix`（`utils/placement_ab/add_next_suffix.py <trace> <表>`：下一轮的 input 长度减去它和这一轮
  共享的整块 × 512）；在线预测时用 tool 响应长度分布（base + 响应长度）。
- 日志：`KVPilot placement applied` 的结果多 `layout=[h, d, x]`；`KVPilot exclusive` 行多 `hbm_candidates=`，stats
  多 `placed_*` / `repen`。

**真机结果**（2026-11-22，`swereq`，默认 HBM 2361 块、DRAM 每 rank 100 GiB、水位 0.9，oracle 判据；日志
`dynamo-log/pab-swereq-exclb-tf1-logs`，对比 11-21 的 `pab-swereq-{lru,incl,excl}-rf1-logs`）。只比新算 ≤ 8192 token 的
5229 条：rf1 三组跑在计量修复之前，多 chunk 请求少记最后一个 chunk（见 [observability.md](observability.md)），
全量平均会把 blockwise 多算约 6 ms。

| | LRU | inclusive-safe | exclusive（旧：D 的实测 tier） | blockwise |
|---|---|---|---|---|
| TTFT 均值 / p50 / p90 | 183.9 / 154.1 / 337.4 | 159.9 / 131.8 / 288.0 | 143.4 / 118.8 / 248.6 | 148.6 / 123.7 / 255.0 |
| 相对 LRU | | −13.1% | −22.0% | −19.2% |
| 露出加载 `gpu_stall` 合计 | 190.3 s | 62.6 s | 23.5 s | 44.0 s |

- blockwise 比 exclusive 慢 5.2 ms（+3.6%），全在下一轮实测 HBM-safe（+5.5 ms）和 DRAM-safe（+5.9 ms）的请求上：
  HBM-safe 的 prefix 从 HBM 读的比例 99.9% → 20.9%，DRAM-safe 的从 SSD 读 1.4% → 26.8%；SSD-safe 的持平。
- 这个工作点上容量不紧：HBM 空闲块最少 1084（exclusive 771），没有下沉、借块；exclusive 下 DRAM-safe 请求也只有 1.4%
  从 SSD 读。按块往下放（每条请求都用掉「露出 < 5%」的余量，加上时间模型的误差）只付了代价，腾出的 HBM / DRAM
  没有请求要用。
- 两边还差着别的：exclusive 用的是 D 回传的实测 tier，blockwise 用时间模型；rf1 没有请求内从尾部往前踢。按块和按请求
  的单独差别要用新代码（tier 也来自时间模型）重跑 exclusive 才能分开。

## 混合模型（Qwen3.6-35B-A3B）

图解见飞书「混合注意力」（`Gb5fdK0g5oznNXxJRoUcia8mnxe`）。**线性注意力状态从不卸载，只从 HBM 命中**；KV 照旧跨
HBM / DRAM / SSD 放置、逐层加载，五个策略都是这样（2026-11-23 定的方案一）。

正确性（2026-11-23 通过，TP2 和 TP4）：`utils/hybrid_check/run_check.sh` 起 1P1D（LRU、SSD offload；`TP_SIZE`、
`SEEDS` 可设），`check_state_reuse.py` 让同一 prompt 的 KV 分别从 HBM / DRAM / SSD 复用、状态从 HBM；prefill 日志核对
来源（`hit=` HBM KV，`dram=` 外部加载）。判据是 **DRAM / SSD 复用和 HBM 命中的输出逐字一致**。和冷算（[3]）比只是
参考：TP4 下 4 个种子里有 2 个，HBM 命中（vLLM 原生前缀缓存，不经过 LMCache）和冷算在几十个字符后分叉，DRAM 复用
和 HBM 命中则始终相同。冷算分几步算到边界、复用从缓存的状态接着算，数值不逐位相同，argmax 近似并列时会翻；冷算重跑
两次是一样的。

**模型与显存**：40 层里 10 层全注意力（L3、L7、…、L39，2 个 KV 头 × 256 维），30 层 Gated DeltaNet。vLLM 的混合
KV 缓存管理把层分成 4 组：1 个注意力组、3 个线性注意力组（各 10 层）。10 个张量各由连续 4 层共用，一个块号在 10 个张量里
各占一页，同一时刻只属于一个组。页大小统一，一页要装下一层在一个 rank 上的一份状态（TP2：fp32 SSM 1 MiB + conv
24 KiB），所以块从 512 被抬到 1536 token。TP4 时状态按头切成一半（512 KiB + 12 KiB），KV 只有 2 个头、每 rank 仍是
1 个（复制），块变成 1024；每 rank 每 token 的 KV 不变（10 KiB），计算却分到 4 张卡上，加载占 prefill 的比例更大。

**启动**：`MODEL=/cfs_cloud_code/kiry/model/Qwen/Qwen3.6-35B-A3B`。`disagg_lmcache.sh` 读 config.json 的
`layer_types`，有 `linear_attention` 就：

- 块取装得下一层状态的最小 512 倍数（按 config 和 `TP_SIZE` 算：TP2 1536，TP4 1024），`LMCACHE_CHUNK_SIZE` 跟着变；
- prefill 加 `--enable-prefix-caching`：混合模型默认关前缀缓存，打开后是 mamba align 模式（Qwen3.5 / 3.6 不支持 all 模式）；
- 两边 `--language-model-only`，不再 `--disable-hybrid-kv-cache-manager`；
- `VLLM_SSM_CONV_STATE_LAYOUT=DS`（NIXL 传状态要求）；
- `MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT` 默认等于全注意力层数（10）。

回放的 `--vocab-size` 从 config 读（248320）。

**对齐模式的 checkpoint**：调度器把每步 prefill 切在块边界上。第 T 个 token 之后的状态写在这一步的最后一块（第
(T−1)//1536 块），按这块的 hash 缓存；下一步开始前拷到新的最后一块，旧块两步后释放（仍按 hash 缓存）。所以一轮 prompt
只在每步的结尾留 checkpoint，其中最后一个完整块的边界一定有；本地命中只能停在有 checkpoint 的边界上。

**命中**（`KVCacheManager._kvpilot_hybrid_hit`，所有策略含 LRU）：从右往左找 3 个线性注意力组在 HBM 里都有状态的
最远块边界 k（不超过 prompt − 1），再从头数注意力块在 HBM 里连续的长度 h（不超过 k）。h = k 是普通的 HBM 命中；h < k
时 [0, h) 的 KV 用 HBM，[h, k) 由 LMCache 加载（adapter 把 lookup 封顶在 k，`request.kvpilot_state_hit`），状态用
HBM 里 k 处那块（vLLM 在前向前拷进当前块）。LMCache 找不到 [h, k) 的全部 KV 时请求先不调度、下一步再查（connector
返回 None），最多 30 s 后报错：exclusive 下上一轮最后几块可能还在被写往下层（写完前 pin 在 HBM，但 HBM 的前缀匹配在
前面第一个不在 HBM 的块就停），同 session 的下一轮紧跟着来时接不上状态（2026-11-24 Eval-EXP1 Qwen3.8 第 670 条，
当时直接报错、P 挂掉）。等待时要清掉 lookup client 按请求缓存的结果（`clear_lookup_status`），否则下一步拿到的
还是旧的命中数：第一版没清，写出 6 ms 后就完成了，仍等满 30 s 报错。等的时间算在这条请求的 TTFT 里；日志
`KVPilot hybrid wait req=...` / `waited req=... ms`（SWE 3000 条里 3 次，9–260 ms）。只有缺口里有 job 正在往下写
（exclusive，`request.kvpilot_pending_hit` 超过 KV 的末端）才等；否则下层是真丢了，不等：adapter 把
`request.kvpilot_state_cap` 设成 KV 的末端、下一步再查，vLLM 只在它之前找状态，后面重算，日志
`KVPilot hybrid cap req=...`。2026-11-25 Eval-EXP1 TraceLab Qwen3.8 LRU 第 298 条：Mooncake 丢了 TP0 的一块
inclusive 副本（另三个 rank 都在），等满 30 s 报错、P 挂掉，之后改成这样。HBM 没有状态就什么都不复用。vLLM 现有的分配代码
（注意力组 h 块、线性注意力组 k 块的命中）不用改就能分对块。`cache_use_control` 不含 1 但含 2 / 4 时只禁 HBM 的 KV，
状态照样从 HBM；连 2 / 4 也不含（[3]）则不复用，即冷算。

**LMCache**：只搬注意力组，10 层 KV 照旧逐层存取，`num_layers` = 10（`get_num_layers_by_block_type(..., "attention")`）。
connector 实现 `SupportsHMA`，`request_finished_all_groups` 只交注意力组的块；adapter 只登记注意力层的 KV，按层号排序。
**单 KV 头的 HND 缓存要按 NHD 读**（`gpu_connector/utils.py`）：TP2 每 rank 只有 1 个 KV 头，vLLM 的 KV 张量
`(块, 2, 64, 1, 256)` 是逻辑 NHD 顺序，`is_contiguous` 忽略长度 1 的头那一维，所以不会被重排；照 vLLM 给的 HND 提示读，
会读成 64 个头、每块 1 个 token，slot s 落在第 s 个 kernel 块。2026-11-23 之前这样存取的 KV 全放错了位置，slot
超过 83,640 时非法访问（「GPU 上的现象」见 [pitfalls.md](pitfalls.md)）。
**存数据用的 GPU buffer 按 `max_model_len` 开**（`gpu_connectors.py` `max_buffer_tokens`，adapter 设）：上游按一层
整个 HBM 缓存的 token 数开，不在 vLLM 的显存预算里，Qwen3.6 TP4 是 6.75 GiB，第几个请求时就 CUDA OOM。prefill 侧
`--max-num-seqs 1`，一次只存一条 prompt。

**状态留在 HBM**（`kvpilot_states.StatePins`）：知道 session 还有下一轮的策略，持有它最新一轮 prompt 结尾的状态
（每个线性注意力组 1 块；引用计数，不在空闲队列里），直到下一轮释放时取代它（新 prompt 包含这个边界），或判据说
没有下一轮：

- 取：请求结束时 vLLM 在把块表交给 connector 前会释放除运行中状态以外的状态块（D 侧只要运行中的那块），所以
  `scheduler._connector_finished` 先调 `KVCacheManager.kvpilot_take_end_state` 拿一个引用，释放（`free()`）时交给策略。
- belady：这个 prompt 以后还会被用到才持有。inclusive / exclusive：持有到判据为「没有下一轮」或被取代。持有的状态不算
  空闲块：每个活跃 session 3 块（每 rank 45 MiB）。
- LRU：A/B 里和 belady 一样，这个 prompt 以后还会被用到才持有（`run_ab.sh` 在混合模型上把回放的 trace 经
  `KVPILOT_BELADY_TRACE` 交给 LRU，P 启动日志 `KVPilot LRU end states held by trace: <路径>`），所以五个策略留下的
  状态相同，只差 KV 放在哪一层。没给 trace（request-safe 测量）就不持有，结尾状态在这个请求的 KV 之后释放，按释放
  顺序淘汰时 KV 先走、状态后走。2026-11-23 晚之前 A/B 里的 LRU 也不持有：压力一大就把状态逐出、整段重算，模拟中
  TTFT 是其余策略的 4–8 倍，那是状态保留的差别，不算放置收益。更早 LRU 持有到被取代，request-safe 测量里每个测试
  都是死胡同，状态越积越多，约 670 行后 HBM 被占满、请求分不到块。
- 没被持有的状态（prefill 中途、被取代的、预热请求的）释放后在按键排序的空闲队列里排在最前（`_STATE` = −2e9，
  只在没 hash 的块之后）；LRU 用上游队列，按释放顺序。
- exclusive / blockwise 只搬 KV（后台写、借块写出、直写 SSD 都不变），HBM 留前缀头部、其余放 DRAM / SSD 也能命中。

**P 只算 prompt 的前 N−1 个 token**：P/D 下 NIXL 把线性注意力模型的最后一个 token 留给 D
（`_truncate_mamba_request_for_prefill`），所以一轮留下的块和状态到 (N−1) 向下取整到 1536 的边界为止。判据表的
`next_suffix` 按这个算（`add_next_suffix.py --drops-last`）；request-safe 的 warm 请求发 p + 1 个 token，状态才落在 p。

**时间模型**：prefill 的切步按对齐模式（`kvpilot_time_model.compute_features(align=1536)`：非最后一步向下取整到
1536 的倍数，并在最后一个完整块边界切一刀，10,000 token 冷算是 7680 + 1536 + 784）。系数由
`paper-figure/Figure3-request-safe` 用同一套 request-safe 测法在 Qwen3.6 上测（`make_trace.py --model qwen3.6`，prefix
按 1536 对齐），`fit_times2.py --block 1536 --align` 拟合，文件里带 `align`，P 启动时核对。

**块大小**：D 侧阈值和在线预测按 engine 的块大小；Belady 的 `next_uses` 每 1536 token 取 3 个 trace hash id；判据表
`signal_<trace>_b<块>.json`，tier 和三档时间按**这个模型自己**的：新 trace 用时间模型算（`make_model_table.py <trace>
<系数> <输出> --block <块> --drops-last`，和 Qwen3-30B 的 k160 表同一做法，不用 request-safe 实测）；k160 的两张是
request-safe 实测（`make_req_table.py <trace> raw/qwen3.6/<ds> ...`，TP4 用 `raw/qwen3.6-tp4/<ds>`，再
`add_next_suffix.py --block <块> --drops-last`）。k160 上 TP4 时间模型算的 tier 与实测一致 84%，不一致的主要是实测
DRAM、模型 HBM（123 / 1102 行），即模型偏保守。2026-11-23 之前的 `_b1536` 表只重算了 next_suffix，tier 和时间
沿用了 Qwen3-30B 的（Qwen3-30B 下一轮多为 DRAM，Qwen3.6 多为 SSD），没用它跑过 A/B。`run_ab.sh` 看 `MODEL` 是混合模型就按 `TP_SIZE` 和是不是 FP8 权重换判据表和系数（TP2：`_b1536`、
`fit_times2_qwen3.6_k160.json`；TP4：`_b1024`、`fit_times2_qwen3.6-tp4_k160.json`；FP8 TP2：`_q36f8`、
`fit_times2_qwen3.6-fp8_k160.json`，开专家并行测的；Qwen3.8：`_q38`、`fit_times2_qwen3.8-ep_k160.json`）。模拟器（`datasets/block-priority-trace/sim_placement.py --model`）按 `PROFILES`
里每个模型的块、HBM 块数、每块字节数和时间模型模拟；混合模型的状态每个 3 块、只在 HBM，持有规则同上。prefill 之外
的开销默认用 Qwen3-30B 拟合的那套（exclusive 的接纳前开销每千命中 token 少约 0.4 ms），Qwen3.6 FP8 和 Qwen3.8 用各自
实测的 `pre`（和放置无关）；exclusive 不 write-through、prefill 比时间模型快的部分是 `excl_save`。两项都是 2026-11-25
从 Eval-EXP1 第二轮 SWE 的实测拟合的，之前的模拟高估了混合模型上 exclusive 的收益（见 `paper-figure/Eval-EXP1/README.md`）。
`search_trace_models.py` 在同一条候选 trace 上把几个模型都模拟一遍。

**Qwen3.6 TP2 上放置几乎没有收益**（2026-11-23，模拟器 + 时间模型）：每 rank 每 token 的 KV 只有 10 KiB，H20 上
prefill 又不比 Qwen3-30B 短，按层加载基本被计算盖住。swe_full 的复用轮次，prefix 全从 SSD 比全在 HBM 平均只慢
7.3%（Qwen3-30B 是 48%），全从 DRAM 只慢 0.5%；状态也卸载（约 31 MiB/rank，相当于 3.1K token 的 KV）只到 8.9%，块
缩到 576 也只到 9.8%。默认容量下 30 条候选 trace 里，belady / incp / exclu / exclb 彼此相差不超过约 3%；压力再大时
LRU 丢状态整段重算，TTFT 是其余四个的 4–8 倍，但那四个仍然一样。所以改测 TP4（计算减半、加载不变）。

**TP4**（2026-11-23）：request-safe 已完成，4420 个请求、3315 次来源检查全部通过；TP4 时间模型和
`signal_swe_blockprio_k160_b1024.json` 已在本地产出。k160 实测上限：prefix 全从 SSD 比全在 HBM 平均慢 72%（TP2 是
8.4%），全从 DRAM 只慢 0.6%；每 rank 计算降到 TP2 的约 0.7 倍，SSD 每 1K token 3.37 ms，是 TP2 的两倍（4 个 rank
共用 SSD）。但默认 HBM 6912 块（708 万 token）装得下整条 k160，五个策略模拟几乎相同；要 LRU 读 SSD 得让活跃 prompt
超过 DRAM（944 万 token），所以 A/B 改用各模型共用的 `sweblock512`（`swe_blockprio_k512.jsonl`，见
[block-priority-trace README](../datasets/block-priority-trace/README.md) 末节），尚无真机结果。详见
[Figure3 README](../paper-figure/Figure3-request-safe/README.md)「Qwen3.6 TP4」。

### Qwen3.8-Flash-Next（TP 4）

模型从 vLLM 0.30（PR #53896 合入点）移植到 0.23（`vllm/models/qwen4_exp/`，V1 runner，无 MTP），只带跑它缺的部分。
48 层：36 层 Gated DeltaNet，12 层 QSA（稀疏注意力：indexer 4 个头 × 128 维、1 个 key 头，每 4 个 token 压成一个
key，打分后选 2048 个 token；主注意力 24 个 Q 头、2 个 KV 头 × 256，TP4 下每 rank 1 个，复制）；第 2 层有 PLE
（3-gram 哈希嵌入加短卷积，TP 间复制）；512 个专家 top-10，FP8 权重（每 rank 44.4 GiB）。块 1024。

**缓存与放置**（每 rank）：

| 缓存 | 页 | 放置 |
|---|---|---|
| QSA 主 KV（12 层） | 每块 1 MiB | 和其他模型的 KV 一样跨 HBM / DRAM / SSD |
| indexer 压缩 key（12 层，每 4 token 一行 128 维） | 每块 64 KiB，与主 KV 同一块号 | 作为「侧页」附在同层主 KV 的 value 后面，一起存取 |
| indexer raw-key 环形缓冲（12 层，4 行） | 每请求 1 块，1 KiB | 不参与前缀缓存；块边界上它是空的（1024 是 4 的倍数），命中用不到 |
| GDN 状态（3 组 × 12 层）、PLE 短卷积状态（1 层） | 补齐到 1 MiB，与主 KV 共用张量 | 只在 HBM，同 Qwen3.6；每个 session 持有 4 块 |

- **LMCache 侧页**：每层的 value = 该 chunk 的 token-major KV（1024 × 1 KiB）+ 该块的压缩 key 页（64 KiB）。
  chunk = 块，所以一个 chunk 正好一个块号，侧页在显存里连续。存储时主 KV 照旧 gather + D2H，侧页从压缩 key 张量的
  块页直接拷进副本（`batched_memcpy_async`）；DRAM / SSD 两个 C++ loader 在每层那一批 `cudaMemcpyBatchAsync` 里
  顺带把侧页直接写回块页，在这层的 event 之前；有借块时侧页排在等待写出事件之后（否则会先覆盖还没写出的旧页）。
  exclusive 的写出（`ExclusiveWriter`）和借块写出（`LayerEvictor`）同样带侧页。`value_bytes()` 替代各处的
  `chunk × token_bytes`。adapter 按层号把主 KV 和压缩 key 配对（`set_side_caches`）。
- **逐层 hook**：QSA 是自定义 op，0.23 的 `maybe_transfer_kv_layer` 挂在它调用的内部函数上（op 本身的 schema 要从
  它模块里的字符串注解推断，不能套别的模块的装饰器）。
- **NIXL（P→D）**：0.23 的 NIXL 要求各张量页大小相同。Qwen3.8 有 1 MiB / 64 KiB / 1 KiB 三种，加了只在它身上生效的
  分支（`worker.py` 的 `_is_qsa`）：按张量注册各自的页长；主 KV 组读主 KV 和压缩 key 的 region，环形缓冲组只读环形
  缓冲的 region；GDN 用原有的三段读，只读主 KV 那 12 个 region；PLE 读整页状态。只支持 P、D 同为 TP 4。上游 0.30
  的这部分改动叠在 0.30 重构过的 NIXL 和打包分配上，不能直接 cherry-pick。
- **调度**：同 Qwen3.6（P 少算最后一个 token、状态从 HBM 命中、`_kvpilot_hybrid_hit`），状态组是 3 个 GDN 组加 PLE 组，
  环形缓冲组算作 scratch。异步调度自动关闭（PLE 的 n-gram 上下文，见 [pitfalls.md](pitfalls.md)）。开专家并行
  （2026-11-25 起；之前不开，见 [pitfalls.md](pitfalls.md)「FP8 模型开专家并行」），时间模型 `fit_times2_qwen3.8-ep_k160.json`。

**正确性**（2026-11-24，`utils/hybrid_check/run_check.sh`，`ROOT=/root/q38 TP_SIZE=4 MODEL=<Qwen3.8> SEEDS="0 1"`）：
DRAM / SSD 复用与 HBM 命中的输出逐字一致（两个种子全部）；P 日志核对来源（`dram=6144`、SSD 的 `ssd_values=72` =
12 层 × 6 chunk）。和冷算有分叉，现象同 Qwen3.6（argmax 近似并列）。单机还用 needle（6.7K token 前埋的数字，冷算
和前缀命中都答对）和「prompt + 输出整体 prefill，看每个生成位置的 argmax」检查过 decode 路径（`tools/kvpilot/qwen38_smoke.py`）。

**request-safe 与时间模型**：见 [Figure3 README](../paper-figure/Figure3-request-safe/README.md)「Qwen3.8」。prefill
很慢（每千 token 约 85 ms，Qwen3.6 TP4 的数倍：0.23 上没有融合 kernel），加载大多被盖住：prefix 全从 SSD 只比
全在 HBM 慢 12%。

## Mooncake master

- 组 = LMCache key（`model@ws@worker@chunk_hash@dtype@layer`）的 chunk-hash 字段，即一个 chunk 的所有层、所有 TP
  rank；不是这个格式的 key 自成一组。组状态 `{tier, penalty, access_ms, placed, order_seq, order_pos}`，由
  `SetGroupPlacement(groups, tiers, penalties)` 写入（tier 码 0 无 / 1 SSD / 2 DRAM / 3 HBM，4 = 释放顺序，
  penalty ≥ 0，否则报错）。
- **所有策略都按组整组淘汰**：`BatchEvict` 直接调 `KvpBatchEvict`，上游按 lease 的淘汰删了。每轮目标 = 有内存副本的
  对象数 × `eviction_ratio`（0.1）；按全部对象数算的话，只剩 SSD 副本的对象比内存里的还多，每轮会踢掉约两倍。
  每个请求从尾部往前（见「从尾部往前踢」），各请求的尾组之间的排序键：
  - LRU：释放顺序
  - inclusive-safe：(组 tier, 释放顺序)
  - inclusive-penalty：(组 penalty, 释放顺序)，没收到判据的算 ∞；删内存副本
  - exclusive：(组 penalty, 释放顺序)；挑没在 offload 的对象，已有磁盘副本的删内存副本，其余下沉（offload）
- **读保护用 refcnt pin**：`GetReplicaList` pin 住内存副本，client 拷完 / 零拷贝用完后 `BatchUnpinMemoryReplicas`，
  `refcnt > 0` 的不淘汰。lease 照发，只管 `Remove` 和 client 拷贝的软超时，淘汰不看它；soft pin 也不看。
  `ExistKey` / `GetReplicaList` 刷新组的访问时间。
- **PutStart**：只剩磁盘副本的对象追加一个内存副本（SSD 命中 load-through；exclusive 下是留着 SSD 副本的块写 DRAM）。**PutEnd**：
  没有磁盘副本的才进 offload 队列，exclusive 下只有 SSD 组进。exclusive 判据为 SSD 的块由 worker 直写 SSD、不走
  PutEnd，所以 exclusive 下进 offload 队列的只有 DRAM 满时的下沉。
- **HBM→DRAM 写（所有策略）**：layerwise store（LRU / inclusive 的 write-through）和 exclusive 写 DRAM 都是：先
  `batch_put_start_local` 在本 rank 的 segment 里给每个（chunk, 层）分配内存副本，每层 gather 到 GPU 暂存区后，一次
  `cudaMemcpyBatchAsync`（c_ops `batched_memcpy_d2h`）直接写进各副本（segment 做过 `cudaHostRegister`），拷完再
  `put_end`（store 在后台线程，不占 forward）。不经 LMCache 的主机缓冲区，也没有主机侧拷贝：以前 write-through 是
  D2H 进 LMCache 缓冲区再由 Mooncake 的 memcpy 线程池拷进副本，exclusive 是 D2H 进暂存区再单线程 memmove，单线程
  memmove 只有约 12.5 GB/s，写 DRAM 因此比加载慢约 3 倍（`utils/hbm_dram_bw/bench.py`）。副本只分配在本 rank 的
  segment，分不到（满了或落到别处）的值不写。
- **磁盘副本直接登记**：`NotifyOffloadSuccess` 遇到 master 不认识的 key 会新建对象、只挂磁盘副本，直写 SSD 靠的就是这个。
- 日志 `[KVPILOT-EVICT] used_ratio=... objects=... mem_objects=... candidates=... target=... evicted=... freed_bytes=...
  spilled=... offloading=... scans_skipped=... offloads_expired=...`；启动一行 `KVPilot placement=<策略|lru>`。
- 单测 `tests/master_service_test.cpp`：`KvpAddMemoryReplicaToDiskOnlyObject`、`KvpExclusivePlacement`、
  `KvpExclusiveSpillInsteadOfEvict`、`KvpExclusiveRespillAfterTierChange`、`KvpExclusiveDropsDramCopyKeepsSsd`、
  `KvpExclusiveDemotionDropsDramCopyOfSsdChunk`、`KvpEvictionTakesRequestTailFirst`、`KvpInclusivePenaltyDropsSmallestPenaltyTail`。上游测 lease / soft pin 淘汰顺序的用例
  （`EvictObject`、`TryEvictLeasedObject`、`SoftPin*`、`HardPinWithSoftPinEvictionOrder` 等）描述的是已删掉的淘汰，
  预期会失败。

## A/B

`bash utils/placement_ab/run_ab.sh <lru|belady|incl|incp|excl|exclu|exclb> <数据集> [tag]`（`incp` = inclusive-penalty，`exclu` = unblocking，`exclb` = blockwise；
除 `lru` 外各臂的系数默认 `paper-figure/Figure3-request-safe/fit_times2_swe.json`），几臂连着跑用
`run_series.sh lru:swereq:<tag> incl:swereq:<tag> excl:swereq:<tag>`。1P1D，HBM 2361 块、DRAM 每 rank 100 GiB、
SSD offload，回放 `mooncake_trace_replay.py -p 1 --release-at-first-token --forced-output`：P 上一次只有一个 prefill，
TTFT 不含排队。各臂只差 `KVPILOT_PLACEMENT`；判据默认查判据表 `signal_<trace 文件名>.json`，`KVPILOT_ONLINE_PREDICT=1` 时在线预测。

- 数据集 `sweblock`：`datasets/block-priority-trace/swe_blockprio.jsonl`（`gen_trace_capacity.py`），HBM 默认 600 块；blockwise 放置下
  HBM、DRAM 需求也分别是容量的 1.4、1.2 倍，用来测 exclusive / blockwise 省容量的收益，见该目录 README。
- 数据集 `sweblock160`：`datasets/block-priority-trace/swe_blockprio_k160.jsonl`（`search_trace.py` 用模拟器挑的），默认 HBM，
  1300 条，160 个 session 同时轮转、开头的 160 个从随机一轮加入，前 160 条预热。每臂回放约 12 分钟。

  **真机结果**（2026-11-22，`r1`，第 160 条以后 1140 条，配对差的 95% 区间按 50 条一批的批均值算）：

  | | TTFT 均值 | 相对 LRU | 模拟 | prefill e2e | prefix 从 HBM / DRAM / SSD |
  |---|---|---|---|---|---|
  | LRU | 259.8 ms | — | 259.8 ms | 228.1 ms | 1.5 / 0.0 / 98.5% |
  | belady | 216.6 ms | −16.6% | 212.9 ms | 182.9 ms | 35.1 / 19.7 / 45.3% |
  | inclusive-penalty | 208.8 ms | −19.6% | 207.7 ms | 175.4 ms | 23.5 / 24.4 / 52.2% |
  | unblocking | 190.8 ms | −26.5% | 199.2 ms | 160.7 ms | 14.7 / 49.4 / 35.9% |
  | blockwise | 187.2 ms | −28.0% | 188.0 ms | 156.3 ms | 6.9 / 48.8 / 44.3% |

  - 五臂都没有等 HBM 的阻塞，P 上也不排队（`-p 1`）。
  - 配对差（真机 / 模拟）：blockwise 比 unblocking 快 3.7 ± 2.4 / 11.3 ms，比 inclusive-penalty 快 21.7 ± 2.2 / 19.8 ms，
    比 belady 快 29.4 ± 10.6 / 25.0 ms；unblocking 比 inclusive-penalty 快 18.0 ± 2.6 / 8.5 ms，比 belady 快 25.8 ± 9.8 /
    13.7 ms；inclusive-penalty 比 belady 快 7.8 ± 10.7 / 5.2 ms，分不开（模拟也预测分不开）。
  - 表里的「模拟」是挑 trace 时的预测，偏差集中在 unblocking（慢 8.4 ms）。两处原因已改进模拟器（`sim_placement.py`）：
    exclusive 的 DRAM 下沉照搬了 inclusive 的「一次删一成」，真机 master 每轮最多下沉 85 块、DRAM 贴着水位，模拟少装约
    100 块、多读 SSD；准入前开销 inclusive 的 HBM / DRAM 命中每千 token 比 exclusive 多约 0.4 ms，原来没分。改后五臂
    TTFT 和真机差在 ±3.3 ms 内（unblocking +3.0，blockwise −3.3）。剩下的：exclusive writer 写 SSD 时同时从 SSD 加载的
    请求慢 6–8 ms，blockwise 撞上写盘更多（62% 对 31%），模拟没有这一项，blockwise 对 unblocking 模拟 9.9 ms、真机 3.7 ms。
- 数据集 `swereqhbm`：trace 同 `swereq`，判据表 `signal_swe_reqprio_allhbm.json`（`make_allhbm_table.py`）把每一轮的下一轮
  都标成 HBM（保留实测 t1 / t2 / t4 算 penalty），用来压 HBM 分配。真实逐轮都 HBM-safe 的 session 构造不出来：SWE 全集
  只有 9.6% 的轮次 HBM-safe，连续 HBM-safe 最多 5 轮；HBM-safe 要 suffix ≲ 128–256 token，而 prefix 按 512 token 整块
  命中、最后不满的块每轮重算，上下文一增长 suffix 平均就过 256。全标 HBM 后，默认 HBM 下 84% 的准入时活跃 KV 超过
  2361 块，超出最多约 1851 块，放得进 0.9 水位的 DRAM。

- `analyze.py <日志目录...> --table <判据表>`：TTFT、KV 从哪层来、按请求自己的分档统计错放；
  `analyze_excl.py <判据表> LRU=<目录> excl=<目录>`。
- **TTFT 取 prefill 结束**：P 的 `KVPilot prefill` 行末尾 `done=`（`time.time()`）减 client `[ok ]` 行的 `sent=`，
  同一台机器同一个时钟。client 收到首 token 的 TTFT 还包含 D 从 Mooncake 拉整段 prompt KV（78→188 ms，随 prompt
  长度涨，各臂逐请求相同），只作参考。
- **判据表** `utils/placement_ab/signal_<trace 文件名>.json`：按回放行号索引，每行的字段：

  | 字段 | 含义 | 谁用 |
  |---|---|---|
  | `own_tier` | 这条请求自己实测的 request-safe 层 | 分析脚本统计错放 |
  | `next_tier` | 同一 session 下一轮实测的 request-safe 层（没有下一轮为 null） | 判据的 `tier` |
  | `next_time` | 下一轮实测的 prefill e2e [t1, t2, t4]（ms，prefix 整段从 HBM / DRAM / SSD 读） | 判据的 `time` |
  | `next_suffix` | 下一轮的 suffix（token，`add_next_suffix.py` 加的；没有下一轮为 null） | 判据的 `suffix`（blockwise） |

  SWE 的两张由 `make_req_table.py` 从 request-safe 实测生成；block-priority 的 trace 用时间模型算
  （`gen_trace_capacity.py`、`make_model_table.py`，`own_tier` 是上一轮给它算的 tier）；toolagent 没有实测，`toolagent_prep.py` 按前缀推出
  session（后一个请求包含前一个请求的全部整块就是它的下一轮），分档和 t1 / t2 / t4 用 `gen_trace.py` 的 Estimator
  （SWE request-safe 实测的近邻）估计。TraceLab（`tlreq`）没有判据表：当时在 GPU 上由 request-safe 日志生成的旧表
  只有分档、没有 t1 / t2 / t4，日志没拉回本机、机器已释放；要用得重跑 TraceLab 的 request-safe
  （`paper-figure/Figure3-request-safe/raw/run_one.sh tracelab_reqprio`）再生成。
- 离线估计：`dram_miss_sim.py`（LRU 与请求粒度分档，SSD 命中全 load-through）、`excl_sim.py --trace <trace> --raw
  raw/swe_reqprio --lag 10`（exclusive）。

## 历史结果（重构前的代码）

下面的实验跑在 2026-11-19 重构之前的代码上：开关名不同，部分机制已经删掉（按块 penalty、HBM 同档 MRU、DRAM 互斥标记、
exclusive 的 TTL 和预留空闲块、SSD 重新挂回旧位置），只作记录，复现要用当时的提交。

### 按块 penalty（`KVPILOT_PLACEMENT=penalty`，已删）

两层都是**被动淘汰、主动改优先级**：KV 写入是 write-through（HBM → DRAM → SSD 各有一份），
判据出来后不搬数据，只改 block / chunk 在淘汰序里的位置。

**penalty 的含义**：把这一块踢到下一层，下一轮预计多出的 delay（ms）。D 侧在 turn 结束时算
（`vllm/v1/core/kvpilot_placement.py`，D/P 共用）：下一轮 suffix
`S = (prompt mod 512) + decode + tool_resp`，tool_resp 取该 tool type 的分布；尾部 k 块缺失时
`exposed(k,S) = max(L_m(512k) − t_hbm(p,S), floor_m(512k))`，对分布取期望得 `D_m(k)`；
每块 penalty 是 `D_m` 下凸包在该块处的斜率（从尾往头不减，尾巴先走）。
DRAM penalty `M = max(0, 斜率(D_ssd) − 斜率(D_dram))`；HBM penalty `H = 斜率(D_dram) + q·M`——
踢出 HBM 后 DRAM 那份也可能已经没了，`q = KVPILOT_HBM_SSD_WEIGHT`（默认 0.5；q=0 时模拟里
HBM 命中掉一半，收益基本消失）。系数来自 Figure3 request-safe 拟合。

**分桶**：排序用的不是 penalty 本身，而是 `log2(penalty / 0.05 ms)` 取整（低于 0.05 ms 归 0 档，
`KVPILOT_PENALTY_BUCKET_MS` / `KVPILOT_MOONCAKE_BUCKET_MS`）。精确 penalty 排序在 TraceLab 上比
LRU 还差：拟合误差级别的差别把释放顺序的信息全盖掉了。

**同档先踢最近释放的（MRU）**：agent session 在一次 tool 调用之后回来，所以释放得最早的那个请求
恰恰最快回来，LRU 正好踢掉下一个要用的前缀（interleaved 轮转下 LRU 的 HBM 命中只有 penalty 的
四成）。同档按「释放批次新→旧、批次内尾块先」出队：堆的 seq 取 `-batch·2^24 + 位置`
（`IndexedHeap.push_many_seq`）。

**DRAM 互斥**：write-through 让 HBM 里的块在 DRAM 也有一份，SWE 下 DRAM（4266 块）里最多有
2361 块是 HBM 的重复。判据到达时块还在 HBM free queue 里（`OnlinePlacement.apply` 返回的
`resident`），P 就把这块的 DRAM penalty 换成 `-(1 + HBM key)`；master 对负数不分桶，排在所有
正常 penalty 之前，HBM 越不可能踢的越先走。读不清判据，冗余标记一直留到下一个判据覆盖它。
只在判据到达那一刻判断、不跟踪之后 HBM 是否踢掉它；模拟里与实时跟踪只差 0.2 ms。

**离线模拟的结论**（`sim.py`，已删除，SWE；当时的真值系数是按已删除的 CPU 墙钟 stall 校准的，数字只作历史记录）：

| HBM / DRAM 策略 | prefill 平均 | stall |
|---|---|---|
| LRU / LRU | 128.7 ms | 109.9 s |
| penalty（LRU 同档） / penalty | 119.7 ms | 63.8 s |
| penalty（MRU 同档） / penalty | 117.3 ms | 51.2 s |
| penalty（MRU 同档） / penalty + 互斥 | 114.7 ms | 37.8 s |
| Belady（按下次使用） / penalty + 互斥 | 115.5 ms | 42.0 s |

（判据延迟 6 个请求。）

**真机结果（SWE，interleaved，5140 条，`-p 1` 首 token 放行，TTFT 取 prefill 结束，两臂各一次）**：

| 加载配置 | 时间模型 | TTFT 均值 LRU → penalty | P50 | P90 | P99 | 露出加载 `gpu_stall` 合计 | 逐请求 prefill 差 |
|---|---|---|---|---|---|---|---|
| C++ loader | `safe_boundary_coeffs.json` | 138.5 → 131.5 ms（−5.0%） | −4.2% | −7.4% | +0.7% | 58.3 → 19.1 s | −7.7 ms |
| DRAM 走 Python 逐层取数* | 该配置下的 request-safe 拟合（已删） | 152.0 → 138.6 ms（−8.8%） | | −7.9% | | 99.2 → 43.3 s | −13.4 ms |
| 同上 + `KVPILOT_SSD_READ_GIBPS=1.6`* | 同上（已删） | 228.2 → 154.6 ms（−32.2%） | −18.0% | −43.9% | −50.6% | 486.2 → 128.3 s | −73.2 ms |

\* 为放慢 DRAM 加载临时加过一个开关，让只从 DRAM 加载的请求走上游逐层 fetch 而不是 C++
`LayerwiseLoader`；该开关已从代码里删掉（DRAM 加载只保留 C++ 版），这两行只作历史记录，现在无法复现。

露出用 `gpu_stall`（计算流在加载 event 上的空闲，见 [observability.md](observability.md)）。

日志在 GPU 的 `dynamo-log/pab-swe-{lru,penalty}-{v2,py,cap}-logs`。时间模型用同一加载配置下的 request-safe
拟合（`paper-figure/Figure3-request-safe/`），经 `KVPILOT_TIME_COEFFS_PATH` 传给 P / D；`run_ab.sh` 按其
`c_ssd` 算 Mooncake 的 pending penalty。快加载下 LRU 的露出约 11 ms/请求（`gpu_stall`），收益上限约 8%；
加载越慢、藏不住的请求越多，收益越大。

模拟里拆开看（`sim.py`，已删除；Python DRAM + SSD 限速）：文档的 penalty（两层都按 penalty，同档 LRU）单独 −16%，
去掉 HBM penalty 里的 SSD 项（q=0）−25%；只把同档改 MRU −15%；当前版本 −34%。SSD 限速后
SSD 加载几乎都藏不住，q·M 对每块近乎常数，HBM penalty 66–71% 落进同一档，同档次序决定了大部分
淘汰，所以 MRU 与 q 的取值相互影响（有 MRU 时 q=0.25–0.5 最好，无 MRU 时 q=0 最好）。

几个反直觉的点：用下一轮真实 tool_resp 代替分布（oracle）反而更差；
DRAM penalty 乘剩余轮数没用；用校准后的系数算 penalty 也不比原系数好。收益来自「同档 MRU」
与「DRAM 互斥」这两个结构性改动，不来自 penalty 的精度。

### 请求粒度优先级（`KVPILOT_REQ_PRIORITY_OFFLINE`，现在的 inclusive-safe-reqwise）

走按块 penalty 的同一条链路，只换 D 侧算出来的东西：不预测、不按块算 penalty，而是按请求
id 开头的行号查表（`utils/placement_ab/make_req_table.py`），取同一 session 下一轮的实测
request-safe 分档，整个请求的所有 prompt 块共用这一档（`kvpilot_placement.REQ_CLASS_MS`）：

| 下一轮 | HBM penalty | DRAM penalty |
|---|---|---|
| 没有下一轮 | 0（和没 hash 的空 / 半满块同排最前） | 0 |
| SSD safe | 0.1 | 0.1 |
| DRAM safe | 1 | 1 |
| HBM safe | 10 | 10 |
| 判据未到（新写入、从没收到过判据） | 10（`pending_penalty`） | 10（`KVPILOT_MOONCAKE_PENDING_MS`） |

0.05 ms 分桶下 0.1 / 1 / 10 落在第 2 / 5 / 8 档，所以 HBM 和 Mooncake 都是四个队列。队列内部
保持原来的顺序：HBM `KVPILOT_PENALTY_TIE=lru`（同档先踢最早释放的，即上游 FIFO；改档不改入队顺序），
Mooncake 同档按组最近一次读或判据的时间。`KVPILOT_DRAM_EXCLUSIVE=0`。A/B（两臂都全写回 SSD 命中）：
`bash utils/placement_ab/run_series.sh reqprio:swe1to2b:<tag> lru:swe1to2b:<tag>`，
`python3 utils/placement_ab/analyze.py <两个日志目录> --table utils/placement_ab/req_swe_hbm1to2_budget220.json`
另外按每个请求自己的分档统计 KV 从哪层来（错读）。数据集 `swereq` / `tlreq` 是 `datasets/req-priority-trace/` 的两条
trace，分档表 `req_{swe,tracelab}_reqprio.json`。

**离线模拟**：`utils/placement_ab/dram_miss_sim.py --trace <trace> --raw <request-safe raw 目录>`，本机约 10 s。
按发送顺序一次一个请求回放，缓存语义同 `utils/trace_order/tier_sim.py`，另加 SSD 命中全写回 DRAM；`lru` 是基线，
`none` 是请求粒度优先级（分档取下一轮的实测 request-safe 分档），其余变体（`exclusive`、`horizonN`、`opt` 等）
见脚本开头。e2e 整段从一层读时用实测 t1 / t2 / t4，混合时用 `fit_times_swe.json` 的时间模型，露出 = e2e − t1。
同一 trace（`--raw raw/swe_hbm1to2`）：LRU 露出 78.9 s、e2e 均值 92.7 ms，请求粒度优先级 12.9 s、81.4 ms；
两臂都全写回的真机结果见下文「req-priority trace」。

**真机结果**（2026-11-10，**不写回**，`swe_hbm1to2_budget220.jsonl` 6026 条，1P1D，HBM 2361 块、DRAM 每 rank
100 GiB，`-p 1` 首 token 放行，两臂各一次，日志 `dynamo-log/pab-swe1to2b-{lru,reqprio}-r1-logs`；括号里是当时
`dram_miss_sim.py` 不写回模式的离线估计，该模式已删除）：

| | LRU | 请求粒度优先级 |
|---|---|---|
| 错放（按 prefix token，对照请求自己的分档） | 60.3%（59.4%） | 42.5%（43.7%） |
| 其中比应在层慢 | 26.1%（24.1%） | 6.2%（5.5%） |
| 其中比应在层快 | 34.2%（35.3%） | 36.3%（38.2%） |
| 应在 HBM 的 token 从 HBM 读到 | 41.8%（43.3%） | 98.5%（98.1%） |
| 从 HBM / DRAM / SSD 读 | 41.6 / 44.7 / 13.8% | 62.2 / 29.7 / 8.1% |
| 露出加载 `gpu_stall` 合计 | 29.3 s | 12.7 s（−57%） |
| prefill e2e 均值（不含排队） | 82.4 ms | 80.3 ms（−2.6%） |
| TTFT（prefill 结束 − 发出）均值 / p50 / p90 / p99 | 100.4 / 76.2 / 164.9 / 401.3 ms | 98.2 / 72.0 / 163.8 / 407.1 ms |

逐请求配对：e2e 平均 −2.2 ms，`gpu_stall` 平均 −2.7 ms，中位数差 0（多数请求本来就没有露出）。
LRU 的露出平均 4.9 ms/请求，是这个 trace 上 e2e 收益的上限（约 6%），请求粒度优先级拿回了其中约 56%。
剩下比应在层慢的主要是应在 DRAM 的 token 从 SSD 读（3.19M，LRU 4.59M）：只需 SSD 的一轮把块踢出 DRAM 后
SSD 命中不写回，见「SSD 命中 load-through 进 DRAM」。

**两臂都开写回**（LRU 全写回，请求粒度优先级只写回下一轮需要 DRAM / HBM 的请求；这份写回列表、`run_ab.sh` 的
`WRITEBACK` 开关和模拟里对应的模式都已删除，现在两臂都全写回；日志 `pab-swe1to2b-{lru,reqprio}-wb1-logs`）：

| | LRU + 写回 | 请求粒度优先级 + 写回 |
|---|---|---|
| 错放 / 其中比应在层慢 | 70.7% / 38.3% | 40.6% / 6.4% |
| 从 HBM / DRAM / SSD 读 | 42.1 / 23.3 / 34.6% | 62.2 / 27.2 / 10.5% |
| 露出加载 `gpu_stall` 合计 | 79.4 s（模拟 78.9） | 21.9 s（模拟 11.7） |
| prefill e2e 均值 | 100.1 ms | 82.3 ms（−17.8%） |
| TTFT 均值 / p50 / p90 / p99 | 118.7 / 94.9 / 197.9 / 419.0 ms | 100.9 / 74.2 / 172.1 / 410.7 ms |
| Mooncake 淘汰轮数（不写回时） | 72（9） | 28（9） |

- 差距主要来自 LRU 变差：LRU 开写回后 e2e 从 82.4 涨到 100.1 ms；请求粒度优先级从 80.3 涨到 82.3 ms，和
  不写回的 LRU 差不多。
- 写回在服务里生效：请求粒度优先级下写回过的 615 个请求，下一轮完全不读 SSD 的占 86%（同一批请求不写回时 75%）。
  LRU 下正相反，写回过的 2490 个请求下一轮完全不读 SSD 的只有 7%（不写回时 49%）：写回的块和别的块互相挤掉，
  和模拟预测的抖动一致。
- 写回本身有关键路径开销，不在 `gpu_stall` 里：和不写回的同一臂逐请求比，写回的请求 `e2e − gpu_stall` 多
  2.6 ms（请求粒度优先级）/ 19.9 ms（LRU），紧接其后的请求多 1.5 / 15.4 ms，其余请求 0.2 / 1.5 ms。来源还没定位
  （后台 put 和 forward 线程抢 GIL / CPU、forward 线程上分配几百个 buffer、DRAM 紧张时 Mooncake 分配变慢都有可能）。

**req-priority trace**（2026-11-11 / 12，两臂都全写回，配置同上）：`datasets/req-priority-trace/` 里按模型估计的收益
挑 session、轮转发送的两条 trace，分档表来自同一条 trace 的 request-safe 实测。复现、口径、日志都在
[`paper-figure/Eval-three-req-priority/`](../paper-figure/Eval-three-req-priority/)，数字以它的 `plot.py` 输出为准；「估计」是
`dram_miss_sim.py` 用实测分档的结果：

| | SWE（5252 条） |
|---|---|
| prefill e2e 均值 LRU → 请求粒度优先级 | 163.7 → 131.7 ms（−19.5%） |
| 同上，只算有 prefix 的 5198 条 | 164.4 → 132.2 ms（−19.6%；估计 167.1 → 125.3，−25.0%） |
| 露出加载 `gpu_stall` 合计 | 240.4 → 75.6 s |
| e2e − `gpu_stall` 均值 | 117.9 → 117.3 ms |
| TTFT（prefill 结束 − 发出）均值 | 193.5 → 164.1 ms（−15.2%） |
| 从 HBM / DRAM / SSD 读 | 7.6 / 29.3 / 63.1% → 22.9 / 58.3 / 18.8%（估计 8.5 / 26.0 / 65.5% → 22.4 / 65.7 / 11.9%） |
| 比应在层慢的 prefix token | 63.0% → 15.4% |

- 收益全部来自露出加载：两臂 e2e − `gpu_stall` 只差 0.6 ms，写回的关键路径开销在这条 trace 上看不出来。
- LRU 臂和估计吻合；请求粒度优先级从 SSD 读的比例比估计高（18.8% 对 11.9%），和 11-10 那次的偏差方向相同。

### 单副本（`KVPILOT_PLACEMENT=exclusive`，现在的 exclusive-penalty-reqwise）

当时和现在的差别：HBM 预留 256 个空闲块（`KVPILOT_EXCL_HBM_FREE`），不够就提前下沉，现在只在准入失败时按差额下沉；DRAM → SSD 的 penalty 是 t4 − t1，现在是 t4 − t2；淘汰水位 0.85；有 `KVPILOT_EXCL_TTL_S` 兜底没有后继的 session；回到 SSD 的 key 挂回原来的磁盘位置，现在删掉重新 offload。x2 是更早的会丢数据的版本。

**真机结果**（2026-11-13，`swe_reqprio.jsonl` 5252 条，配置同上面 req-priority trace 的 A/B：1P1D、HBM 2361 块、
DRAM 每 rank 100 GiB、`-p 1` 首 token 放行；LRU 臂全写回，单副本臂不写回；两臂各一次，日志 GPU 上
`dynamo-log/pab-swereq-{lru-x1,excl-x2}-logs`；请求粒度优先级取 11-12 的 r6，同一台机器、同一份 LRU 代码，
新跑的 LRU 和 r6 的 LRU 均值只差 0.2%）：

| | LRU | 请求粒度优先级（r6） | 单副本（x2） | 单副本，不丢数据（x5） |
|---|---|---|---|---|
| prefill e2e 均值 / p50 / p90 / p99 | 165.1 / 136.7 / 315.9 / 527.6 ms | 130.6 / 100.7 / 249.8 / 525.4 ms | 119.1 / 92.6 / 209.6 / 520.9 ms | 120.0 / 93.6 / 211.8 / 529.8 ms |
| 对 LRU（逐请求配对） | | −20.9% | **−27.9%**（4606 / 5252 条变快） | **−27.3%**（4621 / 5252 条变快） |
| 露出加载 `gpu_stall` 均值 | 47.4 ms | 13.1 ms | 4.5 ms | 4.7 ms |
| e2e − `gpu_stall` 均值 | 117.8 ms | 117.6 ms | 114.6 ms | 115.3 ms |
| TTFT（prefill 结束 − 发出）均值 | 195.5 ms | 163.4 ms | 146.0 ms（−25.3%） | 147.0 ms（−24.8%） |
| 从 HBM / DRAM / SSD 读 | 7.6 / 29.3 / 63.1% | 22.9 / 62.1 / 15.0% | 15.6 / 77.7 / 6.7%（估计 14.6 / 77.9 / 7.5） | 15.6 / 76.2 / 8.2% |
| 比应在层慢的 prefix token | 63.0% | 11.4% | 0.0% | 1.5% |
| 复用的 prefix token | 104.9M | 104.9M | 104.9M | 104.9M |
| 写到 HBM 以下（DRAM / SSD） | 373 GB / 374 GB | | 约 2.4 TB 经 DRAM / 266 GB | 写 DRAM 35113 块 / SSD 15613 块，另下沉 7.8 万个 value 到 SSD |
| Mooncake 淘汰轮数 | 219 | | 0 | 10（全是下沉，没删没落盘的副本） |

- `excl_sim.py` 的估计（判据延迟 10 个请求）是 e2e 121.1 ms、写 DRAM / SSD 35011 / 13760 块；实测 119.1 ms、
  35112 / 13761 块。DRAM 里 DRAM-safe 的块最多 3598 块（84%），Mooncake 一直没到 90% 的淘汰水位。
- 按请求自己的分档拆：HBM safe 的 750 条和请求粒度优先级一样（50.3 对 50.7 ms，LRU 108.9）；比请求粒度优先级多
  出来的收益全在 DRAM safe 的 3854 条（123.2 → 108.1 ms，`gpu_stall` 17.2 → 5.1）：DRAM 不再存 HBM 和 SSD 那几层
  的重复，DRAM-safe 的块全放得下；SSD safe 的 594 条本来就是计算 bound，三臂差不多（280–285 ms）。
- e2e − `gpu_stall` 少 3.2 ms：prefill 的 forward 不再逐层写 DRAM（write-through 每层存完要在 forward 线程上等）。
  判据触发的写在后台：每个 job 每个 rank 中位 14 ms、p90 109 ms、p99 265 ms，没有拖慢 forward。
- 代价是写得多：同层只写新块，但换层要把整段 prompt 重写一遍，SWE 相邻两轮换层很常见（SSD safe 后面 80% 跟着
  DRAM / HBM safe），经 DRAM 写的字节是 write-through 的约 6 倍（28 分钟平均约 1.5 GB/s，PCIe 远没用满）。
  SSD 反而少写 29%：只有 SSD safe 的块落盘，回到 SSD 的 key 挂回原来的磁盘位置（这次 76 万个 value）。
- 修 offload 之前那次（`pab-swereq-excl-x1-logs`）：SSD 层的内存副本一直删不掉，DRAM 涨到 95% 后 master 开始踢
  DRAM-safe 的组，第 3000 条以后 315 个请求的 prefix 被截短、重算，e2e 均值和 LRU 持平（p99 1532 ms）；出事之前的
  第 1000–3000 条已经是 −18.7%（请求粒度优先级同段 −14.3%）。
- x2 是「不丢数据」之前的版本：HBM 满时直接丢判据未到的块、DRAM 满时 master 照常淘汰没落盘的组；那次 DRAM 没到
  90% 的淘汰水位，所以没有丢。x5 是不丢数据的版本（淘汰水位 0.85）：没有下一轮的 session 留在 DRAM 的块（组
  penalty 0）超过水位后先下沉到 SSD，所以从 SSD 读的多了 1.5 个百分点，e2e 比 x2 多 0.9 ms；HBM 一直没满，没有
  HBM → DRAM 的下沉。x3 / x4 两次卡死分别是全表扫描饿死 offload、和上面说的 offload 任务泄漏。
- x5 有 5 条请求复用的 prefix 比 LRU 短（共 8192 token）。不是数据被删：上一轮最新算的块在 HBM 里等判据触发的
  写，下一轮恰好在这次写完成前 10–50 ms 内做 lookup；这些块前面的块早已离开 HBM，vLLM 的 HBM 前缀匹配在第一个
  不在 HBM 的块就停，LMCache 在 DRAM / SSD 又还找不到它们，只能重算。write-through 下块在 prefill 时就写进了
  DRAM，没有这个窗口。可以让 scheduler 在请求的块还在某个 job 里写时把它多留一个 step。

**toolagent**（2026-11-13，`datasets/mooncake-trace/toolagent_sessions.jsonl` 前 4000 条，`toolagent_prep.py` 按
前缀推出 session、分档和 t1 / t2 / t4 用 Estimator 估计；配置同上；日志 `pab-toolagent-{lru-t1,excl-t1,excl-t3}-logs`）：

| | LRU | 单副本（会丢，t1） | 单副本（不丢数据，t3） |
|---|---|---|---|
| prefill e2e 均值 / p50 / p99 | 350.8 / 64.3 / 8304 ms | 345.9 / 60.8 / 8208 ms | 338.3 / 60.6 / 8178 ms |
| 对 LRU（逐请求配对） | | −1.4% | **−3.5%**（3362 / 4000 条变快） |
| `gpu_stall` 均值 / e2e − `gpu_stall` | 5.0 / 345.8 ms | 0.7 / 345.3 ms | 1.1 / 337.2 ms |
| TTFT 均值 | 366.5 ms | 360.0 ms（−1.8%） | 352.3 ms（−3.9%） |
| 复用的 prefix token / 从 HBM / DRAM / SSD 读 | 19.1M / 65.2 / 3.1 / 31.8% | 18.2M / 58.0 / 36.4 / 5.6% | 19.1M / 55.4 / 34.1 / 10.5% |

- toolagent 基本是计算 bound：复用大半在 HBM，LRU 露出的加载只有 5 ms，单副本最多省这 5 ms，再加上 forward 不再
  逐层写 DRAM（不丢的版本 e2e − `gpu_stall` 少 8.5 ms；会丢的版本只少 0.5 ms，重算的 token 把它抵掉了）。
- 会丢的版本 166 个请求的 prefix 被截短（0.9M token 重算）：HBM 满时丢了判据未到的块。不丢的版本逐请求复用的
  prefix 和 LRU 完全一样（0 条变短）；DRAM 最高 87.7%，下沉期间没有 offload 超时。修 offload 泄漏之前的 t2
  （`pab-toolagent-excl-t2-logs`）是 −3.2%，同样 0 条变短。

## 按请求限定 KV 复用来源：`cache_use_control`

请求体字段（不是环境变量），列表，取值：

| 值 | 来源 | 在哪里过滤 |
|---|---|---|
| 1 | 本 P worker 的 HBM prefix cache | vLLM `Request.get_skip_reading_prefix_cache()`，不含 1 就跳过本地 prefix 查找 |
| 2 | 本机 Mooncake DRAM（含同机其他 worker 的 shm 段） | LMCache lookup server，`lmcache/v1/kvpilot_cache_use.py` |
| 3 | 跨机 Mooncake DRAM | 同上 |
| 4 | 本机 Mooncake SSD offload 文件（含同机其他 prefill worker 写的） | LMCache prelookup（`cache_engine.py:_lookup_views`），见下文「L4：SSD offload 与逐层 SSD 加载」 |

- 不传 = 不限制（含 4）；`[2]` = 只从本机 DRAM 复用；`[1,2]` = HBM + 本机 DRAM；`[4]` = 只从 SSD 复用，
  即使 DRAM 里也有同一份。
- **`cache_use_max_tokens`** 是整数时，限制所有层级合计复用的 prefix 长度。也可以是和
  `cache_use_control` 等长的列表：从 prompt 开头起按顺序划成几段，每段只从对应层级取（chunk 归入它第一个
  token 所在的段），总上限是各段之和。例如 `[2, 4]` + `[8192, 4096]` = 前 8K 从 DRAM、后 4K 从 SSD，用来
  构造 DRAM + SSD 混合命中。HBM（1）只能是第一段（`kvpilot_cache_use.local_tiers`、vLLM
  `KVCacheManager.get_computed_blocks`）。
- **只管查找，不管写入**：存 KV 时的去重检查不经过这个过滤，所以被禁掉的层里已有的 chunk 不会重复写入。
- LMCache 只看 HBM 已覆盖部分之后的 chunk（这些才是要加载的），逐个查 Mooncake 副本的
  `transport_endpoint` 主机名，第一个落在不允许的层就截断。主机名取 `ip:port` 去掉端口；
  同机 peer 的 shm 副本已被 Mooncake fork 改写成本 worker 的 endpoint，天然算 2。每个 worker 启动时
  preload 本机**所有**其他 prefill worker 的 segment（不分 TP / PP rank）：本 worker 的 segment 写满后，
  Mooncake 会把 key 放进任何有空间的 segment，包括别的 rank 的。只映射同 rank 时这种 key 会让整次 lookup
  失败（见 [pitfalls.md](pitfalls.md)）。代价是每个进程多映射、多 `cudaHostRegister` 几块：每块 64 GiB
  约 4 s、200 GiB 约 13 s，各 worker 并行；**每注册 1 GiB host 内存占 2 MiB 显存**（GPU 页表），而注册发生在
  vLLM 分完 KV cache 之后，不在它的显存预算里。2P × TP2 下每个进程注册 4 块：64 GiB 的 segment 共 512 MiB，
  200 GiB 的共 1.6 GiB，后者在默认 `--gpu-memory-utilization` 下第一次 forward 就 OOM。
- 每次受限查找打一行：
  `KVPilot cache_use req=... allow=[2] hbm=N hit=N reusable=N local_chunks=N remote_chunks=N`。
- 只作用于 prefill：Dynamo decode handler 收到 `kv_transfer_params` 时把它从 extra_args 删掉。
- 链路：请求体 → Dynamo Rust `PASSTHROUGH_EXTRA_FIELDS` 校验（`cache_use_control` 必须是 1–4 的整数数组；
  `cache_use_max_tokens` 是非负整数，或和它等长的非负整数数组）+
  `sampling_passthrough_args` → handler `build_sampling_params` 写入
  `sampling_params.extra_args["cache_use_control"]` → vLLM `Request` / LMCache adapter
  （作为 `lmcache.kvpilot.*` 放进 request_configs，不以 `lmcache.tag.` 开头，不改变 cache key）。
- 回放：`mooncake_trace_replay.py --cache-use-control 2`。
- **改了 Rust，需要重编 dynamo**，否则前端 400 `Unsupported parameter(s): cache_use_control`。

**已知限制**

- 跨机（3）的读取路径没有实测过：当前单机部署下不会出现跨机副本。一个 batch 里只要有 chunk 只在
  别的机器上，就整批退回 `batch_get_into` 走网络（原代码这里直接报错）。
- 过滤只查每个 chunk **层 0** 的副本位置，假设同一 chunk 各层副本在同一台机器（
  `prefer_local_alloc` 下成立）。

## L4：SSD offload 与逐层 SSD 加载

开关 `KVPILOT_SSD_OFFLOAD_PATH`（空 = 关）。

**写入沿用 Mooncake 自带的 write-through**，不另写一套（exclusive 下只有 SSD 组落盘，见上文）：

- master 加 `--enable_offload=true`（`dynamo_script.sh` 看到开关就加）。每次 `PutEnd` 把对象推进
  持有该副本的 client 的 offload 队列；client 的心跳线程把对象从 DRAM 拷进 bucket 文件，写完才在
  master 上补一个 `LOCAL_DISK` 副本。DRAM 和 SSD 各存一份。
- 每个 TP rank 一个 real client、一份 `FileStorage`，目录是 `$KVPILOT_SSD_OFFLOAD_PATH/pid<pid>/`
  （`mooncakestore_connector.py:_configure_ssd_offload`）。**不能共用目录**：bucket 文件名是
  毫秒时间戳 + 序号，两个 client 同目录会互相覆盖；上一轮留下的目录在启动时会被扫描回 master，
  所以 `dynamo_script.sh` 在 setup / clean 时删掉 `pid*`。
- bucket 记录布局是 `[key 字节][value 字节]`，记录之间顺序随机，所以 value 偏移不对齐；value 就是
  LMCache 的 KV_T2D 原始字节（`save_chunk_meta: False`，无头部）。
- **凑不满 bucket 的 key 等到下一次空心跳**：bucket 攒满 `MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT`（默认
  500）个 key 才写，剩下的先放进 `ungrouped_offloading_objects_`。上游只在下一轮心跳**有新 key**时
  才带上它们，一个之后不再有 put 的 client 就永远不写，这些层在 SSD 上缺失，lookup 在那个 chunk
  截断。Mooncake fork 在 `GroupOffloadingKeysByBucket` 里补了：心跳没有新 key 而有剩余时，把剩余
  单独写成一个 bucket。`disagg_lmcache.sh` 把上限设成 48（一个 chunk 的层数）；fork 把心跳固定成
  100 ms（上游默认 10 s），于是一个 value 最多晚约 0.2 s 落到 SSD。
  - 不满一桶只在 DRAM 写满后出现：同一个 chunk 的 48 层被分到多个 client 的 segment，每个 client
    收到的 key 数不再是 48 的倍数。`paper-figure/Figure2b-safe-suffix-boundary/raw/v1` 第 3 遍
    A 写满后，后面没有新 put（suffix < 512 不存）的 10 条 `[4]` 全部只找到部分 prefix，有新 put 的
    20 条全部完整。
- Mooncake fork 在非 io_uring 写路径上补了 `fdatasync` + `POSIX_FADV_DONTNEED`
  （`storage_backend.cpp:WriteBucket`），和 io_uring 路径一样「刷盘后元数据才可见」。否则 O_DIRECT
  读到还有脏页的范围时，内核要先同步写回，测出来的是写 + 读。
- **`MOONCAKE_OFFLOAD_USE_URING` 必须关**：Mooncake 的 io_uring 路径用 `IORING_OP_READ`，5.4 内核没有。

**读取绕开 Mooncake 的 RPC / Transfer Engine**，直接读本机 bucket 文件：

1. lookup（`_lookup_views`）对每个 value（一层一个 chunk）选源，DRAM 地址和 SSD 位置来自**同一次**
   master 查询（`batched_local_sources` → `batch_get_replica_desc`，顺带拿 lease）：DRAM 允许且
   有本进程可见的副本就用 DRAM；否则 SSD 允许就用该 key 的 `LOCAL_DISK` 副本里记的
   `(文件, value 偏移, value 大小)`，只要写它的 client 在本机（RPC 地址主机名相同），**哪个 prefill
   worker 写的都行**。文件在 lookup 时就打开，之后 Mooncake 再淘汰这个 bucket，已打开的 fd 照样能读。
   master 不知道位置时（快照恢复后）退回查本 client 自己的索引 `batch_locate_local_ssd`。HBM 已覆盖的
   chunk 不加载，任一层级都算命中。
2. 结果存成 `PrelookupPlan`（`[layer][chunk]` 的 host 地址 / 文件下标 / 偏移）。lookup 已按层级选过源，
   `limit_hit` 只再做 `cache_use_max_tokens` 截断。
3. retrieve 时加载范围里只要有一个 SSD value，就走 C++ `SsdLayerwiseLoader`（`csrc/ssd_loader.cu`），
   否则仍走原来的 DRAM `LayerwiseLoader`：
   - I/O 线程池（64 个，`gpu_connectors.py` 构造 loader 时给定）按层序用 O_DIRECT `pread` 读进 pinned 槽位；
     槽位总量 256 MiB（至少两层），决定能提前多少层读。
   - 在途读数从第一层的读数起步、每交付一层翻倍。同时在途的读平分盘的带宽，一开始就让 64 个
     线程全上，第 0 层要等前 ~64 个读一起完成：4K prefix 时第 0 层 9.7 ms 才就绪，起步限制后
     3.6 ms，总加载时间不变。
   - 槽位和线程在 loader 构造时就建好（开了 `KVPILOT_SSD_OFFLOAD_PATH` 的 prefill worker 初始化
     时构造），否则第一条 SSD 请求要多付约 85 ms。
   - 一层的读全部完成后，由完成它的 I/O 线程在 load stream 上按层序发一次 `cudaMemcpyBatchAsync`
     （该层 SSD chunk 从槽位、DRAM chunk 直接从 host 地址，同一层可以混合）+ scatter + event。
   - `wait_for_layer_load(i)` 只在 CPU 上等第 i 层已交给 GPU，再让计算流等它的 event。
   - 槽位 H2D 完就复用；LRU / inclusive 下读到的 value 另拷一份 load-through 进 DRAM，见下一节；exclusive
     下不保留。
4. 日志：`KVPilot prelookup retrieve ... ssd_values=N`、`KVPilot cache_use ... ssd_chunks=N`、
   `KVPilot load ... load=X ssd=N`（`load` 从第一次读 SSD 前算到最后一层 scatter 完）。

**实测带宽**（`utils/ssd_load_check/`，两个 TP rank 同时读同一组盘，`/root` 底下 2 块 NVMe 条带，裸读
峰值 12.8 GiB/s）：

| prefix | 4K | 16K | 32K–64K |
|---|---|---|---|
| 两 rank 合计 | 10.5 GiB/s | 11.5 GiB/s | 12.1–12.2 GiB/s |
| c_ssd（ms / 1K token） | 8.7 | 8.0 | 7.5 |

DRAM 的 c 看测法：两个 rank 同时、没有计算（`test_loader.py bench-dram`）1.4 ms / 1K token；服务内 H2D
合计最快 0.94，和计算重叠时中位 1.85（`paper-figure/prefix-suffix-compute-vs-load-dram-ssd`）。SSD 约慢 5–8 倍，
服务内和 bench 一致。SSD 线程数 32 低 5–10%，128 反而更差，所以固定 64。

端到端（`utils/ssd_load_check`，2026-11-08，1P2D + cudagraph，warm 只发 prefix，`[1]` / `[2]` /
`[4]` 各用一份不同的、长度相同的 suffix hash，delay = e2e − 同一次 `[1]` 的 e2e）：suffix 4K 的 5 格
全部 ssd-safe（delay 1.1–2.1%）；suffix 1K 只有 4K prefix 一格（4.6%），其余 12–83%；suffix ≤ 256
没有一格 ssd-safe，prefix 短时第 0 层等待单独就超过 5%，prefix 长时加载远长于计算。DRAM suffix ≥ 256
时 delay 1.3–6.2 ms（< 6%），suffix 64 时 8%（4K）到 111%（64K）。之前那批三个测试共用 suffix
的网格（含 suffix 1.5K–8K 的长网格）已删除，长网格未重测。

第 0 层前面没有计算可掩盖：发起加载的 CPU 开销 `start` 加上第 0 层在 GPU 上就绪的 `first`（SSD 第
0 层就要从盘上读 prefix/512 × 516 KiB）总要露出一次。要压到 0 得在请求进 forward 之前就开始读
（`--max-num-seqs 1` 下 lookup 和调度在同一步，没有提前量），或者改成 GPU 侧等待
（`cuStreamWaitValue32`）。

**跨 prefill worker**：同一个 key 只存一份（LMCache 存之前查 Mooncake 有没有），offload 派给持有内存
副本的那个 client，所以 2P 下大量 SSD 命中落在别的 worker 的文件里。Mooncake fork 让 client 在
`NotifyOffloadSuccess` 里带上 bucket 文件路径，master 存进 `LocalDiskDescriptor.file_path /
value_offset`（需要同时重编 `mooncake_master` 和 `store`）。A 读自己的文件和 B 读 A 的文件，加载
时间待用修好的 `make_trace_xworker.py` 重测（`utils/ssd_load_check/run_xworker.sh`，旧数据三个测试
共用 suffix，已删除）。master 快照（msgpack）不存这两个字段。

**已知限制**

- 只读本机的 SSD 文件（任一 prefill worker 写的），不读别的机器的（remote SSD）。
- SSD 读期间 forward 线程在 `wait_for_layer_load` 里 CPU 阻塞到该层读完（DRAM 路径是 GPU 侧等待）；
  读领先于 forward 时不阻塞，否则该层数据到了还要等 CPU 重新发射。第 0 层总要等一次。
- 文件在 lookup 和 retrieve 之间被 Mooncake 淘汰、且 fd 缓存恰好被清掉（超过 16384 个文件）时，
  retrieve 会报错。正常容量下不会发生。
- `--max-num-seqs 1` 的前提和 DRAM 路径一样：一个 loader 同一时间只服务一个请求，第二个请求的
  `start()` 会等第一个读完。

## SSD 命中 load-through 进 DRAM

没有开关：LRU 和 inclusive 下总是开，exclusive 下代码里绕开（`_kvpilot_writeback_wanted`，会多一份副本）。
一个请求从 SSD 读的 value 直接拷进 Mooncake 为它新分配的内存副本，加载结束后在后台标记完成，成为 DRAM 副本
（数据只多一次「槽位 → 副本」，由拷贝线程在加载期间完成）。代码里的函数名和日志仍叫 writeback。
`utils/placement_ab/dram_miss_sim.py` 按全 load-through 建模。

对已有的分层测量（`[4]` 强制从 SSD 读，`utils/ssd_load_check`、Figure2/3）没有影响：那时 DRAM 还有这份 value，
master 拒绝追加副本，分配返回 0，什么都不拷。只有 value 真的只剩磁盘副本时才会写进 DRAM。2026-11-11 之前的实验
（Figure2/3、trace_order、placement_ab 除 `wb*` 之外的臂）都是不写回跑的，现在的代码复现不了这种配置。

不走 save 路径（2026-11-04 那次走 save 路径的写回背上了关键路径的查询，见 [pitfalls.md](pitfalls.md)），
不从 GPU 回拷，也不经过 LMCache 的 buffer 再由 Mooncake 单线程 memcpy 一遍（那版每次写回 1.5 GiB/rank
要 0.8 s，写回请求多 7–9 ms）：

1. **分配**（`cache_engine.py:_kvpilot_wb_prepare`，lookup 时）：lookup 已经按 value 选好源，lookup 的那次
   master 查询顺带返回 key 名，存进 `PrelookupPlan.key_strs`。计划里在 SSD 上、chunk 是整块的 value 交给
   后台「alloc」线程：一次 `kvpilot_put_start_local`（Mooncake `batch_put_start_local`）为它们在本 client
   的 segment 里分配内存副本，返回本进程地址（已在内存、没空间、落到别的 segment 的给 0 并当场撤销），
   并算好 `[层, chunk]` 地址表。lookup 线程和 forward 线程都不等这个 RPC。
2. **交给 loader**（`_retrieve_prelookup`）：地址表已回来就切出加载的 chunk 范围，随 `start()` 交给
   `SsdLayerwiseLoader`；没回来（约 10 ms 的分配常常晚于开始加载）就由后台「finish」线程等它回来后调
   `set_writeback`：只挂还没开始的读，返回挂上了哪些（32K prefix 通常挂上 97–98%，漏的是第一层）。
3. **拷贝**（`ssd_loader.cu`）：一个 value 读完，I/O 线程把「槽位 → 副本」的 memcpy 交给独立的拷贝线程
   （8 个，`ssd_loader.cu` 的 `kCopyThreads`）就去读下一个，所以拷贝既不推迟这一层的 H2D，也不占读的并发；槽位
   要等它的拷贝做完才复用（`copying_`）。
4. **完成**（`_kvpilot_submit_writeback`，开始加载后立即提交到 finish 线程）：`wait_writeback(generation)`
   等拷贝结束，挂上的 value `kvpilot_put_end`，其余（被 HBM 覆盖的 chunk、晚到时已开始的读、加载失败）
   `kvpilot_put_revoke`。计划没用上（lookup 后未加载、重复 lookup、`lookup_unpin`）时 `_kvpilot_wb_discard`
   撤销。日志 `KVPilot writeback req=… values=N alloc_ms=… copied_ms=… end_ms=… failed=… revoked=… late=…`。
5. **Mooncake**：master（`master_service.cpp`，exclusive 以外）对只有完成的
   `LOCAL_DISK` 副本、没有内存副本、也没在写 / 复制 / offload 的对象，`PutStart` 追加内存副本并只返回它，
   磁盘副本一直可读；`PutEnd` 对已有磁盘副本的对象不再进 offload 队列；`PutRevoke` 和超时只删新加的内存副本
   （测试 `tests/master_service_test.cpp:KvpAddMemoryReplicaToDiskOnlyObject`）。client
   （`client_service.cpp:KvpBatchPut{Start,End,Revoke}`、`real_client.cpp:batch_put_{start_local,end,revoke}`）
   把一次 put 拆成三个 master 调用给 Python。

**开销**（2026-11-10，当时的 `utils/ssd_load_check/run_wb_diag.sh half`，按请求交替开关写回，脚本随开关一起删了；
`analyze_wb.py` 还在）：每轮先 warm 一段 32K
prefix，8 轮后它已被 8 GiB/rank 的 DRAM 挤出；写回请求只从 SSD 读它（`[4]`），之后 4 个只命中 HBM 的固定形状
探针。偶数轮的写回请求写回（约 3000 value/rank ≈ 1.5 GiB），奇数轮不写回，两种在**同一个服务**里对照
（两次起服务之间，这个请求的 e2e 本身就能差约 6 ms，见 [pitfalls.md](pitfalls.md)）：

| 2026-11-10，各 32 个 | 写回 | 不写回 | 差 |
|---|---|---|---|
| e2e 均值 / 中位数 | 278.3 ± 3.7 / 275.7 ms | 273.7 ± 1.3 / 274.2 ms | +4.6 ± 3.9 / +1.5 ms |
| `gpu_stall` 均值 | 115.2 ms | 111.1 ms | +4.1 ± 3.9 ms |
| e2e − `gpu_stall` | 163.17 ms | 162.66 ms | +0.51 ± 0.20 ms |
| SSD 加载 | 249.9 ms | 250.2 ms | −0.3 ± 1.3 ms |
| 之后的探针 | 101.6 ms | 101.5 ms | 0 |

写回请求自己的 e2e 差别在噪声以内，加载时间不变，不在 `gpu_stall` 里的只有 0.5 ms；后续请求不受影响。
forward 线程上发起加载的 CPU 时间约 1.2 ms（不写回 1.0 ms）。分配约 10 ms，64 次里约 50 次晚于开始加载，
晚到挂表后每 rank 仍写回约 3000 / 3072 个 value。纯 SSD 加载、无计算时（`test_loader.py bench-wb`，拷贝目标是
普通内存）8K 以下不变，16K–64K 慢 2–4%。

之前的两版（都已替换）：拷进 LMCache buffer 再由 Mooncake 单线程 memcpy 一遍，每次写回 1.5 GiB/rank 要 0.8 s；
拷贝在 I/O 线程上做，纯 SSD 加载 8K 慢约 7%。

验证（要编 `c_ops`、`store` + `mooncake_master`，见 [gpu-restore.md](gpu-restore.md)）：
`test_loader.py check`（写回 buffer 逐字节比对、晚到挂表、坏文件时 `wait_writeback` 不挂住）、Mooncake 单元测试。

## 在线链路的几个坑

来自设计文档，实现时踩过：

- **拿不到 `output.text`，明文必须自己 decode**。
- **tokenizer 要从 engine 拿**，不能自己新建一个。
- forced_output 最终采用的是 **3 行 Rust，走 `sampling_options` 通道**。

## dynamo 编译

改了 Rust 部分才需要重编；Python 部分用 `PYTHONPATH` 即可，不需要安装。增量重编命令、
首次工具链搭建、三个容易踩的坑、以及「验证改动真的进去了」的方法，都在飞书
`SnUXdzCJvoyczZxWTCec08nRnBc` 的「dynamo 编译」一节。
