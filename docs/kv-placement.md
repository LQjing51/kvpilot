# KV placement 的实现

概念背景见 [concepts.md](concepts.md)。

## 开关

都在 `dynamo-dev/examples/backends/vllm/launch/disagg_lmcache.sh` 里 export。

| 变量 | 默认 | 作用 |
|---|---|---|
| `KVPILOT_OFFLINE_DRAM_SAFE` | `false` | 启用 `FreeKVCacheOfflineDramSafeQueue`，判据来自离线 trace 标注。开启时必须同时给 metadata 路径，否则启动即报错 |
| `KVPILOT_TRACE_METADATA_PATH` | mooncake trace | 离线标注好的 trace（每条含 `kvpilot` 字段） |
| `KVPILOT_ONLINE_DRAM_SAFE` | `false` | 启用 `FreeKVCacheOnlineDramSafeQueue`，判据来自 D 侧反馈。与离线开关**相互独立**，同开时在线优先 |
| `KVPILOT_MACHINE` | `h20` | `h20` / `h200` / `a100`，决定成本模型常数 |
| `KVPILOT_TOOL_RESP_LEN_PATH` | swe-smith 的表 | **开在线判据时必填**。tool 响应长度查表 JSON |

`KVPILOT_TOOL_RESP_LEN_PATH` 是 **fail-closed** 的（文件不存在就启动报错），因为用错表
会 100% 落 `_default`、判据退化成常数**且日志里看不出来**。两个 trace 的 label 命名
空间实测零重叠，见 [datasets.md](datasets.md)。worker 启动时读表，所以**换 trace 必须
重启服务**。

## 文件分工

改动集中在 `vllm-dev/vllm/v1/core/`：

| 文件 | 作用 |
|---|---|
| `kv_cache_utils.py` | `KVCacheBlock` 上的 `kvpilot_*` 字段；空闲队列实现 |
| `block_pool.py` | 按环境变量选择队列；`free_blocks()` 落标记 |
| `kv_cache_manager.py` | `free()` 里算 kv_hit_rate、读 trace metadata 标 dram-safe；在线模式下把 blocks 交给 `OnlineDramSafe` |
| `kvpilot_online.py` | 在线判据的记账：`_pending[request_id] = [(block, hash)]` 与 `track()` / `apply()` |
| `kvpilot_predict.py` | 在线离线共用的判据实现 + `MACHINES`；查表 JSON 不在这里，随数据集存放 |
| `single_type_kv_cache_manager.py` | 把请求级信号透传给 `block_pool.free_blocks()` |
| `kvpilot_hbm_snapshot.py` | HBM block 分布采样（观测用，见 [observability.md](observability.md)） |
| `kvpilot_load_stall.py` | 每次 prefill 的 load 阻塞计量（观测用，见 [observability.md](observability.md)） |

`kvpilot_predict.py` 导出：`derive_constants` / `is_dram_safe` / `turn_labels` /
`predict_dram_safe` / `explain_dram_safe` / `write_resp_len_table`。

在线反馈还接了一条 handler → scheduler 进程的控制链路：
`vllm/v1/engine/{async_llm,core_client,core}.py` 各加一个 `kvpilot_apply_label` 转发
（形状照 `reset_prefix_cache`），最终落到 `sched/scheduler.py`。`forced_output` 走
`vllm/v1/sample/logits_processor/forced_tokens.py`。

## 共用实现放在 vLLM 里

`kvpilot_predict.py` 是在线与离线**唯一的一份实现**（成本模型常数、`is_dram_safe()`、
tool call 正则与标签逻辑、`_lookup_resp_len()`、`write_resp_len_table()`）。

放在服务代码同目录是为了**不可能偏离**——常数和标签逻辑在 D 侧算但决定 P 侧行为，两处
各写一份迟早不一致，且这类偏差极难察觉（离线验证出的准确率会不再描述线上真正在跑的
东西）。`validate_premises.py` 从那里 import，只依赖标准库，import 它不会拉起 torch。

## 空闲块队列

在 `kv_cache_utils.py`：

| 类 | 策略 |
|---|---|
| `FreeKVCacheBlockQueue` | 上游原版 LRU 双向链表 |
| `FreeKVCacheHeap` | 按 `_priority()` 排序的最小堆，公式 `kv_hit_rate * 3000 + counter`（那个 3000 就是 Challenge #2 批评的「参数」） |
| `FreeKVCacheMultiQueue` | 三层 FIFO 骨架，层名 `_first`/`_second`/`_third` 即淘汰顺序；子类实现 `_tier_for()` |
| `FreeKVCacheLengthQueue` | 按请求长度分层（blank / short / long） |
| `FreeKVCacheOfflineDramSafeQueue` | 读 block 上的 `kvpilot_next_dram_safe` |
| `FreeKVCacheOnlineDramSafeQueue` | 入队时判据还不存在，一律落保守层，判据到达后由 `retier(block, dram_safe)` O(1) 挪层；**不在 block 上存判据** |

## 信号传递链路

```
scheduler.py  kv_cache_manager.free(request)
  → kv_cache_manager.py  算 kv_hit_rate、查 trace metadata
    → single_type_kv_cache_manager.free(..., kv_hit_rate)
      → block_pool.free_blocks(..., kvpilot_req_kv_hit_rate=...)
        → block.kvpilot_next_dram_safe / kvpilot_req_kv_hit_rate
          → 队列的 append() 决定它落在哪一层
```

## 离线与在线两条路

**各用一个队列子类，相互独立**：

- **离线**（`KVPILOT_OFFLINE_DRAM_SAFE`）：从 trace metadata 预读
  `kvpilot_next_dram_safe_by_block` 反标 blocks。
- **在线**（`KVPILOT_ONLINE_DRAM_SAFE`）：D 在 turn 结束时 decode 出明文、正则解析
  tool type、查 p50 估 append、算出判据，经 dynamo endpoint 回传 P，P 把已入队的 block
  **挪层**。

**为什么在线必须挪层而非入队时定级**：P 释放 block 发生在 turn N decode **刚开始**
（D 拉完 KV 就通知 P 放行），而判据要等 decode **结束**才算得出，标签天生比入队晚一整个
decode 阶段。

两个开关同开时**在线优先**。设计文档见飞书 `SnUXdzCJvoyczZxWTCec08nRnBc`。

## 在线链路的几个坑

来自设计文档，实现时踩过：

- **拿不到 `output.text`，明文必须自己 decode**。
- **tokenizer 要从 engine 拿**，不能自己新建一个。
- forced_output 最终采用的是 **3 行 Rust，走 `sampling_options` 通道**。

## dynamo 编译

改了 Rust 部分才需要重编；Python 部分用 `PYTHONPATH` 即可，不需要安装。增量重编命令、
首次工具链搭建、三个容易踩的坑、以及「验证改动真的进去了」的方法，都在飞书
`SnUXdzCJvoyczZxWTCec08nRnBc` 的「dynamo 编译」一节。
