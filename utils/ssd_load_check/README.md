# SSD 逐层加载：正确性、带宽与端到端延迟

实现见 [docs/kv-placement.md](../../docs/kv-placement.md)「L4：SSD offload 与逐层 SSD 加载」。

- `test_loader.py check`：C++ `SsdLayerwiseLoader` 的逐字节正确性。value 按 Mooncake bucket 记录写盘
  （`[key][value]`、顺序随机、偏移不对齐、最后一条贴着 EOF），同一层混合 SSD 与 pinned host
  value，和旧路径（`copy_` + `single_layer_kv_transfer`）逐字节比；再测文件不存在、文件被截断，以及
  出错后下一个请求能否正常加载。
- `test_loader.py bench`：全部 value 在 SSD，每个 GPU 一个进程（模拟一个 TP rank），各自写完文件后
  用 barrier 对齐、同时加载，报每 rank 的加载时间和合计带宽。
- `make_trace.py` / `run.sh` / `analyze.py`：端到端。每组先发 warm（**只有 prefix**，存 KV），10 s 后
  同一个 prefix 用 `cache_use_control` 分别限定 `[1]` HBM、`[2]` DRAM、`[4]` SSD 各发一次，
  `cache_use_max_tokens=prefix`。**三个测试各用一份不同的、长度相同的 suffix hash**：共用一份时只有
  第一个测试写 suffix，其余的发现已存在就不写，suffix ≥ 1K 时 delay 被低估。`make_trace_xworker.py`
  同理。
- `make_trace_mixed.py` / `analyze_mixed.py`：DRAM + SSD 混合命中。每组 warm 之后发 `[1]`，再按 SSD 占比
  0 / ¼ / ½ / ¾ / 1 各发一个测试（混合档 `cache_use_control=[2, 4]`、`cache_use_max_tokens=[DRAM 段, SSD 段]`），
  每个测试同样各用一份不同的 suffix hash；逐条核对每个 rank 的 `ssd_values` 等于计划。跑法同 `run.sh`，
  `TRACE=.../trace_mixed.jsonl NAME=ssd-mixed-check`。
- `make_trace_hbmmix.py`：blockwise 的布局，头部 HBM、中间 DRAM、尾部 SSD。prefix 16K / 32K / 64K × suffix
  256 / 1K / 2K × 3 次，每组 `[1]` 之后按四分之一粒度发 13 种 (HBM, DRAM, SSD) 切分（`cache_use_control=[1, 2, 4]`
  配三段 `cache_use_max_tokens`，空段省掉）。跑法同 `run.sh`，`TRACE=.../trace_hbmmix.jsonl NAME=ssd-hbmmix-check`。
- `mixed_model.py <日志目录> <trace>`：两种混合 trace 都认。逐条核对 HBM 命中、加载量和每个 rank 的 SSD 值数，然后
  比三样：实测加载时间 vs max / 相加形式（用同批纯 DRAM、纯 SSD 的加载时间拟合）；露出（e2e − `[1]` 的 e2e）vs
  时间模型（`kvpilot_time_model`，默认 Figure3 的 `fit_times2_swe.json`）；以及按 5% 判 safe 时模型和实测是否一致。
- `ssd_safe.py`：用端到端网格拟合 `e2e_ssd = max(C + δ(p), E(p))`，再推算 toolagent / turn5plus 每个
  请求是否 ssd-safe（判据同 dram-safe：比 HBM 慢 < 5%）。模型和输入的口径写在脚本开头。

**数据**：`logs/ssd-tp-check-logs/`（`trace_tp.jsonl` = `make_trace.py 64,256,1024,4096`）和
`logs/ssd-mixed-check-logs/`（`trace_mixed.jsonl` = `make_trace_mixed.py`），2026-11-08 按上面的测法跑；
`logs/ssd-hbmmix-check-logs/`（`trace_hbmmix.jsonl`），2026-11-22，405 条全部通过核对，
只留 KVPilot 和 `Stored` 行。之前三个测试共用 suffix 的端到端网格和跨 worker
数据已删除；长 suffix 网格和跨 worker 待用现在的 `make_trace*.py` 重测。

## 配置

- 2026-11-02，Taiji-GPU（H20 × 8），HEAD `56977ec` + 未提交改动（本次相关：
  `lmcache-dev/csrc/{ssd_loader.cu,ssd_loader.cuh,pybind.cpp}`、`lmcache-dev/setup.py`、
  `lmcache-dev/lmcache/v1/{cache_engine.py,kvpilot_cache_use.py,kvpilot_load_time.py}`、
  `lmcache-dev/lmcache/v1/gpu_connector/gpu_connectors.py`、
  `lmcache-dev/lmcache/v1/storage_backend/{remote_backend.py,connector/mooncakestore_connector.py,connector/instrumented_connector.py}`、
  `Mooncake/mooncake-store/src/{storage_backend.cpp,file_storage.cpp,real_client.cpp}` 及头文件、
  `Mooncake/mooncake-integration/store/store_py.cpp`、`dynamo-dev/lib/llm/src/protocols/openai/validate.rs`、
  `dynamo-dev/examples/backends/vllm/launch/disagg_lmcache.sh`、`vllm-dev/.../dynamo_script.sh`）
- SSD：`/root` 是 overlay，底下 `dm-0` 由 2 块 SOLIDIGM 5.8T NVMe（Gen4 x4）条带而成，O_DIRECT
  裸读峰值 12.8 GiB/s（`paper-figure/prefix-suffix-compute-vs-load/opt/bench/ssd_bench.py`）
- 端到端：Qwen3-30B-A3B-Instruct-2507，TP=2，1P2D，block 512，prefill `--max-num-seqs 1` +
  cudagraph `[64,128,256,512,1024]`，`KVPILOT_LOAD_STALL=true`、`KVPILOT_SSD_OFFLOAD_PATH=/root/kvpilot-ssd`，其余用默认
- 模型在 CFS 上时 `MODEL=/cfs_cloud_code/kiry/model/Qwen/Qwen3-30B-A3B-Instruct-2507`；`run.sh` 不删
  `$KVPILOT_SSD_OFFLOAD_PATH`，上一轮留下的目录在启动时会被扫描回 master，重测前先清掉

## 复现

```bash
# GPU 上：先按 docs/gpu-restore.md 编好 c_ops / Mooncake store / dynamo
SO=/root/kvpilot/lmcache-dev/build/cops_out/lmcache/c_ops.cpython-312-x86_64-linux-gnu.so
python3 test_loader.py check $SO 7
python3 test_loader.py bench $SO 6,7
python3 make_trace.py > trace.jsonl && bash run.sh
# 本地：拉回 dynamo-log/ssd-load-check-logs/prefill-0.log 后
python3 analyze.py <log dir>
```

## 口径

- bench 的加载时间 = `start_layers_ssd` 调用前到最后一层 scatter 完成（`torch.cuda.synchronize`），
  3 次取中位；合计带宽 = 两 rank 读的字节之和 / 较慢的 rank 的时间。
- 端到端 compute = `[1]` 测试的 `KVPilot prefill ... e2e`；delay = `[2]` / `[4]` 的 e2e − compute，
  按组配对，3 次取中位。每条测试都核对过来源：`[1]` 是 `hit ≥ prefix, dram = 0`，`[2]` 是
  `dram ≥ prefix` 且 `ssd_values = 0`，`[4]` 是 `dram ≥ prefix` 且每个 rank 的
  `ssd_values = 48 × chunks`，不符的剔除并计数。
- ssd load = `[4]` 测试 `KVPilot load ... load=`（两 rank 取大），从第一次读 SSD 前到最后一层 scatter。
