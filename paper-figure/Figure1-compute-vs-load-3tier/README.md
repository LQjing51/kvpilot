# 不同 prefix / suffix 下的计算耗时与 DRAM、SSD 加载耗时

- `compute_vs_load.pdf`：两个 panel 分别是 suffix 64、1K。横轴 prefix，蓝线是计算耗时，两条红线是从 DRAM、SSD 加载整个 prefix 的耗时（与 suffix 无关，两个 panel 相同）

## 配置

- 2026-11-03，Taiji-GPU（H20 × 8），HEAD `56977ec` + 未提交改动（SSD 逐层加载、跨 worker SSD 定位、同机 segment 全映射）
- Qwen3-30B-A3B-Instruct-2507，TP=2，2P2D，block 512，prefill `--max-num-seqs 1` + cudagraph `[64,128,256,512,1024]`，`--gpu-memory-utilization 0.9`，`KVPILOT_PRELOOKUP_VIEWS=true`、`LMCACHE_FIRST_LAYER_LOOKUP=false`，Mooncake SSD offload 开，DRAM 每 TP worker 200 GiB
- 计算：`raw/trace.jsonl` 是 `Figure2b-safe-suffix-boundary` 第一版网格（prefix 4K–64K × suffix 64–4096，每格 3 次，`-p 1`），这里只取 suffix 64 / 1024 的 `[1]` 测试。日志 `raw/prefill-*.log.gz`、`raw/run.out` 原样拷自该次运行
- 加载：`raw/bench_{dram,ssd}.out`，`utils/ssd_load_check/test_loader.py bench-dram / bench`，GPU 6、7 各一个进程（对应两个 TP rank）同时加载，没有计算；SSD 64 个 I/O 线程、256 MB 在途槽位，数据按 Mooncake bucket 格式写盘后 O_DIRECT 读；DRAM 数据在 `cudaHostRegister` 过的内存里

## 复现

`plot.py` 只读 `raw/`，不依赖 `utils/`。下面 GPU 上的命令只在需要重新采集时才跑：

```bash
# GPU 上：计算网格见 ../Figure2b-safe-suffix-boundary（第一版 make_trace，一次服务跑完）
SO=$(python3 -c 'import lmcache.c_ops as c; print(c.__file__)')
python3 /root/kvpilot/utils/ssd_load_check/test_loader.py bench $SO 6,7 | grep '^bench' > bench_ssd.out
python3 /root/kvpilot/utils/ssd_load_check/test_loader.py bench-dram $SO 6,7 | grep '^bench' > bench_dram.out
python3 plot.py
```

## 口径

- 计算 = `[1]` 测试的 `KVPilot prefill ... e2e`（prefix 已在本 worker HBM，只算 suffix），要求 hit ≥ prefix 且 dram = 0，3 次取中位
- 加载 = bench 的 `load`：调用前到最后一层 scatter 完成，两个 rank 取慢的，3 次取中位

## 注意

- 第一版网格第 3 遍有 11 条 `[4]` 只在 SSD 上找到部分 prefix（Mooncake 凑不满 bucket 的 key 不落盘，已修，见 `docs/kv-placement.md`），只影响 `[4]`，这张图只用 `[1]`
