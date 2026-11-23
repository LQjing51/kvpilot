# toolagent turn5plus on HBM + DRAM + SSD：LRU 下 HBM free queue 的层级构成

- `free_queue_tiers.pdf`：两个 prefill worker 的 HBM free queue（可被淘汰的 cached block）随时间的构成：放 SSD 也不拖慢下次使用（SSD safe）、只能放 DRAM（DRAM safe）、必须留 HBM、之后不再复用

## 结果

两个 prefill worker 共 5128 块，2.8 min 后 pool 已满。free queue 中位 4721 块（pool 已满之后），按块：

| 类别 | 占 free queue |
|---|---|
| SSD safe | 8.5% |
| DRAM safe | 37.5% |
| 必须留 HBM | 8.8% |
| 之后不再复用 | 43.4% |

（每类各自取逐秒占比的中位数，所以加起来不是 100%。）

LRU 留在 HBM 里的可淘汰 block，47.3% 下次使用时从 DRAM / SSD 取也不会变慢，其中约 19% 可以直接放到 SSD；
只算之后还会复用的 block，这个比例是 84%（47.3 / (47.3 + 8.8)），真正必须留在 HBM 的只有 8.8%。

## 配置

- 与 `../toolagent-turn5plus-exposed-load-3tier` 是同一次回放（2P2D，`-p 10 --time-scale 10`，HBM 默认池每个 prefill worker 2564 块（1,312,768 token）+ Mooncake DRAM 每个 TP worker 64 GiB + SSD，各层 LRU），开 `GET_HBM_POOL_DISTRIBUTION`，每个 worker 每秒一次全池普查
- 判定来自 `../toolagent-turn5plus-tier-safe-per-request` 的逐请求实测，那组实验用同一套服务配置

## 复现

```bash
# GPU 上：../toolagent-turn5plus-exposed-load-3tier/raw/run.sh
# 拉回两个 prefill worker 的 hbm-pool-worker-<pid>.jsonl（gzip）到 raw/，pid 写进 raw/prefill-{0,1}.pid
cp ../toolagent-turn5plus-tier-safe-per-request/per_request.jsonl raw/tier_safe_per_request.jsonl
cp ../../datasets/mooncake-trace/toolagent_trace_turn5plus.jsonl raw/
python3 plot.py
```

## 口径

- free queue = 普查里 `ref_cnt == 0` 且有 hash 的 block（state 1）
- 每个 block 的下一个使用者：普查行的 `(req_line, req_block_idx)` 定位到 trace 里的整 block 前缀，之后第一个以同样 block 前缀开头的行
- 下一个使用者的实测判定决定 block 的类别：SSD safe → SSD safe；DRAM safe（SSD 不 safe）→ DRAM safe；都不 safe 或没有实测判定 → 必须留 HBM；没有下一个使用者 → 不再复用
- 普查从 scheduler step 触发，空闲的 worker 会跳过某些秒；每个 worker 按秒沿用上一次采样后再相加
- 「pool 已满」= free queue 里有 hash 的 block ≥ 池子的 90%。没有 hash 的空白 block 始终占 5% 左右（每个 worker 116–190 块），所以不能用「空白 block 用完」来判断
