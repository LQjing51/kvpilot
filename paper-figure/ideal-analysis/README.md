# 成本模型：prefill 计算耗时 vs KV 加载耗时

- `context_time.pdf`：H20 × 8 上，计算耗时（三种 suffix）与加载耗时（HBM / DRAM / RNIC）随 prefix 长度变化

## 配置

- 解析模型，不跑实验
- 算力取 H20 × 8 稠密峰值：BF16 1.184 PFLOP/s，FP8 2.368 PFLOP/s
- 带宽：HBM 32 TB/s，DRAM 504 GB/s，RNIC 400 GB/s
- 模型常数：

| 模型 | 精度 | a (FLOP/token²) | b (FLOP/token) | KV (B/token) |
|---|---|---|---|---|
| Qwen3-30B-A3B | bf16 | 786,432 | 5,460,983,808 | 98,304 |
| DeepSeek-V3.2-671B | fp8 | 999,424 | 107,894,800,384 | 43,188 |

## 复现

```bash
python3 plot.py
```

## 口径

- 计算耗时 = `(a · suffix · (prefix + suffix/2) + b · suffix) / FLOPS`
- 加载耗时 = `KV · prefix / BW`
- 常数推导见 `docs/concepts.md` 的 Insight 1；来源与 `utils/plot_context_time.py` 一致
