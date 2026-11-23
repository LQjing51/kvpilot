# Placement algorithm（离线三件套）

| 文件 | 作用 |
|---|---|
| `safe_boundary_fitting.py` | 拟合 (prefix, suffix)→三档 time；给定 prefix → 两个 safe-suffix boundary |
| `resp_length_distribution.py` | SWE / TraceLab：按 tool call type 的 tool-resp 长度分布（桶宽 10） |
| `placement_decision.py` | tool type + prefix + decode_len + prev_input → (P_hbm, P_dram, P_ssd) |

## 复现

```bash
cd paper-figure/algorithm

# (1) 从 Figure3-request-safe 实测点拟合（约 20s，读 gzip 日志）
python3 safe_boundary_fitting.py
# 已有系数时只预测：
python3 safe_boundary_fitting.py --prefix 16384

# (2) 两个数据集的 resp 分布，各写一份 resp_length_dist_<dataset>.json
#     （在线判据 KVPILOT_TOOL_RESP_DISTRIBUTION_PATH 指向其中一份）
python3 resp_length_distribution.py

# (3) 放置概率（prev-input-len = 上一轮 input，用于 %512 残段）
python3 placement_decision.py --dataset swe \
  --tool 'bash:python' --prefix 16384 --decode-len 100 --prev-input-len 16700
# 线性分割示例：
python3 placement_decision.py --dataset swe --tool x --prefix 0 \
  --decode-len 0 --prev-input-len 0 --demo-split
```

产物：`safe_boundary_coeffs.json`、`resp_length_dist_swe.json`（4942 对）、`resp_length_dist_tracelab.json`（1734 对）。
2026-11-19 之前的在线判据实验用的 SWE 分布来自较早的 `swe_mooncake.jsonl`（1925 对），和现在这份不同。

## 口径

- **tool_resp** = 同 session 下一请求 `input_length` − 本请求 `input_length` − 本请求 `output_length`。label = `"+".join(tool_calls)`（与 `kvpilot_signal` 在线判据一致）；无 tool 为 `""`。
- **长度恒等式**（B=512；与 Figure3-request-safe 实测一致）：
  `input_i = input_{i-1} + decode + tool_resp`，
  `prefix_i = align_B(input_{i-1})`，
  **`suffix_i = (input_{i-1} % B) + decode + tool_resp`**
  （不是 `decode + tool_resp`）。
- **boundary 区间**：`[0, dram_b)` HBM safe，`[dram_b, ssd_b)` DRAM safe，`[ssd_b, ∞)` SSD safe。boundary 落在某个 10-token 桶内时，按桶内长度比例把该桶概率分到两侧。
- **time 模型**见 `safe_boundary_fitting.py` 顶部；拟合数据默认来自 `../Figure3-request-safe/raw/{toolagent,swe,tracelab}` 中存在的那些（目前只有 `swe`）。现有的 `safe_boundary_coeffs.json` 是用 Figure3 删掉的旧数据拟合的，重新拟合会被覆盖。
