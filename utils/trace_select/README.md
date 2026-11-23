# 按 session 挑 SWE：HBM safe : DRAM safe ≈ 1:2

SWE 按 request-safe（`paper-figure/Figure3-request-safe`）实测是 HBM 14.3% / DRAM 69.2% / SSD 16.4%，
HBM : DRAM ≈ 1:4.8。这里从全量 SWE 里挑整条 session，让这个比例到 1:2，产出
`datasets/agent-traces/swe-smith-trajectories/` 下两条 trace：

- `swe_hbm1to2.jsonl`：只约束 HBM : DRAM = 1:2，400 GiB
- `swe_hbm1to2_ssdmax.jsonl`：HBM : DRAM = 1:2，且 working set > 300 GiB 的前提下 SSD safe 尽量多（35%）

| 文件 | 作用 |
|---|---|
| `select_safe_mix.py` | 不带 `--sessions-out`：算一条 trace 的期望 HBM / DRAM / SSD safe 占比；带上：挑 session，写出 id。`--mix 1:2` 只约束 H:D，`--mix 1:2:1.68` 约束 H:D:S |
| `swe_hbm1to2_sessions.txt` | `swe_hbm1to2` 选中的 368 个 traj_id，按原始文件顺序 |
| `swe_hbm1to2_ssdmax_sessions.txt` | `swe_hbm1to2_ssdmax` 选中的 156 个 traj_id |

## 复现

```bash
D=datasets/agent-traces/swe-smith-trajectories
# 全量（约 74 min，10 核 Mac；产物 397 MB，已 gitignore）
(cd $D && python3 prepare_swe_trace.py --num-requests 0 --out swe_full.jsonl)
python3 utils/trace_select/select_safe_mix.py $D/swe_full.jsonl \
    --sessions-out utils/trace_select/swe_hbm1to2_sessions.txt
(cd $D && python3 prepare_swe_trace.py --num-requests 0 \
    --sessions ../../../utils/trace_select/swe_hbm1to2_sessions.txt --out swe_hbm1to2.jsonl)
python3 utils/trace_select/select_safe_mix.py $D/swe_hbm1to2.jsonl     # 核对
# SSD safe 尽量多、working set > 300 GiB
python3 utils/trace_select/select_safe_mix.py $D/swe_full.jsonl --mix 1:2:1.68 \
    --min-unique-kv-gib 0 --sessions-out utils/trace_select/swe_hbm1to2_ssdmax_sessions.txt
(cd $D && python3 prepare_swe_trace.py --num-requests 0 \
    --sessions ../../../utils/trace_select/swe_hbm1to2_ssdmax_sessions.txt --out swe_hbm1to2_ssdmax.jsonl)
```

`prepare_swe_trace.py` 用 HF 缓存里的 tokenizer，离线时加 `HF_HUB_OFFLINE=1`。

## 怎么判一个请求是哪类

全量没有实测，按 (prefix, suffix) 查 request-safe 那 4,900 行实测的经验概率：prefix 分
`[0, 4K)`、`[4K, 16K)`、`≥ 16K` 三档，suffix 按 128 / 160 / … / 4096 细分，相邻档凑够 20 行才成一格。
prefix / suffix 的口径与 request-safe 相同（整块前缀、至少留一块计算，prefix = 0 不计）。一条 trace
的期望 H / D / S = 每行三个概率之和。

**不用 `fit_times_swe.json` 的时间模型判**：它在这 4,900 行上只判出 200 行 HBM safe（实测 703），
召回 24%。经验概率还能表达两件时间模型丢掉的事：suffix < 256 时 HBM safe 占 50–100%；prefix < 4K
时 suffix 到 1K 仍有 20–25% 是 HBM safe（加载的固定开销相对计算显眼）。

## 怎么挑

- 每个 session 只按自身计 prefix（不看别的 session），所以它的分数不取决于还挑了谁
- 按期望 H / (H + D) 从高到低排，取最长的、合计仍满足 H ≥ D / 2 的开头一段：这是 1:2 下最大的集合
- 它超过 `--min-unique-kv-gib`（默认 400，和 `swe_mooncake.jsonl` 一致，DRAM 的 2 倍）时，在这段里
  按等间隔抽，保持 H 占比的构成
- 用 `--sessions` 重新走预处理：子集里的跨 session 共享、`prefix_length` 都按子集重算，再核对一次
- 三项 `--mix H:D:S`：把每个 session 相对目标占比的偏差看成平面上一点（H、S 两维，D 随之确定），沿某个
  方向按每行偏差排序取开头一段，方向扫一圈，取三项都在目标 ±0.5 个百分点内的最大一段（`pick_mix`）

**为什么按概率挑而不是按实测标签挑**：在已测的 198 个 session 上，按实测标签挑出来的 1:2，同一批
session 的另一半轮次只有 1:4.3，挑中的主要是测量噪声。按概率挑做了交叉验证（一半 session 建表、
另一半挑，30 次随机划分）：挑中的 session 合计预测 1:1.95、实测 1:1.91，没挑中的那一半是 1:4.97。

## 结果

| | session | 请求（prefix>0） | 不同整块 KV | H | D | S | H : D |
|---|---|---|---|---|---|---|---|
| `swe_full.jsonl` | 23,144 | 701,515（693,259） | 55,735 GiB | 13.1% | 71.5% | 15.4% | 1:5.46 |
| 1:2 最大集合 | 851 | 14,041 | 932 GiB | 28.4% | 56.7% | 14.9% | 1:2.00 |
| **`swe_hbm1to2.jsonl`** | 368 | 6,026（5,833） | 420 GiB | 27.9% | 56.6% | 15.4% | **1:2.03** |

都是期望值；最大集合那行按 session 自身计 prefix，另外两行是 trace 本身（含跨 session 共享）。
子集逐行与 `swe_full.jsonl` 的长度、tool_calls 一致，prefix > 64K（超出校准范围）的只有 0.1%。

**挑出来的 session 偏轻**，用它做实验时要说明：

| | 全量 | `swe_hbm1to2.jsonl` |
|---|---|---|
| 每 session 轮数（中位） | 26 | 14 |
| input（中位 / p90） | 16.3K / 40.6K | 8.7K / 19.1K |
| 每轮 append（`input − prefix_length`，中位） | 416 | 218 |
| 输出 tool：`str_replace` / `view` / `find` / `submit` | 19.3% / 25.0% / 4.1% / 5.8% | 11.5% / 19.7% / 8.5% / 9.8% |

**GPU 实测（2026-11-10，request-safe 同一套测法，`paper-figure/Figure3-request-safe/raw/swe_hbm1to2/`）**：
5833 行 HBM 27.5% / DRAM 57.8% / SSD 14.7%，HBM : DRAM = 1:2.10，和期望的 1:2.03 一致。概率表只用
`raw/swe` 建，这是样本外检验。按预测的 SSD 占比把 368 个 session 分四档，预测 SSD 27.4% / 18.5% /
13.4% / 6.2%，实测 26.2% / 17.9% / 13.0% / 5.7%：按 SSD 挑 session 也成立，只是系统性高估约 0.5–1 个百分点。

## SSD safe 尽量多：`swe_hbm1to2_ssdmax.jsonl`

约束 HBM : DRAM = 1:2、working set（不同整块 KV）> 300 GiB，SSD 占比越高可选的 session 越少。扫 `--mix 1:2:x`
（每行是该比例下的最大集合，按 session 自身计 prefix 的期望值）：

| `--mix` | session | 请求 | KV | H | D | S |
|---|---|---|---|---|---|---|
| 1:2（S 不约束） | 851 | 14,041 | 932 GiB | 28.4% | 56.7% | 14.9% |
| 1:2:1 | 573 | 8,313 | 833 GiB | 25.2% | 50.1% | 24.7% |
| 1:2:1.4 | 276 | 3,846 | 488 GiB | 22.9% | 45.5% | 31.6% |
| 1:2:1.6 | 182 | 2,492 | 337 GiB | 22.1% | 43.5% | 34.4% |
| **1:2:1.68** | 156 | 2,136 | 303 GiB | 21.2% | 42.7% | 36.2% |
| 1:2:1.7 | 152 | 2,056 | 289 GiB | 21.6% | 42.6% | 35.8% |
| 1:2:2 | 91 | 1,219 | 191 GiB | 20.4% | 40.1% | 39.5% |

1:2:1.68 是 KV 仍超过 300 GiB 时 SSD 最高的点。重建后（含子集内跨 session 共享，多出 83 条 round 0 命中）：
156 个 session、2,136 条请求（prefix>0 2,035）、302.6 GiB，期望 **HBM 21.1% / DRAM 43.8% / SSD 35.1%**，
H : D = 1:2.08。其中 59 个 session（688 行）在 `swe_hbm1to2` 里实测过：预测 26.9% / 45.7% / 27.4%，实测
28.9% / 44.6% / 26.5%。照上面的高估幅度，整条实测 SSD 大约 34%，未实测。

这批 session 的 append 两头都多（短的给 HBM safe、长的给 SSD safe）：每 session 轮数中位 12（全量 26），input
中位 10.3K（16.3K），每轮 append 中位 678（416），prefix > 64K 的 3.0%。
