# 起服务与跑 benchmark

请求怎么构造见 [request-paths.md](request-paths.md)。

## 唯一入口

`vllm-dev/examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/dynamo_script.sh`
有四个子命令：

```bash
bash dynamo_script.sh setup [log_name]    # 起 mooncake_master + dynamo frontend + P/D workers
bash dynamo_script.sh benchmark [log_name]
bash dynamo_script.sh clean               # 优雅停止，清理 SHM 与 GPU 显存
bash dynamo_script.sh cleanforce          # 强杀
```

它调用 `dynamo-dev/examples/backends/vllm/launch/disagg_lmcache.sh` 拉起
2 prefill + 2 decode（TP=2，共 8 卡）。所有 KVPilot 开关都在那个脚本里 export；与
placement / 观测相关的分别记在 [kv-placement.md](kv-placement.md) 和
[observability.md](observability.md)，本文只列影响「怎么起、怎么压」的。

**日志目录会自动加 `-logs` 后缀**：`setup foo` 写到 `dynamo-log/foo-logs/`，不是
`dynamo-log/foo/`。踩过一次——事先建好一个叫 `foo` 的空目录，然后对着它找不到日志。

日志文件：`dynamo-log/<name>-logs/{frontend,prefill-N,decode-N,mooncake_master}.log`。

## 一台机器同一时间只跑一个实验

即使两个服务各占 4 张卡、GPU 不重叠，也共用主机 DRAM（Mooncake）和本地 SSD：一边的
加载流量会改变另一边的 delay / compute。2026-11-04 的 1P1D boundary 搜索与 SWE 回放
同时跑了约 25 分钟，4K–34K 的分界点全部作废重跑。先 `cleanforce` 上一个，再起下一个。

## 成组的 Evaluation 实验，日志要拉回本地

一组一组跑的 Evaluation 实验（`paper-figure/Eval-*`、`run_ab.sh` / `run_series.sh` 的多臂 A/B、request-safe 测量等
要拿来下结论的），**每跑完一组就把日志拉回本地**，放进该实验的 `paper-figure/<实验>/raw/`，不等整批跑完。
GPU 机器随时可能被释放，`/root` 下的东西全没：2026-11-24 Eval-EXP1 的 TraceLab 九组在 GPU 上跑完，还没拉回来
机器就释放了，结果全丢。

- 拉什么：prefill / decode 日志（gzip）、`benchmark.log`、`run.out`、master 日志里的 KVPilot 行，即
  `paper-figure/Eval-EXP1/run_exp1.sh` 归档的那几样；大文件的处理同
  [figure-guidelines.md](evalution/figure-guidelines.md)「`raw/`」。
- 怎么拉（`scp` 不通）：`ssh Taiji-GPU 'cd <实验目录> && tar czf - raw/<...>' | tar xzf -`。
- 简单测试（冒烟、正确性检查、试参数、复现 bug）不用拉，在 GPU 上看完日志即可。

## 非交互 ssh 必须自己 export PYTHONPATH

**这是最容易浪费时间的一个坑。** GPU 机器上装了一份 **stock vLLM 0.23.0** 在
`/usr/local/lib/python3.12/dist-packages/vllm`，里面**没有任何 KVPilot 改动**。仓库
`/root/kvpilot/vllm-dev` 不是 pip 安装的（没有 editable install、没有 .pth、没有 bind
mount），它能被 import 只靠 `/root/.bashrc` 里的：

```bash
export PYTHONPATH=/root/kvpilot/vllm-dev:/root/kvpilot/lmcache-dev:\
/root/kvpilot/dynamo-dev/components/src:/root/kvpilot/dynamo-dev/lib/bindings/kvbm/python
```

而 **`.bashrc` 只对交互式 shell 生效**。所以
`ssh Taiji-GPU "bash dynamo_script.sh setup ..."` 会静默跑成 stock vLLM。

**失败症状很有迷惑性**：setup 卡在

```
ERROR: NIXL pair warm failed: ... nixl_log=False
```

但 KV 传输其实是**成功的**（日志里有 `Num successful transfers=2`，4.4 GB/s）。之所以
报错，是因为 `discover_dynamo_workers.py:request_logged()` 去 decode 日志里 grep
`"NIXL recv metadata queued"`，而这行 `logger.info` 只存在于仓库版
`vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py:212`。

**跑之前先自查代码有没有生效**：

```bash
grep -c 'KVPilot placement=' dynamo-log/<name>-logs/prefill-0.log
```

正常是 1 行以上（每个 scheduler 启动一行，LRU 也打），**0 就说明跑的是 stock vLLM**。

## 同步代码到 GPU 机器

```bash
bash ~/Desktop/code/utils/sync_to_devcloud.sh \
     local:$HOME/Desktop/code/kvpilot Taiji-GPU:/root
```

语义是「把 SRC 目录放到 REMOTE 目录**下面**」，所以上面这条得到 `/root/kvpilot`。
脚本在 `~/Desktop/code/utils/`（**不在 kvpilot 仓库里**）。新机器的完整恢复流程见
[gpu-restore.md](gpu-restore.md)。

常用选项：

| 选项 | 作用 |
|---|---|
| `-n` / `--dry-run` | 只列出会增/删/改哪些文件，不写入 |
| `--delete` | 删除目标端多余文件（默认不删，只增改） |
| `-e PAT` | 追加排除规则 |
| `-j N` | 并行分片数（默认 8） |

**默认不删除目标端多余文件**：本地删掉或改名的文件，GPU 上的旧副本会一直留着。需要
清理时加 `--delete`，它无事前确认，看到 `*deleting` 时文件已经删了，想先看影响用
`-n --delete`。即使加了 `--delete`，`dynamo-log/` 也登记在脚本的 `PROTECT` 里，GPU 上的
历史日志不会被删。

`scp` 到这台机器是不通的（报 `Connection closed`），要传单个文件用：

```bash
B64=$(base64 < local_file | tr -d '\n')
ssh Taiji-GPU "echo '$B64' | base64 -d > /root/remote_file"
```

## 压测参数怎么调

benchmark 段的参数含义（`benchmark_serving_multi_turn.py`）：

| 参数 | 含义 |
|---|---|
| `--max-inflight-req` | 在途请求数上限，限制同时有多少请求在飞。0 = 不限 |
| `--max-inflight-session` | 同时活跃的会话数上限。0 = 不限 |
| `-n` | 总请求数上限，达到就停 |
| `--session-arrival-rate` | 每秒新开会话数，泊松过程。**开环到达**：到点就发，不看服务端忙不忙 |
| `--tool-time` | 两轮之间的工具耗时 Z，下一轮到达 = 上一轮完成 + Z。可填 `0`、常量（`1s` / `250ms`）、或 `lognormal[:mu,sigma]` |
| `--forced-output` | 钉死输出长度，见 [request-paths.md](request-paths.md) |

### 两个 cap 管的是不同的事

- `--max-inflight-req` 约束 **GPU 负载**。
- `--max-inflight-session` 约束**同时保温多少条不同前缀**——关键是**等待中的会话仍然
  占着它的 KV 块**。

所以 session cap 能在不动 GPU 负载的前提下单独抬 HBM 压力。这是制造 KV 淘汰、进而制造
DRAM 加载的**主要杠杆**。

### 排队/缓存模型

记 P=prefill worker 数、S=每 session 轮数、n=每轮新增 block 数、N=池子块数、
Tp/Td/Tt=单轮 prefill/decode/tool 耗时、R=会话到达率。稳态下：

```
请求速率  lambda = R·S                             ← 与 Tt 无关
利用率    rho    = R·S·Tp/P                        ← 与 Tt 无关
排队      Wq     = rho·Tp·(1+Cs²)/(2(1-rho))       (M/G/1 Pollaczek-Khinchine)
工作集    W      = R·n·S·(S+1)·(Tp+Td+Tt)/(2P)
命中率    h      = min(1, N/W)
```

`W` 的推导：活跃 session 数 = `R·S·(Tp+Td+Tt)`（Little 定律），每条平均持有
`n·(S+1)/2` 个 block（第 k 轮持 `k·n`），再除以 P 个 worker。vLLM 每轮 prefill 会
`touch()` 整条前缀刷到 LRU 头，所以一条 session 的前缀**整体存活或整体死亡**，复用间隔
就是轮间隔 `Td+Tt`。

**关键：`Tt` 进 `W` 但不进 `rho`。** 工具耗时把一条 session 的 S 条请求在时间轴上摊开，
同时在飞的 session 数等比例变多，总速率不变。所以：

- **R 是负载旋钮**（同时抬排队和缓存压力）
- **Tt 是纯压力旋钮**（只抬缓存压力）

两个阈值：

```
R*     = 2PN / [n·S·(S+1)·(Tp+Td+Tt)]    开始出现 DRAM 加载
R_stab = P / (S·Tp)                       prefill 队列发散
```

要测加载而不是测排队，必须 `R* < R_stab`，即 `N < n(S+1)(Tp+Td+Tt) / (2Tp)`。

### 实测标定值（2026-09-11，SWE-smith trace）

`Tp=0.25s, Td=1.35s, Tt=1.65s, n=2.18, S=32.4, N=2465, P=2`：

| 配置 | 有 DRAM 加载的请求 | blank 最低 |
|---|---|---|
| `session 32`, R=0.8 | 5.2% | 604 |
| `session 128`, R=1.5 | 9.3% | 208 |
| **`session 400`, R=4.0** | **65.7%** | **78** |

第一行 `R*=1.28/s` 而 `R_stab=0.247/s`，比值 5.2——**窗口是关着的**，单靠抬到达率会先
把 prefill 队列压爆 5 倍。所以必须走 session cap。

池子有**硬下界 512 块**（`max_model_len` 262,144 ÷ 512）：低于它 vLLM 直接拒绝启动
（`kv_cache_utils.py:_check_enough_kv_cache_memory`）。

### 不限 inflight 时怎么设（开环，只剩 R 和 Tt 两个旋钮）

把两个 cap 都设成 0 之后，`--max-inflight-req` 那条兜底没了，**开环到达直接撞服务能力**，
可行域会窄很多。推导如下。

**第一步：R 的上界由 Tp 和 S 决定，与 Tt 无关。** 「Tt 不进 rho」意味着到达率是唯一
影响负载的旋钮，稳定条件 `rho < 1` 给出：

```
rho   = R · S · Tp / P
R_max = P / (S · Tp)        ← 就是前面的 R_stab
```

**容量要从 `Tp` 直接算，不要从别的跑反推**。每个 worker 每秒能做 `1/Tp` 条 prefill，
所以 `C = P/Tp`。实测 `Tp = 251 ms`、`P = 2`：

```
C     = 2 / 0.251 = 7.97 req/s
R_max = 7.97 / 32.4 = 0.246 sessions/s
```

这是我踩过的坑：当初拿 session-400 那轮（`--max-inflight-req 16`）的 span 反推
`C ≈ 3.3 req/s`，但那是被 cap **限制**出来的吞吐、不是机器上限，于是 `R_max` 低估了
2.4 倍，白白选了个过于保守的 R。**凡是带 inflight cap 的跑都不能用来标定容量。**

**第二步：Tt 决定要多少活跃 session 才够，也决定 R 的下界。** 工作集和 session 时长
挂钩，而 `T_sess` 由三段构成：

```
T_sess = S · (Tp + Td + Tt)                      ← Tt 通常占 98%
活跃 session 数 A = R · T_sess                    （Little）
W/N   = R · n · (S+1) · T_sess / (2·P·N)
```

要 `W/N ≈ 1.5`（能持续淘汰又不至于全 miss），给定 Tt 就反解出 R；**两个旋钮不是独立
的，是一条等值线上的一对**：

```
R · (Tp + Td + Tt) = 2·P·N·(W/N) / (n·S·(S+1))  = const
```

**第三步：两边一夹，可行域就出来了。**

```
R_min(Tt)：由 W/N ≥ 目标 反解 —— Tt 越大，需要的 R 越小
R_max    = P/(S·Tp) = 0.246 —— 与 Tt 无关
```

代入实测（`Tp=0.251s, Td=1.210s, n=2.18, S=32.4, N=2465, P=2`，目标 `W/N=1.5`，
等值线常数 `R·(Tp+Td+Tt) = 6.27`）：

| Tt 均值 | 需要的 R | rho | T_sess | 活跃 session | 可行？ |
|---|---|---|---|---|---|
| 5 s | 0.970 | 3.95 | 3 min | 203 | ✗ rho≫1 |
| 20 s | 0.292 | 1.19 | 12 min | 203 | ✗ rho>1 |
| **24 s** | **0.246** | **1.00** | 14 min | 203 | ← 边界 |
| 30 s | 0.199 | 0.81 | 17 min | 203 | ✓ |
| 60 s | 0.102 | 0.41 | 33 min | 203 | ✓ |
| 72 s | 0.085 | 0.35 | 40 min | 203 | ✓（实跑这组）|

注意最后一列**活跃 session 数恒为 203**——这不是巧合：`W/N` 目标固定就等于把活跃
session 数固定住了（每条 session 平均持有的 block 数不变）。Tt 和 R 只是在决定用
「少量快 session」还是「大量慢 session」去凑够这 203 条。

**可行域边界**：`rho < 1` 要求 `R < 0.246`，代回等值线得 **`Tt > 24 s`**。低于这个值
无论怎么配都不行——要么工作集不够，要么队列发散。

**在可行域内选哪一点**：Tt 越小、R 越大，跑得越快（`T_sess` 短，爬坡快），但 `rho`
越接近 1、排队越重。实跑选的 `Tt=72s / R=0.085` 偏保守（rho 只有 0.35），代价是爬坡
40 分钟；按上表 `Tt=30s / R=0.20` 同样能达标而 `T_sess` 只有 17 分钟，**总跑时能缩短
一半以上**，是下次更好的起点。

sigma 取 0.6 而非 1.0：`W/h` 只依赖 Tt 的**均值**，sigma 纯粹控制尾巴；均值固定 72 s 时
sigma=0.6 的 p99 是 243 s，再大会让个别 session 长到拖垮总跑时。

实跑命令：

```bash
--max-inflight-req 0 --max-inflight-session 0 -n 12000 \
--session-arrival-rate 0.085 --tool-time lognormal:4.1,0.6
```

（`lognormal:mu,sigma` 给的是 log 空间参数，均值 = `exp(mu + sigma²/2)`；要均值 72 s
就取 `mu = ln 72 − 0.6²/2 ≈ 4.10`。）

**代价必须先知道**：`lambda = R·S ≈ 2.75 req/s`，但**爬坡期就要 40 分钟**（第一批
session 走完一整轮才建立起工作集），总跑时约 65 分钟，比 session-cap 那条路（35 分钟）
慢一倍。

**实测结果（2026-09-11，两个 cap 均为 0）**：跑到 172/300 conversations、5833 条 prefill
时停下。爬坡过程值得记下来，因为中途读数完全不能用：

| runtime | blank 最低 | DRAM 加载占比 |
|---|---|---|
| 23 min | 1159 | 0.6% |
| 40 min | 304 | 1.2% |
| 停止时 | 117 | 7.4% |

前 40 分钟 blank 还在上千，看起来像「配置没压出效果」，其实只是工作集还没建立。**开环下
判断配置好坏至少要等一个 `T_sess`**（这里 40 分钟），否则会误判。

### 复盘：sum(e2e) 与墙上时间差 2.5 倍，不矛盾

**为什么 `sum(e2e)` 只有 1463 s 而测试跑了 61 分钟**：前者是两个 prefill worker 的
**忙碌时间之和**，后者是墙上时间。一条 session 的 40 分钟里：

| 环节 | 单轮 | × 32.4 轮 |
|---|---|---|
| prefill | 251 ms | 8 s |
| decode | 1,210 ms | 39 s |
| **tool time** | **72.4 s** | **2,347 s（98%）** |

tool time 完全主导。这正是它作为「纯压力旋钮」的定义——拉长 session 让更多 session 同时
驻留、撑大工作集，但**不产生任何 GPU 工作量**。

实测 prefill GPU 占用率：

```
1463 s (两 worker 的 sum(e2e)) / (3681 s 墙上 x 2 worker) = 19.9%
```

而上表预测 `rho = 0.35`。两者也不矛盾：公式里的 `lambda = R·S = 2.75 req/s` 是**稳态
值**，而实跑到 172/300 就停了、session 还在陆续启动，实际 `lambda = 5833/3681 = 1.58
req/s`（稳态的 58%），`1.58 × 0.251 / 2 = 0.199` —— 与实测占用率精确吻合。跑满稳态
`rho` 会升到 0.35。

**这也修正了形态差异的归因**：不是队列堆积（队列几乎是空的，`Waiting` 基本为 0），而是
72 s 的轮间隔本身——期间有大量别的 session 写入 block，等自己回来时前缀已被淘汰。是时间
间隔而非并发竞争。

注意这组的形态和 session-cap 那组不同：**只有 7.4% 的请求真的加载了**（session-cap 组是
65.7%），但这些请求单条加载量极大，形成「少数请求承受极重加载」的长尾。

**结论**：开环这条路能制造加载压力，但可行域被 `Tt > 24 s` 卡住，且爬坡期至少一个 `T_sess`。
相比之下 `--max-inflight-session` 那条路是**把负载和压力解耦**（session cap 只抬压力不
抬负载），窗口宽得多、跑得也快——这就是两个 cap 存在的意义。

### 制造 DRAM 加载压力的可用配置

```bash
--max-inflight-req 16 --max-inflight-session 400 -n 4000 \
--session-arrival-rate 4.0 --tool-time lognormal:0,1
```

框架配置一行不用改（prefix HBM 保持开启），65.7% 的请求有 DRAM 加载。代价是 RPS 减半、span
拉长约 3 倍（843 s → 2423 s），一轮约 35 分钟。

**另一条路是跳过本地 prefix cache 查询**：请求体 `cache_use_control` 不含 1（回放用
`--cache-use-control 2`，见 [kv-placement.md](kv-placement.md)），强制每次复用都走 connector，于是 HBM 命中全变成
DRAM 加载。下表是当时用全局开关 `DISABLE_LOCAL_PREFIX_CACHE_LOOKUP`（已删，换成了按请求的 `cache_use_control`）
在同样的 benchmark 参数下测的：

| 跳过本地 prefix 查询 | 有 DRAM 加载的请求 | 平均加载 token | RPS |
|---|---|---|---|
| `false`（默认） | 5.2% | 153 | 3.99 |
| `true` | 89% | 5,648 | 2.13 |

但它是**绕过** HBM 制造 DRAM 流量，不是真的挤爆 HBM——队列几乎是空的（queue 1.3%），
池子也没溢出。上面那条 session 超订的路才是真淘汰。做 baseline vs KVPilot 的 A/B 时这个
开关两边必须一致。

### 其他加载路径开关

prelookup（lookup 时一次查好所有层副本，C++ `LayerwiseLoader` 把全部层的 H2D 一次 enqueue）、Mooncake 零拷贝
（CPU backend / local HBM / shm segment）、SSD loader 的 64 个读线程和 256 MiB 槽位都已固定，不再是开关；
`MOONCAKE_PRELOAD_EXPECTED_ENGINES` 由 `NUM_PREFILL_WORKERS` 推出。见 [kv-placement.md](kv-placement.md)「开关」。

| 变量 | 默认 | 作用 |
|---|---|---|
| `KVPILOT_SSD_OFFLOAD_PATH` | 空（关） | L4：打开 Mooncake SSD offload（master `--enable_offload`，每个 prefill 进程一个 `pid<pid>/` 子目录），并允许 prelookup 从本机 SSD 逐层加载。需要含 `SsdLayerwiseLoader` 的 c_ops 和含 `batch_locate_local_ssd` 的 Mooncake。见 [kv-placement.md](kv-placement.md)「L4」 |
| `KVPILOT_SSD_READ_GIBPS` | 空（不限） | 每个 prefill 进程（TP rank）SSD 读带宽上限，GiB/s。每次读在同一条时间表上按此速率预约，读完不早于预约结束（没有空闲攒下的突发额度）。按「本 GPU 分到的整机 SSD 带宽」设：裸读峰值 12.8 / 8 GPU = `1.6`，1P TP2 合计 2/8 峰值。`test_loader.py bench` 实测两进程合计 3.16 GiB/s、28.9 ms/1K token（不限时 7.5–9.4）。需要重编的 c_ops |
| `MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS` / `MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT` | 开 SSD 时 `1` / 全注意力层数（Qwen3-30B 48，Qwen3.6 10） | 心跳本身 fork 固定成 100 ms，前者只剩 client buffer GC 的间隔；后者是 bucket 大小（Mooncake 默认 500）。凑不满的剩余 key 由 fork 在下一次没有新 key 的心跳里单独写出（见 [kv-placement.md](kv-placement.md)） |
| `MODEL` | CFS 上的 Qwen3-30B 路径 | 默认直接用 CFS（见 [gpu-restore.md](gpu-restore.md)）。冷启动偏慢但不必先拷本地；仅 CFS 超时再改指 `/root/models/...`。混合注意力模型（`/cfs_cloud_code/kiry/model/Qwen/Qwen3.6-35B-A3B`）自动换块大小和相关参数，见 [kv-placement.md](kv-placement.md)「混合模型」 |
| `PREFILL_EXTRA_ARGS` | 空 | 只传给 prefill worker 的额外 vLLM 参数（按空格切分），例如 `--num-gpu-blocks-override 768` 把 HBM prefix cache 压小，用来在 `-p` 不变时制造 DRAM / SSD 命中 |
| `PREFILL_COMPILATION_CONFIG` | 空 | 传给 prefill worker 的 `--compilation-config`。不设时 `--max-num-seqs 1` 使最大捕获尺寸为 2，prefill 全走 eager；设成 `{"cudagraph_capture_sizes":[64,128,256,512,1024]}` 后 suffix ≤ 1024 走 PIECEWISE cudagraph，token 数向上补齐到最近一档，补齐的部分真实计算。**它会改变 compute 基线**，前后两组数据不能直接比 |

实测（`paper-figure/prefix-suffix-compute-vs-load/opt/README.md`）：直接从 pinned host
读的 gather kernel 会占着 SM 等 PCIe，64K 时让计算慢 2–13 倍；copy engine 带宽相同
（约 55 GB/s），几乎不影响计算。

### 跑起来后怎么验证压力够了

```bash
L=dynamo-log/<name>-logs
grep -o 'blank=[0-9]*' $L/prefill-0.log | sed 's/blank=//' | sort -n | head -1   # 应接近 0
grep -oE 'GPU KV cache usage: [0-9.]+%' $L/prefill-0.log | tail -20              # 应到 80%+
grep -o 'need to load: [0-9]*' $L/prefill-0.log | awk '{n++;if($1==0)z++}END{print z/n}'
grep -oE 'Waiting: [0-9]+' $L/prefill-0.log | sed 's/Waiting: //' | sort -n | tail -1
```

`Waiting` 持续两位数 → rho 估低了，降 R。`blank` 不归零 → 抬 session cap 或 Tt。

**注意 `total=NNNN` 是空闲块数，不是池子容量。** 池子 2465 块时 `total=2464` 意味着
**全池 99.96% 空闲**，一次淘汰都没发生——这个最像「出问题了」的数字其实代表零压力。
要看的是 `blank` 的**最低值**。

## nsys profiling

| 变量 | 默认 | 作用 |
|---|---|---|
| `NSYS_PROFILE` | `1` | 是否用 nsys 包住 prefill worker |
| `NSYS_DELAY` | `0` | 延迟多少秒开始采样 |
| `NSYS_DURATION` | `5000` | 采样时长（秒） |

profile 写到 `dynamo-log/<name>-logs/prefill-N.nsys-rep`。`NSYS_PROFILE=1` 时脚本会
`export VLLM_WORKER_MULTIPROC_METHOD=spawn`——nsys 需要 spawn（不是 fork）才能抓到
vLLM 的 worker 子进程。

### session 会卡在 Collection 不落盘

`dynamo_script.sh clean` 的 SIGTERM **不一定能让 nsys 收尾**：实测跑完 48 分钟、CPU
只用了 1 秒、`.nsys-rep` 一个字节没写。要手动停：

```bash
nsys sessions list                      # 找 STATE=Collection 的 session 名
nsys stop --session=profile-<pid>       # 逐个停，每个约几分钟（1 GB profile）
```

停完才会打印 `Generated: .../prefill-0.nsys-rep`。不做这步的话 TTFT 拆解脚本没数据可读。

## 部署形态与运行时变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `NUM_PREFILL_WORKERS` | `2` | P worker 数 |
| `NUM_DECODE_WORKERS` | `2` | D worker 数 |
| `LOG_DIR` | 由 `dynamo_script.sh` 传入 | 所有日志与观测输出的落点 |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | `1800` | 单次 execute_model 超时；长 prompt + 大量加载时容易撞默认值 |
| `KVPILOT_NUMA_BIND` | `1` | P / D 每个 worker 用 `numactl --cpunodebind=N --preferred=N` 起在它的 GPU 所在的 NUMA 节点（按第一张卡的 PCI 设备查，一个 worker 的卡跨节点直接报错）；`0` 不绑。不绑时 TP 4 的部分 rank SSD 加载变慢，见 [pitfalls.md](pitfalls.md)「跨进程/跨机」 |
| `TZ` | `UTC0` | 见下 |

剩下几个是写死的基础设施常量，一般不用动：`DYN_LOG=info`、
`DYN_TCP_RESPONSE_STREAM_HOST=lo`、`UCX_TLS=self,tcp,cuda_copy,cuda_ipc`、
`LMCACHE_TRACK_USAGE=false`、`MC_TCP_ENABLE_CONNECTION_POOL=1`。

**`PYTHONHASHSEED` 必须固定**：它决定 `NONE_HASH`——整条 prefix hash 链的根
（`kv_cache_utils.py:init_none_hash()`）。不设的话 vLLM 用 `os.urandom(32)`，于是各
worker 之间、两次跑之间的 block hash 全都对不上，跨进程的 prefix 复用直接失效（脚本里
的注释写的是「deterministic hash for KV event IDs」，同一个机制）。

注意两个脚本给的值不同：`dynamo_script.sh` 设 `${VLLM_PYTHON_HASH_SEED:-123}`，
`disagg_lmcache.sh` 里写死 `0`。后者在 worker 子进程里后设，所以**实际生效的是 `0`**，
要改就两处一起改。

模型走环境变量 `MODEL`（默认 Qwen3-30B-A3B），块大小由模型定：`disagg_lmcache.sh` 读 config.json，普通模型
`BLOCK_SIZE=512`，混合注意力模型（Qwen3.6-35B-A3B）1536，连同 prefix caching 等参数一起切换，见
[kv-placement.md](kv-placement.md)「混合模型」。

`--max-model-len` 没显式设，走默认推导 262,144——这同时决定了 KV 池子的硬下界 512 块
（见上面的「压测参数怎么调」）。

### `TZ=UTC0` 不能改成 `UTC`

必须是 POSIX 字面量 `UTC0`。在这个镜像里 `/usr/share/zoneinfo/UTC` 和
`/usr/share/zoneinfo/Etc/UTC` **两个文件装的都是 Asia/Shanghai 数据**（内嵌 POSIX 串
`CST-8`），所以任何经 zoneinfo 文件解析的名字都还是 UTC+8。`UTC0` 由 libc 直接解析，
不查文件。验证：

```bash
TZ=UTC0 python3 -c 'import time;time.tzset();print(time.strftime("%H:%M:%S %Z"))'
```

统一时区是必需的：LMCache、vLLM、mooncake 和 benchmark 都按 localtime 渲染时间戳，而
dynamo 发 ISO-8601 UTC，在 UTC+8 容器里两者差 8 小时，跨日志对时会**静默失效**（实测
4000 条请求的 queue wait 全算成 0，nsys 关联也整体偏 8 小时）。

## 一个 pgrep 自噬陷阱

不要在 GPU 上用 bash 循环等 benchmark 结束：

```bash
# 错：循环自己的命令行就包含这个模式，永远等不完
until ! pgrep -f benchmark_serving_multi_turn; do sleep 30; done
```

改成匹配 `python3.*benchmark_serving_multi_turn`，或者数
`ps -eo cmd | grep -c '[p]ython3 .*benchmark_serving'`。
