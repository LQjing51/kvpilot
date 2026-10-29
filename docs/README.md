# KVPilot 文档索引

研究项目：**Toward Zero Overhead KV Cache Loading in Agentic LLM Serving**（iSING Lab, HKUST）。

目标是让 KV cache 从 DRAM 加载的耗时被 prefill 计算完全掩盖。核心论点是现有系统按
「复用概率」（LRU / reuse prediction）放置 KV blocks，没有区分请求是 **load-bound**
还是 **compute-bound**——前者若 KV 不在 HBM 会导致 GPU 空等，后者占着 HBM 却没有收益。

## 从哪读起

| 你想做什么 | 读这个 |
|---|---|
| 理解项目在解决什么问题、成本模型怎么来的 | [concepts.md](concepts.md) |
| 起服务、跑压测、调压力参数、nsys / 时区 | [benchmark.md](benchmark.md) |
| 搞清楚请求是怎么发出去的、chat template 归谁套 | [request-paths.md](request-paths.md) |
| 处理 agent trace、跑离线验证、生成查表 | [datasets.md](datasets.md) |
| 改 KV placement / eviction 代码 | [kv-placement.md](kv-placement.md) |
| 量测 load 阻塞、HBM 分布、TTFT 拆解 | [observability.md](observability.md) |
| 碰到反直觉的现象，怀疑踩坑了 | [pitfalls.md](pitfalls.md) |
| 看已经跑过的实验（六个配置的 exposed load / TTFT / 池子分布） | [evalution/demo_test_results.md](evalution/demo_test_results.md) |
| 找飞书设计文档 | [references.md](references.md) |

**环境变量没有单独的文档**：每个开关记在它所控制的功能那一篇里，因为开关的含义离开
功能上下文就不成立，分开写必然会和实现漂移。找不到某个变量时按功能归属去翻——
placement 类（`KVPILOT_*DRAM_SAFE` / `KVPILOT_MACHINE` / `KVPILOT_TOOL_RESP_LEN_PATH`）
在 [kv-placement.md](kv-placement.md)，观测类（`GET_HBM_POOL_DISTRIBUTION` /
`KVPILOT_LOAD_STALL`）在 [observability.md](observability.md)，其余（加载路径、nsys、
`TZ`、worker 数）在 [benchmark.md](benchmark.md)。

## 顶层结构

```
kvpilot/
├── vllm-dev/         vLLM fork，分支 0.23.0    ← KV placement / eviction 主战场
├── lmcache-dev/      LMCache fork，分支 0.4.6  ← zero-copy 加载路径
├── Mooncake/         Mooncake fork，0.3.10.post2 ← 分布式 KV 存储后端
├── dynamo-dev/       Dynamo fork，分支 1.3.1   ← PD 分离编排、KV router
├── datasets/
│   ├── mooncake-trace/           conversation_trace_turn4_half.jsonl
│   └── agent-traces/
│       └── swe-smith-trajectories/   SWE-smith agent 轨迹 + 预处理与验证脚本
├── utils/            作图脚本与产出的图（见下）
└── docs/             本目录
```

整个目录是一个 git repo（`github.com/LQjing51/kvpilot`），四个 fork 是普通子目录，
不是 submodule。每个 fork 先以一个提交导入上游原样代码并打 tag，KVPilot 改动叠在其后：

| 目录 | 上游 | tag |
|---|---|---|
| `vllm-dev/` | vLLM v0.23.0 | `upstream/vllm-dev` |
| `lmcache-dev/` | LMCache v0.4.6 | `upstream/lmcache-dev` |
| `Mooncake/` | Mooncake v0.3.10.post2 | `upstream/Mooncake` |
| `dynamo-dev/` | Dynamo v1.3.1 | `upstream/dynamo-dev` |

要看某个 fork 改了什么：`git diff --stat upstream/vllm-dev HEAD -- vllm-dev`。
迁移前的独立 fork 仓库（`LQjing51/vllm-dev` 等）不再更新。

`Mooncake/extern/` 下两个 submodule 由根目录 `.gitmodules` 登记，需要时
`git submodule update --init`；Dockerfile 自己装 yalantinglibs，平时不必拉。

## utils/ 里有什么

作图脚本、跑实验的 runner、分析脚本和它们的产出都放在这里，与各 fork 的代码分开。

**作图**

| 文件 | 作用 |
|---|---|
| `plot_context_time.py` | 时间常数推导与作图（a/b/c 常数的单一来源），产出 `context_time.pdf` |
| `plot_hbm_pool_distribution.py` | HBM block 分布二维直方图（`--pdf-output` 出图，`--output-file` 出文本热力图，`--csv-output` 出网格），见 [observability.md](observability.md) |
| `plot_load_stall.py` | 每次 prefill/forward 的 load 阻塞，产出 `load_stall.pdf` |

**跑实验与分析**（`run_demo.sh` / `plot_pools.sh` 在 GPU 机器上跑，其余本机可跑）

| 文件 | 作用 |
|---|---|
| `run_demo.sh` | 一轮实验的完整流程：clean → 等显存 → setup → **自查跑的是仓库代码** → benchmark → clean。配置全走环境变量 |
| `plot_pools.sh` | 批量出池子分布图，并校验产出真是 PDF（`head -c 4` 是否 `%PDF`） |
| `stall_share.py` | 把 load-stall 日志聚合成 exposed load 占比，带自洽检查 |
| `compare_runs.py` | 两轮的 token / stall 分解对比（谁在加载、加载多少、每 1k token 多贵） |
| `compare_workload.py` | 两轮跑的是不是同一套请求：集合重合、顺序、发送时刻 |
| `label_diagnostics.py` | 实测 dram-safe 标签能不能学，四个子命令：`signal` / `overhead` / `noise` / `denominator` |

**产出**

| 文件 | 作用 |
|---|---|
| `context_time.pdf` | 成本模型的曲线图 |
| `load_stall.pdf` | load 阻塞六联图 |
| `kv_replicate.{pdf,png,svg}` | KV 副本示意图（slides 用） |

`plot_context_time.py` 把 PDF 写在**脚本自己所在目录**，所以移动脚本时 PDF 的落点
会跟着走。另外两个脚本的输出路径由命令行参数给。

**出图只产 PDF，不产 PNG**（`kv_replicate.{png,svg}` 是 slides 用的素材，不是脚本产出）。
自己要看图就在 `/tmp` 下渲一份临时的，别放在交付目录旁边。GPU 机器上没有 `pdftoppm` /
`convert`，要看图得改 `savefig` 落到 `/tmp/*.png` 再取回来。

**本机没装 matplotlib**，作图要在 GPU 机器上跑，见 [benchmark.md](benchmark.md) 的
「同步代码到 GPU 机器」一节。分析脚本（`stall_share` / `compare_*` / `label_diagnostics`）
只用标准库，本机可以直接跑。

`run_demo.sh` 和 `plot_pools.sh` 通过 `KVPILOT_ROOT` 定位仓库，默认取脚本自身的上级目录，
找不到再退回 `/root/kvpilot`。**`scp` 到 GPU 机器不通**，传单个文件用
`base64 | ssh ... base64 -d`（见 [benchmark.md](benchmark.md)）。


## 工作约定

- 除非明确要求，**不要在 GPU 机器上起服务、跑测试**。
- 除非明确要求，**不要把修改后的代码同步到 GPU 机器**。
- 同步命令见 [benchmark.md](benchmark.md)。
- 在运行 benchmark 测试时，你检查 log 运行成功之后，你就可以空出时间来做其他的工作（10分钟回去看一次即可），而不是一直等待测试结束
- Monitor 请求不需要经过我批准，直接运行即可