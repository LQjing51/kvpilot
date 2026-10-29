# KVPilot

研究项目：**Toward Zero Overhead KV Cache Loading in Agentic LLM Serving**（iSING Lab, HKUST）。

目标是让 KV cache 从 DRAM 加载的耗时被 prefill 计算完全掩盖。核心论点是现有系统按
「复用概率」（LRU / reuse prediction）放置 KV blocks，没有区分请求是 **load-bound**
还是 **compute-bound**——前者若 KV 不在 HBM 会导致 GPU 空等，后者占着 HBM 却没有收益。

## 文档在 docs/

**详细文档已拆分到 [`docs/`](docs/)，本文件只留索引。** 改动某个板块时，就地修订对应
的那份文档，不要把内容搬回这里。

| 文档 | 内容 |
|---|---|
| [docs/README.md](docs/README.md) | 文档索引、仓库结构、utils/ 里有什么 |
| [docs/concepts.md](docs/concepts.md) | 项目概念：问题背景、两个 Insight、三个 Challenge、成本模型 |
| [docs/benchmark.md](docs/benchmark.md) | 起服务、跑压测、压力参数、nsys、时区、同步代码到 GPU |
| [docs/request-paths.md](docs/request-paths.md) | 三条打请求的路线、chat template 归属、Dynamo 的约束 |
| [docs/datasets.md](docs/datasets.md) | agent trace 处理链路、预处理四件事、离线验证 |
| [docs/kv-placement.md](docs/kv-placement.md) | KV placement / eviction 的开关、代码位置与调用链 |
| [docs/observability.md](docs/observability.md) | HBM 分布图、load 阻塞计量、TTFT 拆解 |
| [docs/pitfalls.md](docs/pitfalls.md) | 易错点合集 |
| [docs/references.md](docs/references.md) | 飞书文档索引 |

**环境变量不单独成篇**，每个开关写在它所控制的功能那一篇里——开关的含义离开功能上下文
就不成立，分开写必然和实现漂移。

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
├── utils/            作图脚本与产出的图
└── docs/             项目文档
```

整个目录是一个 git repo（`github.com/LQjing51/kvpilot`）。每个 fork 先以一个提交导入
上游原样代码并打 tag `upstream/<fork>`，KVPilot 改动叠在其后。要看某个 fork 改了什么：
`git diff --stat upstream/vllm-dev HEAD -- vllm-dev`。

## 最常用的三条命令

```bash
# 起服务 / 压测（GPU 机器上，注意先 export PYTHONPATH，见 docs/benchmark.md）
cd vllm-dev/examples/disaggregated/lmcache/disagg_prefill_lmcache_v1
bash dynamo_script.sh setup <name> && bash dynamo_script.sh benchmark <name>

# 同步代码到 GPU
bash ~/Desktop/code/sync_to_devcloud.sh local:$HOME/Desktop/code/kvpilot Taiji-GPU:/root

# 确认 GPU 上跑的真是仓库代码而不是 stock vLLM（0 就是没生效）
grep -c 'Free blocks distribution' dynamo-log/<name>-logs/prefill-0.log
```

## 注意事项

- 除非我要求，否则**不要在 GPU 机器上起服务、跑测试**。
- 除非我要求，**不要将修改后的代码同步到 GPU 机器**。
