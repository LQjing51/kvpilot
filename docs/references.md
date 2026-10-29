# 参考文档

飞书文件夹：<https://my.feishu.cn/drive/folder/Fu6lf23zkltuHZd37dcc8F6Dnve>

## 核心文档

| 文档 | token | 内容 |
|---|---|---|
| **KVPilot Proposal** | `WFQWdoXPFoAhl0xZR0dc6gQenQf` | 项目立论：background / existing solutions / insight / design。写论文和讲故事从这里开始 |
| **KVPilot Paper Story** | `Aep9dyb78oAHKExRVjWcuNsjnwc` | 论文叙事与 TODO；LP/WQP 的上下界公式 |
| **KVPilot [Testbed]** | `OaDxdSEpcozQflxHYNzchPwknKe` | 总览与实验记录：环境搭建、数据集、特性测试、Motivation 实验、机器拓扑、代码解读 |
| **改动清单与调用流程** | `Bw3DdYVhMoVzSsxtarXcjxVLnHh` | 按文件粒度的 diff 索引 + 两条调用链图，**review 用** |

## 设计文档

| 文档 | token | 内容 |
|---|---|---|
| 时间常数推导 | `QC2Vdtmcqo2Cpzxt1Khc2j9inab` | `a`/`b`/`c` 的来源与各模型取值。代码实现是 `utils/plot_context_time.py` |
| DRAM-safe eviction（离线版） | `LsN8dlfsLopfYVxuTOKcuZEWnTh` | 离线判据的设计 |
| **在线 dram-safe 反馈** | `SnUXdzCJvoyczZxWTCec08nRnBc` | 该功能的**唯一**设计文档：判据、D→P 通道、P 侧重分层、forced_output、代码改动、dynamo 编译、验证。逻辑变更时**就地修订它**，不要新开文档或在末尾追加 |
| Agent trace 处理链路 | `TZ9vdJ8KLo49OVx7cDoc6ClinKc` | SWE-smith 与 TraceLab 两条链路的预处理、四阶段实例、回放 token 映射、在线判据接入 |

## 组件与优化

| 文档 | token |
|---|---|
| Mooncake + LMCache 服务架构全流程报告 | `Tf0tdxLRBo1YrpxbEDTch328nlc` |
| LMCache+Mooncake layerwise 支持 | `GzjodTZAaoZqwJxABttcczhUnGc` |
| Mooncake GET 直通 HBM 优化 | `SWjhdKreqoNLe5xFiXgcsreknHy` |
| Mooncake GET 两层拷贝优化 | `S9D8dcTCDoCZADxnFHScHUsGnBc` |
| Layerwise 查询 layer-key hit 耗时优化 | `F4BVdkyJvoAw13xkkg2cKEZTnmc` |
| Decoder Layerwise Store 失败 [CUDAGraph 设置] | `UjTidebAbo7YtxxMEBkcwyMenoe` |
| GPUConnector 解读 | `E3nWduH8XoO4o1xl6MBc4MFSnVg` |
| Mooncake Trace 适配 | `RIXmdPbpIoFHTgxD6nfcHiwpnQf` |
| PCP 和 DCP | `L37Qdy07xojkblxTZsecltn7nnf` |
| KV Cache Survey | `XETJdYIzSo5mBrxyWimcQCXunyg` |

## 本地资料

- Slides：`~/Desktop/KVPilot.pdf`

## 怎么读飞书文档

装了 lark-cli 的话：

```bash
lark-cli docs +fetch --doc <token> --as user
```

输出是 HTML-ish 的 JSON，直接看结构可以只抽标题：

```bash
lark-cli docs +fetch --doc <token> --as user --format json \
  | python3 -c "
import json,sys,re
c=json.load(sys.stdin)['data']['document']['content']
for m in re.finditer(r'<h([1-4])[^>]*>(.*?)</h\1>', c):
    print('  '*(int(m.group(1))-1)+'- '+re.sub(r'<[^>]+>','',m.group(2)).strip())
"
```

列出整个文件夹：

```bash
lark-cli drive files list \
  --params '{"folder_token":"Fu6lf23zkltuHZd37dcc8F6Dnve","page_size":100}' \
  --as user --format json
```
