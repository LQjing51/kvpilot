# 论文图规范：复现与画图风格

两部分：**怎么存**（让每张论文图都能从原始日志重新跑出来），**怎么画**（OSDI 的风格，
落在 `paper-figure/paperstyle.py`）。日常探索用的图不受本规范约束，放在各自的
`dynamo-log/<run>-logs/` 即可；**只有我说某张图要进论文，才按下面建目录**。

## 一、复现规范

### 目录结构

论文图放在仓库根目录 `paper-figure/`，一张图（或一组共用同一次实验的子图）一个目录：

```
paper-figure/
├── paperstyle.py              共享样式，所有 plot.py import 它
└── <fig-name>/                如 motivation-wait-by-dram-hit
    ├── README.md              测试说明（模板见下）
    ├── raw/                   原始日志，原样拷贝，不改名不裁剪
    │   ├── benchmark.log
    │   ├── prefill-0.log  prefill-1.log
    │   ├── decode-*.log  frontend.log  mooncake_master.log
    │   └── run.out            runner 的完整输出（含开关自查行）
    ├── plot.py                一个脚本：从 raw/ 到 PDF，数据处理和画图都在里面
    └── <fig-name>.pdf         产出
```

`<fig-name>` 用小写短横线，写图的内容而不是编号（编号会随论文改稿变）。

### 每一项的要求

**`raw/`：原始日志**

- 原样从 `dynamo-log/<run>-logs/` 拷过来。**不放中间产物**（csv、sqlite 等），那些由
  `plot.py` 现算。
- 拷之前在 GPU 上 `ls -la` 看一眼大小。单文件 >50MB 的（toolagent 这种 23K 请求的
  prefill log 约 90MB）先 `gzip`，`plot.py` 直接读 `.gz`；**>100MB 的不能进 git**
  （GitHub 单文件上限），在 `.gitignore` 里排除，README 写明它在 GPU 上的绝对路径。
- `.nsys-rep`（每份约 250MB）**一律不入库**。图若依赖 nsys，README 写清 GPU 上的路径
  和重新导出的命令。

**`plot.py`：一个脚本完成全部**

- 输入只有 `raw/`，输出只有同目录下的 PDF。`python3 plot.py` 无参数即可运行。
- 解析日志、拼表、聚合、画图都写在这一个文件里，**不 import `utils/` 下的脚本**——
  `utils/` 会继续改，论文图要冻结在投稿那一刻的口径。可以从 `utils/` 复制代码进来。
- 开头 `sys.path.insert(0, <paper-figure 目录>)` 后 `from paperstyle import *`。
- 运行时打印图上每一个标注数字的来源值（例如「hit ≥ 90%：37%，等待占比 73%」），
  正文引用的数字从这里抄，不要从别处算。
- 只出 PDF，不出 PNG。自己要看就 `qlmanage -t -s 1600
  -o /tmp x.pdf` 渲到 `/tmp`。

**`README.md`：中文，越短越好**

只写复现需要的东西：配置、复现命令、每个量的口径、已知问题。**不写代码解释、不写数据分析和结论**（数字由 `plot.py` 打印，分析放正文）。

```markdown
# <图的内容>

- `<fig>.pdf`：<一句话>

## 配置
- 日期，机器，HEAD 提交 + 相对 HEAD 改动的文件
- 模型，并行，trace 与回放命令
- 与默认不同的开关

## 复现
<GPU 上的命令>
python3 plot.py

## 口径
- 每个量 = 日志里哪个字段

## 注意
<已知偏差，没有就删掉这一节>
```

### 流程

1. 你说「这张图要进论文」。
2. 建目录，拷 `raw/`，写 `plot.py`（套 `paperstyle`），跑出 PDF，写 README。
3. **在干净的拷贝里复跑一遍** `python3 plot.py`，确认不依赖 `/tmp` 或本机的其他文件。
4. **不自己 commit**，全部保持 unstaged，等你说提交。

### 代码版本怎么记

实验常常跑在还没提交的代码上，所以 README 的「配置」里不写 commit sha，而是**列出这次实验相对 HEAD
改了哪些文件**（`git status --short` 里与实验相关的那几行），并注明 HEAD 是哪个提交。
你提交之后，再把 README 里这一行换成实际的 commit sha。

## 二、画图风格（OSDI）

来源：读了 `~/Desktop/paper-all/paper/` 下的 OSDI 论文。渲染查看了 6 篇的图页
（DistServe、Llumnix、InfiniGen、Parrot、ECHO、No Buffer No Bottleneck），另外 4 篇
（Sarathi-Serve、Contextra、Prism、Simple is Better）只读了文字和 `pdffonts`。
样本不大，下面是共性，不是硬规则。实现都在 `paper-figure/paperstyle.py`。

### 尺寸与布局

| 项 | 规范 |
|---|---|
| 单栏 | 最宽 3.33 in（`COL`），通常放 1–2 个小 panel |
| 通栏 | 最宽 7.0 in（`FULL`），一行 2–4 个 panel |
| panel 高 | 约 1.3–1.5 in（`PANEL_H`），扁宽，宽高比 2:1 到 3:1 |
| 子图标注 | 每个 panel 下方 `(a) …`，有 LaTeX subcaption 就用 LaTeX 的，否则用 `panel_label()` |
| y 轴 | 每个 panel 各自一套 y 轴，同一行对齐；很少共享 |

**宽度跟着内容走，尽量紧凑。** `COL` / `FULL` 是上限，不是默认值。单个 panel
如果内容少（一两条线、一团散点），缩到 2.0–2.4 in，刻度和图例仍然读得清就行。
这样两张小图可以在单栏里并排。图例用 `handletextpad=0.3, borderaxespad=0.2`
贴紧坐标轴。

评估部分最常见的是**通栏 + 一行多 panel**（DistServe Fig 8 是 2×3 网格，ECHO Fig 12/13
是 2 panel）。端到端大矩阵（Llumnix Fig 11：6 条 trace × 7 个指标）也是常见套路。

### 字体

- **无衬线**：图里用 Helvetica / Arial（10 篇里 8 篇嵌的是 matplotlib 默认的 DejaVu Sans），
  正文是 Times。两者不一致在 OSDI 是常态，不用强行改成衬线。
- **字号偏小**：刻度 6.5pt，轴标签、legend 7–7.5pt，约为 10pt 正文的 0.65–0.75 倍。
  因为图按实际尺寸画、不缩放，这些就是印出来的大小。
- **只有加速比标注用粗体**（如 ECHO 的「2.15x」），其余都不加粗。
- `pdf.fonttype = 42`：嵌 TrueType，投稿系统会检查 Type 3 字体。

### 颜色

- 一张图 3–6 个系列。
- **「我们的」用最显眼的颜色**：最饱和、最深，或者唯一的实心填充。Parrot 里自己是实心红，
  baseline 是带斜线的橙和蓝；InfiniGen 用一组深浅不同的蓝。
- **本项目的选择**：用 dataviz 校验过的六色，固定顺序（`SERIES`），**系列多少都按这个
  顺序取，不循环**。「我们的」固定取 slot 1 蓝色并实心，baseline 依次取后面的颜色并加 hatch。
- **固定含义的颜色**：全文统一，读者看一次就记住了。所有图里**计算一律蓝、加载一律红**，
  不管是加载总耗时、exposed load 还是理论加载时间。
  - `LOAD` / `WAIT`（红 `#e34948`）：KV 加载、GPU 等 KV、exposed load
  - `COMPUTE`（蓝）：prefill 计算
- **灰度也要能分辨**：柱状图里 baseline 一律加 hatch（`HATCHES`），折线一律加 marker。
  OSDI 论文多数没有做到（tab10 转灰度后分不开），我们做到。
- 子任务建议过一组 Okabe-Ito 配色，我用 validator 测过：深蓝超出亮度带（FAIL），
  粉和绿在 deutan 色盲下 ΔE 只有 7.6，所以没有采用。

### 图的类型选择

| 要表达什么 | 用什么 | 例子 |
|---|---|---|
| 指标随负载变化（SLO 达成率、吞吐 vs 请求速率） | 折线 + 每个点都画 marker | DistServe、ECHO |
| 不同参数下的延迟（长度、chunk、batch） | 分组柱状图 | Parrot、No Buffer、InfiniGen |
| 时间拆解（TTFT 各组成部分） | 堆叠柱状图 | InfiniGen Fig 14、No Buffer Fig 11b |
| 每个请求的分布（命中率、TTFT） | CDF（阶梯线） | 本项目常用；看过的 OSDI 页里没见到，不确定其余几篇有没有 |

### 坐标轴

- **四边框 + 刻度朝里**，这是 OSDI 最常见的样子（DistServe、Parrot、InfiniGen、No Buffer）。
- **只画 y 方向的浅灰虚线网格**（ECHO 有；多数没有，但有网格更好读数值）。
- 柱状图 y 从 0 开始；折线可以从接近数据的地方开始。
- 每个轴 3–5 个主刻度，跨度大才用 log。
- 轴标签写成「量 (单位)」：`TTFT (s)`、`Throughput (req/s)`。
  比例统一写百分号刻度，标签里不再重复写 %。

### 图上的文字：只写名字，不写解释

轴标签、图例、标注只写**量的名字**，**不加括号说明**。解释和样本量写进 caption 或正文。

| 不要 | 要 |
|---|---|
| `DRAM hit rate (share of prompt tokens)` | `DRAM hit rate` |
| `Request with DRAM load (2,836 of 23,608)` | `Request with DRAM load` |
| `Wait share, moving mean (25 req)` | `Moving mean` |
| `Requests sorted by hit rate (n requests; tick = hit rate)` | `Requests sorted by hit rate` |

**唯一例外是单位**：`TTFT (s)` 这种括号里只有单位的可以保留。

量的定义、`n=`、窗口大小、分桶方式，统一放进 `README.md` 的「口径」和论文 caption。

### 图例

- **多 panel 时：所有 panel 共用一个图例，放在顶部图外，单行，无边框**（`top_legend()`）。
  这是看到的最主要的样式（DistServe、Llumnix、ECHO、No Buffer Fig 12）。
- 单个小 panel：图例放在图内左上角。
- 不要每个 panel 各放一个图例。
- **图例只放大类**（如 Compute / Load）。同一大类下的多条线**同色同线型**，不再用深浅或虚实区分，
  **名字直接写在线旁边**（如 `suffix 100`、`DRAM`）：贴在线上方、顺着线的斜率倾斜、和线同色、加白边避免被别的线压住。
  标签放在各条线彼此分开、不和交叉点重叠的位置。见 `paper-figure/context-time/plot.py` 的 `label()`。

### 线与标记

- 线宽 1.2pt，marker 3.5pt，**每个数据点都画 marker**（○ □ △ ◇ ▽，按 `MARKERS` 顺序）。
- 所有系列一律实线，靠颜色 + marker 区分。**虚线只用于参考线**（SLO 阈值、显存上限、
  理论值）。

### 标注

- **加速比**：在最能说明问题的 x 位置画双向箭头，旁边写粗体「2.1×」，颜色用 `OURS`。
- **数值标在柱顶或点旁**：只标关键的几个，不要每个都标。
- **超出范围的柱子**：用 `ylim` 截断，柱顶写真实数值（InfiniGen 的「2007.4」）。
- **关键阈值**：用点线画出曲线与目标的交点（DistServe）。

### 图注（caption）

- USENIX 模板：`Figure N:` 正常字重（ACM 模板是 `Figure N.`）。
- 1–2 句：**第一句交代设置**（模型、trace、硬件），**第二句给结论**
  （Parrot Fig 13：「All applications finish earlier in Parrot.」）。
- 图上有标注的，图注里要解释标注是什么意思（ECHO：「Annotated numbers denote the
  effective batch size」）。

### 和 matplotlib 默认值的区别（自查清单）

出图后对着检查：

- [ ] 宽度跟着内容收窄（不超过单栏 3.33 / 通栏 7.0 in），没有在 LaTeX 里缩放
- [ ] 字号 6.5–7.5pt，sans-serif，`pdf.fonttype` 是 42
- [ ] 轴标签、图例、标注里没有括号说明（单位除外），没有 `n=`
- [ ] 四边框、刻度朝里、只有 y 方向浅灰虚线网格
- [ ] 多 panel 时只有一个顶部单行无边框图例
- [ ] 折线每个点都有 marker；虚线只用于参考线
- [ ] 柱状图 baseline 带 hatch，转灰度后仍然分得清
- [ ] 「我们的」是蓝色实心；等待/exposed load 用 `WAIT` 红色
- [ ] 每个关键数字在 `plot.py` 运行输出里能找到
- [ ] `head -c 4 x.pdf` 是 `%PDF`
