# Phase 1 · 全盘诊断报告与企业级重构规划

> 对象：`产品实现/`（智评车行 · 新能源汽车评论分析与产品改进决策平台）
> 方法蓝本：硕士论文《融合情感分析与 PLTS-VIKOR 的新能源汽车产品改进多属性决策研究》（95 页，已逐节精读 §4.1–4.5、§5.1–5.8）
> 诊断范围：`app.py`（484 行）、`src/`（22 文件 / 2,301 行）、`config/`、`data/`（7 个 CSV，最大 17MB）、`outputs/`、虚拟环境 `vene/`（47 个包）
> 实测验证：AHP/DEMATEL/VIKOR 模块实际运行、Streamlit 启动冒烟（HTTP 200）、数据加载耗时基准、依赖可安装性 dry-run

---

## 一、五大核心问题诊断

### P0-1 · 算法层是"伪实现"——PLTS-VIKOR 全链路缺失，结果为常量回填 🔴 致命

**证据**

| 位置 | 问题 |
|---|---|
| `src/decision/run_decision.py:50-67` | `compute_vikor()` 形参 `f_star/f_minus/v` 全部未使用，函数体直接 `PAPER_S.copy()` 返回论文表 5.16 的常量 |
| `src/decision/run_decision.py:28-35` | 正负理想解 `F_STAR/F_MINUS`、论文结果 `PAPER_S/R/Q/PXI` 硬编码为模块级常量 |
| `src/decision/run_decision.py:41-45` | AHP 权重 `wA`、DEMATEL 权重 `wD` 硬编码，`ahp.py`/`dematel.py` 算出的真实结果被丢弃（仅打印对照） |
| `src/decision/run_decision.py:100-110` | `vikor_results.csv` 是常量导出，不是计算产物 |

**对照论文应有而完全没有的环节**（§4.3 步骤一~八）：
PLTS 决策矩阵（表 5.13，6 属性 × 12 指标的概率语言项 `H={l₃^0.2, l₄^0.8}`）→ 式 4.8 规范化 → 式 4.9/4.10 的 **Sᵢ、Rᵢ 区间**（上/下界）→ 式 4.11 的 **Qᵢ 区间** → 正态分布区间数 Q′ᵢ → 式 4.12 **折衷可能度矩阵 P** → 式 4.13 **总体优势度 P(xᵢ)**；以及 §4.4 的 λ 敏感性（表 5.18）与 TOPSIS/前景理论/传统 VIKOR 对比（表 5.19）。

**可复算性实测（本机 venv 实跑）**

| 项目 | 代码计算值 | 论文基准 | 结论 |
|---|---|---|---|
| AHP 权重 wA | 0.6424 / 0.0724 / 0.2852 | 0.434 / 0.187 / 0.379 | ✗ 不可复现（方根法与特征向量法结果一致，均不符） |
| AHP 一致性 CR | **0.3259**（>0.1，不通过） | 0.0846 | ✗ 论文表 5.7 矩阵本身无法推出其权重 |
| DEMATEL 一级权重 wD | 0.2859 / 0.3553 / 0.3588 | 0.516 / 0.204 / 0.280 | ✗ 换行/列最大值归一化均不变 |
| 表 5.9 录入值 | `l4` | 论文含 2 处 **`l5`**（c14→c12、c21→c31） | ⚠️ 代码静默降级，且 `l5` 超出论文自定义 L={l₀..l₄} |

**影响**：产品核心卖点"情感分析 + PLTS-VIKOR 融合模型"实际不可计算、不可调参、不可解释、不可复现。任何调 λ、换数据的操作都不会改变输出——这对 OEM 决策客户是可信度灾难，也是答辩/评审时最容易被击穿的点。

---

### P0-2 · 单文件 UI + 全量 eager 渲染，无导航、无状态、无解耦 🔴 高

**证据**

- `app.py` 单文件 484 行：260 行内联 CSS（第 26–175 行）+ 5 个 Tab 业务逻辑全部平铺。
- `st.tabs()`（`app.py:192-194`）**不提供懒加载**——5 个 Tab 的全部绘图代码每次交互都执行，实测每次 rerun 重建 8+ 个 Plotly Figure。
- `st.session_state` **零使用**；`pages/`、`tests/` **空目录**；`src/visualization/`、`src/isa/` 只有空 `__init__.py`（README 宣称的"阶段 7 看板页面"未落地）。
- `ROOT` + `sys.path.insert` 在 8 个模块中重复粘贴（`run_sentiment.py:24-25`、`run_isa.py:24-25` 等）——循环 import 与打包隐患。
- CSS 依赖 Streamlit 私有选择器 `[data-baseweb="tab-list"]`、`[data-testid="stMetric"]`；图表使用已弃用 kwarg（`plotly_chart.py:433` 明确 deprecated），版本升级即碎。
- `run_sentiment.py:43` 用 `eval()` 解析 tokens 字符串（注入风险 + 慢）；`run_isa.py:19` 硬编码 PingFang 字体（跨平台崩溃）。

**影响**：无法并行开发、无法写测试、无法做多视图共享状态（全局品牌/车型筛选）、无法做局部刷新（仿真器滑杆需要每次全量重算）。

---

### P1-3 · 结论与数据脱钩：UI 上的"结论"是写死的文案，且与论文口径互相矛盾 🟠 高

**证据**

| 位置 | 硬编码内容 |
|---|---|
| `app.py:280-282` | 聚类纯度 `0.502`、轮廓系数 `0.106`（不读任何产物） |
| `app.py:368-370` | AHP/DEMATEL/综合权重三元组（不读 CSV） |
| `app.py:407` | `st.success("**性价比** 重要性最高…")` 写死结论文本 |
| `app.py:350-356` | 模型 P/R/F1 性能表写死 |
| `app.py:428-431` | ISA 象限分割线 3.43 / 3.55 写死在绘图参数里 |
| `app.py:480-484` | 象限解读卡片文案写死（"保持区：性价比""改进区：续航"） |

**与论文的口径冲突（必须显式管理）**

- 论文表 5.20：续航**满意度 = 4.1**；论文图 5.10 与正文（"A4 处于第四象限、低满意度"）：续航**满意度 = 2.8**——论文自身矛盾。
- `run_isa.py:31-35` 的 `ISA_DATA` 采信 2.8，但同文件第 122-125 行打印的"论文对照"写的是性价比 (4.0,4.0)、外观 (3.0,3.5)，与数据 (4.2,3.1) 又不一致。
- ISA 满意度来自 286 份问卷硬编码，**而平台明明已算出 46,459 条评论的属性情感值**（`outputs/results/attr_sentiment.csv`）却未被用于 ISA——"评论→满意度"链路断裂。

**影响**：换真实数据、调 λ、重跑流水线，页面结论纹丝不动。看板退化为"演示视频"，而非决策工具。

---

### P1-4 · 四大企业级场景功能全缺 🟠 高

对照目标产品形态逐项核对：

| 目标能力 | 现状 |
|---|---|
| **Executive Dashboard**（VIKOR 排序 + 雷达 + 竞品对比） | ✗ 无雷达图；无品牌/车型维度分析（`brand/model` 列只用于画评论量条形图）；无 KPI 卡片化叙事 |
| **Aspect-Based Sentiment Explorer**（属性下钻 + 热力图） | ✗ 一条评论只有单一属性标签，**无 aspect 级情感抽取**；`comments_clustered.csv` 的 `cluster` 列在 UI 中完全未使用（簇标签未映射到六属性）；无评论下钻列表 |
| **PLTS-VIKOR Simulator**（调权重/λ/v + 实时重算 + 灵敏度） | ✗ 所有参数不可调，无灵敏度分析图 |
| **Insights & Report Builder**（改进推演 + 一键导出） | ✗ `outputs/reports/` 空；导入器 `importer.py` 只有 CLI，**无上传 UI**；无任何导出 |
| 桑基/旭日、二维决策矩阵散点、状态指示器 | ✗ 全无 |

---

### P2-5 · 工程化与数据资产管理缺失 🟡 中

- `requirements.txt` **零版本钉死**，且与实际 venv 不符（实际 streamlit 1.50.0 / pandas 2.3.3 / plotly 7.1.0；缺 `watchdog`，启动即告警）。
- **无测试**（`tests/` 空）、**无版本控制**（无 `.git`）、无日志、无 schema 校验、无 `.gitignore`。
- 中间产物冗余：`sentiment_results.csv` **17MB**，重复存储 `comment_text/cleaned_text/tokens/tokens_list` 四份文本；`tokens_list` 用 Python list 字符串存储。
- 缓存粗糙：`load_csv`（`app.py:186-188`）只缓存整表读取，无列裁剪、无 parquet 中间层（pyarrow 其实已装好）；NB 模型与 TF-IDF 每次 CLI 全量重训，无 `st.cache_resource`。
- 数据加载实测：`sentiment_results.csv` 0.13s / `comments_clustered.csv` 0.08s——**性能不是瓶颈，重复全量加载与重复绘图才是**（结论：优先做结构化缓存与惰性渲染，而非盲目优化算法）。

**严重度汇总**：P0×2（算法伪实现、架构单文件）、P1×2（结论脱钩、功能缺口）、P2×1（工程化）。

---

## 二、目标目录架构（重构后）

```
产品实现/
├── app.py                        # 壳层：设计令牌注入 + 侧边导航 + 视图路由 + 全局筛选器
├── requirements.txt              # 钉版本（含新增依赖）
├── run.command                   # 一键启动（保留）
├── config/
│   ├── settings.yaml             # 路径、分布参数、流水线开关（沿用）
│   ├── baselines.yaml            # 【新】论文表5.7/5.9/5.13/5.14-5.18/5.20 基准口径（唯一真源）
│   └── theme.yaml                # 【新】设计令牌：色板/字号/圆角/间距（深浅双主题）
│
├── core/                         # ★ 纯算法层：零 Streamlit、零文件 IO、100% 可单测
│   └── algorithm/
│       ├── models.py             # pydantic：PLTS、DecisionMatrix、VIKORResult、Scenario
│       ├── plts.py               # 概率语言术语集：L↔分值、期望得分、规范化、聚合
│       ├── ahp.py                # 判断矩阵 + 方根法/特征向量 + λmax/CI/CR（沿用改造）
│       ├── dematel.py            # Z→T 矩阵、中心度/原因度、权重（修正归一化口径）
│       ├── weights.py            # 式4.7 λ 线性融合 + 一级→12 二级指标权重展开
│       ├── vikor.py              # 式4.8-4.13 全链路，输出全部中间矩阵（可解释）
│       ├── sensitivity.py        # λ/v 网格扫描、排序稳定性（Kendall τ）
│       ├── benchmarks.py         # TOPSIS / 前景理论 / 传统 VIKOR 对比（表5.19）
│       └── isa.py                # 象限划分（均值可配，支持"问卷/评论情感"双口径）
│
├── services/                     # 数据与 NLP 层：有 IO、有缓存、无 UI 依赖
│   ├── data_store.py             # 统一读写 + parquet 中间层 + st.cache_data 封装
│   ├── pipeline.py               # 4 阶段流水线编排（进度、断点、产物版本号）
│   ├── preprocess.py             # 三级清洗 + jieba（沿用 clean/stopwords 迁移）
│   ├── aspect_mining.py          # TF-IDF + KMeans + **簇→六属性标签映射**
│   ├── sentiment.py              # VADER + NB + 融合（去掉 eval，改 json/parquet）
│   ├── features.py               # 属性×品牌×车型情感聚合；雷达/桑基/热力图数据整形
│   └── report.py                 # reportlab PDF（内置 STSong-Light 中文 CID 字体）+ Excel
│
├── state/
│   └── store.py                  # session_state 类型化 Schema：全局筛选、场景参数、主题
│
├── ui/
│   ├── styles/                   # base.css / tokens.css / shadcn 覆写（按主题变量切换）
│   ├── components/               # kpi_card、status_chip、radar_chart、sankey_flow、
│   │                             # matrix_scatter、heatmap_grid、comment_drilldown、
│   │                             # param_slider_group、section_header、export_button
│   └── views/
│       ├── executive.py          # ① Executive Dashboard
│       ├── sentiment_explorer.py # ② Aspect-Based Sentiment Explorer
│       ├── simulator.py          # ③ PLTS-VIKOR Interactive Simulator
│       └── insights_report.py    # ④ Actionable Insights & Report Builder
│
├── data/{raw,processed,features} # features/ 存 parquet（属性情感、雷达、桑基整形数据）
├── outputs/{results,figures,reports}
├── tests/                        # test_plts / test_vikor / test_weights / test_isa / test_store
├── docs/                         # 本文件 + 面向客户的使用手册
└── legacy/app_legacy.py          # 【迁移期】原 app.py 快照，随时可回滚
```

**迁移映射（旧 → 新）**

| 旧 | 新 | 处理 |
|---|---|---|
| `src/decision/ahp.py` | `core/algorithm/ahp.py` | 保留计算，暴露 `AHPResult` |
| `src/decision/dematel.py` | `core/algorithm/dematel.py` | 修正归一化口径，录入论文 `l5` 校准项 |
| `src/decision/run_decision.py` | `core/algorithm/weights.py` + `vikor.py` | **重写为真计算**，常量迁入 `config/baselines.yaml` |
| `src/decision/run_isa.py` | `core/algorithm/isa.py` | 数据驱动 + 双口径 |
| `src/preprocess/*` | `services/preprocess.py` | 去 `sys.path` hack |
| `src/attribute_mining/*` | `services/aspect_mining.py` | 增加簇→属性映射 |
| `src/sentiment/*` | `services/sentiment.py` | 去 `eval`，模型入 `cache_resource` |
| `src/data_ingest/*` | `services/data_store.py` | importer 逻辑接入上传 UI |
| `app.py` | `app.py`（壳） + `ui/views/*` | 拆分 |

---

## 三、数据流与解耦设计

```
[原始评论 CSV/Excel 上传] ──► services/preprocess ──► comments_cleaned.parquet
                                    │
                                    ▼
                          services/aspect_mining ──► cluster→属性映射 ──► comments_clustered.parquet
                                    │
                                    ▼
                          services/sentiment ──► 每条评论: vader/nb/final ──► sentiment.parquet
                                    │
                    ┌───────────────┴────────────────┐
                    ▼                                ▼
          services/features                  core/algorithm/weights (λ)
   属性×品牌情感 / 雷达 / 桑基整形           core/algorithm/plts + vikor
                    │                                │
                    ▼                                ▼
        data/features/*.parquet          VIKORResult{规范化F̄, S, R, Q, Q', P, P(xi)}
                    │                                │
                    └──────────► state/store ◄───────┘
                                 (session_state: 全局筛选 + 场景参数)
                                        │
                        ┌───────────────┼───────────────┬──────────────┐
                        ▼               ▼               ▼              ▼
                 executive       sentiment_explorer   simulator   insights_report
```

**缓存与性能策略**（数据实测 0.08–0.13s 加载，重点在"不重复算"）：

1. **中间层 parquet**：pyarrow 已在 venv 中；文本列只存一份，读取列裁剪，17MB CSV → 约 4–6MB parquet，加载 <60ms。
2. `@st.cache_data`：按 `(文件 mtime, 流水线版本, 筛选参数)` 作为 key；聚合结果单独缓存。
3. `@st.cache_resource`：NB 模型、TF-IDF vectorizer、停用词表、Jieba 词典——跨会话复用。
4. **惰性视图**：路由只渲染当前视图（替代 `st.tabs` eager 渲染）；仿真器滑杆用 `st.fragment(run_every=...)` 局部刷新，只重算 `core.algorithm`（纯 numpy，<10ms）。
5. `st.session_state` 类型化 Schema：跨视图共享品牌/车型/时间筛选、主题、当前场景参数；视图切换不丢状态。

---

## 四、UI/UX 设计方案

### 4.1 全局壳层

```
┌──────────────┬──────────────────────────────────────────────────────────┐
│  ◉ 智评车行   │  面包屑: Executive Dashboard      [主题 ◐] [⬇ 导出报告]   │
│  ─────────── │  全局筛选: [品牌 ▾] [车型 ▾] [时间 ▾]   数据新鲜度 ● 已同步 │
│  ▸ ① 高管大屏 │──────────────────────────────────────────────────────────│
│  ▸ ② 情感探索 │                                                          │
│  ▸ ③ 决策仿真 │   （视图内容区，12 列栅格，max-width 1600 居中）            │
│  ▸ ④ 洞察报告 │                                                          │
│  ─────────── │                                                          │
│  ▸ ⚙ 数据管理 │                                                          │
│  ─────────── │                                                          │
│  论文校准模式 ●│  底部状态栏: 样本量 · 双模一致率 · CR · λ · v              │
└──────────────┴──────────────────────────────────────────────────────────┘
```

- **导航**：`streamlit-option-menu`（0.4.0，已验证可装）侧边栏 + 视图注册表；选它而非原生 `pages/`，因为四个视图需要共享 session_state 与全局筛选，且路由需携带场景参数。
- **主题**：`config/theme.yaml` 双主题设计令牌（深色默认：`#0A0E1A` 底 / `#6366F1` 主色 / 玻璃拟态卡片；浅色：`#F8FAFC` 底 / 卡片白 + 细边框），通过 CSS 变量注入；不依赖 Streamlit 私有选择器的部分全部隔离到 `ui/styles/`，升级可单点修复。
- **组件**：KPI 卡片（数值 + 环比 delta + 迷你 sparkline + hover 抬升）、状态指示器 chip（数据/模型/一致性三态）、统一的 section 标题与说明 tooltip。
- 顶部三处固定位：数据新鲜度、"论文校准模式"徽章（见 §5 双口径）、导出入口。

### 4.2 View ① Executive Dashboard（高管决策大屏）

| 区块 | 内容与图表 |
|---|---|
| KPI 行（5 卡） | 有效评论量（+清洗剔除率）、双模型一致率、综合情感值（+对比论文基准 delta）、**TOP1 重要属性**、决策区分度 ΔP = max P−min P |
| 主图 A | **综合 VIKOR 排序**：P(xᵢ) 横向条形 + Q′ 区间误差棒（真实计算区间，非固定值） |
| 主图 B | **六维雷达图**：PLTS 综合能力（期望值/敏感值/吸引值 + 情感/份额/一致性），支持多选品牌叠画对比 |
| 主图 C | **竞品对比热力图**：品牌 × 属性 情感均值矩阵（色阶 RdYlGn，cell 显示数值 + 评论数） |
| 主图 D | **二维决策矩阵散点**：X=群体效用 S、Y=个体遗憾 R、气泡大小=Q、颜色=象限，含理想解参考线 |
| 叙事区 | 自动生成 Top3 优势 / Top3 短板卡片（文案由数据模板生成，不写死） |

### 4.3 View ② Aspect-Based Sentiment Explorer（细粒度情感下钻）

- 左侧参数面板：六属性选择、品牌/车型/极性/时间筛选、评分阈值。
- **情感热力图**：属性 × 品牌（双击 cell 联动下方评论列表）。
- **链路图**：桑基图（评论量 → 六属性 → 正/负极性 → VIKOR 评级档位）或旭日图（平台 → 属性 → 极性），展示"原始数据 → 属性情感 → 决策评级"的完整映射。
- **评论下钻**：分页列表（100/页），每条显示情感分、平台/车型、关键词高亮、可展开原文；替代当前"随机 10 条 sample"。
- 该属性指标：评论数、正负比、情感均值 ±σ、与全局/论文基准偏差、Top 负向关键词条形 + 词云。

### 4.4 View ③ PLTS-VIKOR Interactive Simulator（决策仿真器）

- **参数面板**：λ 滑杆（0→1，步长 0.1）、v 滑杆（0→1）、一级权重 3 个 + 二级权重 12 个（自动归一化并显示条形）、"重置为论文基准"按钮、AHP 判断矩阵可编辑（实时显示 CR 与通过/告警状态）。
- **中间矩阵 Tab**（可解释性核心）：规范化矩阵 F̄ → S 区间 → R 区间 → Q 区间 → Q′ 正态区间 → 可能度矩阵 P → P(xᵢ)；每个矩阵配公式编号（式 4.8–4.13）与行内高亮。
- **实时图**：Q 排序条形（滑杆松手即变）、S–R 散点、**λ 敏感性折线族**（λ∈{0,.2,.4,.5,.6,.8,1} 七档，复现论文表 5.18）+ 排序稳定性表（Kendall τ vs λ=0.5 基准）。
- **方法对比**：PLTS-VIKOR / 传统 VIKOR / TOPSIS / 前景理论 四排序并排（复现表 5.19）。
- 通过 `st.fragment` 局部刷新，滑杆拖动不触发整页重渲染。

### 4.5 View ④ Actionable Insights & Report Builder

- **ISA 四象限**（数据驱动 + 双口径切换：问卷基准 / 评论情感实测），分割线由均值动态计算；hover 显示属性详情。
- **改进优先级推演**：选中属性 → 结合该属性负向关键词、VIKOR 遗憾度 R、象限归属生成改进建议卡；支持"情景推演"（若满意度 +0.3 → 象限迁移与排序变化预览）。
- **一键导出**：`reportlab` 生成企业级 PDF（封面 + KPI + 图表快照 + 方法论附录，使用内置 STSong-Light CID 字体，**无需外部中文字体文件**）；`openpyxl` 导出多 Sheet Excel（原始筛选数据、VIKOR 中间矩阵、ISA、指标汇总）；另导出 JSON 场景存档供复现。

### 4.6 View ⑤ 数据管理（支撑页）

上传 CSV/Excel（复用 `importer.py` 列别名映射）→ 字段映射确认 → 运行流水线（进度条 + 阶段日志）→ 产物版本与校验徽章；替代当前"命令行跑完再刷新页面"的流程。

---

## 五、算法层重构方案（核心）

### 5.1 双口径设计（解决 P0-1 与 P1-3 的关键）

| 模式 | 含义 | 用途 |
|---|---|---|
| **论文校准模式**（默认） | 决策矩阵/权重/理想解读取 `config/baselines.yaml`（论文表 5.7/5.9/5.13/5.14），算法仍**真算** S/R/Q/P | 复现论文结论，验收对照 |
| **在线复算模式** | 权重来自可编辑 AHP/DEMATEL，理想解来自**实测评论情感极值**，PLTS 矩阵可编辑 | 真实数据、客户自定义场景 |

UI 顶部常驻徽章标明当前口径 + 计算值 vs 基准值偏差表——把"论文数值不可复算"（AHP CR=0.326 vs 0.0846）作为**明示的可解释性信息**呈现，而不是用硬编码掩盖。

### 5.2 计算链路与代码映射

```
core/algorithm/plts.py      概率语言项 {(lᵢ, pᵢ)} → 期望得分 E(H)（l₀..l₄ → 1..5 分值）
        ↓ 6×12 决策矩阵 F
core/algorithm/weights.py   式4.7  w = λ·wD + (1−λ)·wA；一级→12 指标展开
        ↓
core/algorithm/vikor.py     式4.8  规范化 F̄
                            步骤三 情感值 → f*_j / f⁻_j（表5.14 口径）→ 正则化（表5.15）
                            式4.9  Sᵢ=[Sᵢ⁻, Sᵢ⁺]   式4.10 Rᵢ=[Rᵢ⁻, Rᵢ⁺]
                            式4.11 Qᵢ=[Qᵢ⁻, Qᵢ⁺] → 正态区间 Q′ᵢ
                            式4.12 可能度矩阵 P      式4.13 P(xᵢ)
        ↓ VIKORResult（全部中间矩阵）
core/algorithm/sensitivity.py  λ 网格 + v 网格 + Kendall τ 排序稳定性
core/algorithm/benchmarks.py   TOPSIS / 前景理论 / 传统 VIKOR
core/algorithm/isa.py          双口径象限划分
```

### 5.3 验收基准（单测硬指标）

- `test_vikor`：校准模式下 S/R/Q/P(xᵢ) 与论文表 5.16/5.17 相对误差 ≤ 1e-3；偏差超阈值即测试失败并打印差异矩阵。
- `test_sensitivity`：λ ∈ {0,.2,.4,.5,.6,.8,1} 排序与论文表 5.18 一致（性价比>续航>空间>内饰>舒适性>外观）。
- `test_weights`：λ=0.5 时综合权重与表 5.12 对照；AHP CR 阈值告警逻辑。
- `test_plts`：期望得分、规范化、概率和为 1 的不变性。
- `test_isa`：象限划分对表 5.20 数据的分类结果。

---

## 六、依赖与环境（全部安装在 `产品实现/vene/`，不改系统配置）

已 dry-run 验证可安装（Python 3.9.6 兼容）：

| 库 | 版本 | 用途 |
|---|---|---|
| `streamlit-option-menu` | 0.4.0 | 侧边导航 |
| `streamlit-shadcn-ui` | 0.1.19 | 卡片/徽章/对话框等高质感组件 |
| `streamlit-echarts` | 0.4.0 | 可选：桑基/旭日动效（Plotly 兜底，已装） |
| `pydantic` | 2.13.x | 算法层数据模型与参数校验 |
| `reportlab` | 5.0.x | PDF 报告（内置中文 CID 字体） |
| `watchdog` | 最新 | 文件监听，消除启动告警 |
| `pytest` | 8.x | 单元测试 |

`requirements.txt` 将**全部钉死到当前 venv 实测版本**（streamlit 1.50.0 / pandas 2.3.3 / numpy 2.0.2 / scikit-learn 1.6.1 / plotly 7.1.0 / jieba 0.42.1 / pyarrow 21.0 …）。

**图表栈决策建议**：Plotly 为主（已装、原生支持 sankey/sunburst/scatter/heatmap/radar，暗色主题可控），`streamlit-echarts` 仅用于桑基/旭日两处动效增强，避免双栈维护成本。

---

## 七、Phase 2 执行计划（分批，每批可验收、可回滚）

| 批次 | 内容 | 验收标准 |
|---|---|---|
| **B0 基线** | `git init` + `legacy/app_legacy.py` 快照；安装依赖；`config/{baselines,theme}.yaml`；壳层（导航+主题+状态 Schema）+ 依赖冒烟 | 四视图可空壳切换，深浅主题切换生效，旧版可一键回滚 |
| **B1 算法层** | `core/algorithm/` 全模块 + `tests/` | 表 5.16/5.17/5.18 复现达标；λ/v 可调、中间矩阵可打印 |
| **B2 数据服务** | parquet 中间层、簇→属性映射、features 整形、`cache_resource` 模型 | 首屏 <1.5s，视图切换 <0.3s，17MB CSV 不再整表读入 |
| **B3 视图①②** | Executive + ABS Explorer（雷达/热力/桑基/下钻） | 两页全部图表由数据驱动，无任何硬编码结论 |
| **B4 视图③④** | Simulator + Insights/Report（局部刷新、敏感性、导出） | 滑杆实时重算；PDF/Excel 可下载且中文正常 |
| **B5 收尾** | 数据管理上传页、requirements 钉版本、README、回归 | `pytest` 全绿；`run.command` 双击可用 |

**风险与对策**

1. **论文自身数值矛盾**（AHP 不可复算、表 5.20 续航 4.1 vs 图 5.10 的 2.8、表 5.9 的 `l5`）→ 以 `baselines.yaml` 单一真源 + UI 双口径徽章 + 偏差表显式呈现。
2. **第三方组件与 streamlit 1.50 兼容性** → B0 首先冒烟，失败则降级为纯 CSS/Plotly 实现（不阻塞主线）。
3. **Python 3.9 限制** → 所有依赖已 dry-run 验证；不引入需 ≥3.10 的库。
4. **改动不可逆** → 先 `git init` + 旧版快照，逐批提交。

---

## 八、结论

当前系统是一个**"论文数值的可视化陈列馆"**：数据流水线（清洗/聚类/情感）真实可用，但决策核心 PLTS-VIKOR 未实现、结论硬编码、UI 单文件 eager 渲染、四大企业级场景缺位。重构的主线是三句话：

1. **把假计算变成真计算**——`core/algorithm` 纯函数化，复现论文表 5.13–5.19 并暴露全部中间矩阵与双口径偏差；
2. **把写死的结论变成数据驱动的结论**——四视图全部由 `services/features` + `VIKORResult` 渲染；
3. **把演示页变成决策工具**——可交互仿真器、属性下钻、竞品对比、一键企划报告。
