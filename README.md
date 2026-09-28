# 智评车行 · 产品实现目录

> 融合情感分析与 PLTS-VIKOR 的新能源汽车产品改进多属性决策平台
> 技术蓝本：赵敬华副教授指导硕士论文《融合情感分析与PLTS-VIKOR的新能源汽车产品改进多属性决策研究》
> 交付形态：企业级决策看板（OEM 产品 / 市场团队使用）

## 快速开始

**双击 `run.command`**（或手动执行）：

```bash
cd 产品实现
./vene/bin/streamlit run app.py
```

- 所有依赖隔离在 `vene/`（Python 3.9.6，arm64），**不改动系统配置**
- 首次启动会自动生成仿真评论数据（51,224 条，约 2 秒）

## 架构（企业级重构后）

```
产品实现/
├── app.py                    # 路由 + 主题注入 + 全局筛选（<200 行，零业务逻辑）
├── run.command               # Mac 一键启动
├── requirements.txt          # 依赖清单（已锁定版本）· requirements.lock.txt 全量锁定
│
├── config/                   # ★ 配置层（单一事实来源）
│   ├── baselines.yaml        # 论文表5.7–5.20 全部基准 + 审计发现定义
│   ├── theme.yaml            # 设计令牌：色板 / 字体 / 圆角 / 图表规范
│   └── settings.yaml         # 数据生成参数
│
├── core/algorithm/           # ★ 纯算法层（零 Streamlit 依赖，100% 可单测）
│   ├── models.py             # Scenario / VIKORResult / AHPResult / AuditFinding
│   ├── plts.py               # PLTS 期望得分、概率补全、式(4.8) 列规范化
│   ├── ahp.py                # 方根法 + 特征向量法（CR / λmax）
│   ├── dematel.py            # T = Z̄(I−Z̄)⁻¹、中心度 / 因果度、一级 / 二级权重
│   ├── weights.py            # 式(4.7) w = λ·wD + (1−λ)·wA，二级权重展开
│   ├── vikor.py              # 式(4.8)–(4.13) 全链路
│   ├── sensitivity.py        # λ / v 扫描、Kendall τ、论文表5.18 对照
│   ├── benchmarks.py         # TOPSIS / 前景理论 / 传统 VIKOR 对比（表5.19）
│   ├── isa.py                # ISA 四象限（表5.20 / 图5.10 双口径）
│   └── audit.py              # 一致性审计（9 项发现，均带实时计算数字）
│
├── services/                 # ★ 服务层（IO + 缓存，无 UI）
│   ├── data_store.py         # 路径注册表、CSV→parquet 中间层、流水线状态
│   ├── features.py           # 视图数据契约（KPI / 热力 / 桑基 / 下钻 / 趋势 / 叙事）
│   ├── report.py             # Excel / Markdown / PDF 报告导出
│   ├── preprocess/           # 阶段② 清洗分词
│   ├── aspect_mining/        # 阶段③ TF-IDF + KMeans 属性聚类
│   ├── sentiment/            # 阶段④ VADER + 朴素贝叶斯双模型情感
│   └── data_ingest/          # 阶段① 导入 / 仿真数据
│
├── state/store.py            # ★ 会话状态唯一入口（theme / view / filters / scenario / overrides）
│
├── ui/                       # ★ 表现层
│   ├── theme.py              # 设计令牌 → CSS / Plotly / ECharts
│   ├── components/           # KPI 卡、徽章、图表构建器（Plotly 主 + ECharts 桑基 / 旭日）
│   └── views/                # 五个视图（互不 import）
│       ├── executive.py          # ① 决策总览
│       ├── sentiment_explorer.py # ② 属性情感分析（总览 / 链路 / 下钻 / 竞品）
│       ├── simulator.py          # ③ PLTS-VIKOR 交互式模拟器（参数 / 中间量 / 敏感性 / 矩阵编辑）
│       ├── insights_report.py    # ④ 洞察与报告（ISA / 敏感性 / 方法对比 / 审计 / 导出）
│       └── data_manager.py       # ⑤ 数据管理（流水线 / 上传 / 质量）
│
├── tests/                    # 单测 + 应用级冒烟（streamlit AppTest）
├── legacy/app_legacy.py      # 重构前单文件版本快照（回归对照）
└── docs/PHASE1_诊断与重构规划.md   # Phase 1 诊断报告与规划
```

## 双口径设计（重要）

论文中的若干表格无法由其自身数据复算得到，系统**不静默修正**，而是双轨并列 + 审计面板：

| 发现 | 状态 | 说明 |
|---|---|---|
| `AHP_WEIGHT_MISMATCH` | ✕ 偏差 | 表5.7 矩阵方根法得 (0.642, 0.072, 0.285) CR=0.326，论文表5.8 为 (0.434, 0.187, 0.379) CR=0.085 |
| `DEMATEL_WEIGHT_MISMATCH` | ✕ 偏差 | 表5.9 计算得 (0.286, 0.355, 0.359)，论文表5.11 为 (0.516, 0.204, 0.280) |
| `DEMATEL_L5` | △ 待确认 | 表5.9 含 `l5`，超出正文定义的 L={l0..l4}；按论文原值录入并标注 |
| `Q_NOT_DERIVABLE_FROM_S_R` | ✕ 偏差 | 表5.17 的 Q 无法由表5.16 的 S/R 与式(4.11) 推出（极值属性应为 0 或 1，论文为 0.0763） |
| `SENSITIVITY_VS_RESULT` | ✕ 偏差 | 表5.18 在 λ=0.5 处与表5.17 不一致 |
| `ISA_SATISFACTION_CONFLICT` | △ 待确认 | 续航满意度 表5.20=4.1 vs 图5.10=2.8，导致象限归属变化 |
| `COMBINED_WEIGHT_OK` | ✓ 通过 | 式(4.7) 综合权重可精确复现表5.12 (0.475, 0.196, 0.329) |
| `LIVE_VS_PAPER` | △ 待确认 | 实时复算 P(x) 与论文表5.17 的相关性与排序差异 |
| `PLTS_PROB_SUM` | △ 待确认 | 表5.13 中 16/72 格 Σp<1（最小 0.9），按区间 / 归一化补全 |

- **论文校准模式（默认）**：权重与结论以论文表为基准展示，复算值并列显示 Δ
- **在线复算模式**：AHP / DEMATEL 由（可编辑的）判断矩阵实时计算

两种模式在页面顶部都有徽章标识，绝不混淆；审计面板逐条给出实时计算出来的数字。

## 数据流水线

```
data/raw/comments_raw.csv (51,224)
  → services/preprocess    清洗 + jieba 分词 + 六属性词典
  → services/aspect_mining TF-IDF + KMeans（k=7）属性聚类
  → services/sentiment     VADER + 朴素贝叶斯 双模型融合（一致率约 76%）
  → data/processed/*.csv + data/features/*.parquet（中间层）
  → core/algorithm         PLTS-VIKOR 决策（六属性 × 12 指标）
  → outputs/reports        Excel / Markdown / PDF 报告
```

## 测试

```bash
cd 产品实现
./vene/bin/python -m pytest tests/ -q              # 算法 / 服务单测 + 应用级冒烟
./vene/bin/python scripts_syntax_check.py          # 全量语法检查（不写 __pycache__）
```

## 依赖

见 `requirements.txt`（锁定到实测版本）：
Streamlit 1.50 + streamlit-option-menu + streamlit-shadcn-ui + streamlit-echarts（仅桑基 / 旭日）+ Plotly 7（主力图表）；
pandas / numpy / scipy / scikit-learn / pyarrow；jieba；reportlab / openpyxl / pypdf；pytest。

## 看板功能

| 视图 | 核心能力 |
|---|---|
| ① 决策总览 | 企业 KPI、双口径排序对照、S-R 决策散点、情感-重要性 2D 决策矩阵、品牌雷达、月度趋势、优势/短板叙事、审计摘要 |
| ② 属性情感分析 | 属性总览表 + 正负构成、品牌×属性热力、平台→属性→极性→重要性档位桑基与旭日、词频 / 词云、评论下钻分页、车型口碑榜 |
| ③ PLTS-VIKOR 模拟器 | λ / v / 理想解 / 补全 / 方向 / 权重展开全部可调，四个中间矩阵 Tab（PLTS 期望、规范化与效用、权重体系、可能度矩阵），λ-v 敏感性扫描 + 论文表5.18 对照，AHP / DEMATEL / 决策矩阵在线编辑 |
| ④ 洞察与报告 | ISA 三口径象限 + 改进建议、λ / v 稳健性、四方法对比（表5.19）、完整审计面板、Excel / Markdown / PDF 导出 |
| ⑤ 数据管理 | 六阶段流水线状态、数据上传、流水线重跑、数据质量与字典 |
