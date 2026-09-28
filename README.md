# 智评车行 · 产品实现目录

> 融合情感分析与 PLTS-VIKOR 的新能源汽车产品改进多属性决策平台
> 技术蓝本：赵敬华副教授指导硕士论文《融合情感分析与PLTS-VIKOR的新能源汽车产品改进多属性决策研究》

## 快速开始（你只需要做一件事）

**双击 `run.command`**，浏览器会自动打开看板。
- 第一次启动会自动生成仿真数据（约 2 秒）
- 之后每次启动直接进入

手动命令（等价）：
```bash
cd 产品实现
./vene/bin/streamlit run app.py
```

## 目录结构

```
产品实现/
├── vene/                  # Python 3.9 虚拟环境（M1 arm64），所有依赖隔离在此
├── app.py                  # Streamlit 看板主入口
├── run.command             # Mac 一键启动（双击即可）
├── requirements.txt        # 依赖清单
├── config/
│   └── settings.yaml       # 全局配置（论文表5.1/5.3/5.5 分布参数）
├── src/
│   ├── data_ingest/        # 【阶段1 已完成】数据层
│   │   ├── mock_generator.py  # 仿真数据生成器（按论文分布造5万条）
│   │   └── importer.py       # 真实数据导入适配器（八爪鱼CSV/Excel→标准格式）
│   ├── preprocess/          # 【阶段2】文本预处理（待实现）
│   ├── attribute_mining/    # 【阶段3】TF-IDF+K-means 属性识别（待实现）
│   ├── sentiment/           # 【阶段4】双模型情感分析（待实现）
│   ├── decision/           # 【阶段5】AHP+DEMATEL+PLTS-VIKOR（待实现）
│   ├── isa/                 # 【阶段6】ISA 四象限矩阵（待实现）
│   └── visualization/      # 【阶段7】看板页面（待实现）
├── data/
│   ├── raw/comments_raw.csv   # 原始评论（当前为仿真数据 51,224 条）
│   └── processed/             # 清洗后数据（阶段2产出）
├── outputs/                  # 图表与报告
└── pages/                    # Streamlit 多页（阶段7填充）
```

## 数据说明

| 项目 | 当前状态 |
|---|---|
| 数据来源 | **路线A 仿真数据**（当天跑通用） |
| 总条数 | 51,224 条（严格按论文表5.1的25个平台×品牌格子生成） |
| 平台分布 | 汽车之家/易车网/爱卡汽车/太平洋汽车/网上车市 各约 1 万条 |
| 品牌分布 | 特斯拉/比亚迪/小鹏/理想/蔚来 各约 1 万条 |
| 噪声评论 | 4,765 条（9.3%，模拟水帖，阶段3会被自动剔除） |
| 金标准列 | `_attr_ground_truth`（属性）、`_sentiment_ground_truth`（情感）——仿真数据独有，用于对照验收 |

**切换真实数据**：把八爪鱼导出的 CSV/Excel 放到 `data/raw/`，运行
```bash
./vene/bin/python -m src.data_ingest.importer 你的文件.xlsx
```
导入器会自动映射列名、去重去空，与仿真数据走同一套后续流程。

## 已完成阶段对照论文基准

| 基准项 | 论文值 | 当前仿真 |
|---|---|---|
| 原始评论量 | 50,520（正文）/ 51,224（表5.1明细） | 51,224 ✓ |
| 噪声占比 | 约 9.3%（清洗后 45,836 条有效） | 9.3% ✓ |
| 六属性比例 | 11147/9722/7875/6101/5697/5292 | 按同比例分配 ✓ |
| 各属性正负比 | 表5.5 | 按同比例分配 ✓ |

## 环境信息

- Python 3.9.6（macOS 系统自带，arm64）
- 依赖：streamlit 1.50 / pandas 2.3 / scikit-learn 1.6 / jieba 0.42 / plotly 7.1 等
- 全部安装在 `产品实现/vene/` 内，**不污染系统 Python**
- MATLAB 暂未使用（本项目用 Python 实现，MATLAB 可后续用于算法复现对比）
