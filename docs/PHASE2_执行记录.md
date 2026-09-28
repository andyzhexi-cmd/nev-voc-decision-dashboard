# Phase 2 执行记录 · 企业级重构

> 对应规划：`docs/PHASE1_诊断与重构规划.md`
> 执行范围：B0–B5 全量（用户已批准"按规划全量执行"）

## 一、批次完成情况

| 批次 | 规划内容 | 状态 | 关键产物 |
|---|---|---|---|
| B0 | 依赖安装、git 基线、legacy 快照 | ✅ | `legacy/app_legacy.py`、基线提交 `a14b0d5`、`requirements.txt`/`requirements.lock.txt` |
| B1 | 配置层 + 算法层 | ✅ | `config/baselines.yaml`（论文表5.7–5.20 + 9 项审计发现）、`config/theme.yaml`、`core/algorithm/` 10 个模块 |
| B2 | 服务层 | ✅ | `services/data_store.py`（CSV→parquet 中间层）、`services/features.py`（视图数据契约）、`services/report.py`（Excel/MD/PDF）、流水线包自 `src/` 迁入 |
| B3 | 表现层骨架 + 视图①③ | ✅ | `state/store.py`、`ui/theme.py`、`ui/components/`、`ui/views/executive.py`、`ui/views/simulator.py`、`app.py` 路由壳 |
| B4 | 视图②④⑤ | ✅ | `ui/views/sentiment_explorer.py`、`ui/views/insights_report.py`、`ui/views/data_manager.py` |
| B5 | 回归 / 文档 / 提交 | ✅ | 41 项单测 + AppTest 应用级冒烟、README 重写、分批提交 |

## 二、关键技术决策

1. **双口径 + 审计面板**（而不是"复现论文数字"）
   数值实验已证明：论文表5.16/5.17 的 S/R/Q 无法由其自身数据与式(4.9)–(4.11) 复算得到
   （最好相关性仅 0.763）。因此系统以 **论文基准为 SSOT**、**实时复算为可验证路径**，
   两者并列展示，并把 6 处已知不一致固化为 `config/baselines.yaml → audit_findings`，
   在 9 条审计项中以实时计算的数字呈现（2 项 pass / 4 项 fail / 3 项 warn）。
2. **效用方向显式化**：默认 `direction="attainment"`（论文口径：S 越大越重要），
   同时提供传统 VIKOR 的 `shortfall`；两种口径下 S′ = 1 − S 的互证关系已写入单测。
3. **图表栈**：Plotly 7 为主力（雷达/热力/S-R 散点/象限/趋势/分布），ECharts 仅用于
   桑基图与旭日图（`streamlit-echarts`），符合"Plotly + ECharts 混合"的决策。
4. **中间层缓存**：17MB `sentiment_results.csv` → 746KB parquet，列裁剪读取；
   `store.features_frame()` 以 (kind, filters, data_version) 为 key 的 `st.cache_data` 缓存。
5. **视图降级**：`app.py` 捕获视图异常并把 traceback 存入 `dsh.render_error_tb`，
   单个视图失败不会白屏，其余视图仍可用。

## 三、验收对照

| 验收项 | 结果 |
|---|---|
| 式(4.7) 复现表5.12 (0.475, 0.196, 0.329) | ✅ `test_combine_weights_reproduces_table_5_12`，偏差 5e-4 |
| 表5.15 = 表5.14 + 0.9643 | ✅ `test_ideal_strategy_sentiment` |
| 表5.16/5.17 直接复现（≤1e-3） | ❌ 预期失败 → 由 `Q_NOT_DERIVABLE_FROM_S_R`、`LIVE_VS_PAPER` 两条审计项记录偏差 |
| 表5.13 概率不全 16/72 格（min 0.9） | ✅ `test_decision_matrix_shape_and_diagnostics` |
| 四方法对比（表5.19） | ✅ `compare_methods()`，TOPSIS/前景理论 Top1 = 性价比，与论文一致 |
| 五个视图可渲染 | ✅ `tests/test_smoke_views.py`（streamlit AppTest，明暗双主题） |
| 启动可用性 | ✅ `streamlit run app.py` HTTP 200 + `_stcore/health = ok` |
| 不改系统配置 | ✅ 全部依赖在 `产品实现/vene/`；matplotlib 缓存改写到 `services/.mplconfig/` |

## 四、已知限制

- 论文排序与复算排序存在系统性差异（Top1 论文=性价比，复算=内饰/续航区间），
  根因是论文权重与判断矩阵不可复算，已在审计面板与总览页显式说明；
- 感知情感均值（复算 0.463）与论文问卷均值（0.488）口径不同，仅作对照，不混用；
- ECharts 图在 AppTest 中被跳过（自定义组件），已用裸模式单测验证 option 构造。
