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
   在 9 条审计项中以实时计算的数字呈现（整改后：**1 项 pass / 4 项已解析 / 4 项 warn /
  0 项未解释偏差**，高严重级 3 项；关闭任一修复开关即回落为 fail，见第五节）。
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

## 五、第三阶段 · 三问题整改（版式 / 审计 / 上手）

| 问题 | 处置 | 产物 |
|---|---|---|
| ① 界面"乱飘"、版式不协调 | 参照 NHS dashboard 布局规范（32px 页边 / 16px 卡内 / 32px 卡间距 / 一排 ≤6 KPI）、Shilp Sutra 三档间距节奏（8/16/32）、Fiori 语义页头，收敛为**具名栅格 + 统一面板 + 四档高度**；五个视图里 15 种随手比例归档为 9 组具名栅格，`st.columns` 字面量全站归零 | `ui/layout.py`、`ui/theme.py`、`docs/版式与主题基准.md` |
| ② 四项审计偏差 | 逐条取证 → 根因 → 可开关的修复动作（AHP 反推一致性矩阵 / DEMATEL 自底向上聚合 / Q 按式4.11 重建 / 敏感性锚定表5.17），四项全部 **已解析**，`unexplained=0` | `docs/审计取证与修复.md`、`tests/test_audit_remediation.py`（12 项）、`ui/components/audit_panel.py` |
| ③ 不了解项目难上手 | 新增默认落地页「产品导览」：产品定义、3 分钟路线（点击跳转）、五层架构图、两条数据链路、9 条概念词典、可复现性说明、60 秒讲稿、运行命令 | `ui/views/onboarding.py`、`state/store.py` 默认 `dsh.view="产品导览"` |
| 附：明暗主题不完整 | `theme.sync()` 回写 Streamlit 主题配置（表格/代码块跟随）+ `:root` 覆盖内建组件令牌（滑块/单选/进度条跟随），徽章与面板颜色全部令牌化；选择持久化到 `data/ui_prefs.json` | `ui/theme.py`、`state/store.py` |
| 附：真实数据接入底座 | 编码/分隔符嗅探、列别名映射、列错位检测、干跑体检报告、备份后原子落盘、模板与规范 | `services/data_ingest/importer.py`、`tests/test_ingest.py`（10 项）、`docs/数据接入规范.md` |
| 追加① 真实数据接入（页面） | 「数据管理 → 导入真实评论」：上传（CSV/TSV/TXT/JSON/XLSX）→ 5 列可改映射 → 干跑体检与前 10 行预览 → 校验失败禁用确认 → 覆盖前自动备份落盘 → 缓存失效并指引重跑流水线；标准模板下载 + 最近备份列表 | `ui/views/data_manager.py` |
| 追加② 决策矩阵可编辑化 | AHP 只开上三角（对角线锁定、互反自动回填）+ 实时 CR/诊断与 hints、反推一致性矩阵；DEMATEL 走数值投影诊断、应用时回写 lN 字符串；PLTS 聚合体检 + `build_decision_matrix` 两种补全口径试算；w3 一键归一化与逐项 Δ；`matrix_diff` 摘要 + Δ 热力图；三预设 + 撤销/重做；JSON 导出与白名单校验导入；提交时同步 `use_paper_weights`（默认「论文校准」行为逐字节不变） | `ui/views/simulator.py`、`tests/test_matrix_editor.py`（8 项） |
| 追加③ 架构图审美重构 | 「产品导览 → 产品组织架构」由色条列表改为 **L1→L5 竖向栈**：层号节点 + 渐变导轨 + 卡片（层名/路径 mono/职责/模块数 chip），核心层（算法层）用渐变节点 + 反色卡片高亮，文件走 auto-fit 网格自适应宽度；整块单次渲染，明暗主题全令牌 | `ui/views/onboarding.py`（`_arch_html` + `.arch-*`） |

验收：六视图 × 明暗双主题 AppTest 共 12 组无异常（`TOTAL_BAD = 0`）；`pytest tests/` 全绿（82 项）；`scripts_syntax_check.py` 通过；
`st.columns` / `use_container_width` / 硬编码色值 / 手工空行四类 grep 在 `app.py` + 六视图 + 组件中计数均为 0（仅 `ui/layout.py` 作为唯一列切分入口保留字面量）。
