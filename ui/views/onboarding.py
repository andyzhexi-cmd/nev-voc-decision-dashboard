"""ui.views.onboarding — 0. 产品导览（默认落地页）

面向读者：不了解项目背景的评审老师 / 面试官，3 分钟看懂
「这是什么 → 为什么可信 → 结构怎么组织 → 从哪点开始」。

区块
----
1. 这是什么        产品定义 + 5 张现算 KPI
2. 3 分钟上手路线   4 张卡 + 跳转按钮（决策总览 / 情感分析 / 模拟器 / 报告）
3. 产品组织架构    五层分层图（数据 → 服务 → 算法 → 状态 → 表现）
4. 两条数据链路    论文基准链路 与 业务数据链路 并列（含「在哪看」）
5. 关键概念词典    9 条术语（面向没读过论文的人）
6. 可复现性说明    SSOT 口径 + 9 项审计实时状态
7. 60 秒讲稿      问题—方法—结果—可信度—工程化
8. 本地运行与出处  命令 / 测试状态 / 论文出处 / 数据新鲜度

约定：版式走 ui.layout（split + panel），颜色一律用主题令牌（var(--*)），
不读大 CSV（kpi 走 st.cache_data，新鲜度走 data_store 单次调用）。
"""
from __future__ import annotations

import streamlit as st

from core.algorithm.models import baselines
from services.data_store import store as data_store
from state import store
from ui import layout
from ui.components import (advice_card, audit_badge, callout, kpi_row, legend_html,
                           section_header, severity_tone, status_chip)

# ------------------------------------------------------------------ 局部样式
#: 仅用主题令牌（var(--*)），明暗主题自动跟随；无任何硬编码色值。
GUIDE_STYLE = """
<style>
.ob-intro{font-size:14.5px;line-height:1.9;color:var(--text);margin:2px 0 14px;}

/* ---- 架构图：L1→L5 竖向栈（节点 + 导轨 + 卡片） ---- */
.arch{margin:2px 0 6px;}
.arch-item{display:grid;grid-template-columns:46px minmax(0,1fr);gap:14px;align-items:stretch;}
.arch-rail{position:relative;display:flex;justify-content:center;}
.arch-dot{width:32px;height:32px;border-radius:50%;border:1.5px solid var(--brand);
  background:var(--panel-alt);color:var(--brand);
  font-family:var(--font-mono,"SFMono-Regular",Menlo,monospace);font-size:11.5px;font-weight:700;
  display:flex;align-items:center;justify-content:center;letter-spacing:.03em;}
.arch-item:not(:last-child) .arch-rail::after{content:"";position:absolute;left:50%;top:34px;
  bottom:-14px;width:2px;transform:translateX(-50%);border-radius:2px;
  background:linear-gradient(180deg,var(--brand),var(--border));}
.arch-card{background:var(--panel);border:1px solid var(--border);border-left:3px solid var(--brand);
  border-radius:14px;padding:13px 16px 14px;min-width:0;}
.arch-item.core .arch-card{border-left-color:var(--brand2);background:var(--panel-alt);}
.arch-item.core .arch-dot{background:linear-gradient(135deg,var(--brand),var(--brand2));
  border-color:transparent;color:white;}
.arch-head{display:flex;align-items:baseline;gap:9px;flex-wrap:wrap;}
.arch-name{font-size:14.5px;font-weight:700;color:var(--text);}
.arch-hint{font-family:var(--font-mono,"SFMono-Regular",Menlo,monospace);font-size:11.3px;
  color:var(--muted);word-break:break-all;}
.arch-tag{margin-left:auto;font-size:11px;font-weight:650;color:var(--brand);
  background:var(--panel-alt);border:1px solid var(--brand);border-radius:999px;padding:2px 9px;
  white-space:nowrap;}
.arch-item.core .arch-tag{background:var(--panel);color:var(--brand2);border-color:var(--brand2);}
.arch-duty{font-size:12.4px;color:var(--muted);margin-top:5px;line-height:1.7;}
.arch-files{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
  gap:8px 14px;margin-top:10px;}
.arch-file{background:var(--panel-alt);border:1px solid var(--border);border-radius:10px;
  padding:7px 10px;min-width:0;}
.arch-item.core .arch-file{background:var(--panel);}
.arch-file code{display:block;font-family:var(--font-mono,"SFMono-Regular",Menlo,monospace);
  font-size:11.6px;color:var(--brand);line-height:1.5;word-break:break-all;}
.arch-file span{display:block;font-size:11.7px;color:var(--muted);margin-top:3px;line-height:1.6;}
.arch-flow{display:flex;align-items:center;gap:8px;margin:14px 0 2px 60px;
  font-size:11.8px;color:var(--muted);letter-spacing:.06em;}
.arch-flow::after{content:"";flex:1;height:1px;background:var(--border);}
.ob-file{font-size:12.6px;line-height:1.95;color:var(--text);}
.ob-file code{font-size:11.8px;background:var(--panel-alt);color:var(--brand);
  padding:1px 6px;border-radius:6px;border:1px solid var(--border);}
.ob-file i{font-style:normal;color:var(--muted);}
.ob-arrow{text-align:center;color:var(--muted);font-size:14px;padding:3px 0;letter-spacing:.35em;}
.ob-step{display:flex;gap:10px;align-items:flex-start;padding:9px 12px;border-radius:12px;
  background:var(--panel-alt);border:1px solid var(--border);}
.ob-step .ds-chip{flex:0 0 auto;}
.ob-body{flex:1;min-width:0;}
.ob-t{font-size:13.2px;font-weight:650;color:var(--text);line-height:1.6;}
.ob-d{font-size:12.4px;color:var(--muted);line-height:1.7;margin-top:2px;}
.ob-where{margin-top:9px;}
.ob-term{min-height:84px;}
.ob-talk{margin-bottom:10px;}
.ob-talk .t{display:block;}
.ob-code{font-family:var(--font-mono,"SFMono-Regular",Menlo,monospace);font-size:12.4px;
  background:var(--panel-alt);color:var(--text);border:1px solid var(--border);
  padding:3px 8px;border-radius:7px;}
.ob-kv{font-size:13px;line-height:2.05;color:var(--text);}
.ob-kv b{color:var(--text);}
.ob-kv span{color:var(--muted);}
</style>
"""

# ------------------------------------------------------------------ 内容表
#: 4 张上手卡（视图名必须与 app.NAV 完全一致）
ROUTES: list[dict] = [
    {"view": "决策总览", "eta": "60 秒", "first": True,
     "body": "先看结论：核心 KPI、双口径排序对照（论文 vs 复算）、S-R 效用与四象限、审计摘要。"},
    {"view": "属性情感分析", "eta": "90 秒",
     "body": "再看证据：六属性情感热力图、平台→属性→极性桑基链路、词频与评论下钻。"},
    {"view": "PLTS-VIKOR 模拟器", "eta": "90 秒",
     "body": "自己动手：改 λ、v、理想解与权重，AHP / DEMATEL / PLTS 决策矩阵可直接编辑"
             "（实时体检 + 撤销重做），四个中间矩阵与排序实时跟着变。"},
    {"view": "洞察与报告", "eta": "60 秒",
     "body": "拿走结果：ISA 改进四象限、四方法对比、审计面板，一键导出 Excel / Markdown / PDF。"},
    {"view": "数据管理", "eta": "120 秒",
     "body": "接真实数据：上传 CSV / XLSX → 列映射 → 干跑体检 → 确认落盘（覆盖前自动备份），"
             "再看流水线与数据质量。"},
]

#: 五层架构（自上而下）：code 层号 / duty 一句话职责 / core 标注核心层
LAYERS: list[dict] = [
    {"code": "L1", "name": "数据层", "duty": "采集 → 清洗 → 分词 → 属性聚类 → 双模型情感",
     "hint": "services/* 采集与加工",
     "files": [("services/data_ingest", "导入 CSV / 生成仿真数据"),
               ("services/preprocess", "清洗、去重、jieba 分词"),
               ("services/aspect_mining", "TF-IDF + KMeans 属性聚类"),
               ("services/sentiment", "VADER × 朴素贝叶斯 双模型情感")]},
    {"code": "L2", "name": "服务层", "duty": "中间层读写 · 视图数据契约 · 报告生成",
     "hint": "services/{data_store, features, report}",
     "files": [("services/data_store.py", "CSV→parquet 中间层与阶段状态"),
               ("services/features.py", "筛选后的聚合指标（视图数据契约）"),
               ("services/report.py", "Excel / Markdown / PDF 报告生成")]},
    {"code": "L3", "name": "算法层", "core": True, "tag": "核心 · 零 Streamlit 依赖",
     "duty": "PLTS-VIKOR 主方法 · 权重融合 · 敏感性与 9 项一致性审计",
     "hint": "core/algorithm/*（41 项单测覆盖）",
     "files": [("core/algorithm/plts.py", "PLTS 定义、补全与列归一化"),
               ("core/algorithm/{ahp, dematel, weights}", "主观权重与关联权重及融合"),
               ("core/algorithm/vikor.py", "PLTS-VIKOR 主方法与论文基准"),
               ("core/algorithm/{sensitivity, benchmarks}", "敏感性与四方法对照"),
               ("core/algorithm/{isa, audit}", "四象限与 9 项一致性审计")]},
    {"code": "L4", "name": "状态层", "duty": "唯一状态入口 · 可序列化 · 双主题持久化",
     "hint": "state/store.py",
     "files": [("state/store.py", "主题 / 筛选 / 场景 / 覆盖值唯一入口")]},
    {"code": "L5", "name": "表现层", "duty": "具名栅格 · 设计令牌 · 组件库 · 六个视图",
     "hint": "ui/{layout, theme, components, views}",
     "files": [("ui/layout.py", "具名栅格 split 与统一 panel"),
               ("ui/theme.py", "设计令牌与明暗主题注入"),
               ("ui/components", "徽章 / KPI / 图表可复用组件"),
               ("ui/views", "六个视图（互不 import）")]}
]

#: 两条数据链路
PAPER_STEPS: list[dict] = [
    {"t": "论文表5.7–5.20", "d": "AHP 判断矩阵、DEMATEL 关联、PLTS 决策矩阵、VIKOR 结果、ISA"},
    {"t": "config/baselines.yaml", "d": "人工转录为唯一事实源（SSOT），附出处与单位"},
    {"t": "core/algorithm 复算", "d": "按式(4.7)–(4.13) 重新计算权重、S/R/Q、P(x)"},
    {"t": "双口径对照", "d": "论文基准与实时复算并列展示，偏差逐项列出"},
    {"t": "core/algorithm/audit.py", "d": "9 项一致性检查，把论文内部冲突算给你看"},
    {"t": "洞察与报告 导出", "d": "审计结论写入 Excel / MD / PDF 的「审计发现」章节"},
]
BIZ_STEPS: list[dict] = [
    {"t": "CSV / XLSX 评论（车评）", "d": "「数据管理 → 导入真实评论」体检后落盘，或脚本生成仿真数据"},
    {"t": "清洗分词", "d": "去重停用词 + jieba 分词，产出 comments_cleaned.csv"},
    {"t": "属性聚类", "d": "TF-IDF + KMeans，六属性金标准对照算纯度"},
    {"t": "双模型情感", "d": "VADER 与朴素贝叶斯一致才保留，产出有效评论"},
    {"t": "parquet + features", "d": "中间层列裁剪读取，按筛选聚合出指标表"},
    {"t": "视图与报告", "d": "六视图实时渲染，报告可导出三格式"},
]

#: 关键概念词典（9 条，每条 ≤40 字）
GLOSSARY: list[dict] = [
    {"tone": "info", "title": "PLTS 概率语言分布",
     "body": "用「术语 + 概率」表达打分的不确定性，如 {l3:0.4, l4:0.6}，可算期望与区间。"},
    {"tone": "info", "title": "VIKOR 折中解",
     "body": "在群体效用与个体遗憾之间取折中，给出可排序的妥协方案，而非绝对最优。"},
    {"tone": "info", "title": "AHP 主观权重",
     "body": "专家两两比较打分，方根法求权重，并用一致性比率 CR 校验是否可信。"},
    {"tone": "positive", "title": "DEMATEL 关联权重",
     "body": "从指标间因果影响矩阵算中心度，衡量「谁更影响谁」，属客观赋权。"},
    {"tone": "positive", "title": "λ 权重融合系数",
     "body": "w = λ·wD + (1−λ)·wA：λ=0 纯 AHP，λ=1 纯 DEMATEL，默认 0.5。"},
    {"tone": "positive", "title": "v 决策系数",
     "body": "调群体效用 S 与个体遗憾 R 的相对重要性，v=1 只看群体，v=0 只看遗憾。"},
    {"tone": "negative", "title": "ISA 四象限",
     "body": "重要性 × 满意度分四象限：改进区先动手，保持区筑口碑，低优先级暂缓。"},
    {"tone": "neutral", "title": "双口径",
     "body": "论文校准（基准取论文表）与在线复算（实时计算）并列，绝不混为一谈。"},
    {"tone": "neutral", "title": "审计",
     "body": "9 项一致性检查，把论文自身冲突实时算给你看，不做静默修正。"},
]

#: 60 秒讲稿
TALKING: list[dict] = [
    {"tone": "negative", "title": "1 · 问题",
     "body": "新能源车主改款资源有限：六属性十二指标同时要改，先改谁、改到什么程度，缺可量化依据。"},
    {"tone": "info", "title": "2 · 方法",
     "body": "情感分析给出「满意度」，PLTS-VIKOR 给出「重要度 → 折中排序」，两者在 ISA 四象限合流。"},
    {"tone": "positive", "title": "3 · 结果",
     "body": "输出六属性改进优先级与动作建议，并与 TOPSIS、前景理论、传统 VIKOR 四方法交叉验证。"},
    {"tone": "neutral", "title": "4 · 可信度",
     "body": "论文基准为唯一事实源，9 项审计实时核对，能复现的写通过、冲突的标偏差并给数字。"},
    {"tone": "info", "title": "5 · 工程化",
     "body": "分层架构、41 项单测 + 5 视图应用级冒烟、缓存与中间层，双击 run.command 即可本地运行。"},
]


# ------------------------------------------------------------------ HTML 构造
def _chain_html(steps: list[dict]) -> str:
    """步骤条：编号 chip + 标题 + 说明，竖向箭头连接。"""
    nodes = []
    for i, s in enumerate(steps, 1):
        nodes.append(
            '<div class="ob-step"><span class="ds-chip chip-brand">'
            f"{i}</span>"
            f'<div class="ob-body"><div class="ob-t">{s["t"]}</div>'
            f'<div class="ob-d">{s["d"]}</div></div></div>')
    return '<div class="ob-chain">' + '<div class="ob-arrow">↓</div>'.join(nodes) + "</div>"


def _arch_html() -> str:
    """五层架构栈：L1→L5 节点 + 导轨 + 卡片，整块单次渲染（栅格才不会被切开）。"""
    items = []
    for layer in LAYERS:
        files = "".join(
            f'<div class="arch-file"><code>{p}</code><span>{d}</span></div>'
            for p, d in layer["files"])
        tag = layer.get("tag") or f"模块 ×{len(layer['files'])}"
        items.append(
            '<div class="arch-item{core}">'
            '<div class="arch-rail"><div class="arch-dot">{code}</div></div>'
            '<div class="arch-card">'
            '<div class="arch-head"><span class="arch-name">{name}</span>'
            '<span class="arch-hint">{hint}</span>'
            '<span class="arch-tag">{tag}</span></div>'
            '<div class="arch-duty">{duty}</div>'
            '<div class="arch-files">{files}</div>'
            "</div></div>".format(
                core=" core" if layer.get("core") else "",
                code=layer["code"], name=layer["name"], hint=layer["hint"],
                tag=tag, duty=layer["duty"], files=files))
    flow = '<div class="arch-flow">自上而下逐层供给 → 汇入六视图</div>'
    return '<div class="arch">' + "".join(items) + flow + "</div>"


def _route_card(col, route: dict, index: int) -> None:
    """上手路线卡：标题 + 建议用时 + 一句话说明 + 跳转按钮。"""
    with col:
        with layout.panel(route["view"], route["body"],
                          tag="第一步" if route.get("first") else f"第{index}步"):
            status_chip(f"建议用时 · {route['eta']}", "brand" if route.get("first") else "muted")
            if st.button(f"打开 → {route['view']}", width="stretch",
                         key=f"ob_go_{index}", type="primary" if route.get("first") else "secondary"):
                store.set_view(route["view"])
                st.rerun()


def _status_counts(findings: list) -> dict:
    out = {"pass": 0, "resolved": 0, "warn": 0, "fail": 0, "other": 0}
    for f in findings:
        out[f.status if f.status in out else "other"] += 1
    return out


def _filters_brief(f) -> str:
    """全局筛选摘要（未筛选时明说全量，避免读者误以为被裁剪）。"""
    parts = []
    if f.brand:
        parts.append("品牌 " + "/".join(f.brand))
    if f.model:
        parts.append("车型 " + "/".join(f.model))
    if f.attrs:
        parts.append("属性 " + "/".join(f.attrs))
    if f.polarity:
        parts.append(f"极性 {f.polarity}")
    if f.date_range:
        parts.append(f"时间 {f.date_range[0]} ~ {f.date_range[1]}")
    return " · ".join(parts) if parts else "全量数据（未筛选）"


# ------------------------------------------------------------------ 视图
def render() -> None:
    st.markdown(GUIDE_STYLE, unsafe_allow_html=True)

    filters_json = store.filters_json()
    version = store.data_version()
    k = store.features_frame("kpi", filters_json, version)          # services.features.kpi_summary
    ov = store.features_frame("overview", filters_json, version)    # services.features.attr_overview
    base = baselines()
    findings = store.get_audit()
    counts = _status_counts(findings)
    fresh = data_store().freshness()
    sc = store.scenario()
    mode_label = "论文校准模式" if sc.mode == "calibrated" else "在线复算模式"

    n_raw = int(k.get("n_raw", 0))
    n_kept = int(k.get("n_kept", 0))
    keep_rate = float(k.get("keep_rate", 0.0))
    n_attr = int((ov["评论数"] > 0).sum()) if not ov.empty else len(base["meta"]["attributes"])
    n_crit = len(base["meta"]["second_level"])
    methods = list(base["method_comparison"])

    layout.page_head("产品导览",
                     "不了解项目也能三分钟上手：这是什么 · 怎么组织 · 数据从哪来 · 从哪开始点")

    # ---------------------------------------------------------- 1. 这是什么
    section_header("这是什么",
                   "3 分钟看懂：这是什么、为什么可信、结构怎么组织、从哪开始点", tag="导览")
    st.markdown(
        f'<p class="ob-intro">「智评车行」是一套<b>融合情感分析与 PLTS-VIKOR 的新能源汽车产品改进'
        f"多属性决策平台</b>：输入 {n_raw:,} 条真实车评与论文表5.7–5.20 的专家基准数据，"
        f"经四阶段流水线加工，进入「{n_attr} 属性 × {n_crit} 指标」决策模型，"
        f"输出<b>属性改进优先级</b>与<b>改进动作建议</b>，并实时标注每条结论的可复现性。</p>",
        unsafe_allow_html=True)
    kpi_row([
        {"label": "评论数据规模", "value": f"{n_raw:,}", "delta": f"有效 {n_kept:,} 条",
         "hint": f"保留率 {keep_rate:g}% · {k.get('n_platforms', 0)} 平台 "
                 f"{k.get('n_brands', 0)} 品牌 {k.get('n_models', 0)} 车型", "tone": "up"},
        {"label": "决策维度（属性 × 指标）", "value": f"{n_attr} × {n_crit}",
         "delta": f"一级维度 {len(base['meta']['first_level'])} 个",
         "hint": "属性金标准：词典标注 + 聚类对照"},
        {"label": "决策方法", "value": f"{len(methods)} 种",
         "delta": "主方法 PLTS-VIKOR",
         "hint": "、".join(m for m in methods if m != "PLTS_VIKOR") + " 同框对照"},
        {"label": "一致性审计项", "value": f"{len(findings)} 项",
         "delta": f"通过 {counts['pass']} · 已解析 {counts.get('resolved', 0)} · "
                  f"待确认 {counts['warn']} · 未解释 {counts['fail']}",
         "hint": "每条附根因、修复动作与修复前后数字",
         "tone": "up" if not counts["fail"] else "down"},
        {"label": "数据流水线就绪", "value": f"{fresh['ready']}/{fresh['total']}",
         "delta": f"最近更新 {fresh['age']}",
         "hint": "原始 → 分词 → 聚类 → 情感 → 决策 → ISA"},
    ])
    st.markdown(legend_html())

    # ------------------------------------------------------ 2. 3 分钟上手路线
    section_header("3 分钟上手路线", "五步走完：结论 → 证据 → 算法 → 交付 → 接数据；点按钮直接跳转",
                   tag="路线")
    rows = [(0, 1), (2, 3), (4, 4)]
    for r, (a, b) in enumerate(rows):
        if a >= len(ROUTES):
            break
        cols = layout.split("even")
        _route_card(cols[0], ROUTES[a], a)
        if b != a and b < len(ROUTES):
            _route_card(cols[1], ROUTES[b], b)
        if r == 0:
            st.markdown('<div class="ob-arrow">↓</div>', unsafe_allow_html=True)

    # ------------------------------------------------------ 3. 产品组织架构
    section_header("产品组织架构", "五层分层，每层职责单一、可独立测试；自上而下逐层汇入页面",
                   tag="架构")
    with layout.panel("分层结构", "L1 → L5：节点即层号，卡片里是该层的路径与一句话职责（明暗主题自动跟随）",
                      tag="ARCH"):
        st.markdown(_arch_html(), unsafe_allow_html=True)

    # ------------------------------------------------------ 4. 两条数据链路
    section_header("两条数据链路", "论文基准链路 与 业务数据链路 并行，在「双口径对照」处汇合",
                   tag="数据流")
    left, right = layout.split("even")
    with left:
        with layout.panel("① 论文基准链路", "不改动论文数据，先忠实转录，再独立复算", tag="SSOT",
                          footer=layout.note("在哪看：决策总览 · 洞察与报告", "ok")):
            st.markdown(_chain_html(PAPER_STEPS), unsafe_allow_html=True)
    with right:
        with layout.panel("② 业务数据链路", "四阶段流水线把原始车评加工成可聚合指标", tag="PIPELINE",
                          footer=layout.note("在哪看：属性情感分析 · 数据管理", "ok")):
            st.markdown(_chain_html(BIZ_STEPS), unsafe_allow_html=True)

    # ------------------------------------------------------ 5. 关键概念词典
    section_header("关键概念词典", "没读过论文也能看懂页面在说什么（9 条，先扫一遍再往下滑）",
                   tag="词典")
    for r in range(3):
        cols = layout.split("even3")
        for c in range(3):
            term = GLOSSARY[r * 3 + c]
            with cols[c]:
                advice_card({"title": term["title"], "body": term["body"],
                             "tone": term["tone"]})

    # ------------------------------------------------------ 6. 可复现性说明
    section_header("可复现性说明", "哪些能复现、哪些是论文自身冲突，逐条列示，不做静默修正",
                   tag="可信度")
    callout(
        "系统以<b>论文基准为唯一事实源（SSOT）</b>：实时复算结果与论文基准<b>并列展示</b>，"
        "两者不一致时给出偏差数值与解释；论文内部口径冲突（例如判断矩阵算不出表5.8 权重、"
        "表5.17 的 Q 无法由表5.16 的 S/R 推出）由审计面板<b>显式列出</b>，"
        "既不改论文数字，也不让算法悄悄凑数。")
    chips = " ".join(
        f'<span class="ds-chip chip-{severity_tone(f.status)}">'
        f"{audit_badge(f.status)} · {f.title}</span>"
        for f in findings)
    st.markdown(f'<div style="line-height:2.4">{chips}</div>', unsafe_allow_html=True)
    with st.expander("展开：9 项审计的当前状态与关键数字"):
        for f in findings:
            st.markdown(
                f'<div class="ob-file" style="line-height:2.0">'
                f'<span class="ds-chip chip-{severity_tone(f.status)}">'
                f"{audit_badge(f.status)}</span> "
                f"<b>{f.title}</b><i> — {f.detail}</i></div>", unsafe_allow_html=True)

    # ------------------------------------------------------ 7. 60 秒讲稿
    section_header("评审 / 面试 60 秒讲稿", "问题 — 方法 — 结果 — 可信度 — 工程化，五句话讲完",
                   tag="话术")
    with layout.panel("照着说即可", "数字均来自页面现算值，与看板一致", tag="PITCH"):
        for pt in TALKING:
            advice_card({"title": pt["title"], "body": pt["body"], "tone": pt["tone"]})

    # ------------------------------------------------------ 8. 运行与出处
    section_header("本地运行与出处", "命令、测试状态、论文出处与数据新鲜度", tag="附录")
    with layout.panel("如何本地运行", "依赖已随仓库提供，无需改动系统配置", tag="RUN"):
        cols = layout.split("even")
        with cols[0]:
            st.markdown(
                '<div class="ob-kv">'
                '<b>方式一 · 命令行</b><br><span>进入 <code>产品实现/</code> 后执行：</span><br>'
                '<span class="ob-code">./vene/bin/python -m streamlit run app.py</span><br><br>'
                '<b>方式二 · 双击</b><br><span>直接运行 <code>run.command</code>'
                "（自动选择虚拟环境）</span><br><br>"
                '<b>自检</b><br><span>41 项单元测试 + 5 视图应用级冒烟：'
                '<span class="ob-code">./vene/bin/python -m pytest tests/ -q</span></span>'
                "</div>",
                unsafe_allow_html=True)
        with cols[1]:
            age = fresh["age"] if fresh.get("all_ready") else "存在缺失产物"
            st.markdown(
                '<div class="ob-kv">'
                "<b>论文出处</b><br><span>上海理工大学硕士学位论文 ·"
                "《融合情感分析与PLTS-VIKOR的新能源汽车产品改进多属性决策研究》</span><br>"
                "<b>数据新鲜度</b><br>"
                f'<span>六阶段就绪 {fresh["ready"]}/{fresh["total"]} · '
                f'最近产物 {fresh["newest"]}（{age}）</span><br>'
                "<b>运行环境</b><br><span>Python 3.9.6 · Streamlit 1.50 · "
                "依赖锁定 requirements.txt / requirements.lock.txt</span><br>"
                "<b>当前口径</b><br><span>"
                + f"{mode_label} · λ={sc.lam:g} · v={sc.v:g} · "
                + ("达标度方向" if sc.direction == "attainment" else "差值度方向") + "</span><br>"
                "<b>当前筛选</b><br><span>" + _filters_brief(store.filters()) + "</span>"
                "</div>",
                unsafe_allow_html=True)
