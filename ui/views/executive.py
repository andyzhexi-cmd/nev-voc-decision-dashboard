"""ui.views.executive — 决策总览（执行摘要 / Executive Dashboard）

版式（ui.layout）：页头 page_head → 区块 section_header → 面板 layout.panel → 图表/表格；
栅格只用具名比例（main_rail / even / head_actions / head_action），高度只取 layout.H 档位
（150/260/340/440），占满宽度一律 width="stretch"，不手工插入空行撑间距。

区块（自上而下）
  页头 layout.page_head（含算法口径徽章）+ 1) KPI 行（6 卡，等高 .ds-kpi）
  2) 双口径排序对照（论文表5.17 基准 vs 当前场景实时复算）—— 核心模块
  3) S-R 效用散点 + S/R/Q/Q′/P(x) 复算对照表
  4) 情感-重要性 2D 决策矩阵（象限分割线）
  5) 品牌雷达 + 车型口碑榜
  6) 月度情感趋势
  7) 叙事卡片（优势 / 短板）
  8) 审计口径声明 + 完整核查清单

约定：顶层不读大数据；数据一律 store.filters() → services.features.*（走
store.features_frame 缓存）；主题 MODE = store.theme()，所有图表传 mode=MODE。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.algorithm.vikor import paper_reference
from services.data_store import ATTRS
from state import store
from ui import layout
from ui.components import (advice_card, audit_badge, callout, empty_state, kpi_row,
                           section_header, severity_tone, status_chip)
from ui.components import charts
from ui.components.indicators import algorithm_badges
from ui.theme import attribute_colors, finish, palette

# ------------------------------------------------------------------ 数据入口
def _frame(kind: str):
    """带缓存的 features 取数（kind ∈ features_frame 的表名）。"""
    return store.features_frame(kind, store.filters_json(), store.data_version())


# ------------------------------------------------------------------ KPI
def _kpi_cards(k: dict, dp_live: float, dp_paper: float) -> list[dict]:
    tone = "up" if k["mean_sentiment"] >= k["paper_mean_sentiment"] else "down"
    return [
        {"label": "原始评论数", "value": f"{k['n_raw']:,}",
         "delta": f"噪声 {k['n_noise']:,} 条", "hint": "论文表5.1 基准 50,520 条",
         "tone": "flat"},
        {"label": "双模型一致有效评论", "value": f"{k['n_kept']:,}",
         "delta": f"剔除 {k['n_dropped']:,} 条", "hint": "VADER 与朴素贝叶斯判定一致方保留",
         "tone": "flat"},
        {"label": "数据保留率", "value": f"{k['keep_rate']}%",
         "delta": f"正向占比 {k['pos_ratio']}%", "hint": "剔除噪声与双模型分歧后的保留比例",
         "tone": "flat"},
        {"label": "评论平台 / 品牌", "value": f"{k['n_platforms']} / {k['n_brands']}",
         "delta": f"车型 {k['n_models']} 款", "hint": "覆盖平台数 / 品牌数 / 车型数",
         "tone": "flat"},
        {"label": "情感均值（复算）", "value": f"{k['mean_sentiment']:+.3f}",
         "delta": f"{k['mean_sentiment'] - k['paper_mean_sentiment']:+.3f} vs 论文",
         "hint": f"论文基准 {k['paper_mean_sentiment']:+.3f}", "tone": tone},
        {"label": "决策区分度 ΔP", "value": f"{dp_live:.3f}",
         "delta": f"论文基准 {dp_paper:.3f}",
         "hint": "max P(x) − min P(x)，越大越能拉开属性差距", "tone": "flat"},
    ]


# ------------------------------------------------------------------ 排序对照
def _consistency(live: list[str], paper: list[str]) -> tuple[str, str]:
    """返回 (判定文案, tone)。tone: ok / warn / bad。"""
    if list(live) == list(paper):
        return "同序", "ok"
    if list(live)[:3] == list(paper)[:3]:
        return "Top-k 一致", "warn"
    return "不一致", "bad"


def _rank_df(res, paper) -> pd.DataFrame:
    """分组条形图数据：属性 / P(x) / 口径（论文基准、当前场景）。"""
    order = list(paper.ranking)[::-1]          # 条形 y 轴反向：最高重要度在最上
    live = dict(zip(res.attributes, [float(v) for v in res.Pxi]))
    base = dict(zip(paper.attributes, [float(v) for v in paper.Pxi]))
    rows = []
    for a in order:
        rows.append({"属性": a, "P(x)": round(base.get(a, 0.0), 4), "口径": "论文基准"})
        rows.append({"属性": a, "P(x)": round(live.get(a, 0.0), 4), "口径": "当前场景"})
    return pd.DataFrame(rows)


def _dev_table(res, paper) -> pd.DataFrame:
    """S / R / Q / Q′ / P(x) 复算值 vs 论文值 + Δ（长表）。"""
    metrics = [("S", res.S, paper.S),
               ("R", res.R, paper.R),
               ("Q", res.Q, paper.Q),
               ("Q′", res.Q_prime, paper.Q_prime),
               ("P(x)", res.Pxi, paper.Pxi)]
    cur = dict(zip(res.attributes, range(len(res.attributes))))
    order = [a for a in paper.ranking if a in cur]
    rows = []
    for name, a_cur, a_paper in metrics:
        for a in order:
            i = cur[a]
            c, p = float(a_cur[i]), float(a_paper[i])
            rows.append({"属性": a, "指标": name, "复算值": round(c, 4),
                         "论文值": round(p, 4), "Δ": round(c - p, 4)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 情感-重要性矩阵
def _importance_matrix(ov: pd.DataFrame, paper, mode: str):
    """x=情感均值(features.attr_overview) · y=重要度 P(xi)(论文表5.17) · 气泡=评论数。"""
    sub = ov.copy()
    sub = sub[sub["评论数"] > 0]
    if sub.empty:
        return None
    pm = dict(zip(paper.attributes, [float(v) for v in paper.Pxi]))
    sub["重要度"] = sub["属性"].map(pm)
    sub = sub.dropna(subset=["重要度"])
    if sub.empty:
        return None
    n = sub["评论数"].astype(float)
    span = float(n.max() - n.min()) or 1.0
    size = 26.0 + 210.0 * (n - n.min()) / span
    colors = attribute_colors()
    fig = go.Figure(go.Scatter(
        x=sub["情感均值"], y=sub["重要度"], mode="markers+text",
        text=sub["属性"], textposition="top center",
        marker=dict(size=size.to_list(),
                    color=[colors.get(a, palette(mode)["brand"]) for a in sub["属性"]],
                    opacity=0.85,
                    line=dict(width=1.5, color=palette(mode)["text"])),
        customdata=np.stack([sub["评论数"], sub["正向"], sub["负向"]], axis=-1),
        hovertemplate=("属性 %{text}<br>情感均值 %{x:.4f} · 重要度 P(x) %{y:.4f}"
                       "<br>评论 %{customdata[0]:.0f} 条（正 %{customdata[1]:.0f} / "
                       "负 %{customdata[2]:.0f}）<extra></extra>")))
    xm, ym = float(sub["情感均值"].mean()), float(sub["重要度"].mean())
    fig.add_vline(x=xm, line_dash="dash", line_color=palette(mode)["muted"], line_width=1,
                  annotation_text="情感均值", annotation_position="top")
    fig.add_hline(y=ym, line_dash="dash", line_color=palette(mode)["muted"], line_width=1,
                  annotation_text="重要度均值", annotation_position="right")
    xr = float(sub["情感均值"].max() - sub["情感均值"].min()) or 1.0
    yr = float(sub["重要度"].max() - sub["重要度"].min()) or 1.0
    fig.update_layout(
        xaxis_title="情感均值（评论复算）", yaxis_title="重要度 P(x)（论文表5.17）",
        annotations=[
            dict(x=xm + 0.42 * xr, y=ym + 0.36 * yr, text="保持区（高重要·高满意）",
                 showarrow=False, font=dict(size=11, color=palette(mode)["ok_fg"])),
            dict(x=xm - 0.42 * xr, y=ym + 0.36 * yr, text="机会区（低重要·高满意）",
                 showarrow=False, font=dict(size=11, color=palette(mode)["brand_fg"])),
            dict(x=xm + 0.42 * xr, y=ym - 0.40 * yr, text="改进区（高重要·低满意）",
                 showarrow=False, font=dict(size=11, color=palette(mode)["bad_fg"])),
            dict(x=xm - 0.42 * xr, y=ym - 0.40 * yr, text="低优先级区",
                 showarrow=False, font=dict(size=11, color=palette(mode)["muted_fg"])),
        ])
    return finish(fig, mode, height=layout.height("l"))


# ------------------------------------------------------------------ 审计
def _audit_section() -> None:
    findings = store.get_audit()
    from core.algorithm.audit import audit_summary
    s = audit_summary(findings)

    with layout.panel("审计口径声明",
                      f"共 {s['total']} 项核查 · 高严重级 {s['high']} 项 · "
                      f"偏差来自论文数据本身，系统如实列示、不做静默修正"):
        try:
            import streamlit_shadcn_ui as ui_s
            ui_s.alert(title="审计口径声明",
                       description="论文基准与实时复算的差异全部公开，不掩盖、不修正。",
                       class_name="warning", key="dsh_audit_notice")
        except Exception:
            callout("论文基准与实时复算的差异全部公开，不掩盖、不修正。")
        notice = layout.split("head_actions")
        with notice[0]:
            status_chip(f"✓ 通过 {s['pass']}", "ok")
        with notice[1]:
            status_chip(f"◆ 已解析 {s.get('resolved', 0)}", "ok")
        with notice[2]:
            status_chip(f"△ 待确认 {s['warn']}", "warn")
        with notice[3]:
            status_chip(f"✕ 未解释偏差 {s['fail']}", "bad")

    with layout.panel("核查清单",
                      f"共 {s['total']} 项 · 根因 → 修复动作 → 修复前后数字，逐条可复核"):
        with st.expander("查看完整审计报告"):
            for f in findings:
                head, badge = layout.split("head_action")
                with head:
                    st.markdown(f"**{f.title}**")
                with badge:
                    status_chip(audit_badge(f.status), severity_tone(f.status))
                st.caption(f.detail)
                if f.numbers:
                    st.json(f.numbers)
                st.markdown("---")


# ------------------------------------------------------------------ 视图
def render() -> None:
    MODE = store.theme()
    sc = store.scenario()
    findings = store.get_audit()

    # ---------- 页头（每页唯一） ----------
    layout.page_head(
        "执行摘要",
        "论文表5.17 主口径 × 当前场景实时复算 · 六属性重要性与情感双维决策一页总览",
        chips=[algorithm_badges(sc, findings)])

    # ---------- 1) KPI 行 ----------
    section_header(
        "核心指标速览",
        subtitle="数据规模、情感基线与决策区分度，随全局筛选与场景实时重算",
        tag="KPI")
    k = _frame("kpi")
    res = store.compute_result()
    paper = paper_reference()
    dp_live = float(np.max(res.Pxi) - np.min(res.Pxi))
    dp_paper = float(np.max(paper.Pxi) - np.min(paper.Pxi))
    with layout.panel("数据规模 · 情感基线 · 决策区分度",
                      "论文基准见卡片 hint；ΔP = max P(xᵢ) − min P(xᵢ)，越大越能拉开属性差距"):
        kpi_row(_kpi_cards(k, dp_live, dp_paper))

    # ---------- 2) 双口径排序对照 ----------
    section_header(
        "双口径排序对照",
        subtitle="P(xᵢ) 分组条形：论文表5.17 基准与当前场景实时复算并排展示",
        tag="核心")
    left, right = layout.split("main_rail")
    with left:
        with layout.panel("P(xᵢ) 分组条形对照", "条形按论文基准 P(xᵢ) 由高到低排列"):
            st.plotly_chart(
                charts.rank_bars(_rank_df(res, paper), x="P(x)", y="属性", color="口径",
                                 barmode="group", height=layout.height("m"), mode=MODE))
    with right:
        with layout.panel("一致性判定与场景参数", "排序一致时方可直接引用论文结论"):
            verdict, tone = _consistency(res.ranking, paper.ranking)
            status_chip(f"排序一致性 · {verdict}", tone)
            st.markdown(
                f'<span class="ds-chip chip-muted">论文基准</span> '
                f'{" > ".join(paper.ranking)}', unsafe_allow_html=True)
            st.markdown(
                f'<span class="ds-chip chip-brand">当前场景</span> '
                f'{" > ".join(res.ranking)}', unsafe_allow_html=True)
            ideal_txt = ("按指标列理想解" if sc.ideal_strategy == "criterion" else "情感理想解")
            dir_txt = "达标度（越大越优）" if sc.direction == "attainment" else "差值度（越小越优）"
            st.markdown(
                f'<span class="ds-chip chip-muted">λ={sc.lam:g}</span> '
                f'<span class="ds-chip chip-muted">v={sc.v:g}</span> '
                f'<span class="ds-chip chip-muted">{dir_txt}</span> '
                f'<span class="ds-chip chip-muted">{ideal_txt}</span> '
                f'<span class="ds-chip chip-muted">'
                f'{"论文权重" if sc.use_paper_weights else "实时权重"}</span> '
                f'<span class="ds-chip chip-muted">'
                f'权重来源 {res.diagnostics.get("weight_source", "—")}</span>',
                unsafe_allow_html=True)
            callout("论文校准模式以论文表5.17 为主口径，实时复算并列展示，偏差由审计面板解释。")
            st.caption(f"决策区分度 ΔP：当前场景 {dp_live:.4f} · 论文基准 {dp_paper:.4f}；"
                       f"Top1 当前为「{res.ranking[0]}」，论文为「{paper.ranking[0]}」。")

    # ---------- 3) 决策散点 ----------
    section_header(
        "S-R 效用散点",
        subtitle="气泡大小 = 折衷评价值 Q；右表为 S / R / Q / Q′ / P(x) 复算值与论文值对照")
    sc1, sc2 = layout.split("main_rail")
    with sc1:
        with layout.panel("S 群体效用 × R 个体遗憾", "点划线 = S̄ 与 R̄ 均值线 · 气泡大小 = Q"):
            st.plotly_chart(charts.sr_scatter(res, mode=MODE, height=layout.height("l")))
    with sc2:
        with layout.panel("S / R / Q / Q′ / P(x) 逐项对照", "Δ = 复算值 − 论文值，按论文排序排列"):
            st.dataframe(_dev_table(res, paper), width="stretch",
                         hide_index=True, height=layout.height("l"))

    # ---------- 4) 情感-重要性矩阵 ----------
    section_header(
        "情感-重要性矩阵",
        subtitle="x = 情感均值（评论复算） · y = 重要度 P(xᵢ)（论文表5.17） · 气泡 = 评论数 · 虚线 = 均值分割")
    ov = _frame("overview")
    fig_m = _importance_matrix(ov, paper, MODE) if isinstance(ov, pd.DataFrame) else None
    with layout.panel("情感 × 重要度四象限", "气泡面积 = 评论数 · 虚线 = 情感与重要度均值分割"):
        if fig_m is None:
            empty_state("暂无情感数据", "请先在「数据管理」页完成情感分析流水线，或调整全局筛选。")
        else:
            st.plotly_chart(fig_m)

    # ---------- 5) 品牌雷达 + 车型榜 ----------
    section_header("品牌雷达与车型榜", subtitle="六属性情感雷达（按品牌叠画） 与 车型口碑 Top 10")
    rc1, rc2 = layout.split("even")
    with rc1:
        with layout.panel("六属性情感雷达", "按品牌叠画，取值为该品牌在该属性上的情感均值"):
            series = _frame("radar")
            if not series:
                empty_state("暂无品牌数据", "请先完成情感分析流水线或调整筛选条件。")
            else:
                st.plotly_chart(charts.radar(series, labels=ATTRS, mode=MODE,
                                             height=layout.height("l")))
    with rc2:
        with layout.panel("车型口碑榜 Top 10", "按当前筛选下的情感均值降序"):
            ranking = _frame("ranking")
            if ranking is None or ranking.empty:
                empty_state("暂无车型榜数据", "请先完成情感分析流水线或调整筛选条件。")
            else:
                st.dataframe(ranking.head(10), width="stretch", hide_index=True,
                             height=layout.height("l"))

    # ---------- 7) 月度趋势 ----------
    section_header("月度情感趋势", subtitle="各属性情感均值按月走势（气泡点标注评论数）")
    trend = _frame("trend")
    with layout.panel("六属性月度走势", "点大小 = 当月评论数；随全局筛选实时重算"):
        if trend is None or trend.empty:
            empty_state("暂无趋势数据", "请先完成情感分析流水线或调整筛选条件。")
        else:
            st.plotly_chart(charts.trend_line(trend, MODE, height=layout.height("m")))

    # ---------- 8) 叙事卡片 ----------
    section_header("洞察叙事", subtitle="由数据自动生成的优势与短板（随筛选实时重算）")
    insight = _frame("insight")
    with layout.panel("优势 strengths / 短板 weaknesses",
                      "由属性情感均值与重要度实时排序，不写死结论"):
        if insight.get("headline"):
            callout(insight["headline"])
        ic1, ic2 = layout.split("even")
        with ic1:
            st.markdown("**优势 strengths**")
            for card in insight.get("strengths", []):
                advice_card(card)
        with ic2:
            st.markdown("**短板 weaknesses**")
            for card in insight.get("weaknesses", []):
                advice_card(card)

    # ---------- 9) 审计摘要条 ----------
    section_header("一致性审计", subtitle="论文基准 vs 实时复算的偏差逐项列示，不掩盖、不静默修正")
    _audit_section()
