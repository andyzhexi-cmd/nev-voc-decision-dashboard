"""ui.components.indicators — 版式与指标组件（HTML 注入类，全部走主题令牌）

约定：
  * 所有组件函数都接收 `mode`（'dark'/'light'）以保证明暗一致
  * 只负责渲染，不读 session_state、不拉数据
"""
from __future__ import annotations

import streamlit as st

from ui.theme import attribute_colors


def section_header(title: str, subtitle: str | None = None, tag: str | None = None) -> None:
    tag_html = f'<span class="ds-chip chip-brand" style="margin-left:8px;">{tag}</span>' if tag else ""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="ds-section"><h3>{title}{tag_html}</h3>{sub}<div class="rule"></div></div>',
        unsafe_allow_html=True)


def kpi_card(label: str, value: str, delta: str | None = None, hint: str | None = None,
             tone: str = "flat") -> None:
    """单个 KPI 卡片。tone ∈ up/down/flat（用于 delta 颜色）。"""
    delta_cls = {"up": "k-up", "down": "k-down"}.get(tone, "k-flat")
    delta_html = f'<div class="k-delta {delta_cls}">{delta}</div>' if delta else ""
    hint_html = f'<div class="k-hint">{hint}</div>' if hint else ""
    st.markdown(
        f'<div class="ds-kpi"><div class="k-label">{label}</div>'
        f'<div class="k-value">{value}</div>{delta_html}{hint_html}</div>',
        unsafe_allow_html=True)


def kpi_row(cards: list[dict]) -> None:
    """一行 KPI。cards: [{label, value, delta, hint, tone}]。"""
    cols = st.columns(len(cards))
    for c, card in zip(cols, cards):
        with c:
            kpi_card(card.get("label", ""), card.get("value", ""),
                     card.get("delta"), card.get("hint"), card.get("tone", "flat"))


def status_chip(text: str, tone: str = "muted") -> None:
    st.markdown(f'<span class="ds-chip chip-{tone}">{text}</span>', unsafe_allow_html=True)


def mode_badge(mode: str) -> str:
    """当前算法口径徽章文本。"""
    return "论文校准模式" if mode == "calibrated" else "在线复算模式"


def algorithm_badges(scenario, findings: list | None = None) -> str:
    """口径徽章的 HTML（纯函数，供整行徽章条复用）。"""
    from core.algorithm.models import Scenario
    sc: Scenario = scenario
    if sc.mode == "calibrated":
        tone, label = "ok", "● 论文校准模式 · 基准取自论文表5.8/5.11/5.12/5.16"
    else:
        tone, label = "warn", "● 在线复算模式 · AHP/DEMATEL 实时计算"
    n_fail = sum(1 for f in (findings or []) if f.status == "fail")
    n_res = sum(1 for f in (findings or []) if f.status == "resolved")
    n_warn = sum(1 for f in (findings or []) if f.status == "warn")
    audit_txt = ((f"{n_fail} 项偏差 · " if n_fail else (f"{n_res} 项已解析 · " if n_res else ""))
                 + f"{n_warn} 项待确认")
    audit_tone = "bad" if n_fail else ("ok" if not n_warn else "warn")
    return (
        f'<span class="ds-chip chip-{tone}">{label}</span>'
        f'<span class="ds-chip chip-muted">λ={sc.lam:g} · v={sc.v:g} · '
        f'{"达标度" if sc.direction == "attainment" else "差值度"} · '
        f'{"按指标列理想解" if sc.ideal_strategy == "criterion" else "情感理想解"}</span>'
        f'<span class="ds-chip chip-{audit_tone}">审计 {audit_txt}</span>'
    )


def algorithm_badge(scenario, findings: list | None = None) -> None:
    """顶部口径徽章：论文校准 / 在线复算 + 模式参数摘要。"""
    st.markdown(algorithm_badges(scenario, findings), unsafe_allow_html=True)


def callout(text: str) -> None:
    st.markdown(f'<div class="ds-callout">{text}</div>', unsafe_allow_html=True)


def advice_card(card: dict) -> None:
    tone = card.get("tone", "neutral")
    st.markdown(
        f'<div class="ds-adv adv-{tone}"><span class="t">{card.get("title", "")}</span>'
        f'{card.get("body", "")}</div>',
        unsafe_allow_html=True)


def empty_state(title: str = "暂无数据", body: str = "请先在「数据管理」页完成流水线，或调整筛选条件。") -> None:
    st.markdown(
        f'<div class="ds-card" style="text-align:center;padding:44px 16px;">'
        f'<div style="font-size:16px;font-weight:700;">{title}</div>'
        f'<div style="color:var(--muted);font-size:13px;margin-top:6px;">{body}</div></div>',
        unsafe_allow_html=True)


def legend_html(extra: dict | None = None) -> str:
    colors = attribute_colors()
    if extra:
        colors = {**colors, **extra}
    spans = "".join(
        f'<span class="ds-chip chip-muted" style="border-color:{v};">'
        f'<span class="dot" style="background:{v}"></span>{k}</span>&nbsp;'
        for k, v in colors.items())
    return f'<div style="margin:2px 0 8px">{spans}</div>'


def dev_table(df, column: str = "Δvs论文") -> None:
    """带正负色条的偏差表。"""
    st.dataframe(
        df, width="stretch", hide_index=True,
        column_config={column: st.column_config.ProgressColumn(
            column, help="复算值与论文基准的偏差", format="%.4f",
            min_value=0.0, max_value=0.3)}
        if column in df.columns and str(df[column].dtype).startswith("float") else None,
    )


def audit_badge(status: str) -> str:
    return {"pass": "✓ 通过", "resolved": "◆ 已解析", "warn": "△ 待确认",
            "fail": "✕ 偏差"}.get(status, status)


def severity_tone(status: str) -> str:
    return {"pass": "ok", "resolved": "ok", "warn": "warn", "fail": "bad"}.get(status, "muted")
