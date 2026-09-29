"""ui.components.audit_panel — 审计与整改面板

一次性呈现「根因 → 修复动作 → 修复前后数字」：

  * KPI 行：核查 / 通过 / 已解析 / 待确认 / 未解释
  * 逐条卡片：状态徽章（audit_badge + severity_tone）、detail、根因、修复动作与效果、
    修复前/后数字对照（st.json）
  * 每条带修复开关的卡片有 `st.toggle`，通过回调 `on_toggle(flag_id, value)` 回写
    `Scenario.repair_flags`（本组件不直接依赖 state.store，便于独立测试与复用）

典型接线（视图层）::

    from ui.components.audit_panel import audit_panel

    def _on_toggle(fid: str, value: bool) -> None:
        flags = dict(store.scenario().repair_flags)
        flags[fid] = value
        store.update_scenario(repair_flags=flags)   # 缓存键随之变化，审计自动重算
        st.rerun()

    audit_panel(findings, store.scenario().repair_flags, on_toggle=_on_toggle, mode=MODE)
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

import streamlit as st

from ui import layout
from ui.components.indicators import (audit_badge, callout, kpi_row,
                                      section_header, severity_tone, status_chip)

STATUS_TONE = {"pass": "ok", "resolved": "ok", "warn": "warn", "fail": "bad"}


def _jsonable(d) -> dict:
    """把 numpy / 嵌套结构转成 st.json 可序列化的纯 Python。"""
    import numpy as np

    def conv(x):
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, np.ndarray):
            return [conv(v) for v in x.tolist()]
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return x.item()
        if isinstance(x, float) and x != x:      # NaN → 可读文本
            return "—"
        return x
    return {str(k): conv(v) for k, v in (d or {}).items()}


def _summary_cards(findings: Sequence) -> list[dict]:
    s = _audit_summary(findings)
    return [
        {"label": "核查项", "value": str(s["total"]), "hint": "审计条目总数", "tone": "flat"},
        {"label": "通过", "value": str(s["pass"]), "hint": "可精确复现", "tone": "up"},
        {"label": "已解析", "value": str(s["resolved"]), "hint": "根因已定位 + 修复已生效", "tone": "up"},
        {"label": "待确认", "value": str(s["warn"]), "hint": "论文原值如实标注", "tone": "flat"},
        {"label": "未解释", "value": str(s["unexplained"]), "hint": "默认修复全开时应为 0",
         "tone": "down" if s["unexplained"] else "up"},
    ]


def _audit_summary(findings: Sequence) -> dict:
    try:
        from core.algorithm.audit import audit_summary
        return audit_summary(list(findings))
    except Exception:                               # 兜底：不因审计模块异常整页崩
        return {"total": len(findings),
                "pass": sum(1 for f in findings if getattr(f, "status", "") == "pass"),
                "resolved": sum(1 for f in findings if getattr(f, "status", "") == "resolved"),
                "warn": sum(1 for f in findings if getattr(f, "status", "") == "warn"),
                "fail": sum(1 for f in findings if getattr(f, "status", "") == "fail"),
                "unexplained": sum(1 for f in findings if getattr(f, "status", "") == "fail"),
                "high": 0}


def _toggle(fid: str, enabled: bool, label: str, effect: str,
            on_toggle: Callable[[str, bool], None] | None) -> None:
    """单条修复开关。key 随当前状态变化：开关变化后重建 widget，显示值即时对齐。"""
    key = f"dsh_repair_{fid}_{int(bool(enabled))}"
    hint = f"修复动作：{label} —— {effect}" if label else effect

    def _cb():
        if on_toggle:
            on_toggle(fid, bool(st.session_state.get(key)))

    st.toggle(f"启用修复 · {label}" if label else "启用修复",
              value=bool(enabled),
              key=key,
              disabled=on_toggle is None,
              help=hint or None,
              on_change=_cb if on_toggle else None)


def audit_panel(findings: Iterable, repair_flags: dict | None = None,
                on_toggle: Callable[[str, bool], None] | None = None,
                mode: str = "dark") -> dict:
    """渲染审计与整改面板，返回 `audit_summary(...)` 结果。

    Parameters
    ----------
    findings : list[AuditFinding]
    repair_flags : 当前开关状态（与 finding.repair.enabled 对照，缺失时按 finding 自带状态）
    on_toggle : 用户切换开关时回调 `on_toggle(flag_id, value)`（写回 repair_flags + rerun）
    mode : 'dark' | 'light'（当前仅预留，版式由全局 CSS 控制）
    """
    findings = list(findings)
    flags = dict(repair_flags or {})
    summary = _audit_summary(findings)

    section_header("一致性审计与整改",
                   subtitle="根因（取证结论）→ 修复动作 → 修复前后数字；论文原值始终保留在「修复前」可见",
                   tag="AUDIT")
    kpi_row(_summary_cards(findings))

    if summary.get("unexplained", 0):
        callout(f"⚠ 存在 {summary['unexplained']} 条未解释偏差（修复开关已关闭或修复失效），"
                "下方对应卡片已回落为「✕ 偏差」。")
    else:
        callout("✔ 默认修复全开时无无法解释的 fail：每条偏差都有根因、修复动作与修复前后数字。")

    severity_order = {"high": 0, "medium": 1, "low": 2}
    ordered = sorted(findings, key=lambda f: (severity_order.get(getattr(f, "severity", "low"), 3),
                                              f.id))
    for f in ordered:
        status = getattr(f, "status", "info")
        rep = getattr(f, "repair", {}) or {}
        switch = rep.get("switch")
        enabled = bool(flags.get(switch, rep.get("enabled", True))) if switch else True

        sev_label = {"high": "高", "medium": "中", "low": "低"}.get(
            getattr(f, "severity", ""), getattr(f, "severity", ""))
        with layout.panel(f.title,
                          subtitle=(getattr(f, "detail", "") or "").strip(),
                          tag=sev_label):
            head, chip_col = layout.split("head_action")
            with head:
                st.caption(f"{f.id}"
                           + (f" · 修复开关：{rep.get('label') or f.id}" if switch else ""))
            with chip_col:
                status_chip(audit_badge(status),
                            tone=STATUS_TONE.get(status, severity_tone(status)))

            if getattr(f, "root_cause", ""):
                st.markdown(layout.note(f"根因（取证）：{f.root_cause.strip()}", "warn"),
                            unsafe_allow_html=True)

            label = rep.get("label") or ""
            effect = rep.get("effect") or ""
            if label:
                st.markdown(f"**修复动作**：{label}" + (f" —— {effect}" if effect else ""))

            c_before, c_after = layout.split("even")
            with c_before:
                st.caption("修复前（论文原值）")
                st.json(_jsonable(getattr(f, "before", {}) or {}))
            with c_after:
                st.caption("修复后" if enabled else "修复后（修复已关闭，回落为 ✕ 偏差）")
                st.json(_jsonable(getattr(f, "after", {}) or {}))

            if switch:
                _toggle(f.id, enabled, label, effect, on_toggle)
            else:
                st.caption("无需开关：" + (label or "论文原值如实标注，系统不做静默修正"))

    return summary
