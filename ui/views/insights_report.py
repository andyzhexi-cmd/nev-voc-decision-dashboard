"""ui.views.insights_report — ④ 洞察与报告（Actionable Insights & Report Builder）

区块
----
1. 改进优先级（ISA）      三口径四象限 + 象限建议卡片 + 口径冲突提示
2. 敏感性分析            λ / v 滑块（写回场景）+ P(x) 扫描折线 + 排序稳定性 + 论文表5.18 对照
3. 方法对比              PLTS-VIKOR / 传统VIKOR / TOPSIS / 前景理论（论文表5.19）
4. 审计面板              逐条 finding（徽章 + 说明 + 关键数字）+ audit_summary KPI 行
5. 报告导出              Excel / Markdown / PDF 下载（services.report，现场生成并落盘）

约定：所有图表传 mode=MODE；重 import 放在函数内；空数据用 empty_state。
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from state import store
from ui import theme
from ui.components import (advice_card, audit_badge, callout, empty_state, kpi_row,
                           quadrant_chart, rank_bars, section_header, severity_tone,
                           status_chip)

SRC_KEYS = {"论文表5.20": "paper", "图5.10": "figure", "实测情感": "sentiment"}
QUAD_ORDER = ["改进区", "保持区", "机会区", "低优先级区"]
QUAD_TONE = {"改进区": "bad", "保持区": "ok", "机会区": "brand", "低优先级区": "muted"}
SEV_LABEL = {"high": "高", "medium": "中", "low": "低"}


# ================================================================ 1. ISA
def _isa_section(MODE: str, overview: "pd.DataFrame | None") -> tuple[str, dict | None]:
    """返回 (口径标签, isa dict)；数据不足时 isa 为 None。"""
    from core.algorithm.isa import quadrant_advice
    from services import report as R

    section_header("改进优先级（ISA）",
                   "重要性-满意度四象限：改进区优先攻关，保持区构筑口碑护城河", tag="ISA")

    label = st.radio("数据口径", ["论文表5.20", "图5.10", "实测情感"],
                     horizontal=True, index=0)
    src = SRC_KEYS.get(label, "paper")

    if src == "sentiment":
        isa = R.sentiment_isa(overview)
        if isa is None:
            empty_state("暂无实测情感数据",
                        "当前筛选下没有可用的属性情感均值：请先在「数据管理」完成流水线，或清空全局筛选。")
            return label, None
    else:
        from core.algorithm.isa import isa_quadrants
        isa = isa_quadrants(source=src)

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.plotly_chart(quadrant_chart(isa, mode=MODE), use_container_width=True,
                        key="isa_quadrant_chart")
        st.caption(
            f"口径：{label} · 重要性均值 {isa['importance_mean']:.3f} · "
            f"满意度均值 {isa['satisfaction_mean']:.3f} · "
            f"标准差 {isa.get('importance_std', 0):.3f} / {isa.get('satisfaction_std', 0):.3f}")
        if src == "sentiment":
            st.caption("实测情感口径：评论情感均值（VADER 与朴素贝叶斯融合，取值 -1~+1）"
                       "经「情感值 → 李克特」线性换算到 1-5 分；重要性沿用论文表5.20。")

    with right:
        for q in QUAD_ORDER:
            recs = [r for r in isa["records"] if r["象限"] == q]
            status_chip(f"{q} · {len(recs)} 项", tone=QUAD_TONE[q])
            if not recs:
                st.caption("（无属性落入该象限）")
                st.markdown("")
                continue
            for r in recs:
                advice_card(quadrant_advice(r))

    st.markdown("")
    st.markdown("**ISA 象限数据表**")
    st.dataframe(pd.DataFrame(isa["records"]), use_container_width=True, hide_index=True)

    _isa_conflict_note()
    return label, isa


def _isa_conflict_note() -> None:
    """论文表5.20 与 图5.10 两套满意度口径的冲突提示（取自 baselines()['isa']）。"""
    from core.algorithm.models import baselines

    cfg = baselines()["isa"]
    p, f510 = dict(cfg["satisfaction"]), dict(cfg["satisfaction_figure510"])
    diffs = {a: (float(p[a]), float(f510[a])) for a in p
             if abs(float(p[a]) - float(f510[a])) > 1e-9}
    if not diffs:
        return
    parts = [f"{a}：表5.20={x:.3f}，图5.10={y:.3f}（Δ={y - x:+.3f}）"
             for a, (x, y) in diffs.items()]
    status_chip("⚠ 满意度口径冲突", tone="warn")
    callout("论文内部存在两套 ISA 满意度口径，" +
            "；".join(parts) +
            "。两套口径下象限归属可能不同（本页按所选口径实时划分，系统不做静默修正；"
            "详见审计项 ISA_SATISFACTION_CONFLICT）。")
    with st.expander("查看两套口径明细"):
        rows = [{"属性": a, "表5.20满意度": x, "图5.10满意度": y,
                 "差值": round(y - x, 3),
                 "表5.20象限": _quad_of(a, "paper"), "图5.10象限": _quad_of(a, "figure"),
                 "象限变化": "是" if _quad_of(a, "paper") != _quad_of(a, "figure") else "否"}
                for a, (x, y) in diffs.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _quad_of(attr: str, source: str) -> str:
    from core.algorithm.isa import isa_quadrants
    for r in isa_quadrants(source=source)["records"]:
        if r["属性"] == attr:
            return r["象限"]
    return "—"


# ================================================================ 2. 敏感性
def _sweep_chart(sweep: dict, x_name: str, MODE: str, key: str,
                 title: str) -> None:
    import plotly.express as px

    xs = sweep["lambdas"] if x_name == "λ" else sweep["vs"]
    df = pd.DataFrame({a: [round(float(v), 4) for v in vals]
                       for a, vals in sweep["Pxi"].items()})
    df.insert(0, x_name, [float(x) for x in xs])
    long = df.melt(id_vars=x_name, var_name="属性", value_name="P(x)")
    fig = px.line(long, x=x_name, y="P(x)", color="属性", markers=True,
                  color_discrete_map=theme.attribute_colors(),
                  category_orders={"属性": list(sweep["Pxi"].keys())})
    fig = theme.finish(fig, MODE, height=300, title=title)
    st.plotly_chart(fig, use_container_width=True, key=key)


def _sensitivity_section(MODE: str, sc, res) -> tuple[dict, dict]:
    from core.algorithm.sensitivity import (compare_to_paper_sensitivity,
                                            kendall_tau, paper_sensitivity_table,
                                            sweep_lambda, sweep_v)

    section_header("敏感性分析",
                   "λ（DEMATEL 一级权重占比）与 v（群体效用偏好）扰动下的 P(x) 轨迹与排序稳健性",
                   tag="Sensitivity")

    c_left, c_mid, c_right = st.columns([1.0, 2.0, 1.6], gap="medium")

    with c_left:
        lam = st.slider("λ (DEMATEL 一级权重占比)", 0.0, 1.0, float(sc.lam), 0.05,
                        key=f"ins_lam_{sc.lam:g}")
        if lam != sc.lam:                       # 守卫：只有值真正变化才写回，避免无限 rerun
            sc = store.update_scenario(lam=float(lam))
        v = st.slider("v (群体效用偏好)", 0.0, 1.0, float(sc.v), 0.05,
                      key=f"ins_v_{sc.v:g}")
        if v != sc.v:
            sc = store.update_scenario(v=float(v))
        sc = store.scenario()
        st.caption(
            f"当前场景 λ={sc.lam:g}、v={sc.v:g}；λ=0 仅 AHP 主观权重，λ=1 仅 DEMATEL 关联权重；"
            f"v=0 仅看个体遗憾，v=1 仅看群体效用。改动会即时写入场景并同步到其他视图。")
        from core.algorithm.models import baselines
        paper_rank = list(baselines()["vikor"]["ranking"])
        tau = float(kendall_tau(list(store.compute_result(sc).ranking), paper_rank))
        status_chip(f"当前排序 vs 论文排序 · Kendall τ={tau:.4f}",
                    tone="ok" if tau >= 0.999 else "warn" if tau >= 0.6 else "bad")

    sw_l = sweep_lambda(store.scenario())
    sw_v = sweep_v(store.scenario())

    with c_mid:
        _sweep_chart(sw_l, "λ", MODE, "sweep_lambda_chart", "P(x) 随 λ 的变化（每属性一条线）")
        _sweep_chart(sw_v, "v", MODE, "sweep_v_chart", "P(x) 随 v 的变化（每属性一条线）")

    with c_right:
        st.markdown("**排序稳定性（Kendall τ vs 基准排序）**")
        stab = pd.DataFrame({
            "λ": [float(x) for x in sw_l["lambdas"]],
            "Kendall τ": [round(float(t), 4) for t in sw_l["stability"]],
            "排序": [" > ".join(o) for o in sw_l["orders"]],
            "一致": ["是" if o == sw_l["base_order"] else "否" for o in sw_l["orders"]],
        })
        st.dataframe(stab, use_container_width=True, hide_index=True, height=184)
        stab_v = pd.DataFrame({
            "v": [float(x) for x in sw_v["vs"]],
            "Kendall τ": [round(float(t), 4) for t in sw_v["stability"]],
            "一致": ["是" if o == sw_v["base_order"] else "否" for o in sw_v["orders"]],
        })
        st.dataframe(stab_v, use_container_width=True, hide_index=True, height=150)
        st.caption(f"λ 扫描{'全部排序一致 ✔' if sw_l['stable'] else '存在排序翻转 △'} · "
                   f"v 扫描{'全部排序一致 ✔' if sw_v['stable'] else '存在排序翻转 △'}")

        st.markdown("**复算 vs 论文表5.18**")
        try:
            cmp_df = pd.DataFrame(compare_to_paper_sensitivity(sw_l))
            st.dataframe(cmp_df, use_container_width=True, hide_index=True, height=210)
        except Exception as e:                                   # noqa: BLE001
            st.caption(f"对照表生成失败：{type(e).__name__}: {e}")
        with st.expander("论文表5.18 基准（只读）"):
            paper = paper_sensitivity_table()
            pdf_df = pd.DataFrame(paper["Pxi"])
            pdf_df.insert(0, "λ", [float(x) for x in paper["lambdas"]])
            st.dataframe(pdf_df, use_container_width=True, hide_index=True)

    return sw_l, sw_v


# ================================================================ 3. 方法对比
def _methods_section(MODE: str, sc, res) -> dict:
    from core.algorithm.benchmarks import compare_methods

    section_header("方法对比",
                   "同一规范化决策矩阵与同一组权重下的四种多属性决策方法（论文表5.19）",
                   tag="Benchmarks")
    cm = compare_methods(sc)

    left, right = st.columns([1.35, 1], gap="large")
    rows_df = pd.DataFrame(cm["rows"])
    if not rows_df.empty:
        rows_df["与论文一致"] = rows_df["与论文一致"].map({True: "是", False: "否"})
    try:
        n_top = int(str(cm.get("top1_agreement") or "0/4").split("/")[0])
    except (TypeError, ValueError):
        n_top = -1
    with left:
        status_chip(f"Top1 一致率 {cm.get('top1_agreement', '—')}",
                    tone="ok" if n_top == 4 else "warn" if n_top >= 3 else "bad")
        st.dataframe(rows_df, use_container_width=True, hide_index=True)
        st.caption("「是否一致」= 该方法的完整排序与论文表5.19 完全相同；"
                   "Top1 一致率 = 复算与论文首位属性相同的方法数 / 4。")
    with right:
        df = pd.DataFrame({"属性": list(res.attributes),
                           "P(x)": [round(float(x), 4) for x in res.Pxi]})
        df = df.sort_values("P(x)", ascending=False).reset_index(drop=True)
        st.plotly_chart(rank_bars(df, x="P(x)", y="属性", height=330,
                                  title="PLTS-VIKOR 主方法 · 复算 P(x)", mode=MODE),
                        use_container_width=True, key="method_rank_bars")

    st.markdown("")
    st.markdown("**四方法打分（各方法自身量纲，分数仅供同方法内比较）**")
    attrs = list(res.attributes)
    score_rows = []
    for m in cm.get("methods", []):
        row = {"方法": m.get("name", ""),
               "排序": " > ".join(m.get("ranking", []))}
        for i, a in enumerate(attrs):
            arr = m.get("scores")
            row[a] = round(float(arr[i]), 4) if arr is not None and i < len(arr) else None
        score_rows.append(row)
    if score_rows:
        st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)
    else:
        empty_state("暂无方法对比数据", "无法在当前场景下执行四方法对比，请稍后重试。")
    return cm


# ================================================================ 4. 审计面板
def _audit_section() -> list:
    from core.algorithm.audit import audit_summary

    section_header("审计面板", "论文基准 vs 实时复算的一致性核查，逐条附带可复核的数字",
                   tag="Audit")
    findings = store.get_audit()
    if not findings:
        empty_state("暂无审计发现", "审计函数未返回任何条目，请检查 core.algorithm.audit。")
        return []

    summ = audit_summary(findings)
    kpi_row([
        {"label": "核查项", "value": str(summ["total"]), "hint": "每次进入页面自动执行", "tone": "flat"},
        {"label": "通过", "value": str(summ["pass"]), "hint": "复算与论文一致", "tone": "up"},
        {"label": "待确认", "value": str(summ["warn"]), "hint": "口径差异需人工确认", "tone": "down"},
        {"label": "偏差", "value": str(summ["fail"]), "hint": "论文数值不可复算", "tone": "down"},
        {"label": "高严重度", "value": str(summ["high"]), "hint": "严重度为「高」的条目", "tone": "down"},
    ])
    st.markdown("")
    callout("以下问题来自论文数据本身，系统不做静默修正：每条核查均实时计算并展示关键数字，"
            "页面以论文基准为主口径、复算值并列展示，便于逐项复核。")

    for f in findings:
        tone = severity_tone(f.status)
        with st.container(border=True):
            st.markdown(
                f'<span class="ds-chip chip-{tone}">{audit_badge(f.status)}</span> '
                f'<span class="ds-chip chip-muted">{SEV_LABEL.get(f.severity, f.severity)} 严重度</span> '
                f'<span class="ds-chip chip-muted">{f.id}</span>'
                f'<b style="margin-left:6px;">{f.title}</b>',
                unsafe_allow_html=True)
            st.caption(f.detail)
            with st.expander("查看关键数字"):
                if f.numbers:
                    st.json(f.numbers)
                else:
                    st.caption("该项无数值明细。")
    return findings


# ================================================================ 5. 导出
@st.cache_data(show_spinner=False, ttl=900, max_entries=8)
def _payload(kind: str, ctx: dict) -> tuple[str, bytes]:
    """现场生成报告并读回 bytes；ctx（含场景与结果）整体作为缓存 key 的一部分。"""
    from services import report as R
    return R.build_payload(dict(ctx), kind)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=2)
def _pdf_probe() -> tuple[bool, str]:
    from services import report as R
    try:
        return R.pdf_probe()
    except Exception as e:                                    # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _filters_desc() -> str:
    f = store.filters()
    parts: list[str] = []
    if f.brand:
        parts.append("品牌=" + "/".join(f.brand))
    if f.model:
        parts.append("车型=" + "/".join(f.model))
    if f.polarity:
        parts.append("情感=" + {"positive": "仅正面", "negative": "仅负面"}.get(f.polarity, f.polarity))
    if f.date_range:
        parts.append(f"时间={f.date_range[0]} ~ {f.date_range[1]}")
    if not f.only_kept:
        parts.append("含双模型不一致评论")
    return "；".join(parts) if parts else "全量数据（未筛选）"


def _export_section(sc, res, overview, insight, kpi, findings,
                    isa_label: str, isa, sw_l, sw_v, methods) -> None:
    from services import report as R

    section_header("报告导出",
                   "Excel 多 Sheet · Markdown 结构化中文报告 · PDF 中文报告（同时落盘 outputs/reports）",
                   tag="Export")

    ctx = R.context_from(
        scenario=sc, result=res, findings=findings, overview=overview,
        insight=insight, kpi=kpi, filters_desc=_filters_desc(),
        data_version=store.data_version(), isa=isa,
        isa_source=SRC_KEYS.get(isa_label, "paper"),
        sweep_lambda_res=sw_l, sweep_v_res=sw_v, methods=methods,
    )

    cards = st.columns(3, gap="medium")
    specs = [
        ("excel", "📊 Excel 工作簿", "8 个 Sheet：执行摘要 / 决策结果（含论文基准 Δ）/ "
                                 "权重与中间量 / 敏感性 λ 与 v / 方法对比 / ISA 象限 / "
                                 "审计发现 / 属性情感明细"),
        ("markdown", "📝 Markdown 报告", "九章节结构化中文报告：背景与口径 → 执行摘要 → "
                                     "决策结果 → 权重体系 → 敏感性 → 方法对比 → ISA → 审计 → 附录"),
        ("pdf", "📄 PDF 报告", "标题页 + 执行摘要 + 决策表 + ISA 表 + 审计摘要（reportlab 中文）"),
    ]
    pdf_ok, pdf_msg = _pdf_probe()

    for col, (kind, title, desc) in zip(cards, specs):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(desc)
                if kind == "pdf" and not pdf_ok:
                    status_chip("PDF 已降级为 Markdown", tone="warn")
                    st.caption(f"{pdf_msg}。请使用左侧 Markdown 下载，内容与 PDF 等价。")
                    continue
                try:
                    name, data = _payload(kind, ctx)
                except Exception as e:                        # noqa: BLE001
                    if kind == "pdf":
                        status_chip("PDF 生成失败 · 已降级为 Markdown", tone="warn")
                        st.caption(f"{type(e).__name__}: {e}。请使用左侧 Markdown 下载。")
                        continue
                    st.error(f"{title} 生成失败：{type(e).__name__}: {e}")
                    continue
                st.download_button(
                    f"⬇ 下载 {'PDF' if kind == 'pdf' else 'Excel' if kind == 'excel' else 'Markdown'}",
                    data=data, file_name=name, mime=R.KIND_MIME[kind],
                    use_container_width=True, key=f"dl_{kind}")
                st.caption(f"{len(data) / 1024:.0f} KB · 已落盘 {name}")

    st.markdown("")
    st.markdown(
        f"**报告目录**：`{R.REPORTS_DIR}` —— 每次生成都会在该目录写入一份带时间戳的同名文件；"
        "点击上方下载按钮即触发生成（约 1 秒内）。")
    st.code(f'open "{R.REPORTS_DIR}"', language="bash")
    st.caption(f"缓存口径：场景 λ={sc.lam:g}、v={sc.v:g} · 数据版本 {store.data_version()} · "
               f"筛选 {_filters_desc()} · 口径 {isa_label}；任一参数变化后报告会自动重新生成。")


def _frame(kind: str, filters_json: str, version: int):
    """store.features_frame 的三参封装（kind / filters_json / data_version）。"""
    return store.features_frame(kind, filters_json, version)


# ================================================================ 入口
def render() -> None:
    MODE = store.theme()
    sc = store.scenario()
    res = store.compute_result(sc)
    filters_json = store.filters_json()
    version = store.data_version()
    overview = _frame("overview", filters_json, version)
    insight = _frame("insight", filters_json, version)
    kpi = _frame("kpi", filters_json, version)

    section_header("洞察与报告",
                   "改进优先级 · 敏感性与稳健性 · 方法对比 · 一致性审计 · 一键导出",
                   tag="Insights")

    isa_label, isa = _isa_section(MODE, overview)
    sw_l, sw_v = _sensitivity_section(MODE, sc, res)
    # 敏感性区的滑块可能已改写场景 → 后续区块统一用最新场景与结果
    sc = store.scenario()
    res = store.compute_result(sc)
    methods = _methods_section(MODE, sc, res)
    _audit_section()
    _export_section(sc, res, overview, insight, kpi, store.get_audit(),
                    isa_label, isa, sw_l, sw_v, methods)
