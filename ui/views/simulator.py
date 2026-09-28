"""PLTS-VIKOR 交互式模拟器视图

布局：左栏（参数，320px） · 中栏（解释性结果与中间量） · 右栏（结果面板，280px）

所有参数写入 state.store 的 Scenario；矩阵/权重的实验性修改写入 overrides，
应用后由 `store.compute_result()` 统一执行。论文基准始终并列展示，不被覆盖。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.algorithm.models import Scenario, baselines
from core.algorithm.sensitivity import compare_to_paper_sensitivity, kendall_tau, sweep_lambda, sweep_v
from core.algorithm.vikor import paper_reference
from services.features import attr_overview
from state import store
from ui import theme
from ui.components import (callout, empty_state, kpi_row, matrix_heatmap, rank_bars,
                           section_header, sr_scatter, status_chip)

MODELS = ["dematel_within", "equal_within", "pure_sub", "first_only"]
MODEL_LABEL = {
    "dematel_within": "组内按 DEMATEL 分配（默认）",
    "equal_within": "组内均分",
    "pure_sub": "只用 12 个 DEMATEL 分值",
    "first_only": "只用一级权重均分",
}
B = baselines()


# ---------------------------------------------------------------- 参数面板
def _param_panel(sc: Scenario) -> tuple[bool, bool]:
    """左侧参数栏，返回 (已应用, 已重置)。"""
    with st.form("sim_params"):
        st.markdown("**计算参数**")
        mode = st.radio("计算口径", ["论文校准", "在线复算"],
                        index=0 if sc.mode == "calibrated" else 1,
                        help="论文校准：权重取自论文表5.8/5.11；在线复算：由矩阵实时计算")
        lam = st.slider("λ · DEMATEL 一级权重占比", 0.0, 1.0, float(sc.lam), 0.05,
                        help="式(4.7)：w = λ·wD + (1−λ)·wA")
        v = st.slider("v · 决策系数", 0.0, 1.0, float(sc.v), 0.05,
                      help="v→1 更看重群体效用 S；v→0 更看重个体遗憾 R")
        ideal = st.radio("理想解策略", ["按指标列（列内极值）", "按属性情感值（表5.15）"],
                         index=0 if sc.ideal_strategy == "criterion" else 1,
                         help="表5.14/5.15 给出按属性的理想解；按指标列是标准 VIKOR 口径")
        completion = st.radio("PLTS 概率补全", ["区间补全", "归一化"],
                              index=0 if sc.prob_completion == "interval" else 1,
                              help="表5.13 中有 16 个格子 Σp<1（最小 0.9）")
        direction = st.radio("效用方向", ["达标度 attainment（论文口径）", "差值度 shortfall（传统 VIKOR）"],
                             index=0 if sc.direction == "attainment" else 1,
                             help="论文按 S 越大越重要解释；传统 VIKOR 按 S 越小越优")
        weight_mode = st.selectbox("二级权重展开方式", MODELS,
                                   index=MODELS.index(sc.weight_mode),
                                   format_func=lambda k: MODEL_LABEL[k])
        with st.expander("高级 · 可能度参数"):
            sigma_floor = float(st.slider("σ 下限", 0.01, 0.30, 0.06, 0.01,
                                          help="式(4.12) 正态比较的离散度下限，越大排序越平滑"))
            pd_mode = st.radio("可能度实现", ["正态分布 Φ", "硬比较"], horizontal=True,
                               index=0)
        submitted = st.form_submit_button("应用参数", use_container_width=True, type="primary")
    reset = st.button("↺ 恢复论文基准场景", use_container_width=True)
    if reset:
        store.reset_scenario()
        st.session_state.pop("sigma_floor", None)
        return False, True
    if submitted:
        store.update_scenario(
            mode="calibrated" if mode == "论文校准" else "live",
            lam=lam, v=v,
            ideal_strategy="criterion" if ideal.startswith("按指标列") else "sentiment",
            prob_completion="interval" if completion == "区间补全" else "normalize",
            direction="attainment" if direction.startswith("达标度") else "shortfall",
            weight_mode=weight_mode,
        )
        st.session_state["dsh.sigma_floor"] = sigma_floor
        st.session_state["dsh.possible_degree"] = "normal" if pd_mode.startswith("正态") else "crisp"
        return True, False
    return False, False


# ---------------------------------------------------------------- 编辑器
def _matrix_editor(sc: Scenario):
    """底部实验性编辑器：w3 / AHP / DEMATEL / PLTS 决策矩阵。"""
    st.markdown("")
    section_header("实验性修改（覆盖当前场景）",
                   subtitle="修改只写入会话 overrides，不影响论文基准；点击「应用修改」后生效")
    tab_w, tab_a, tab_d, tab_p = st.tabs(["一级权重 w", "AHP 判断矩阵", "DEMATEL 关联矩阵", "PLTS 决策矩阵"])

    with tab_w:
        st.caption("式(4.7) 三个一级权重（自动归一化），可用于快速压力测试")
        cols = st.columns(3)
        vals = []
        cur = store.overrides().get("w3", list(sc.weight_overrides_3 or B["combined_weights"]["w"]))
        for i, c in enumerate(cols):
            with c:
                vals.append(st.slider(B["meta"]["first_level"][i], 0.0, 1.0, float(cur[i]), 0.01,
                                      key=f"w3_{i}"))
        st.caption(f"合计 {sum(vals):.3f} → 应用时归一化")
        if st.button("应用一级权重", key="apply_w3"):
            s = sum(vals) or 1.0
            store.set_override("w3", [round(v / s, 6) for v in vals])
            st.success("已写入一级权重覆盖，页面结果已按新权重重算")
            st.rerun()

    with tab_a:
        st.caption("论文表5.7 判断矩阵（3×3，比例标度 1-9）；保存后在线复算将使用它")
        M = pd.DataFrame(np.array(B["ahp"]["matrix"], dtype=float),
                         index=B["meta"]["first_level"], columns=B["meta"]["first_level"])
        edited = st.data_editor(M, use_container_width=True, key="ahp_editor",
                                column_config={c: st.column_config.NumberColumn(c, min_value=1, max_value=9)
                                               for c in M.columns})
        if st.button("应用 AHP 矩阵", key="apply_ahp"):
            arr = edited.to_numpy(dtype=float)
            if np.allclose(arr, arr.T, atol=1e-9) and np.allclose(np.diag(arr), 1.0):
                store.set_override("ahp_matrix", arr.tolist())
                st.success("AHP 矩阵已覆盖（已通过一致性形状校验）")
                st.rerun()
            else:
                st.error("判断矩阵必须对称且对角线为 1")

    with tab_d:
        st.caption("论文表5.9 关联语言矩阵（12×12，取值 l0–l5）；l5 超出正文定义的 L={l0..l4}，按论文原样保留")
        Z = pd.DataFrame(B["dematel"]["Z"], index=B["meta"]["second_level"],
                         columns=B["meta"]["second_level"])
        z_edited = st.data_editor(Z, use_container_width=True, key="dematel_editor")
        if st.button("应用 DEMATEL 矩阵", key="apply_dematel"):
            store.set_override("dematel_z", z_edited.to_numpy().tolist())
            st.success("DEMATEL 关联矩阵已覆盖")
            st.rerun()

    with tab_p:
        st.caption("表5.13 决策矩阵：格式 `l3:0.4,l4:0.6`（术语:概率），Σp<1 会被按补全策略处理")
        codes = B["meta"]["attribute_codes"]
        attrs = B["meta"]["attributes"]
        code_of = dict(zip(attrs, codes))
        rows = {}
        for c in B["meta"]["second_level"]:
            rows[c] = {
                a: ",".join(f"{t}:{p:g}" for t, p in
                            zip(B["plts_decision_matrix"][c][code]["terms"],
                                B["plts_decision_matrix"][c][code]["probs"]))
                for code, a in zip(codes, attrs)}
        P = pd.DataFrame(rows).T.rename_axis("指标")
        p_edited = st.data_editor(P, use_container_width=True, key="plts_editor",
                                  height=340)
        if st.button("应用决策矩阵", key="apply_plts"):
            try:
                cfg = {}
                for c in p_edited.index:
                    cfg[c] = {}
                    for a in attrs:
                        terms, probs = [], []
                        for piece in str(p_edited.loc[c, a]).split(","):
                            t, p = piece.split(":")
                            terms.append(t.strip())
                            probs.append(float(p))
                        cfg[c][code_of[a]] = {"terms": terms, "probs": probs}
                store.set_override("plts_matrix", cfg)
                st.success("决策矩阵已覆盖，结果已重算")
                st.rerun()
            except Exception as exc:
                st.error(f"解析失败：{exc}（应形如 l3:0.4,l4:0.6）")

    if store.overrides():
        st.warning(f"当前存在 {len(store.overrides())} 项覆盖：{', '.join(store.overrides())}")
        if st.button("清除全部覆盖", key="clear_ov"):
            store.clear_overrides()
            st.rerun()


# ---------------------------------------------------------------- 中间矩阵
def _intermediate_tabs(res, MODE: str):
    st.markdown("")
    section_header("可解释中间量", subtitle="从 PLTS 期望得分到式(4.13) 总体优势度的完整链路")
    t1, t2, t3, t4 = st.tabs(["① PLTS 决策矩阵", "② 规范化与效用", "③ 权重体系", "④ 可能度矩阵"])

    with t1:
        raw = []
        for a, code in zip(B["meta"]["attributes"], B["meta"]["attribute_codes"]):
            row = {}
            for c in B["meta"]["second_level"]:
                cell = B["plts_decision_matrix"][c][code]
                row[c] = ",".join(f"{t}:{p:g}" for t, p in zip(cell["terms"], cell["probs"]))
            raw.append({"属性": a, **row})
        st.dataframe(pd.DataFrame(raw), use_container_width=True, hide_index=True)
        st.caption(f"期望得分 F（6×12）：区间补全下有 {int(res.prob_sum_deviation)} 个格子 Σp<1")
        matrix_heatmap(res.F_point, B["meta"]["attributes"], B["meta"]["second_level"],
                       mode=MODE, title="PLTS 期望得分 F", colorscale="Viridis")

    with t2:
        matrix_heatmap(res.F_norm, B["meta"]["attributes"], B["meta"]["second_level"],
                       mode=MODE, title="式(4.8) 规范化 F̄（列单位范数）", colorscale="Cividis")
        st.caption("效用矩阵 A = (F̄ − f⁻)/(f* − f⁻)，达标度口径下越大越优")
        matrix_heatmap(res.attainment, B["meta"]["attributes"], B["meta"]["second_level"],
                       mode=MODE, title="效用矩阵 A", colorscale="RdYlGn")
        ide = pd.DataFrame({"理想解 f*": np.round(res.f_star, 4),
                            "负理想 f⁻": np.round(res.f_minus, 4)},
                           index=(B["meta"]["attributes"] if res.diagnostics["ideal_scope"] == "attribute"
                                  else B["meta"]["second_level"]))
        st.dataframe(ide, use_container_width=True)

    with t3:
        w3 = pd.DataFrame({
            "当前 w3": np.round(res.w3, 4),
            "论文表5.12": np.round(B["combined_weights"]["w"], 4),
        }, index=B["meta"]["first_level"])
        w3["Δ"] = (w3["当前 w3"] - w3["论文表5.12"]).round(4)
        st.dataframe(w3, use_container_width=True)
        st.caption("式(4.7) w = λ·wD + (1−λ)·wA；二级权重按所选方式展开到 12 个指标")
        w12 = pd.DataFrame({"权重": np.round(res.w12, 4),
                            "指标": [B["meta"]["second_level_names"][c] for c in B["meta"]["second_level"]],
                            "所属": [B["meta"]["group_of"][c] for c in B["meta"]["second_level"]]})
        rank_bars(w12.assign(口径="当前").rename(columns={"权重": "P(x)", "指标": "属性"}),
                  x="P(x)", y="属性", color=None, mode=MODE, height=360,
                  title="二级权重 w12（12 个指标）")

    with t4:
        P = pd.DataFrame(np.round(res.P, 4), index=B["meta"]["attributes"],
                         columns=B["meta"]["attributes"])
        st.markdown("**式(4.12) 折衷可能度矩阵 P(xᵢ ≥ xⱼ)**")
        st.dataframe(P.style.background_gradient(cmap="Blues", axis=None),
                     use_container_width=True)
        st.markdown("**式(4.13) 总体优势度 P(xᵢ) = Σⱼ P**")
        px = pd.DataFrame({"P(xᵢ)": np.round(res.Pxi, 4),
                           "复算排名": [res.ranking.index(a) + 1 for a in B["meta"]["attributes"]],
                           "论文 P(x)": np.round(B["vikor"]["Pxi"], 4),
                           "论文排名": [list(B["vikor"]["ranking"]).index(a) + 1
                                        for a in B["meta"]["attributes"]]},
                          index=B["meta"]["attributes"])
        px["ΔP(x)"] = (px["P(xᵢ)"] - px["论文 P(x)"]).round(4)
        st.dataframe(px, use_container_width=True)
        st.caption("σ 下限决定了可能度矩阵的平滑度；区间越宽，排序越接近软比较")


# ---------------------------------------------------------------- 敏感性
def _sensitivity_tabs(sc: Scenario, MODE: str):
    section_header("敏感性与稳健性", subtitle="λ（权重融合）与 v（决策系数）对排序的影响")
    sw = sweep_lambda(sc)
    sv = sweep_v(sc)
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        fig = go.Figure()
        colors = theme.attribute_colors()
        for a, vals in sw["Pxi"].items():
            fig.add_trace(go.Scatter(x=sw["lambdas"], y=vals, mode="lines+markers",
                                     name=a, line=dict(color=colors.get(a, "#64748B"), width=2.2)))
        fig.update_layout(xaxis_title="λ", yaxis_title="P(xᵢ)",
                          legend_title="属性")
        st.plotly_chart(theme.finish(fig, MODE, height=360), use_container_width=True)
        st.caption("P(xᵢ) 随 λ 的变化；折线越平，说明结论对权重来源越不敏感")
        fig2 = go.Figure()
        for a, vals in sv["Pxi"].items():
            fig2.add_trace(go.Scatter(x=sv["vs"], y=vals, mode="lines+markers",
                                      name=a, line=dict(color=colors.get(a, "#64748B"), width=2.2)))
        fig2.update_layout(xaxis_title="v（群体效用权重）", yaxis_title="P(xᵢ)", legend_title="属性")
        st.plotly_chart(theme.finish(fig2, MODE, height=320), use_container_width=True)
    with right:
        st.markdown("**λ 扫描下的排序**")
        rows = []
        for i, lam in enumerate(sw["lambdas"]):
            rows.append({"λ": lam, "排序": " > ".join(sw["orders"][i]),
                         "Kendall τ": round(sw["stability"][i], 3),
                         "稳定": "✓" if sw["orders"][i] == sw["base_order"] else "—"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("**论文表5.18 与复算对照**")
        st.dataframe(pd.DataFrame(compare_to_paper_sensitivity(sw)),
                     use_container_width=True, hide_index=True)
        st.markdown("**v 扫描**")
        st.dataframe(pd.DataFrame({"v": sv["vs"],
                                   "排序": [" > ".join(o) for o in sv["orders"]],
                                   "τ": [round(x, 3) for x in sv["stability"]]}),
                     use_container_width=True, hide_index=True)


# ---------------------------------------------------------------- 结果面板
def _result_rail(res, ref, MODE: str):
    st.markdown("**当前排序（P(xᵢ) 降序）**")
    lo, hi = float(res.Pxi.min()), float(res.Pxi.max())
    span = (hi - lo) or 1.0
    for i, a in enumerate(res.ranking):
        k = res.ranking.index(a)
        val = float(res.Pxi[B["meta"]["attributes"].index(a)])
        pct = 8 + 92 * (val - lo) / span
        color = theme.attribute_colors().get(a, "#4F46E5")
        st.markdown(
            f'<div style="margin:6px 0 10px;">'
            f'<div style="display:flex;justify-content:space-between;font-size:12.5px;">'
            f'<span style="font-weight:700;">{k+1}. {a}</span>'
            f'<span style="color:var(--muted);">{val:.4f}</span></div>'
            f'<div style="height:7px;background:var(--panel-alt);border-radius:6px;overflow:hidden;">'
            f'<div style="height:100%;width:{pct:.1f}%;background:{color};border-radius:6px;"></div>'
            f'</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown("**Q 与区间 Q′**")
    q = pd.DataFrame({"Q": np.round(res.Q, 4),
                      "Q′": np.round(res.Q_prime, 4),
                      "σ": np.round(res.sigma, 4)},
                     index=B["meta"]["attributes"]).loc[res.ranking]
    st.dataframe(q, use_container_width=True)

    st.markdown("")
    same = res.ranking == list(ref.ranking)
    st.markdown(
        f'<span class="ds-chip chip-{"ok" if same else "warn"}">'
        f'{"与论文排序一致" if same else "与论文排序不一致"}</span>',
        unsafe_allow_html=True)
    st.caption(f"Kendall τ = {kendall_tau(res.ranking, list(ref.ranking)):.3f} | "
               f"Top1 复算 {res.ranking[0]} · 论文 {ref.ranking[0]}")


# ---------------------------------------------------------------- 主入口
def render():
    MODE = store.theme()
    sc = store.scenario()
    ref = paper_reference()

    _param_panel(sc)
    sc = store.scenario()
    res = store.compute_result(
        sc,
        sigma_floor=float(st.session_state.get("dsh.sigma_floor", 0.06)),
        possible_degree=st.session_state.get("dsh.possible_degree", "normal"))

    center_left, center, center_right = st.columns([0.62, 3.0, 0.78], gap="medium")
    with center_left:
        st.markdown("")
        _result_rail(res, ref, MODE)
    with center:
        section_header("PLTS-VIKOR 模拟器",
                       subtitle="论文式(4.8)–(4.13) 全链路可调 · 论文基准始终并列展示",
                       tag="论文校准" if sc.mode == "calibrated" else "在线复算")
        top1_match = res.ranking[0] == ref.ranking[0]
        kpi_row([
            {"label": "复算 Top1", "value": res.ranking[0],
             "delta": f"论文：{ref.ranking[0]}", "tone": "up" if top1_match else "down",
             "hint": f"P(xᵢ)={res.Pxi[B['meta']['attributes'].index(res.ranking[0])]:.4f}"},
            {"label": "排序一致性", "value": "一致" if res.ranking == list(ref.ranking) else "部分一致",
             "delta": f"Kendall τ={kendall_tau(res.ranking, list(ref.ranking)):.3f}",
             "tone": "up" if res.ranking == list(ref.ranking) else "flat",
             "hint": "论文基准表5.17"},
            {"label": "决策区分度 ΔP", "value": f"{float(res.Pxi.max()-res.Pxi.min()):.3f}",
             "delta": f"论文 ΔP={float(np.ptp(B['vikor']['Pxi'])):.3f}",
             "tone": "flat", "hint": "复算值区间宽度"},
            {"label": "概率不全格子", "value": f"{int(res.prob_sum_deviation)}/72",
             "delta": f"补全方式：{sc.prob_completion}", "tone": "flat",
             "hint": "表5.13 原文如此"},
        ])

        callout(
            f"当前口径：{'论文校准' if sc.mode == 'calibrated' else '在线复算'} · "
            f"λ={sc.lam:g} · v={sc.v:g} · 理想解={'按指标列' if sc.ideal_strategy=='criterion' else '按属性情感值'} · "
            f"方向={'达标度' if sc.direction=='attainment' else '差值度'} · "
            f"二级权重={MODEL_LABEL[sc.weight_mode]}")

        tab_r, tab_m, tab_s, tab_a = st.tabs(["排序与结果", "中间矩阵", "敏感性", "审计"])
        with tab_r:
            cmp = pd.concat([
                pd.DataFrame({"属性": ref.ranking, "P(x)": np.round(ref.Pxi, 4), "口径": "论文基准"}),
                pd.DataFrame({"属性": res.ranking,
                              "P(x)": np.round([res.Pxi[B["meta"]["attributes"].index(a)] for a in res.ranking], 4),
                              "口径": "当前场景"}),
            ], ignore_index=True)
            st.plotly_chart(rank_bars(cmp, x="P(x)", y="属性", color="口径", mode=MODE, height=330),
                            use_container_width=True)
            tbl = pd.DataFrame({
                "S 群体效用": np.round(res.S, 4), "R 个体遗憾": np.round(res.R, 4),
                "Q 折衷值": np.round(res.Q, 4), "Q′ 区间均值": np.round(res.Q_prime, 4),
                "P(xᵢ)": np.round(res.Pxi, 4),
                "论文 P(xᵢ)": np.round(B["vikor"]["Pxi"], 4),
                "Δ": np.round(res.Pxi - np.array(B["vikor"]["Pxi"]), 4),
                "复算排名": [res.ranking.index(a) + 1 for a in B["meta"]["attributes"]],
                "论文排名": [list(B["vikor"]["ranking"]).index(a) + 1 for a in B["meta"]["attributes"]],
            }, index=B["meta"]["attributes"])
            st.dataframe(tbl, use_container_width=True, hide_index=True)
            st.plotly_chart(sr_scatter(res, MODE), use_container_width=True)
            st.caption("气泡大小 = Q；点越靠右上，群体效用与个体遗憾同时偏高")

        with tab_m:
            _intermediate_tabs(res, MODE)
        with tab_s:
            _sensitivity_tabs(sc, MODE)
        with tab_a:
            findings = store.get_audit()
            fail = [f for f in findings if f.status == "fail"]
            warn = [f for f in findings if f.status == "warn"]
            kpi_row([{"label": "审计通过", "value": str(sum(1 for f in findings if f.status == "pass")), "tone": "up"},
                     {"label": "存在偏差", "value": str(len(fail)), "tone": "down"},
                     {"label": "待确认", "value": str(len(warn)), "tone": "flat"}])
            for f in findings:
                with st.expander(f"{f.title}（{f.severity}）"):
                    st.write(f.detail)
                    st.json(f.numbers)
            st.caption("完整审计与论文口径差异说明见「洞察与报告」页")

        st.markdown("")
        _matrix_editor(sc)

    with center_right:
        st.markdown("**口径摘要**")
        st.dataframe(pd.DataFrame({
            "参数": ["mode", "λ", "v", "理想解", "补全", "方向", "权重展开"],
            "取值": [sc.mode, f"{sc.lam:g}", f"{sc.v:g}", sc.ideal_strategy,
                     sc.prob_completion, sc.direction, sc.weight_mode],
        }), use_container_width=True, hide_index=True)
        ov = store.overrides()
        if ov:
            status_chip(f"{len(ov)} 项矩阵覆盖生效", tone="warn")
        else:
            status_chip("使用论文基准输入", tone="ok")
        with st.expander("实时数据对照"):
            ovw = attr_overview(store.filters())
            if not ovw.empty:
                st.dataframe(ovw[["属性", "评论数", "情感均值", "论文基准情感", "Δvs论文"]],
                             use_container_width=True, hide_index=True)
            st.caption("平台实测情感与论文问卷基准的差异，用于校验决策输入")
