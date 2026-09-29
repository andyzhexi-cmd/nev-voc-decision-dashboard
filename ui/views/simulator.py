"""PLTS-VIKOR 交互式模拟器视图

版式（ui.layout 具名栅格 console_canvas_result）：
  左栏  计算参数 · 口径摘要   —— 控制台
  中栏  结果速览 + 四个 Tab + 实验性编辑器 —— 画布
  右栏  排序结果 · Q/Q′ · 一致性 · 实时数据对照 —— 结论栏

所有参数写入 state.store 的 Scenario；矩阵/权重的实验性修改写入 overrides，
应用后由 `store.compute_result()` 统一执行。论文基准始终并列展示，不被覆盖。
"""
from __future__ import annotations

import copy
import json
import re
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.algorithm.models import Scenario, baselines
from core.algorithm.plts import build_decision_matrix
from core.algorithm.sensitivity import compare_to_paper_sensitivity, kendall_tau, sweep_lambda, sweep_v
from core.algorithm.validate import (ahp_diagnostics, consistent_matrix_from_weights,
                                     dematel_diagnostics, matrix_diff, mirror_upper_tri,
                                     plts_diagnostics)
from core.algorithm.vikor import paper_reference
from services.features import attr_overview
from state import store
from ui import layout, theme
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
    """左栏控制台（计算参数面板），返回 (已应用, 已重置)。"""
    with layout.panel("计算参数",
                      "式(4.7) 权重融合与 VIKOR 决策参数 · 应用后写入会话场景，论文基准不被覆盖",
                      footer=layout.note("仅影响当前会话；「恢复论文基准场景」可一键复位")):
        with st.form("sim_params"):
            mode = st.radio("计算口径", ["论文校准", "在线复算"],
                            index=0 if sc.mode == "calibrated" else 1,
                            help="论文校准：权重取自论文表5.8/5.11（编辑的 AHP/DEMATEL 矩阵只保存不参与）；"
                                 "在线复算：由下方编辑的矩阵实时计算权重")
            lam = st.slider("λ · DEMATEL 一级权重占比", 0.0, 1.0, float(sc.lam), 0.05,
                            help="式(4.7)：w = λ·wD + (1−λ)·wA",
                            key=f"sim_lam_{sc.lam:g}")
            v = st.slider("v · 决策系数", 0.0, 1.0, float(sc.v), 0.05,
                          help="v→1 更看重群体效用 S；v→0 更看重个体遗憾 R",
                          key=f"sim_v_{sc.v:g}")
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
            submitted = st.form_submit_button("应用参数", width="stretch", type="primary")
        reset = st.button("↺ 恢复论文基准场景", width="stretch")
    if reset:
        store.reset_scenario()
        st.session_state.pop("sigma_floor", None)
        return False, True
    if submitted:
        store.update_scenario(
            mode="calibrated" if mode == "论文校准" else "live",
            # 论文校准沿用论文表5.8/5.11 权重；只有在线复算口径才让 AHP/DEMATEL 矩阵参与权重
            use_paper_weights=(mode == "论文校准"),
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


# ---------------------------------------------------------------- 编辑器基建
#: 可导入/可导出的覆盖键（JSON 交换的白名单）
MX_KEYS = ("w3", "ahp_matrix", "dematel_z", "plts_matrix")
MX_HIST = "mx_hist_stack"        # 撤销栈：写入覆盖前的 overrides 快照
MX_FUT = "mx_hist_redo"          # 重做栈
MX_HIST_CAP = 10                 # 会话内最多保留 10 步


def _flash(msg: str) -> None:
    """跨 st.rerun 传递一条提示（rerun 会丢弃本轮的 st.success）。"""
    st.session_state["mx_flash"] = msg


def _pop_flash() -> None:
    msg = st.session_state.pop("mx_flash", None)
    if msg:
        st.success(msg)


def _seed_widget(key: str, base) -> None:
    """基准（论文原值或已应用覆盖）变化时，在控件实例化前把 widget 重置为基准值。

    必须在 widget 创建之前调用：Streamlit 允许在实例化前写 key，之后写会抛异常。
    """
    marker = f"mx_base::{key}"
    if st.session_state.get(marker) != base:
        st.session_state[marker] = base
        st.session_state[key] = base


def _seed_editor(key: str, base_sig) -> None:
    """data_editor 的已编辑状态存在它自己的 key 上；基准变化时丢弃旧状态回基准。"""
    marker = f"mx_base::{key}"
    if st.session_state.get(marker) != base_sig:
        st.session_state[marker] = base_sig
        st.session_state.pop(key, None)


def _snapshot() -> dict:
    return copy.deepcopy(store.overrides())


def _record_change() -> None:
    """覆盖写入前调用：压入撤销快照（上限 MX_HIST_CAP 步）并清空重做栈。"""
    snap = _snapshot()
    stack = st.session_state.setdefault(MX_HIST, [])
    if stack and stack[-1] == snap:      # 连续相同快照不重复入栈
        return
    stack.append(snap)
    del stack[:-MX_HIST_CAP]
    st.session_state.setdefault(MX_FUT, []).clear()


def _restore_overrides(d: dict) -> None:
    store.clear_overrides()
    for k, v in d.items():
        store.set_override(k, v)


def _undo() -> bool:
    stack = st.session_state.setdefault(MX_HIST, [])
    if not stack:
        return False
    st.session_state.setdefault(MX_FUT, []).append(_snapshot())
    _restore_overrides(stack.pop())
    return True


def _redo() -> bool:
    fut = st.session_state.setdefault(MX_FUT, [])
    if not fut:
        return False
    st.session_state.setdefault(MX_HIST, []).append(_snapshot())
    _restore_overrides(fut.pop())
    return True


def _drop_override(key: str) -> None:
    """移除单项覆盖（store 未提供删除接口，就地删同一份会话字典）。"""
    store.overrides().pop(key, None)
    st.session_state["dsh.scenario_dirty"] = True


def _commit(key: str, value, msg: str) -> None:
    """写入一项覆盖：先压撤销快照，再写入、留提示、重跑刷新结果。"""
    _record_change()
    store.set_override(key, value)
    _flash(msg)
    st.rerun()


def _drop_then_flash(keys, msg: str) -> None:
    """撤销若干项覆盖（恢复原值类操作）。"""
    _record_change()
    for k in keys:
        _drop_override(k)
    _flash(msg)
    st.rerun()


def _diff_summary(diff: dict, unit: str = "格") -> str:
    """matrix_diff 结果 → 一行中文摘要（原值 vs 当前）。"""
    if not diff.get("comparable"):
        return f"形状不一致，无法对照：当前 {diff.get('shape_current')} vs 原值 {diff.get('shape_reference')}"
    if diff["n_changed"] == 0:
        return "未修改（与论文原值完全一致）"
    return f"已修改 {diff['n_changed']} {unit} · 最大偏离 Δ={diff['max_abs']:.3f}"


def _delta_toggle(delta: np.ndarray, rows, cols, MODE: str, key: str) -> None:
    """原值 vs 当前的 Δ 热力图开关（每个矩阵一个独立 key）。"""
    if st.checkbox("显示 Δ 热力图", key=key):
        st.plotly_chart(matrix_heatmap(np.asarray(delta, dtype=float),
                                       list(rows), list(cols), mode=MODE,
                                       colorscale="RdBu", height=layout.height("m")),
                        key=f"{key}_chart")


def _dematel_numeric(grid) -> tuple[np.ndarray | None, list[str]]:
    """语言项矩阵（'l0'..'l5' 或 0–5 的数字）→ 数值矩阵；解析失败逐格记入 errors。"""
    errs, rows = [], []
    for i, row in enumerate(grid):
        vals = []
        for j, raw in enumerate(row):
            s = str(raw).strip().lower()
            if re.fullmatch(r"l[0-5]", s):
                vals.append(float(s[1:]))
                continue
            try:
                v = float(s)
            except ValueError:
                errs.append(f"第 {i + 1} 行第 {j + 1} 列「{raw}」不是合法语言项（应为 l0–l5）")
                vals.append(np.nan)
                continue
            if 0.0 <= v <= 5.0 and float(v).is_integer():
                vals.append(v)
            else:
                errs.append(f"第 {i + 1} 行第 {j + 1} 列「{raw}」超出语言标度 0–5")
                vals.append(np.nan)
        rows.append(vals)
    if errs:
        return None, errs[:12]
    return np.asarray(rows, dtype=float), []


def _plts_frame(cfg: dict) -> pd.DataFrame:
    """{指标: {A1: {terms, probs}}} → 编辑器表格（单元格形如 l3:0.4,l4:0.6）。"""
    codes = B["meta"]["attribute_codes"]
    attrs = B["meta"]["attributes"]
    rows = {}
    for c in B["meta"]["second_level"]:
        rows[c] = {a: ",".join(f"{t}:{p:g}" for t, p in zip(cfg[c][code]["terms"],
                                                            cfg[c][code]["probs"]))
                   for code, a in zip(codes, attrs)}
    return pd.DataFrame(rows).T.rename_axis("指标")


#: PLTS 决策矩阵的合法语言项（正文定义 l0..l4；l5 只出现在 DEMATEL 表5.9）
PLTS_EDIT_TERMS = tuple(f"l{i}" for i in range(5))


def _parse_plts_frame(df: pd.DataFrame) -> tuple[dict, list[str]]:
    """编辑器表格 → plts_diagnostics 可用结构；解析问题逐条给中文可执行提示。"""
    codes = B["meta"]["attribute_codes"]
    attrs = B["meta"]["attributes"]
    code_of = dict(zip(attrs, codes))
    cfg, errs = {}, []
    for c in B["meta"]["second_level"]:
        if c not in df.index:
            errs.append(f"缺少指标行 {c}（表格应为 12 行指标 × 6 列属性）")
            continue
        row = {}
        for a in attrs:
            if a not in df.columns:
                errs.append(f"缺少属性列 {a}（表格应为 12 行指标 × 6 列属性）")
                continue
            cell = {"terms": [], "probs": []}
            raw = str(df.loc[c, a]).strip()
            if not raw:
                errs.append(f"{c}/{a} 为空（应形如 l3:0.4,l4:0.6）")
                row[code_of[a]] = cell
                continue
            for piece in raw.split(","):
                piece = piece.strip()
                if not piece:
                    continue
                if ":" not in piece:
                    errs.append(f"{c}/{a}：段「{piece}」缺少冒号（应形如 l3:0.4）")
                    continue
                t, p = piece.split(":", 1)
                term = t.strip()
                if term not in PLTS_EDIT_TERMS:
                    errs.append(f"{c}/{a}：语言项「{term}」越界（PLTS 决策矩阵只有 l0–l4；"
                                f"l5 属 DEMATEL 表5.9，不进 PLTS）")
                    continue
                try:
                    prob = float(p)
                except ValueError:
                    errs.append(f"{c}/{a}：概率「{p.strip()}」不是数字（应形如 l3:0.4）")
                    continue
                cell["terms"].append(term)
                cell["probs"].append(prob)
            row[code_of[a]] = cell
        cfg[c] = row
    return cfg, errs


def _validate_plts_cfg(obj) -> tuple[dict | None, list[str]]:
    """导入用：PLTS 覆盖结构匹配校验（12 指标 × 6 属性码 × terms/probs）。"""
    crits = B["meta"]["second_level"]
    codes = B["meta"]["attribute_codes"]
    if not isinstance(obj, dict):
        return None, [f"plts_matrix 必须是对象（指标 → 属性码 → terms/probs），"
                      f"当前是 {type(obj).__name__}"]
    missing = [c for c in crits if c not in obj]
    extra = [k for k in obj if k not in crits]
    if missing:
        return None, [f"plts_matrix 缺少指标行：{'、'.join(missing)}（需要 12 个：{'、'.join(crits)}）"]
    if extra:
        return None, [f"plts_matrix 含未知指标：{'、'.join(str(k) for k in extra)}"]
    cfg: dict = {}
    for c in crits:
        row = obj[c]
        if not isinstance(row, dict):
            return None, [f"plts_matrix[{c}] 必须是对象，键为属性码 {'/'.join(codes)}"]
        bad = [k for k in row if k not in codes]
        if bad:
            return None, [f"plts_matrix[{c}] 含未知属性码：{'、'.join(str(k) for k in bad)}"
                          f"（应为 {'/'.join(codes)}）"]
        lack = [k for k in codes if k not in row]
        if lack:
            return None, [f"plts_matrix[{c}] 缺少属性码：{'、'.join(lack)}（6 个属性都要有）"]
        cells = {}
        for code in codes:
            cell = row[code]
            if (not isinstance(cell, dict) or not isinstance(cell.get("terms"), list)
                    or not isinstance(cell.get("probs"), list)):
                return None, [f"plts_matrix[{c}][{code}] 必须形如 "
                              f'{{"terms": ["l3", "l4"], "probs": [0.4, 0.6]}}']
            terms = [str(t) for t in cell["terms"]]
            try:
                probs = [float(p) for p in cell["probs"]]
            except (TypeError, ValueError):
                return None, [f"plts_matrix[{c}][{code}] 的 probs 含非数字"]
            if len(terms) != len(probs):
                return None, [f"plts_matrix[{c}][{code}]：terms({len(terms)}) 与 "
                              f"probs({len(probs)}) 长度不一致"]
            bad_terms = [t for t in terms if t not in PLTS_EDIT_TERMS]
            if bad_terms:
                return None, [f"plts_matrix[{c}][{code}] 含越界语言项 "
                              f"{'、'.join(bad_terms)}（PLTS 只有 l0–l4；l5 属 DEMATEL 表5.9）"]
            cells[code] = {"terms": terms, "probs": probs}
        cfg[c] = cells
    dia = plts_diagnostics(cfg)
    if not dia["ok"]:
        return None, [f"plts_matrix：{e}" for e in dia["errors"]]
    return cfg, []


def _validate_overrides(payload: dict) -> tuple[dict, list[str]]:
    """导入用：键白名单 + 形状校验（3 / 3×3 / 12×12 / PLTS 结构匹配）。"""
    clean: dict = {}
    errs: list[str] = []
    unknown = [k for k in payload if k not in MX_KEYS]
    if unknown:
        errs.append(f"未知键：{'、'.join(str(k) for k in unknown)}；只允许 {' / '.join(MX_KEYS)}")

    if "w3" in payload:
        try:
            w = [float(x) for x in payload["w3"]]
        except (TypeError, ValueError):
            w = []
        if len(w) != 3 or any(not np.isfinite(x) or x <= 0 for x in w):
            errs.append("w3 必须是 3 个正数的一维数组，如 [0.475, 0.196, 0.329]")
        else:
            clean["w3"] = w

    if "ahp_matrix" in payload:
        try:
            m = np.array(payload["ahp_matrix"], dtype=float)
        except (TypeError, ValueError):
            m = np.zeros((0, 0))
        if m.shape != (3, 3):
            errs.append(f"ahp_matrix 必须是 3×3 二维数组，当前形状 {list(m.shape)}")
        else:
            d = ahp_diagnostics(m.tolist())
            if d["ok"]:
                clean["ahp_matrix"] = m.tolist()
            else:
                errs.extend(f"ahp_matrix：{e}" for e in d["errors"])

    if "dematel_z" in payload:
        g = payload["dematel_z"]
        shape_ok = (isinstance(g, (list, tuple)) and len(g) == 12
                    and all(isinstance(r, (list, tuple)) and len(r) == 12 for r in g))
        if not shape_ok:
            errs.append("dematel_z 必须是 12×12 二维数组（单元格为 l0–l5，或 0–5 的数字）")
        else:
            num, cell_errs = _dematel_numeric(g)
            if cell_errs:
                errs.extend(f"dematel_z：{e}" for e in cell_errs[:6])
            else:
                d = dematel_diagnostics(num)
                if d["ok"]:
                    clean["dematel_z"] = [[f"l{int(v)}" for v in row] for row in num.tolist()]
                else:
                    errs.extend(f"dematel_z：{e}" for e in d["errors"])

    if "plts_matrix" in payload:
        cfg, plts_errs = _validate_plts_cfg(payload["plts_matrix"])
        if plts_errs:
            errs.extend(plts_errs)
        else:
            clean["plts_matrix"] = cfg
    return clean, errs


def _import_overrides(raw: str) -> None:
    """JSON 导入：解析 → 校验 → 全部通过才写入会话覆盖；失败给中文可执行提示。"""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        st.error(f"JSON 解析失败：{exc.msg}（第 {exc.lineno} 行第 {exc.colno} 列）——"
                 f"请检查逗号、引号是否成对；可先导出一份对照格式")
        return
    if not isinstance(payload, dict):
        st.error(f"顶层必须是对象（{{键: 值}}），当前是 {type(payload).__name__}；"
                 f"允许的键：{' / '.join(MX_KEYS)}")
        return
    clean, errs = _validate_overrides(payload)
    if errs:
        for e in errs:
            st.error(e)
        st.caption("以上任一项修复后再重新导入；校验未通过时不会写入任何覆盖")
        return
    if not clean:
        st.warning(f"JSON 里没有可导入的键（允许：{' / '.join(MX_KEYS)}），未做任何修改")
        return
    _record_change()
    for k, v in clean.items():
        store.set_override(k, v)
    _flash(f"已导入 {len(clean)} 项覆盖：{', '.join(clean)}，页面结果已重算")
    st.rerun()


def _scope_chip(sc: Scenario, key: str) -> bool:
    """矩阵参与计算的口径提示：论文校准下 AHP/DEMATEL 不参与，给出一键切换。"""
    if sc.use_paper_weights:
        status_chip("论文校准口径 · 该矩阵只保存、不参与权重计算", tone="warn")
        if st.button("切换到「在线复算」使矩阵参与计算", key=key, width="stretch"):
            store.update_scenario(mode="live", use_paper_weights=False)
            _flash("已切换到在线复算口径，AHP/DEMATEL 矩阵开始参与权重计算")
            st.rerun()
        return False
    status_chip("在线复算口径 · 该矩阵已参与权重计算", tone="ok")
    return True


# ---------------------------------------------------------------- 编辑器
def _matrix_editor(sc: Scenario, MODE: str):
    """底部实验性编辑器：w3 / AHP / DEMATEL / PLTS 决策矩阵。"""
    section_header("实验性修改（覆盖当前场景）",
                   subtitle="修改只写入会话 overrides，不影响论文基准；点击「应用修改」后生效")
    st.markdown(layout.note(
        "口径：本区块的改动只写入会话 overrides、只影响在线重算；论文基准"
        "（表5.7 / 5.9 / 5.12 / 5.13）在各面板并列展示、原值不变。其中 w3 与 PLTS 决策矩阵在两种计算口径下"
        "都立即生效；AHP / DEMATEL 矩阵在「在线复算」口径下才参与权重（面板内可一键切换）。"),
        unsafe_allow_html=True)
    _pop_flash()

    # —— 预设与撤销（写在标签页之前，方便任何操作后回退）
    with layout.panel(
            "预设与撤销",
            "论文原值 = 清除 w3 与 AHP 覆盖（权重链回表5.7 / 5.12，不动 DEMATEL/PLTS 实验）· "
            "恢复全部实验修改 = 清除四类覆盖 · 撤销/重做只在本次会话内有效（上限 10 步）",
            footer=layout.note(
                "一致性重建：a_ij = w_i/w_j，方根法精确还原论文表5.8 权重 (0.434, 0.187, 0.379)、CR=0")):
        p1, p2, p3 = layout.split("even3")
        with p1:
            if st.button("论文原值", key="mx_preset_paper", width="stretch",
                         help="清除 w3 与 ahp_matrix 覆盖，权重链回到论文表5.7 / 表5.12"):
                _drop_then_flash(["w3", "ahp_matrix"],
                                 "已恢复论文原值：w3 与 AHP 覆盖已清除（表5.7 / 表5.12）")
        with p2:
            if st.button("一致性重建", key="mx_preset_consistent", width="stretch",
                         help="按论文表5.8 权重 (0.434, 0.187, 0.379) 反推 a_ij=w_i/w_j，CR=0"):
                _commit("ahp_matrix", consistent_matrix_from_weights(B["ahp"]["weights"]),
                        "一致性重建：CR=0，方根法精确还原 (0.434, 0.187, 0.379)")
        with p3:
            if st.button("均匀权重", key="mx_preset_uniform", width="stretch",
                         help="w3 = (1/3, 1/3, 1/3)，做消除一级权重差异的压力测试"):
                _commit("w3", [round(1 / 3, 6)] * 3,
                        "均匀权重已写入：w3 = (0.333333, 0.333333, 0.333333)")
        r1, r2, r3 = layout.split("even3")
        hist = st.session_state.get(MX_HIST) or []
        fut = st.session_state.get(MX_FUT) or []
        with r1:
            if st.button("恢复全部实验修改", key="mx_clear_all", width="stretch",
                         disabled=not store.overrides(),
                         help="清除 w3 / ahp_matrix / dematel_z / plts_matrix 四类覆盖，回到论文基准输入"):
                _record_change()
                store.clear_overrides()
                _flash("已恢复全部实验修改：四类覆盖全部清除，回到论文基准输入")
                st.rerun()
        with r2:
            if st.button(f"↩ 撤销（{len(hist)} 步可撤）", key="mx_undo", width="stretch",
                         disabled=not hist):
                _undo()
                _flash("已撤销上一步覆盖修改")
                st.rerun()
        with r3:
            if st.button(f"↪ 重做（{len(fut)} 步可重做）", key="mx_redo", width="stretch",
                         disabled=not fut):
                _redo()
                _flash("已重做上一步覆盖修改")
                st.rerun()

    tab_w, tab_a, tab_d, tab_p = st.tabs(["一级权重 w", "AHP 判断矩阵", "DEMATEL 关联矩阵", "PLTS 决策矩阵"])

    with tab_w:
        names1 = B["meta"]["first_level"]
        paper_w = [float(x) for x in B["combined_weights"]["w"]]
        cur_w = [float(x) for x in
                 (store.overrides().get("w3") or sc.weight_overrides_3 or paper_w)]
        with layout.panel(
                "一级权重 w（式 4.7）",
                "三个一级权重写入会话覆盖 · 计算时按比例归一化到 Σw=1 · "
                "存在 w3 覆盖时优先于 AHP/DEMATEL 矩阵融合",
                footer=layout.note(
                    "w3 覆盖在「论文校准 / 在线复算」两种口径下都立即生效；"
                    "「一键归一化」把当前滑杆值按比例缩放到 Σw=1 并写入覆盖")):
            cols = layout.split("even3")
            vals = []
            for i, c in enumerate(cols):
                with c:
                    _seed_widget(f"mx_w3_{i}", round(cur_w[i], 6))
                    vals.append(st.slider(names1[i], 0.0, 1.0, float(cur_w[i]), 0.01,
                                          key=f"mx_w3_{i}",
                                          help="滑杆为压力测试入口；写入前可一键归一化，"
                                               "计算时无论是否归一都会按比例归一"))
            total = float(sum(vals))
            s1, s2 = layout.split("even")
            with s1:
                if abs(total - 1.0) < 5e-4:
                    status_chip(f"Σw = {total:.3f} · 已归一化", tone="ok")
                else:
                    status_chip(f"Σw = {total:.3f} · 未归一化（计算时按比例自动归一）", tone="warn")
            with s2:
                status_chip("w3 覆盖在两种口径下都立即生效", tone="ok")

            b1, b2 = layout.split("even")
            with b1:
                if st.button("一键归一化（Σ→1 并写入）", key="mx_w3_normalize", width="stretch",
                             help="把三个滑杆值按比例缩放到合计 1 后写入覆盖"):
                    s = total or 1.0
                    norm = [round(v / s, 6) for v in vals]
                    _commit("w3", norm, f"已归一化并写入一级权重：Σw = {sum(norm):.3f}")
            with b2:
                if st.button("应用一级权重（按滑杆原样写入）", key="apply_w3", width="stretch",
                             type="primary",
                             help="原样写入（可 Σw≠1）；下游会按比例归一，数值结果与归一化后一致"):
                    _commit("w3", [round(v, 6) for v in vals],
                            f"一级权重已写入：Σw = {total:.3f}"
                            f"{'（未归一，可用「一键归一化」修正）' if abs(total - 1.0) >= 5e-4 else ''}")

            eff = np.array(store.overrides().get("w3") or sc.weight_overrides_3 or paper_w,
                           dtype=float)
            if eff.sum() > 0:
                eff = eff / eff.sum()
            wtab = pd.DataFrame({"论文表5.12": np.round(paper_w, 4),
                                 "当前生效": np.round(eff, 4),
                                 "Δ": np.round(eff - np.array(paper_w), 4)},
                                index=names1)
            st.dataframe(wtab, width="stretch")
            wd = matrix_diff(eff.reshape(1, 3), np.array(paper_w).reshape(1, 3))
            st.caption("原值 vs 当前：" + _diff_summary(wd, unit="项"))
            _delta_toggle(eff.reshape(1, 3) - np.array(paper_w).reshape(1, 3),
                          ["w"], names1, MODE, "mx_delta_w3")

    with tab_a:
        names = B["meta"]["first_level"]
        paper_M = np.array(B["ahp"]["matrix"], dtype=float)
        ov_M = store.overrides().get("ahp_matrix")
        base_M = np.array(ov_M, dtype=float) if ov_M is not None else paper_M
        if base_M.shape != (3, 3):
            base_M = paper_M                     # 导入校验已保证 3×3，这里仅兜底
        pairs = [(0, 1), (0, 2), (1, 2)]         # 上三角：只让用户改这 3 格
        with layout.panel(
                "AHP 判断矩阵（论文表5.7）",
                "3×3 · 只编辑上三角 3 格 · 对角线锁定为 1（只读） · 下三角按 a_ji = 1/a_ij 自动回填",
                footer=layout.note(
                    "Saaty 比例标度 1–9 · CR ≥ 0.1 属警告级（仍可计算）；"
                    "方阵 / 对角线 / 互反性被破坏才是阻断级，无法应用")):
            draft = np.array(base_M, dtype=float, copy=True)
            np.fill_diagonal(draft, 1.0)
            ucols = layout.split("even3")
            vals = []
            for k, (i, j) in enumerate(pairs):
                with ucols[k]:
                    _seed_widget(f"mx_ahp_{i}{j}", float(base_M[i, j]))
                    vals.append(st.number_input(
                        f"a{i + 1}{j + 1} · {names[i]} ÷ {names[j]}",
                        min_value=0.01, max_value=99.0, step=0.01, format="%.4f",
                        key=f"mx_ahp_{i}{j}",
                        help="只改上三角；下三角自动按 1/a 回填，对角线恒为 1"))
            for (i, j), v in zip(pairs, vals):
                draft[i, j] = float(v)
            draft = np.array(mirror_upper_tri(draft.tolist()), dtype=float)
            dia = ahp_diagnostics(draft)

            # —— 状态条：CR / 结构 / 参与口径（三列 chip）
            s1, s2, s3 = layout.split("even3")
            with s1:
                if dia["cr"] is None:
                    status_chip("CR=— · 结构错误，无法计算", tone="bad")
                else:
                    if dia["cr"] < 1e-9:
                        cr_text, cr_tone = f"CR={dia['cr']:.4f} · 完全一致", "ok"
                    elif dia["consistent"]:
                        cr_text, cr_tone = f"CR={dia['cr']:.4f} · 一致性可接受（<0.1）", "ok"
                    else:
                        cr_text, cr_tone = f"CR={dia['cr']:.4f} · 不一致（≥0.1）", "warn"
                    status_chip(cr_text, tone=cr_tone)
            with s2:
                if dia["ok"]:
                    status_chip(f"结构合法 · λmax={dia['lambda_max']:.4f}", tone="ok")
                else:
                    status_chip(f"{len(dia['errors'])} 项阻断级结构错误", tone="bad")
            with s3:
                _scope_chip(sc, "ahp_go_live")

            # —— 只读全矩阵（对角线 1 锁定显示 + 回填后的下三角）
            st.dataframe(pd.DataFrame(np.round(draft, 4), index=names, columns=names),
                         width="stretch")
            st.caption("上三格外的单元格只读：对角线恒为 1，下三角为 1/a 回填值（显示保留 4 位小数）")

            # —— 诊断：errors 阻断 / warnings 提示 / hints 可点修复
            for e in dia["errors"]:
                st.error(e)
            for w in dia["warnings"]:
                st.warning(w)
            for hi, hint in enumerate(dia["hints"]):
                if "互反" in hint:
                    if st.button(f"🔧 修复提示：{hint}", key=f"ahp_hint_{hi}", width="stretch"):
                        _commit("ahp_matrix", mirror_upper_tri(draft.tolist()),
                                "已同步互反值：对角线 1、a_ji = 1/a_ij")
                elif "反推" in hint:
                    if st.button(f"🔧 修复提示：{hint}", key=f"ahp_hint_{hi}", width="stretch"):
                        _commit("ahp_matrix", consistent_matrix_from_weights(B["ahp"]["weights"]),
                                "已反推一致性矩阵：CR=0，方根法精确还原 (0.434, 0.187, 0.379)")
                else:
                    st.info(hint)

            # —— 快捷操作：论文原值 / 按权重反推 / 同步互反值
            a1, a2, a3 = layout.split("even3")
            with a1:
                if st.button("① 恢复论文表5.7 原值", key="ahp_preset_paper", width="stretch",
                             help="移除 AHP 覆盖，回到论文原判断矩阵与表5.8 权重"):
                    _drop_then_flash(["ahp_matrix"], "AHP 判断矩阵已恢复论文表5.7 原值")
            with a2:
                if st.button("② 按权重反推一致性矩阵", key="ahp_preset_consistent", width="stretch",
                             help="a_ij = w_i/w_j，取论文表5.8 权重 (0.434, 0.187, 0.379)"):
                    _commit("ahp_matrix", consistent_matrix_from_weights(B["ahp"]["weights"]),
                            "已写入一致性矩阵：CR=0，方根法精确还原 (0.434, 0.187, 0.379)")
            with a3:
                if st.button("③ 同步互反值", key="ahp_preset_mirror", width="stretch",
                             help="以当前上三角为准回填下三角 a_ji = 1/a_ij，对角线置 1"):
                    _commit("ahp_matrix", mirror_upper_tri(draft.tolist()),
                            "已按当前上三角同步互反值（对角线 1，a_ji = 1/a_ij）")

            applied = st.button("应用 AHP 矩阵", key="apply_ahp", width="stretch", type="primary",
                                disabled=not dia["ok"])
            if not dia["ok"]:
                st.caption("存在阻断级结构错误（见上方红字），修复后才能应用")
            elif applied:
                _commit("ahp_matrix", draft.tolist(),
                        f"AHP 判断矩阵已写入：CR={dia['cr']:.4f} · "
                        f"{'完全一致' if dia['cr'] < 1e-9 else ('一致性可接受' if dia['consistent'] else '不一致')}"
                        f"{' · 仅在线复算口径参与计算' if sc.use_paper_weights else ''}")
            # —— 原值 vs 当前（已应用覆盖）：一行摘要 + Δ 热力图开关
            st.caption("原值 vs 当前：" + _diff_summary(matrix_diff(base_M, paper_M)))
            _delta_toggle(base_M - paper_M, names, names, MODE, "mx_delta_ahp")

    with tab_d:
        names2 = B["meta"]["second_level"]
        paper_Z = B["dematel"]["Z"]
        ov_Z = store.overrides().get("dematel_z")
        base_Z = ov_Z if ov_Z is not None else paper_Z
        paper_num, _ = _dematel_numeric(paper_Z)
        with layout.panel(
                "DEMATEL 关联矩阵（论文表5.9）",
                "12×12 语言值 l0–l5（数值 0–5） · 对角线必须为 0（自身不影响自身） · "
                "论文表5.9 含 l5，审计项 DEMATEL_L5 如实标注",
                footer=layout.note(
                    "l5 超出正文定义的 L={l0..l4}，按论文原值保留、不静默降级；"
                    "全零矩阵无法计算 T = Z̄(I−Z̄)⁻¹")):
            _seed_editor("dematel_editor", [list(r) for r in base_Z])
            z_edited = st.data_editor(
                pd.DataFrame(base_Z, index=names2, columns=names2),
                width="stretch", key="dematel_editor")
            raw_terms = z_edited.to_numpy().tolist()
            num, cell_errs = _dematel_numeric(raw_terms)
            dia_d = None if cell_errs else dematel_diagnostics(num)
            n_l5 = int((num == 5).sum()) if num is not None else 0
            nonstd = sum(1 for row in raw_terms
                         for v in row if not re.fullmatch(r"l[0-5]", str(v).strip().lower()))

            # —— 状态条：结构 / l5 审计项 / 参与口径
            d1, d2, d3 = layout.split("even3")
            with d1:
                if dia_d is None:
                    status_chip(f"{len(cell_errs)} 格无法解析", tone="bad")
                elif dia_d["ok"]:
                    status_chip(f"{dia_d['n']}×{dia_d['n']} · 对角线 0 · 值域 0–5", tone="ok")
                else:
                    status_chip(f"{len(dia_d['errors'])} 项阻断级错误", tone="bad")
            with d2:
                status_chip(f"含 l5 单元格 {n_l5} 个 · 审计项 DEMATEL_L5",
                            tone="warn" if n_l5 else "ok")
            with d3:
                _scope_chip(sc, "dematel_go_live")

            for e in cell_errs:
                st.error(e)
            if dia_d is not None:
                for e in dia_d["errors"]:
                    st.error(e)
                for h in dia_d["hints"]:
                    st.info(h)
            if nonstd:
                st.caption(f"提示：{nonstd} 格不是标准语言项写法（如数字 3），应用时将统一规范化为 l3")

            ok = dia_d is not None and dia_d["ok"]
            applied = st.button("应用 DEMATEL 矩阵", key="apply_dematel", width="stretch",
                                type="primary", disabled=not ok)
            if not ok:
                st.caption("存在阻断级错误（见上方红字），修复后才能应用")
            elif applied:
                terms = [[f"l{int(v)}" for v in row] for row in num.tolist()]
                _commit("dematel_z", terms,
                        f"DEMATEL 关联矩阵已写入（已规范化为 l0–l5 · l5 单元格 {n_l5} 个）"
                        f"{' · 仅在线复算口径参与计算' if sc.use_paper_weights else ''}")
            # —— 原值 vs 当前（已应用覆盖）：一行摘要 + Δ 热力图开关
            base_num, _ = _dematel_numeric(base_Z)
            cur_num = base_num if base_num is not None else (num if num is not None else paper_num)
            st.caption("原值 vs 当前：" + _diff_summary(matrix_diff(cur_num, paper_num)))
            _delta_toggle(cur_num - paper_num, names2, names2, MODE, "mx_delta_dematel")

    with tab_p:
        base_cfg = store.overrides().get("plts_matrix") or B["plts_decision_matrix"]
        with layout.panel(
                "PLTS 决策矩阵（论文表5.13）",
                "6 属性 × 12 指标 · 单元格 `l3:0.4,l4:0.6`（术语:概率，语言项 l0–l4） · "
                "Σp<1 按左栏所选补全口径处理",
                footer=layout.note(
                    "表5.13 原文有 16 个格子 Σp<1（最小 0.90），属原文事实、只警告不阻断；"
                    "缺行缺列、语言项越界、概率越界、terms 与 probs 不等长才是阻断级")):
            _seed_editor("plts_editor", json.dumps(base_cfg, ensure_ascii=False, sort_keys=True))
            P = _plts_frame(base_cfg)
            p_edited = st.data_editor(P, width="stretch", key="plts_editor",
                                      height=layout.height("m"))
            cfg, perr = _parse_plts_frame(p_edited)
            dia_p = plts_diagnostics(cfg)
            blockers = perr[:8] + dia_p["errors"]

            # —— 聚合状态条
            p1, p2, p3 = layout.split("even3")
            with p1:
                if blockers:
                    status_chip(f"{len(blockers)} 项阻断级错误", tone="bad")
                else:
                    status_chip(f"{dia_p['cells']} 格 · 结构合法", tone="ok")
            with p2:
                if dia_p["cells"]:
                    status_chip(f"{dia_p['cells']} 格 · {dia_p['incomplete_cells']} 格 Σp<1 · "
                                f"最小 {dia_p['min_prob_sum']:.2f}",
                                tone="warn" if dia_p["incomplete_cells"] else "ok")
                else:
                    status_chip("无可解析单元格", tone="bad")
            with p3:
                status_chip("PLTS 矩阵在两种口径下都参与计算", tone="ok")

            for e in blockers:
                st.error(e)
            for w in dia_p["warnings"]:
                st.warning(w)
            if dia_p["bad_cells"]:
                st.caption("问题格明细（最多 20 条）：")
                for bad in dia_p["bad_cells"]:
                    st.caption(f"· {bad}")

            # —— 两种补全口径：调用 core.algorithm.plts 的现成函数做试算对照
            if not blockers:
                try:
                    F_int, _, _, diag_int = build_decision_matrix(cfg, completion="interval")
                    F_nrm, _, _, diag_nrm = build_decision_matrix(cfg, completion="normalize")
                    comp = pd.DataFrame({
                        "补全口径": ["区间补全 interval（余量记入最低/最高术语 → 区间 [E⁻, E⁺]）",
                                     "归一化 normalize（按 Σp 归一 → 点估计）"],
                        "Σp<1 格数": [diag_int["n_incomplete"], diag_nrm["n_incomplete"]],
                        "最小 Σp": [f"{diag_int['min_prob_sum']:.2f}", f"{diag_nrm['min_prob_sum']:.2f}"],
                        "期望矩阵均值": [f"{F_int.mean():.4f}", f"{F_nrm.mean():.4f}"],
                        "两口径最大差": [f"{np.abs(F_int - F_nrm).max():.4f}"] * 2,
                        "当前生效": ["✓" if sc.prob_completion == "interval" else "",
                                     "✓" if sc.prob_completion == "normalize" else ""],
                    })
                    st.dataframe(comp, width="stretch", hide_index=True)
                    st.caption("口径在左栏「计算参数 · PLTS 概率补全」切换；"
                               "两口径只改变 Σp<1 格子的取值，不改原始概率。")
                except Exception as exc:
                    st.error(f"补全口径试算失败：{type(exc).__name__}: {exc}")

            applied = st.button("应用决策矩阵", key="apply_plts", width="stretch",
                                type="primary", disabled=bool(blockers))
            if blockers:
                st.caption("存在阻断级错误（见上方红字），修复后才能应用")
            elif applied:
                _commit("plts_matrix", cfg,
                        f"决策矩阵已写入：{dia_p['cells']} 格 · "
                        f"{dia_p['incomplete_cells']} 格 Σp<1 · 最小 {dia_p['min_prob_sum']:.2f}")
            # —— 原值 vs 当前（已应用覆盖，按归一化口径折算成期望得分）：摘要 + Δ 热力图
            try:
                F_cur, _, _, _ = build_decision_matrix(base_cfg, completion="normalize")
                F_pap, _, _, _ = build_decision_matrix(B["plts_decision_matrix"], completion="normalize")
                st.caption("原值 vs 当前：" + _diff_summary(matrix_diff(F_cur, F_pap)))
                _delta_toggle(F_cur - F_pap, B["meta"]["attributes"], B["meta"]["second_level"],
                              MODE, "mx_delta_plts")
            except Exception as exc:
                st.caption(f"原值 vs 当前：无法对照（{type(exc).__name__}: {exc}）")

    ov = store.overrides()
    with layout.panel(
            "覆盖状态与 JSON 交换",
            "实验性修改只写入会话，不影响 config/baselines.yaml 中的论文基准 · "
            "导出文件仅含用户修改项",
            footer=layout.note("导入先校验键与形状，全部通过才写入会话；"
                               "撤销/重做只在本次会话内有效（最多 10 步）")):
        if ov:
            status_chip(f"{len(ov)} 项覆盖生效：{', '.join(ov)}", tone="warn")
        else:
            status_chip("无覆盖 · 使用论文基准输入", tone="ok")
        # —— 导出：只含用户改过的键
        payload = json.dumps(ov, ensure_ascii=False, indent=2)
        st.download_button(
            f"⬇ 导出当前覆盖 JSON（{len(ov)} 项 · 仅含用户修改项）",
            data=payload,
            file_name=f"plts_overrides_{date.today():%Y%m%d}.json",
            mime="application/json", key="mx_export", width="stretch",
            help="只导出会话里被改过的键；论文基准原值不在该文件内")
        # —— 导入：粘贴 JSON，校验通过才写入
        raw_text = st.text_area(
            "导入：粘贴覆盖 JSON（键 ∈ w3 / ahp_matrix / dematel_z / plts_matrix）",
            key="mx_import_text", height=layout.height("xs"),
            placeholder='{"w3": [0.475, 0.196, 0.329]}',
            help="可先导出一份看格式；键与形状全部校验通过才会写入会话")
        if st.button("校验并导入", key="mx_import_apply", width="stretch", type="primary",
                     disabled=not str(raw_text).strip()):
            _import_overrides(str(raw_text))


# ---------------------------------------------------------------- 中间矩阵
def _intermediate_tabs(res, MODE: str):
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
        with layout.panel("决策矩阵明细（术语 : 概率）",
                          "论文表5.13 原样列示 · 6 属性 × 12 指标 · 单位：语言术语标度 l0–l4"):
            st.dataframe(pd.DataFrame(raw), width="stretch", hide_index=True)
        with layout.panel("PLTS 期望得分 F",
                          f"逐格求期望 E(x)=Σ 标度×概率 · 区间补全下有 "
                          f"{int(res.prob_sum_deviation)} 个格子 Σp<1（表5.13 原文如此）"):
            st.plotly_chart(
                matrix_heatmap(res.F_point, B["meta"]["attributes"], B["meta"]["second_level"],
                               mode=MODE, colorscale="Viridis", height=layout.height("m")),
                key="sim_f_point")

    with t2:
        with layout.panel("式(4.8) 规范化 F̄",
                          "列单位范数归一化 · 消除 12 个指标的量纲差异 · 无量纲 0–1"):
            st.plotly_chart(
                matrix_heatmap(res.F_norm, B["meta"]["attributes"], B["meta"]["second_level"],
                               mode=MODE, colorscale="Cividis", height=layout.height("m")),
                key="sim_f_norm")
        with layout.panel("效用矩阵 A",
                          "A=(F̄−f⁻)/(f*−f⁻) · 达标度口径下越大越优 · 取值裁剪到 0–1"):
            st.plotly_chart(
                matrix_heatmap(res.attainment, B["meta"]["attributes"], B["meta"]["second_level"],
                               mode=MODE, colorscale="RdYlGn", height=layout.height("m")),
                key="sim_attainment")
        ide = pd.DataFrame({"理想解 f*": np.round(res.f_star, 4),
                            "负理想 f⁻": np.round(res.f_minus, 4)},
                           index=(B["meta"]["attributes"] if res.diagnostics["ideal_scope"] == "attribute"
                                  else B["meta"]["second_level"]))
        with layout.panel("理想解与负理想",
                          f"口径：{res.diagnostics['ideal_scope']} 极值 · "
                          f"{'表5.15（按属性情感值）' if res.diagnostics['ideal_scope'] == 'attribute' else '表5.14（按指标列）'}"):
            st.dataframe(ide, width="stretch")

    with t3:
        w3 = pd.DataFrame({
            "当前 w3": np.round(res.w3, 4),
            "论文表5.12": np.round(B["combined_weights"]["w"], 4),
        }, index=B["meta"]["first_level"])
        w3["Δ"] = (w3["当前 w3"] - w3["论文表5.12"]).round(4)
        with layout.panel("一级权重 w3 对照",
                          "式(4.7) w = λ·wD + (1−λ)·wA · 复算值 vs 论文表5.12 · 差值列 Δ"):
            st.dataframe(w3, width="stretch")
        w12 = pd.DataFrame({"权重": np.round(res.w12, 4),
                            "指标": [B["meta"]["second_level_names"][c] for c in B["meta"]["second_level"]],
                            "所属": [B["meta"]["group_of"][c] for c in B["meta"]["second_level"]]})
        with layout.panel("二级权重 w12（12 个指标）",
                          "按所选展开方式从三个一级权重分配 · 权重合计 1 · 条长 = 权重大小"):
            st.plotly_chart(
                rank_bars(w12.assign(口径="当前").rename(columns={"权重": "P(x)", "指标": "属性"}),
                          x="P(x)", y="属性", color=None, mode=MODE,
                          height=layout.height("m")),
                key="sim_w12_bars")

    with t4:
        P = pd.DataFrame(np.round(res.P, 4), index=B["meta"]["attributes"],
                         columns=B["meta"]["attributes"])
        with layout.panel("式(4.12) 折衷可能度矩阵 P(xᵢ ≥ xⱼ)",
                          "正态分布 CDF 可能度 · 取值 0–1 · 行 i 对列 j 表示 xᵢ 优于 xⱼ 的可能度"):
            st.dataframe(P.style.background_gradient(cmap="Blues", axis=None),
                         width="stretch")
        px = pd.DataFrame({"P(xᵢ)": np.round(res.Pxi, 4),
                           "复算排名": [res.ranking.index(a) + 1 for a in B["meta"]["attributes"]],
                           "论文 P(x)": np.round(B["vikor"]["Pxi"], 4),
                           "论文排名": [list(B["vikor"]["ranking"]).index(a) + 1
                                        for a in B["meta"]["attributes"]]},
                          index=B["meta"]["attributes"])
        px["ΔP(x)"] = (px["P(xᵢ)"] - px["论文 P(x)"]).round(4)
        with layout.panel("式(4.13) 总体优势度 P(xᵢ)",
                          "P(xᵢ)=Σⱼ P · 复算 vs 论文表5.17 · 含排名与差值 ΔP",
                          footer=layout.note("σ 下限决定可能度矩阵的平滑度；区间越宽，排序越接近软比较")):
            st.dataframe(px, width="stretch")


# ---------------------------------------------------------------- 敏感性
def _sensitivity_tabs(sc: Scenario, MODE: str):
    section_header("敏感性与稳健性", subtitle="λ（权重融合）与 v（决策系数）对排序的影响")
    sw = sweep_lambda(sc)
    sv = sweep_v(sc)
    left, right = layout.split("main_rail")
    colors = theme.attribute_colors()
    muted = theme.palette(MODE)["muted"]
    with left:
        fig = go.Figure()
        for a, vals in sw["Pxi"].items():
            fig.add_trace(go.Scatter(x=sw["lambdas"], y=vals, mode="lines+markers",
                                     name=a, line=dict(color=colors.get(a, muted), width=2.2)))
        fig.update_layout(xaxis_title="λ", yaxis_title="P(xᵢ)",
                          legend_title="属性")
        with layout.panel("λ 扫描 · P(xᵢ) 轨迹",
                          "纵轴：总体优势度 P(xᵢ)（式4.13）· 折线越平，结论对权重来源越不敏感"):
            st.plotly_chart(theme.finish(fig, MODE, height=layout.height("m")),
                            key="sim_sweep_lam")
        fig2 = go.Figure()
        for a, vals in sv["Pxi"].items():
            fig2.add_trace(go.Scatter(x=sv["vs"], y=vals, mode="lines+markers",
                                      name=a, line=dict(color=colors.get(a, muted), width=2.2)))
        fig2.update_layout(xaxis_title="v（群体效用权重）", yaxis_title="P(xᵢ)", legend_title="属性")
        with layout.panel("v 扫描 · P(xᵢ) 轨迹",
                          "纵轴：总体优势度 P(xᵢ)（式4.13）· 折线越平，结论对决策系数越不敏感"):
            st.plotly_chart(theme.finish(fig2, MODE, height=layout.height("m")),
                            key="sim_sweep_v")
    with right:
        rows = []
        for i, lam in enumerate(sw["lambdas"]):
            rows.append({"λ": lam, "排序": " > ".join(sw["orders"][i]),
                         "Kendall τ": round(sw["stability"][i], 3),
                         "稳定": "✓" if sw["orders"][i] == sw["base_order"] else "—"})
        with layout.panel("λ 扫描 · 排序稳定性",
                          "λ ∈ [0,1] · Kendall τ 以当前基准排序为参照 · ✓ 表示排序完全一致"):
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        with layout.panel("论文表5.18 与复算对照",
                          "同一组 λ 下的排序与 P(x) 逐项对照 · 差异来自论文权重不可复算"):
            st.dataframe(pd.DataFrame(compare_to_paper_sensitivity(sw)),
                         width="stretch", hide_index=True)
        with layout.panel("v 扫描 · 排序稳定性",
                          "v 从 0（个体遗憾）到 1（群体效用）· τ 为与当前基准排序的 Kendall 相关"):
            st.dataframe(pd.DataFrame({"v": sv["vs"],
                                       "排序": [" > ".join(o) for o in sv["orders"]],
                                       "τ": [round(x, 3) for x in sv["stability"]]}),
                         width="stretch", hide_index=True)


# ---------------------------------------------------------------- 结果面板
def _result_rail(res, ref, MODE: str):
    lo, hi = float(res.Pxi.min()), float(res.Pxi.max())
    span = (hi - lo) or 1.0
    with layout.panel("当前排序（P(xᵢ) 降序）",
                      "条长按总体优势度 P(xᵢ) 线性缩放 · 属性色取自设计令牌",
                      footer=layout.note(f"Top1 复算 {res.ranking[0]} · 论文 {ref.ranking[0]}")):
        for i, a in enumerate(res.ranking):
            k = res.ranking.index(a)
            val = float(res.Pxi[B["meta"]["attributes"].index(a)])
            pct = 8 + 92 * (val - lo) / span
            color = theme.attribute_colors().get(a, theme.palette(MODE)["brand"])
            st.markdown(
                f'<div style="margin:6px 0 10px;">'
                f'<div style="display:flex;justify-content:space-between;font-size:12.5px;">'
                f'<span style="font-weight:700;">{k+1}. {a}</span>'
                f'<span style="color:var(--muted);">{val:.4f}</span></div>'
                f'<div style="height:7px;background:var(--panel-alt);border-radius:6px;overflow:hidden;">'
                f'<div style="height:100%;width:{pct:.1f}%;background:{color};border-radius:6px;"></div>'
                f'</div></div>', unsafe_allow_html=True)

    q = pd.DataFrame({"Q": np.round(res.Q, 4),
                      "Q′": np.round(res.Q_prime, 4),
                      "σ": np.round(res.sigma, 4)},
                     index=B["meta"]["attributes"]).loc[res.ranking]
    with layout.panel("Q 与区间 Q′",
                      "式(4.10) 折衷值 Q · 式(4.11) 区间均值 Q′ · σ 为区间标准差（下限 σ₀）"):
        st.dataframe(q, width="stretch")

    same = res.ranking == list(ref.ranking)
    with layout.panel("与论文排序一致性",
                      "参照：论文表5.17 排序 · Kendall τ 衡量两条排序链的相关程度"):
        status_chip("与论文排序一致" if same else "与论文排序不一致",
                    tone="ok" if same else "warn")
        st.caption(f"Kendall τ = {kendall_tau(res.ranking, list(ref.ranking)):.3f} | "
                   f"Top1 复算 {res.ranking[0]} · 论文 {ref.ranking[0]}")


# ---------------------------------------------------------------- 主入口
def render():
    MODE = store.theme()
    sc = store.scenario()
    ref = paper_reference()

    # 三层具名栅格：宽控制台 · 中画布 · 右结论栏
    layout.page_head("PLTS-VIKOR 模拟器",
                     "拖动 λ / v / 权重，实时看排序怎么变；论文基准始终并列对照，不被覆盖")
    rail_l, canvas, rail_r = layout.split("console_canvas_result")

    with rail_l:
        # 参数面板必须先执行：它会写回会话场景，随后才计算结果
        _param_panel(sc)
        sc = store.scenario()
        res = store.compute_result(
            sc,
            sigma_floor=float(st.session_state.get("dsh.sigma_floor", 0.06)),
            possible_degree=st.session_state.get("dsh.possible_degree", "normal"))
        with layout.panel("口径摘要", "当前场景的算法参数 · 与顶栏徽章、审计面板同源"):
            st.dataframe(pd.DataFrame({
                "参数": ["mode", "λ", "v", "理想解", "补全", "方向", "权重展开"],
                "取值": [sc.mode, f"{sc.lam:g}", f"{sc.v:g}", sc.ideal_strategy,
                         sc.prob_completion, sc.direction, sc.weight_mode],
            }), width="stretch", hide_index=True)
            if store.overrides():
                status_chip(f"{len(store.overrides())} 项矩阵覆盖生效", tone="warn")
            else:
                status_chip("使用论文基准输入", tone="ok")

    with canvas:
        section_header("场景结果与可解释链路",
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
            section_header("排序与结果对照",
                           subtitle="复算与论文基准并列 · P(xᵢ)/S/R/Q 全量指标")
            cmp = pd.concat([
                pd.DataFrame({"属性": ref.ranking, "P(x)": np.round(ref.Pxi, 4), "口径": "论文基准"}),
                pd.DataFrame({"属性": res.ranking,
                              "P(x)": np.round([res.Pxi[B["meta"]["attributes"].index(a)] for a in res.ranking], 4),
                              "口径": "当前场景"}),
            ], ignore_index=True)
            with layout.panel("双口径 P(xᵢ) 排序对照",
                              "横轴：总体优势度 P(xᵢ)（式4.13）· 分组柱 · 左=论文基准，右=当前场景"):
                st.plotly_chart(rank_bars(cmp, x="P(x)", y="属性", color="口径", mode=MODE,
                                          height=layout.height("m")),
                                key="sim_rank_cmp")
            tbl = pd.DataFrame({
                "S 群体效用": np.round(res.S, 4), "R 个体遗憾": np.round(res.R, 4),
                "Q 折衷值": np.round(res.Q, 4), "Q′ 区间均值": np.round(res.Q_prime, 4),
                "P(xᵢ)": np.round(res.Pxi, 4),
                "论文 P(xᵢ)": np.round(B["vikor"]["Pxi"], 4),
                "Δ": np.round(res.Pxi - np.array(B["vikor"]["Pxi"]), 4),
                "复算排名": [res.ranking.index(a) + 1 for a in B["meta"]["attributes"]],
                "论文排名": [list(B["vikor"]["ranking"]).index(a) + 1 for a in B["meta"]["attributes"]],
            }, index=B["meta"]["attributes"])
            with layout.panel("S / R / Q / P(xᵢ) 指标明细",
                              "式(4.9)–(4.11) 与式(4.13) · 复算 vs 论文基准 · 含差值 Δ 与双排名"):
                st.dataframe(tbl, width="stretch", hide_index=True)
            with layout.panel("S-R 效用散点",
                              "横轴 S 群体效用 · 纵轴 R 个体遗憾 · 气泡大小 = Q · 虚线为均值 S̄/R̄"):
                st.plotly_chart(sr_scatter(res, MODE, height=layout.height("l")),
                                key="sim_sr_scatter")

        with tab_m:
            _intermediate_tabs(res, MODE)
        with tab_s:
            _sensitivity_tabs(sc, MODE)
        with tab_a:
            section_header("一致性审计",
                           subtitle="论文口径与实时复算的偏差逐项列示 · 不掩盖、不静默修正")
            findings = store.get_audit()
            fail = [f for f in findings if f.status == "fail"]
            warn = [f for f in findings if f.status == "warn"]
            kpi_row([{"label": "审计通过", "value": str(sum(1 for f in findings if f.status == "pass")), "tone": "up"},
                     {"label": "存在偏差", "value": str(len(fail)), "tone": "down"},
                     {"label": "待确认", "value": str(len(warn)), "tone": "flat"}])
            for f in findings:
                with layout.panel(f.title, f.detail, tag=f.severity):
                    st.json(f.numbers)
            st.caption("完整审计与论文口径差异说明见「洞察与报告」页")

        _matrix_editor(sc, MODE)

    with rail_r:
        _result_rail(res, ref, MODE)
        with layout.panel("实时数据对照",
                          "平台实测情感 vs 论文问卷基准 · 用于校验决策输入的可信度"):
            ovw = attr_overview(store.filters())
            if not ovw.empty:
                st.dataframe(ovw[["属性", "评论数", "情感均值", "论文基准情感", "Δvs论文"]],
                             width="stretch", hide_index=True)
                st.caption("两列口径不同：实测为 VADER 感知情感，论文为问卷均值")
            else:
                empty_state("当前筛选下没有数据", "请在左侧或顶栏调整筛选条件后重试。")
