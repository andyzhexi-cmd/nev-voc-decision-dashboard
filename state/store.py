"""state.store — 会话状态唯一入口（规范化 session_state + 计算缓存）

所有 View 只允许通过本模块读写运行时状态：
  * theme      'dark' | 'light'
  * view       当前导航
  * filters    全局筛选（品牌/车型/日期/属性/极性）
  * scenario   core.algorithm.Scenario（算法场景，序列化存 session_state）
  * overrides  用户在模拟器中对矩阵/权重的临时修改
"""
from __future__ import annotations

import hashlib

import streamlit as st

from core.algorithm import Scenario, run_vikor
from services.features import Filters

DEFAULTS = {
    "dsh.theme": "dark",
    "dsh.view": "决策总览",
    "dsh.filters": {},
    "dsh.scenario": Scenario().model_dump(),
    "dsh.overrides": {},          # {"ahp_matrix": [...], "dematel_z": [...], "w3": [...]}
    "dsh.scenario_dirty": False,  # 用户是否改过参数（决定是否展示"复算"口径）
    "dsh.uploads": {},            # 上传文件登记
    "dsh.data_version": 0,        # 上传/重跑后 +1，用于失效缓存
}


def init() -> dict:
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v.copy() if isinstance(v, dict) else v
    return st.session_state


# ---------------------------------------------------------------- theme / view
def theme() -> str:
    return st.session_state.get("dsh.theme", "dark")


def set_theme(v: str) -> None:
    st.session_state["dsh.theme"] = "light" if v in ("light", "亮色") else "dark"


def toggle_theme() -> str:
    set_theme("light" if theme() == "dark" else "dark")
    return theme()


def view() -> str:
    return st.session_state.get("dsh.view", "决策总览")


def set_view(v: str) -> None:
    st.session_state["dsh.view"] = v


# ---------------------------------------------------------------- scenario
def scenario() -> Scenario:
    return Scenario(**st.session_state["dsh.scenario"])


def update_scenario(**kw) -> Scenario:
    cur = dict(st.session_state["dsh.scenario"])
    cur.update(kw)
    st.session_state["dsh.scenario"] = cur
    if kw:
        st.session_state["dsh.scenario_dirty"] = True
    return scenario()


def reset_scenario() -> None:
    st.session_state["dsh.scenario"] = DEFAULTS["dsh.scenario"].copy()
    st.session_state["dsh.overrides"] = {}
    st.session_state["dsh.scenario_dirty"] = False


def overrides() -> dict:
    return st.session_state.get("dsh.overrides", {})


def set_override(key: str, value) -> None:
    st.session_state["dsh.overrides"][key] = value
    st.session_state["dsh.scenario_dirty"] = True


def clear_overrides() -> None:
    st.session_state["dsh.overrides"] = {}


# ---------------------------------------------------------------- filters
def filters() -> Filters:
    d = st.session_state.get("dsh.filters", {}) or {}
    return Filters(
        brand=d.get("brand") or None,
        model=d.get("model") or None,
        attrs=d.get("attrs") or None,
        polarity=d.get("polarity") or None,
        date_range=tuple(d["date"]) if d.get("date") else None,
    )


def update_filters(**kw) -> None:
    d = dict(st.session_state.get("dsh.filters", {}) or {})
    d.update(kw)
    st.session_state["dsh.filters"] = d


def clear_filters() -> None:
    st.session_state["dsh.filters"] = {}


def data_version() -> int:
    return int(st.session_state.get("dsh.data_version", 0))


def bump_data_version() -> int:
    st.session_state["dsh.data_version"] = data_version() + 1
    return data_version()


# ---------------------------------------------------------------- 计算入口
def compute_result(sc: Scenario | None = None, **kw):
    """执行 VIKOR（单次 <50ms，无需缓存）；overrides 会自动合并进场景。"""
    import numpy as np
    sc = sc or scenario()
    clean = {k: v for k, v in kw.items() if v is not None}
    ov = overrides()
    if "w3" in ov and "w3" not in clean:
        clean["w3"] = ov["w3"]
    for k in ("ahp_matrix", "dematel_z", "plts_matrix"):
        if clean.get(k) is None and ov.get(k) is not None:
            clean[k] = ov[k]
        if clean.get(k) is not None:
            clean[k] = np.asarray(clean[k]).tolist() if k != "plts_matrix" else clean[k]
    if "w3" in clean:
        sc = sc.model_copy(update={"weight_overrides_3": tuple(clean.pop("w3"))})
    return run_vikor(sc, **clean)


@st.cache_data(show_spinner=False, max_entries=8)
def audit_findings(payload: str, version: int) -> list:
    import json
    from core.algorithm.audit import run_audit
    sc = Scenario(**json.loads(payload))
    return [f.model_dump() for f in run_audit(sc)]


def get_audit() -> list:
    from core.algorithm.models import AuditFinding
    raw = audit_findings(scenario().model_dump_json(), data_version())
    return [AuditFinding(**f) for f in raw]


@st.cache_data(show_spinner=False, ttl=600, max_entries=8)
def features_frame(kind: str, filters_json: str, version: int):
    """按 kind + filters 缓存 services.features 结果（DataFrame / dict）。"""
    import json
    from services import features as F
    d = json.loads(filters_json or "{}")
    f = Filters(**d) if d else F.Filters()
    table = {"overview": lambda: F.attr_overview(f),
             "brand_attr": lambda: F.brand_attr_matrix(f),
             "sankey": lambda: F.sankey_data(f),
             "sunburst": lambda: F.sunburst_data(f),
             "trend": lambda: F.trend_series(f),
             "ranking": lambda: F.model_ranking(f),
             "insight": lambda: F.insight_cards(f),
             "kpi": lambda: F.kpi_summary(f),
             "radar": lambda: F.radar_series(f)}
    fn = table.get(kind)
    if fn is None:
        raise KeyError(kind)
    return fn()


def filters_json(f: Filters | None = None) -> str:
    f = f or filters()
    import json
    return json.dumps({"brand": f.brand, "model": f.model, "attrs": f.attrs,
                       "polarity": f.polarity,
                       "date_range": list(f.date_range) if f.date_range else None,
                       "only_kept": f.only_kept}, ensure_ascii=False, default=str)
