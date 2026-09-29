"""智评车行 · 企业级决策看板主入口

架构（重构后）：
    app.py           路由 + 主题注入 + 全局筛选（无业务逻辑）
    state/store.py   会话状态唯一入口
    ui/layout.py     版式系统（具名栅格 / 统一面板 / 高度档位）
    ui/theme.py      设计令牌（config/theme.yaml）+ 明暗同步
    ui/components/   可复用组件（徽章、KPI、图表）
    ui/views/        六个视图（互不 import）
    services/        数据访问 / 聚合 / 报告
    core/algorithm/  纯算法层（零 Streamlit 依赖）

运行： streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

from state import store
from ui import theme

st.set_page_config(
    page_title="智评车行 · 决策看板",
    layout="wide",
    page_icon="🚗",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ 初始化
store.init()
MODE = store.theme()
theme.inject(MODE)   # 内部会同步 Streamlit 内建主题（明暗一致）

NAV = ["产品导览", "决策总览", "属性情感分析", "PLTS-VIKOR 模拟器", "洞察与报告", "数据管理"]
ICONS = ["compass", "house", "diagram-3", "sliders", "file-earmark-bar-graph", "database"]
SECTION_OF = {
    "产品导览": "开始", "决策总览": "分析", "属性情感分析": "分析",
    "PLTS-VIKOR 模拟器": "分析", "洞察与报告": "分析", "数据管理": "数据",
}


@st.cache_data(show_spinner=False, ttl=300)
def _dim_options() -> dict:
    from services.data_store import PATHS
    if not PATHS["raw"].exists():
        return {"brand": [], "model": [], "date": ["—", "—"], "brand_models": {}}
    df = pd.read_csv(PATHS["raw"], usecols=["brand", "model", "comment_date"])
    return {
        "brand": sorted(df["brand"].dropna().unique().tolist()),
        "model": sorted(df["model"].dropna().unique().tolist()),
        "date": [str(df["comment_date"].min()), str(df["comment_date"].max())],
        "brand_models": {b: sorted(sub["model"].dropna().unique().tolist())
                         for b, sub in df.groupby("brand")},
    }


dims = _dim_options()

# ------------------------------------------------------------------ 侧栏
with st.sidebar:
    st.markdown(
        '<div style="padding:4px 4px 12px;">'
        '<div style="font-size:18px;font-weight:800;letter-spacing:.02em;">'
        '<span class="ds-grad-text">智评车行</span> · 决策看板</div>'
        '<div style="font-size:11.5px;color:var(--muted);margin-top:3px;">'
        '新能源汽车产品改进多属性决策平台</div></div>',
        unsafe_allow_html=True)

    view = option_menu(
        "工作台", NAV, icons=ICONS, menu_icon="grid-1x2-fill",
        default_index=NAV.index(store.view()) if store.view() in NAV else 0,
        styles={
            "container": {"padding": "0", "background": "transparent"},
            "nav-link": {"font-size": "13.5px", "font-weight": "600",
                         "margin": "2px 0", "border-radius": "10px"},
            "nav-link-selected": {"background": "linear-gradient(90deg,#4F46E5,#7C3AED)",
                                  "color": "#fff"},
        },
    )
    if view:
        store.set_view(view)

    st.markdown("---")
    st.markdown('<div class="ds-chip chip-brand">全局筛选</div>', unsafe_allow_html=True)
    st.caption("作用于所有分析视图，需点击「应用」生效")
    f = store.filters()
    brand = st.multiselect("品牌", dims["brand"], default=f.brand or [],
                           placeholder="全部品牌")
    model_opts = (sorted({m for b in brand for m in dims["brand_models"].get(b, [])})
                  if brand else dims["model"])
    model = st.multiselect("车型", model_opts,
                           default=[m for m in (f.model or []) if m in model_opts],
                           placeholder="全部车型")
    picked: list[str] | None = None
    d0 = pd.to_datetime(dims["date"][0], errors="coerce")
    d1 = pd.to_datetime(dims["date"][1], errors="coerce")
    if pd.notna(d0) and pd.notna(d1):
        sel = st.date_input("评论时间", value=(d0.date(), d1.date()),
                            min_value=d0.date(), max_value=d1.date())
        if isinstance(sel, tuple) and len(sel) == 2:
            picked = [str(sel[0]), str(sel[1])]
            if picked == [str(d0.date()), str(d1.date())]:
                picked = None
    polarity = st.radio("情感倾向", ["全部", "仅正面", "仅负面"],
                        index={"全部": 0, "仅正面": 1, "仅负面": 2}
                        .get({"positive": "仅正面", "negative": "仅负面"}.get(f.polarity, "全部"), 0),
                        horizontal=True)
    st.caption(f"数据范围 {dims['date'][0]} ~ {dims['date'][1]}")
    c1, c2 = st.columns(2, gap="small")
    with c1:
        if st.button("应用", width="stretch", type="primary"):
            store.update_filters(
                brand=brand or None, model=model or None, date=picked,
                polarity=None if polarity == "全部"
                else ("positive" if polarity == "仅正面" else "negative"))
            st.rerun()
    with c2:
        if st.button("清空", width="stretch"):
            store.clear_filters()
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="ds-chip chip-brand">视图与场景</div>', unsafe_allow_html=True)
    t1, t2 = st.columns(2, gap="small")
    with t1:
        if st.button(("🌙 暗色" if MODE == "light" else "☀️ 亮色"), width="stretch"):
            store.toggle_theme()
            st.rerun()
    with t2:
        if st.button("↺ 重置场景", width="stretch"):
            store.reset_scenario()
            st.rerun()

    # 数据新鲜度
    from services.data_store import store as ds_store
    fresh = ds_store().freshness()
    st.markdown(
        f'<div style="font-size:11.5px;color:var(--muted);margin-top:12px;line-height:1.8;">'
        f'数据流水线 <b style="color:{"#34D399" if fresh["all_ready"] else "#FBBF24"}">'
        f'{fresh["ready"]}/{fresh["total"]}</b> 就绪 · 更新于 {fresh["age"]}<br>'
        f'口径 <b>{"论文校准" if store.scenario().mode == "calibrated" else "在线复算"}'
        f'</b> · λ={store.scenario().lam:g} · v={store.scenario().v:g}</div>',
        unsafe_allow_html=True)

# ------------------------------------------------------------------ 顶栏（单一状态条）
sc = store.scenario()
findings = store.get_audit()
from ui.components.indicators import algorithm_badges          # noqa: E402
from services.data_store import store as ds_store              # noqa: E402
fr = ds_store().freshness()
from services.features import kpi_summary                      # noqa: E402
k = kpi_summary(store.filters())
badges = (
    algorithm_badges(sc, findings)
    + f'<span class="ds-chip chip-muted">评论库 {fr["ready"]}/{fr["total"]} 阶段</span>'
    + f'<span class="ds-chip chip-muted">筛选后 {k["n_kept"]:,} 条 · '
      f'情感均值 {k["mean_sentiment"]:+.3f}</span>'
)
st.markdown(
    f'<div class="ds-topbar">'
    f'<div><div class="tb-title">智评车行 · 决策看板</div>'
    f'<div class="tb-sub">{SECTION_OF.get(store.view(), "")} · 论文校准与在线复算双口径 · 全链路可审计'
    f'</div></div><div class="tb-chips">{badges}</div></div>',
    unsafe_allow_html=True)

# ------------------------------------------------------------------ 路由
def _render():
    name = store.view()
    if name == "产品导览":
        from ui.views import onboarding
        onboarding.render()
    elif name == "决策总览":
        from ui.views import executive
        executive.render()
    elif name == "属性情感分析":
        from ui.views import sentiment_explorer
        sentiment_explorer.render()
    elif name == "PLTS-VIKOR 模拟器":
        from ui.views import simulator
        simulator.render()
    elif name == "洞察与报告":
        from ui.views import insights_report
        insights_report.render()
    elif name == "数据管理":
        from ui.views import data_manager
        data_manager.render()
    else:
        st.error(f"未知视图：{name}")


st.session_state.pop("dsh.render_error", None)
st.session_state.pop("dsh.render_error_tb", None)
try:
    _render()
except Exception as exc:  # 视图失败不拖垮整个应用
    import traceback
    tb = traceback.format_exc()
    st.session_state["dsh.render_error"] = f"{type(exc).__name__}: {exc}"
    st.session_state["dsh.render_error_tb"] = tb
    st.error(f"视图渲染异常：{type(exc).__name__}: {exc}")
    with st.expander("查看错误详情", expanded=False):
        st.code(tb)
    st.info("页面加载出错，已降级。可切换其他视图或重置场景后重试。")

st.markdown("---")
st.caption("智评车行 · 融合情感分析与 PLTS-VIKOR 的新能源汽车产品改进多属性决策 | "
           "论文校准与在线复算双口径 · 全链路可审计 | 论文出处：上海理工大学硕士学位论文")
