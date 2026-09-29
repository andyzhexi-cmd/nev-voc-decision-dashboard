"""ui.theme — 设计令牌与主题注入（config/theme.yaml → CSS / Plotly / ECharts / Streamlit 内建组件）

职责：
  * 读取设计令牌（颜色、圆角、字体、栅格）
  * 生成明/暗两套 CSS 变量与组件样式，注入 st.markdown
  * 把令牌同步给 Streamlit 内建主题（st.dataframe / st.code / 输入控件等），保证明暗彻底一致
  * 为 Plotly / ECharts 提供与令牌一致的布局工厂

版式节奏（与 ui/layout.py 配套）：
  页头 28px → 区块 22px → 面板 16px → 图表 24/340/…
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
THEME_PATH = ROOT / "config" / "theme.yaml"

LIGHT = {
    "bg": "#F6F7FB", "panel": "#FFFFFF", "panel_alt": "#F1F3FA", "border": "#E3E6F0",
    "text": "#101828", "muted": "#667085", "brand": "#4F46E5", "brand2": "#8B5CF6",
    "grid": "#E7EAF3", "shadow": "0 8px 24px rgba(16,24,40,.06)",
    # 文字色（浅色底需要更深的实色，暗色底需要提亮）
    "brand_fg": "#4338CA", "ok_fg": "#047857", "warn_fg": "#B45309", "bad_fg": "#B91C1C",
    "muted_fg": "#475467",
}
DARK = {
    "bg": "#0B0F1A", "panel": "#141A2A", "panel_alt": "#1B2236", "border": "#26304A",
    "text": "#E8ECF7", "muted": "#98A2B8", "brand": "#6366F1", "brand2": "#8B5CF6",
    "grid": "#22304C", "shadow": "0 10px 30px rgba(0,0,0,.35)",
    "brand_fg": "#A5B4FC", "ok_fg": "#34D399", "warn_fg": "#FBBF24", "bad_fg": "#F87171",
    "muted_fg": "#94A3B8",
}
SENTIMENT = {"positive": "#10B981", "negative": "#EF4444", "neutral": "#94A3B8"}

#: 布局尺寸令牌
LAYOUT = {"max_width": 1440, "sidebar": 300, "gap_block": 14, "header_h": 56}


@lru_cache(maxsize=4)
def tokens() -> dict:
    try:
        with open(THEME_PATH, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


@lru_cache(maxsize=8)
def attribute_colors() -> dict:
    t = tokens()
    base = t.get("chart", {}).get("attribute_colors") or {}
    default = {"外观": "#8B5CF6", "内饰": "#6366F1", "空间": "#3B82F6",
               "续航": "#22D3EE", "性价比": "#34D399", "舒适性": "#FBBF24"}
    out = dict(default)
    out.update(base)
    return out


def palette(mode: str) -> dict:
    return DARK if mode != "light" else LIGHT


def mode_label(mode: str) -> str:
    return "暗色" if mode != "light" else "亮色"


def sync(mode: str) -> None:
    """把令牌写回 Streamlit 内建主题，使 dataframe/code/输入控件与看板同色系。

    必须在每次 rerun 早期调用（theme.inject 内已调用）。
    """
    import streamlit as st
    p = palette(mode)
    opts = {
        "theme.base": "light" if mode == "light" else "dark",
        "theme.primaryColor": p["brand"],
        "theme.backgroundColor": p["bg"],
        "theme.secondaryBackgroundColor": p["panel_alt"],
        "theme.textColor": p["text"],
        "theme.font": "Inter, sans-serif",
    }
    for key, value in opts.items():
        try:
            st._config.set_option(key, value)
        except Exception:  # 内部 API 变化时降级为纯 CSS 主题
            pass


# ------------------------------------------------------------------ CSS
def css(mode: str) -> str:
    p = palette(mode)
    attrs = attribute_colors()
    attr_vars = "\n".join(f"  --attr-{k}: {v};" for k, v in attrs.items())
    struct = _STRUCTURE.format(
        header_h=LAYOUT["header_h"], sidebar=LAYOUT["sidebar"],
        max_width=LAYOUT["max_width"],
        brand_grad1="#818CF8" if mode != "light" else "#6366F1",
        brand_grad2="#22D3EE",
    )
    widgets = _WIDGETS.format()
    chips = _CHIPS.format()
    return f"""
<style id="dsh-theme">
:root {{
{attr_vars}
  --bg: {p['bg']}; --panel: {p['panel']}; --panel-alt: {p['panel_alt']};
  --border: {p['border']}; --text: {p['text']}; --muted: {p['muted']};
  --brand: {p['brand']}; --brand2: {p['brand2']}; --grid: {p['grid']};
  --brand-fg: {p['brand_fg']}; --ok-fg: {p['ok_fg']};
  --warn-fg: {p['warn_fg']}; --bad-fg: {p['bad_fg']}; --muted-fg: {p['muted_fg']};
  --radius: 16px; --radius-sm: 10px; --shadow: {p['shadow']};
  --gap-s: 8px; --gap-m: 16px; --gap-l: 32px;
  --font: "Inter", "PingFang SC", "Microsoft YaHei", -apple-system, sans-serif;
  /* 覆盖 Streamlit 内建主题变量：明暗切换不刷新也完全一致（滑块/单选/复选/进度条跟随） */
  --primary-color: {p['brand']}; --background-color: {p['bg']};
  --secondary-background-color: {p['panel_alt']}; --text-color: {p['text']};
}}
{struct}
{widgets}
{chips}
</style>"""

_STRUCTURE = """
/* ============ 1. 页面骨架：侧栏 / 顶栏 / 主区 ============ */
.stApp {{ background: var(--bg); }}
[data-testid="stHeader"] {{
  height: {header_h}px; background: color-mix(in srgb, var(--bg) 88%, transparent);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid color-mix(in srgb, var(--border) 70%, transparent);
}}
section[data-testid="stSidebar"] {{ width: {sidebar}px; border-right: 1px solid var(--border); }}
section[data-testid="stSidebar"] > div {{
  padding: 1.1rem 1.1rem 2rem; background: linear-gradient(180deg, var(--panel) 0%, var(--panel-alt) 100%);
}}
[data-testid="stMainBlockContainer"] {{
  max-width: {max_width}px; padding: 16px 32px 48px;
}}
/* 间距三档制（参考 Shilp Sutra 三档节奏 8/16/32 与 NHS 卡片规范：卡内 16、卡间 32） */
[data-testid="stVerticalBlock"] > div {{ gap: var(--gap-m); }}
[data-testid="stHorizontalBlock"] > div {{ gap: var(--gap-m); }}

/* ============ 2. 页头（每页唯一） ============ */
.ds-pagehead {{ margin: 2px 0 4px; }}
.ds-pagehead .ph-title {{ font-size: 26px; font-weight: 800; letter-spacing: .01em; color: var(--text); }}
.ds-pagehead .ph-sub {{ font-size: 13px; color: var(--muted); margin-top: 4px; line-height: 1.6; }}
.ds-pagehead .ph-chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
/* 顶部状态条：左侧视图名 + 右侧徽章行（整行一条基线，不再各自漂浮） */
.ds-topbar {{ display: flex; align-items: center; justify-content: space-between;
  gap: 18px; flex-wrap: wrap; padding: 2px 0 12px;
  border-bottom: 1px solid color-mix(in srgb, var(--border) 80%, transparent);
  margin-bottom: 6px; }}
.ds-topbar .tb-title {{ font-size: 23px; font-weight: 800; letter-spacing: .01em; color: var(--text); }}
.ds-topbar .tb-sub {{ font-size: 12px; color: var(--muted); margin-top: 3px; }}
.ds-topbar .tb-chips {{ display: flex; gap: 8px; flex-wrap: wrap;
  justify-content: flex-end; align-items: center; }}

/* ============ 3. 区块标题（固定节奏） ============ */
.ds-section {{ margin: var(--gap-l) 0 0; }}
.ds-section:first-child {{ margin-top: 0; }}
.ds-section h3 {{ font-size: 16.5px; font-weight: 700; color: var(--text); margin: 0 0 3px; }}
.ds-section p {{ font-size: 12.5px; color: var(--muted); margin: 0; line-height: 1.6; }}
.ds-section .rule {{ height: 3px; width: 42px; border-radius: 3px; margin-top: 9px;
  background: linear-gradient(90deg, var(--brand), var(--brand2)); }}

/* ============ 4. 统一面板（卡片） ============ */
[data-testid="stVerticalBlockBorderWrapper"] {{
  background: var(--panel); border-color: var(--border); border-radius: var(--radius);
  box-shadow: var(--shadow);
}}
[data-testid="stVerticalBlockBorderWrapper"] > div {{ padding: 16px; }}
.ds-panelhead {{ padding-bottom: 2px; border-bottom: 1px solid color-mix(in srgb, var(--border) 80%, transparent);
  margin-bottom: 4px; }}
.ds-panelhead .pn-title {{ font-size: 13.5px; font-weight: 700; color: var(--text);
  display: flex; align-items: center; gap: 8px; }}
.ds-panelhead .pn-sub {{ font-size: 11.5px; color: var(--muted); margin-top: 3px; }}
.ds-panelfoot {{ font-size: 11.5px; color: var(--muted); border-top: 1px solid color-mix(in srgb, var(--border) 80%, transparent);
  padding-top: 8px; margin-top: 4px; line-height: 1.6; }}
.ds-note {{ color: var(--muted); }} .ds-note.note-ok {{ color: var(--ok-fg); }}
.ds-note.note-warn {{ color: var(--warn-fg); }} .ds-note.note-bad {{ color: var(--bad-fg); }}
/* 面板内的图表/表格贴边对齐，不再各自留白 */
.stPlotlyChart, [data-testid="stPlotlyChart"] {{ padding: 0; }}
[data-testid="stDataFrame"] {{ border-radius: var(--radius-sm); overflow: hidden; }}

/* ============ 5. KPI 卡（等高对齐） ============ */
.ds-kpi {{
  background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 13px 15px; box-shadow: var(--shadow); height: 100%;
  transition: transform .18s ease, box-shadow .18s ease;
}}
.ds-kpi:hover {{ transform: translateY(-2px); }}
.ds-kpi .k-label {{ font-size: 11.5px; color: var(--muted); letter-spacing: .06em; white-space: nowrap; }}
.ds-kpi .k-value {{ font-size: 25px; font-weight: 700; color: var(--text); line-height: 1.3; }}
.ds-kpi .k-delta {{ font-size: 12px; font-weight: 600; }}
.ds-kpi .k-hint {{ font-size: 11px; color: var(--muted); margin-top: 4px; line-height: 1.5; }}
.k-up {{ color: #10B981; }} .k-down {{ color: #EF4444; }} .k-flat {{ color: var(--muted); }}

/* ============ 6. 小构件 ============ */
.ds-section .rule, .ds-card {{ background-clip: padding-box; }}
.ds-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 16px; box-shadow: var(--shadow); }}
.ds-callout {{ border-left: 3px solid var(--brand); background: var(--panel-alt);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0; padding: 10px 14px; font-size: 13px;
  color: var(--text); line-height: 1.7; }}
.ds-grad-text {{ background: linear-gradient(90deg, {brand_grad1}, {brand_grad2});
  -webkit-background-clip: text; color: transparent; }}
.ds-adv {{ padding: 10px 14px; border-radius: var(--radius-sm); border: 1px solid var(--border);
  background: var(--panel); font-size: 13px; line-height: 1.7; color: var(--text); }}
.ds-adv .t {{ font-weight: 700; display: block; margin-bottom: 3px; }}
.adv-positive {{ border-left: 3px solid #10B981; }}
.adv-negative {{ border-left: 3px solid #EF4444; }}
.adv-info {{ border-left: 3px solid #3B82F6; }}
.adv-neutral {{ border-left: 3px solid #94A3B8; }}
div[data-testid="stMetric"] {{ background: var(--panel); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 12px; }}
"""

_WIDGETS = """
/* ============ 控件 ============ */
.stButton > button, .stDownloadButton > button {{
  border-radius: 10px; border: 1px solid var(--border); background: var(--panel);
  color: var(--text); font-weight: 600; min-height: 38px; transition: all .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  border-color: var(--brand); background: var(--panel-alt); color: var(--brand-fg);
}}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
  background: linear-gradient(90deg, var(--brand), var(--brand2)); border-color: transparent; color: #fff;
}}
[data-testid="stExpander"] details {{
  background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius-sm);
}}
[data-testid="stExpander"] summary {{ font-weight: 600; font-size: 13px; color: var(--text); }}
[data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
[data-baseweb="tab"] {{
  background: var(--panel-alt); border: 1px solid var(--border); border-radius: 10px;
  font-weight: 600; font-size: 13px; padding: 6px 14px;
}}
[data-baseweb="tab"][aria-selected="true"] {{ background: var(--brand); color: #fff; border-color: var(--brand); }}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea, [data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stMultiSelect"] [data-baseweb="tag"], [data-testid="stDateInput"] {{
  background: var(--panel); border-color: var(--border); color: var(--text); border-radius: 10px;
}}
[data-testid="stSidebar"] .stRadio > div, [data-testid="stSidebar"] label {{ font-size: 13px; }}
code {{ background: var(--panel-alt); color: var(--brand2); padding: 1px 5px; border-radius: 6px;
  font-size: .92em; }}
pre code {{ color: var(--text); }}
footer, [data-testid="stStatusWidget"] {{ color: var(--muted) !important; font-size: 11.5px; }}
[data-testid="stToolbar"] {{ display: none; }}
/* ---- 明暗兜底：滑块 / 单选 / 复选 / 进度 / 提示（跟随 --primary-color） ---- */
div[data-baseweb="slider"] [role="slider"] {{ background: var(--brand); border-color: var(--brand); }}
[data-testid="stRadio"] input:checked + div > div {{ background: var(--brand) !important; }}
[data-testid="stRadio"] input:checked + div {{ border-color: var(--brand) !important; }}
[data-testid="stCheckbox"] input:checked + div {{ background: var(--brand) !important;
  border-color: var(--brand) !important; }}
[data-testid="stToggle"] input:checked + div {{ background: var(--brand) !important; }}
div[data-testid="stProgress"] div[data-testid="stProgressFill"] {{ background: var(--brand); }}
[data-testid="stSpinner"] svg {{ border-top-color: var(--brand); }}
"""

_CHIPS = """
/* ============ 徽章 ============ */
.ds-chip {{ display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px;
  border-radius: 999px; font-size: 12px; font-weight: 600; border: 1px solid transparent; white-space: nowrap; }}
.ds-chip .dot {{ width: 7px; height: 7px; border-radius: 50%; }}
.chip-brand {{ background: color-mix(in srgb, var(--brand) 14%, transparent);
  color: var(--brand-fg); border-color: color-mix(in srgb, var(--brand) 32%, transparent); }}
.chip-ok {{ background: rgba(16,185,129,.14); color: var(--ok-fg); border-color: rgba(16,185,129,.34); }}
.chip-warn {{ background: rgba(245,158,11,.15); color: var(--warn-fg); border-color: rgba(245,158,11,.36); }}
.chip-bad {{ background: rgba(239,68,68,.15); color: var(--bad-fg); border-color: rgba(239,68,68,.36); }}
.chip-muted {{ background: color-mix(in srgb, var(--muted) 14%, transparent);
  color: var(--muted-fg); border-color: color-mix(in srgb, var(--muted) 32%, transparent); }}
"""


def inject(mode: str) -> None:
    """注入 CSS 并把令牌同步给 Streamlit 内建主题。"""
    import streamlit as st
    sync(mode)
    st.markdown(css(mode), unsafe_allow_html=True)


# ------------------------------------------------------------------ Plotly
def plotly_layout(mode: str, height: int = 340, **kw) -> dict:
    p = palette(mode)
    base = dict(
        height=height, margin=dict(l=10, r=10, t=46 if kw.get("title") else 24, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='Inter, "PingFang SC", "Microsoft YaHei", sans-serif',
                  size=12, color=p["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.12, x=0),
        hoverlabel=dict(bgcolor=p["panel"], font_color=p["text"], bordercolor=p["border"]),
        xaxis=dict(gridcolor=p["grid"], zeroline=False, showgrid=True,
                   tickfont=dict(size=11, color=p["muted"]),
                   title_font=dict(size=11, color=p["muted"])),
        yaxis=dict(gridcolor=p["grid"], zeroline=False, showgrid=True,
                   tickfont=dict(size=11, color=p["muted"]),
                   title_font=dict(size=11, color=p["muted"])),
        colorway=list(attribute_colors().values()),
    )
    base.update(kw)
    return base


def finish(fig, mode: str, height: int = 340, **kw):
    """给 Plotly Figure 应用主题并返回。"""
    fig.update_layout(**plotly_layout(mode, height=height, **kw))
    fig.update_xaxes(showgrid=True, gridcolor=palette(mode)["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=palette(mode)["grid"])
    return fig


# ------------------------------------------------------------------ ECharts
def echarts_theme(mode: str) -> dict:
    p = palette(mode)
    return {
        "backgroundColor": "rgba(0,0,0,0)",
        "textStyle": {"color": p["text"], "fontFamily": "Inter, PingFang SC, Microsoft YaHei"},
        "color": list(attribute_colors().values()) + ["#F87171", "#60A5FA", "#34D399", "#FBBF24"],
    }


def echarts_tooltip(mode: str) -> dict:
    p = palette(mode)
    return {"backgroundColor": p["panel"], "borderColor": p["border"],
            "textStyle": {"color": p["text"], "fontSize": 12}}
