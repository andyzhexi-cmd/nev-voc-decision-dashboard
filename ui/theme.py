"""ui.theme — 设计令牌与主题注入（config/theme.yaml → CSS / Plotly / ECharts）

职责：
  * 读取设计令牌（颜色、圆角、字体、栅格）
  * 生成明/暗两套 CSS 变量与组件样式，注入 st.markdown
  * 为 Plotly / ECharts 提供与令牌一致的布局工厂
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
}
DARK = {
    "bg": "#0B0F1A", "panel": "#141A2A", "panel_alt": "#1B2236", "border": "#26304A",
    "text": "#E8ECF7", "muted": "#98A2B8", "brand": "#4F46E5", "brand2": "#8B5CF6",
    "grid": "#22304C", "shadow": "0 10px 30px rgba(0,0,0,.35)",
}
SENTIMENT = {"positive": "#10B981", "negative": "#EF4444", "neutral": "#94A3B8"}


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


# ------------------------------------------------------------------ CSS
def css(mode: str) -> str:
    p = palette(mode)
    attrs = attribute_colors()
    attr_vars = "\n".join(f"  --attr-{k}: {v};" for k, v in attrs.items())
    body = _BODY_CSS if mode != "light" else _BODY_CSS_LIGHT
    return f"""
<style id="dsh-theme">
:root {{
{attr_vars}
  --bg: {p['bg']}; --panel: {p['panel']}; --panel-alt: {p['panel_alt']};
  --border: {p['border']}; --text: {p['text']}; --muted: {p['muted']};
  --brand: {p['brand']}; --brand2: {p['brand2']}; --grid: {p['grid']};
  --radius: 16px; --radius-sm: 10px; --shadow: {p['shadow']};
  --font: "Inter", "PingFang SC", "Microsoft YaHei", -apple-system, sans-serif;
}}
{body}
.ds-kpi {{
  background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 14px 16px; box-shadow: var(--shadow); position: relative; overflow: hidden;
  transition: transform .18s ease, box-shadow .18s ease;
}}
.ds-kpi:hover {{ transform: translateY(-2px); }}
.ds-kpi .k-label {{ font-size: 12px; color: var(--muted); letter-spacing: .06em; }}
.ds-kpi .k-value {{ font-size: 26px; font-weight: 700; color: var(--text); line-height: 1.25; }}
.ds-kpi .k-delta {{ font-size: 12px; font-weight: 600; }}
.ds-kpi .k-hint {{ font-size: 11px; color: var(--muted); margin-top: 4px; }}
.k-up {{ color: #10B981; }} .k-down {{ color: #EF4444; }} .k-flat {{ color: var(--muted); }}
.ds-chip {{
  display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px;
  border-radius: 999px; font-size: 12px; font-weight: 600; border: 1px solid transparent;
}}
.ds-chip .dot {{ width: 7px; height: 7px; border-radius: 50%; }}
.chip-brand {{ background: rgba(79,70,229,.14); color: #A5B4FC; border-color: rgba(79,70,229,.35); }}
.chip-ok {{ background: rgba(16,185,129,.14); color: #34D399; border-color: rgba(16,185,129,.35); }}
.chip-warn {{ background: rgba(245,158,11,.14); color: #FBBF24; border-color: rgba(245,158,11,.35); }}
.chip-bad {{ background: rgba(239,68,68,.14); color: #F87171; border-color: rgba(239,68,68,.35); }}
.chip-muted {{ background: rgba(148,163,184,.14); color: #94A3B8; border-color: rgba(148,163,184,.3); }}
.ds-section {{ margin-top: 8px; }}
.ds-section h3 {{ font-size: 17px; font-weight: 700; color: var(--text); margin: 0 0 2px; }}
.ds-section p {{ font-size: 12.5px; color: var(--muted); margin: 0; }}
.ds-section .rule {{ height: 3px; width: 44px; background: linear-gradient(90deg, var(--brand), var(--brand2));
  border-radius: 3px; margin-top: 8px; }}
.ds-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 16px; box-shadow: var(--shadow); }}
.ds-callout {{ border-left: 3px solid var(--brand); background: var(--panel-alt); border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  padding: 10px 14px; font-size: 13px; color: var(--text); }}
.ds-grad-text {{ background: linear-gradient(90deg, #818CF8, #22D3EE); -webkit-background-clip: text; color: transparent; }}
.ds-adv {{ padding: 10px 14px; border-radius: var(--radius-sm); border: 1px solid var(--border);
  background: var(--panel); font-size: 13px; line-height: 1.6; color: var(--text); margin-bottom: 8px; }}
.ds-adv b {{ color: var(--text); }}
.ds-adv .t {{ font-weight: 700; display: block; margin-bottom: 3px; }}
.adv-positive {{ border-left: 3px solid #10B981; }}
.adv-negative {{ border-left: 3px solid #EF4444; }}
.adv-info {{ border-left: 3px solid #3B82F6; }}
.adv-neutral {{ border-left: 3px solid #94A3B8; }}
div[data-testid="stMetric"] {{ background: var(--panel); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 12px; }}
</style>"""

_BODY_CSS = """
/* ---------- 通用结构 ---------- */
.stApp { background: var(--bg); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--panel) 0%, var(--panel-alt) 100%);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
[data-testid="stMainBlockContainer"] { padding-top: 1rem; max-width: 1500px; }
/* 卡片容器 */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--panel); border-color: var(--border); border-radius: var(--radius);
  box-shadow: var(--shadow);
}
/* 表格 */
[data-testid="stDataFrame"] { border-radius: var(--radius-sm); overflow: hidden; }
/* 按钮 / 输入 */
.stButton > button, .stDownloadButton > button {
  border-radius: 10px; border: 1px solid var(--border); background: var(--panel);
  color: var(--text); font-weight: 600; transition: all .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--brand); color: #A5B4FC; background: var(--panel-alt);
}
[data-testid="stExpander"] details {
  background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius-sm);
}
/* 侧栏导航项 */
[data-testid="stSidebarNav"] a { font-weight: 600; border-radius: 8px; }
[data-testid="stSidebarNav"] [data-testid="stSidebarNavLink"] { border-radius: 10px; }
/* tabs */
[data-baseweb="tab-list"] { gap: 6px; }
[data-baseweb="tab"] {
  background: var(--panel-alt); border: 1px solid var(--border); border-radius: 10px;
  font-weight: 600; font-size: 13px; padding: 6px 14px;
}
[data-baseweb="tab"][aria-selected="true"] { background: var(--brand); color: #fff; border-color: var(--brand); }
/* 代码 / 脚注 */
code { background: var(--panel-alt); color: var(--brand2); padding: 1px 5px; border-radius: 6px; }
footer, [data-testid="stStatusWidget"] { color: var(--muted) !important; }
"""

_BODY_CSS_LIGHT = _BODY_CSS


def inject(mode: str) -> None:
    import streamlit as st
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
        xaxis=dict(gridcolor=p["grid"], zeroline=False, showgrid=True),
        yaxis=dict(gridcolor=p["grid"], zeroline=False, showgrid=True),
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
