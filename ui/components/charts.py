"""ui.components.charts — 图表构建器（Plotly 主力 + ECharts 桑基/旭日）

约定：
  * 函数名 <对象>_<形式>，入参是纯数据，返回 Plotly Figure 或直接渲染 ECharts
  * 所有颜色取自 ui.theme.attribute_colors()，保证与文案色标一致
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui.theme import (SENTIMENT, attribute_colors, echarts_theme, echarts_tooltip,
                      finish, palette)

# ------------------------------------------------------------------ 排行条形图
def rank_bars(df: pd.DataFrame, x: str, y: str = "属性", color: str | None = None,
              barmode: str = "group", height: int = 320, title: str | None = None,
              mode: str = "dark", hover: str | None = None) -> go.Figure:
    colors = attribute_colors()
    fig = go.Figure()
    if color is None:
        fig.add_trace(go.Bar(
            y=df[y], x=df[x], orientation="h",
            marker_color=[colors.get(v, palette(mode)["brand"]) for v in df[y]],
            text=df[x].round(4) if pd.api.types.is_float_dtype(df[x]) else df[x],
            textposition="outside", hoverinfo="text",
            hovertext=df[hover] if hover else None, name=x))
    else:
        groups = df[color].unique()
        pal = palette(mode)
        # 语义色：论文口径用弱化色，复算口径用主色/次色，随明暗主题切换
        palette_map = {"论文基准": pal["muted"], "当前场景": pal["brand"], "实时复算": pal["brand2"],
                       "论文": pal["muted"], "复算": pal["brand2"]}
        for g in groups:
            sub = df[df[color] == g]
            fig.add_trace(go.Bar(
                y=sub[y], x=sub[x], orientation="h", name=str(g),
                marker_color=palette_map.get(str(g), pal["brand"]),
                text=sub[x].round(4) if pd.api.types.is_float_dtype(sub[x]) else sub[x],
                textposition="outside"))
        fig.update_layout(barmode=barmode)
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_xaxes(title=x)
    return finish(fig, mode, height=height, title=title)


# ------------------------------------------------------------------ 雷达图
def radar(series: list[dict], labels: list[str], mode: str = "dark",
          height: int = 400) -> go.Figure:
    """series: [{'name':..., 'values':[...6]}]"""
    colors = attribute_colors()
    attr_colors = list(colors.get(l, palette(mode)["brand"]) for l in labels)
    fig = go.Figure()
    for i, s in enumerate(series):
        fig.add_trace(go.Scatterpolar(
            r=list(s["values"]) + [s["values"][0]],
            theta=labels + [labels[0]],
            fill="toself", name=s.get("name", f"系列{i+1}"),
            opacity=0.55, line=dict(width=2),
            hovertemplate="%{theta}: %{r:.1f}<extra>" + str(s.get("name", "")) + "</extra>"))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, gridcolor=palette(mode)["grid"],
                            tickfont=dict(size=10), linecolor=palette(mode)["border"]),
            angularaxis=dict(gridcolor=palette(mode)["grid"], tickfont=dict(size=13)),
        ))
    return finish(fig, mode, height=height, title=None)


# ------------------------------------------------------------------ 热力图
def heatmap(pivot: pd.DataFrame, mode: str = "dark", height: int = 340,
            diverging: bool = True, fmt: str = ".3f", title: str | None = None) -> go.Figure:
    values = pivot.values.astype(float)
    lim = float(np.nanmax(np.abs(values))) or 1.0
    colorscale = "RdYlGn" if diverging else "Blues"
    fig = go.Figure(go.Heatmap(
        z=values, x=pivot.columns, y=pivot.index,
        colorscale=colorscale,
        zmid=0 if diverging else None,
        zmin=-lim if diverging else 0, zmax=lim if diverging else None,
        text=np.vectorize(lambda v: ("%" + fmt) % v)(values),
        texttemplate="%{text}", textfont={"size": 12},
        hovertemplate="品牌 %{y} · 属性 %{x}<br>值 %{z:.4f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="")
    return finish(fig, mode, height=height, title=title)


def matrix_heatmap(M: np.ndarray, rows: list[str], cols: list[str], mode: str = "dark",
                   height: int = 340, colorscale: str = "Viridis",
                   title: str | None = None, fmt: str = ".3f") -> go.Figure:
    M = np.asarray(M, dtype=float)
    fig = go.Figure(go.Heatmap(
        z=M, x=cols, y=rows, colorscale=colorscale,
        text=np.vectorize(lambda v: ("%" + fmt) % v)(M), texttemplate="%{text}",
        textfont={"size": 11},
        hovertemplate="行 %{y} · 列 %{x}<br>值 %{z:.4f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="")
    return finish(fig, mode, height=height, title=title)


# ------------------------------------------------------------------ S-R 散点
def sr_scatter(res, mode: str = "dark", height: int = 420) -> go.Figure:
    colors = attribute_colors()
    S, R, Q, attrs = res.S, res.R, res.Q, res.attributes
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=S, y=R, mode="markers+text", text=attrs, textposition="top center",
        marker=dict(size=14 + 46 * (Q - Q.min()) / max(float(Q.max() - Q.min()), 1e-9),
                    color=[colors.get(a, palette(mode)["brand"]) for a in attrs],
                    line=dict(width=1.5, color=palette(mode)["text"])),
        customdata=np.stack([Q, res.Pxi], axis=-1),
        hovertemplate="属性 %{text}<br>S=%{x:.4f} · R=%{y:.4f}<br>"
                      "Q=%{customdata[0]:.4f} · P(x)=%{customdata[1]:.4f}<extra></extra>",
        name="属性"))
    fig.add_hline(y=float(np.mean(R)), line_dash="dot", line_color=palette(mode)["muted"],
                  annotation_text="R̄", annotation_position="right")
    fig.add_vline(x=float(np.mean(S)), line_dash="dot", line_color=palette(mode)["muted"],
                  annotation_text="S̄", annotation_position="top")
    fig.update_layout(xaxis_title="S 群体效用", yaxis_title="R 个体遗憾")
    return finish(fig, mode, height=height)


# ------------------------------------------------------------------ ISA 象限
def quadrant_chart(isa: dict, mode: str = "dark", height: int = 440) -> go.Figure:
    colors = attribute_colors()
    rec = pd.DataFrame(isa["records"])
    xm, ym = isa["importance_mean"], isa["satisfaction_mean"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rec["重要性"], y=rec["满意度"], mode="markers+text",
        text=rec["属性"], textposition="top center",
        marker=dict(size=18, color=[colors.get(a, palette(mode)["brand"]) for a in rec["属性"]],
                    line=dict(width=1.5, color=palette(mode)["text"])),
        customdata=np.stack([rec["象限"], rec["Δ重要性"], rec["Δ满意度"]], axis=-1),
        hovertemplate="属性 %{text}<br>重要性 %{x:.3f} · 满意度 %{y:.3f}"
                      "<br>%{customdata[0]}<extra></extra>"))
    fig.add_vline(x=xm, line_dash="dash", line_color=palette(mode)["muted"], line_width=1)
    fig.add_hline(y=ym, line_dash="dash", line_color=palette(mode)["muted"], line_width=1)
    xr = float(rec["重要性"].max() - rec["重要性"].min()) or 1.0
    yr = float(rec["满意度"].max() - rec["满意度"].min()) or 1.0
    pal = palette(mode)
    ann = [
        dict(x=xm + 0.45 * xr, y=ym + 0.42 * yr, text="保持区（高重要·高满意）", showarrow=False,
             font=dict(size=11, color=pal["ok_fg"])),
        dict(x=xm - 0.45 * xr, y=ym + 0.42 * yr, text="机会区（低重要·高满意）", showarrow=False,
             font=dict(size=11, color=pal["brand_fg"])),
        dict(x=xm + 0.45 * xr, y=ym - 0.45 * yr, text="改进区（高重要·低满意）", showarrow=False,
             font=dict(size=11, color=pal["bad_fg"])),
        dict(x=xm - 0.45 * xr, y=ym - 0.45 * yr, text="低优先级区（低重要·低满意）", showarrow=False,
             font=dict(size=11, color=pal["muted_fg"])),
    ]
    fig.update_layout(xaxis_title="重要性（论文表5.20 / 图5.10）",
                      yaxis_title="满意度（李克特 1-5）", annotations=ann)
    return finish(fig, mode, height=height)


# ------------------------------------------------------------------ 趋势 / 分布
def trend_line(df: pd.DataFrame, mode: str = "dark", height: int = 340) -> go.Figure:
    colors = attribute_colors()
    fig = px.line(df, x="月份", y="情感均值", color="属性", markers=True,
                  color_discrete_map=colors, hover_data=["评论数"])
    return finish(fig, mode, height=height)


def sentiment_hist(df: pd.DataFrame, mode: str = "dark", height: int = 300) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=df["final_sentiment"], nbinsx=60, marker_color=palette(mode)["brand"],
        hovertemplate="情感值 %{x}<br>条数 %{y}<extra></extra>"))
    fig.update_layout(bargap=0.06)
    fig.update_xaxes(title="VADER 情感值")
    fig.update_yaxes(title="评论数")
    return finish(fig, mode, height=height)


def sentiment_bars_by_attr(df: pd.DataFrame, mode: str = "dark", height: int = 340) -> go.Figure:
    colors = attribute_colors()
    fig = go.Figure()
    fig.add_trace(go.Bar(name="正面", x=df["属性"], y=df["正向"],
                         marker_color=SENTIMENT["positive"]))
    fig.add_trace(go.Bar(name="负面", x=df["属性"], y=df["负向"],
                         marker_color=SENTIMENT["negative"]))
    fig.add_trace(go.Scatter(name="情感均值", x=df["属性"], y=df["情感均值"] * 0 + df["正向"] * 0.02,
                             yaxis="y2", mode="lines+markers",
                             line=dict(color=palette(mode)["brand"], width=2),
                             customdata=np.stack([df["情感均值"], df["论文基准情感"]], axis=-1),
                             hovertemplate="%{x}<br>复算均值 %{customdata[0]:.4f}"
                                           " · 论文 %{customdata[1]:.4f}<extra></extra>"))
    fig.update_layout(barmode="stack", yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                                   title="情感均值"))
    fig.update_xaxes(title="")
    fig.update_yaxes(title="评论数")
    return finish(fig, mode, height=height)


# ------------------------------------------------------------------ ECharts（桑基 / 旭日）
def sankey(data: dict, mode: str = "dark", height: int = "560px", key: str = "sankey") -> None:
    from streamlit_echarts import st_echarts
    if not data or not data.get("nodes"):
        st.info("当前筛选下没有可绘制的数据。")
        return
    p = palette(mode)
    options = {
        **echarts_theme(mode),
        "tooltip": {**echarts_tooltip(mode), "trigger": "item", "triggerOn": "mousemove"},
        "series": [{
            "type": "sankey",
            "layout": "none",
            "orient": "horizontal",
            "nodeAlign": "justify",
            "emphasis": {"focus": "adjacency"},
            "lineStyle": {"color": "gradient", "opacity": 0.28, "curveness": 0.5},
            "label": {"color": p["text"], "fontSize": 12},
            "itemStyle": {"borderWidth": 0},
            "data": data["nodes"],
            "links": data["links"],
        }],
    }
    st_echarts(options=options, height=height, key=key)


def sunburst(data: dict, mode: str = "dark", height: str = "520px", key: str = "sunburst") -> None:
    from streamlit_echarts import st_echarts
    if not data or not data.get("children"):
        st.info("当前筛选下没有可绘制的数据。")
        return
    p = palette(mode)
    options = {
        **echarts_theme(mode),
        "tooltip": {**echarts_tooltip(mode), "trigger": "item"},
        "series": [{
            "type": "sunburst",
            "radius": ["12%", "92%"],
            "data": data["children"],
            "label": {"color": p["text"], "fontSize": 12, "minAngle": 12},
            "itemStyle": {"borderColor": p["bg"], "borderWidth": 2},
            "emphasis": {"focus": "ancestor"},
        }],
    }
    st_echarts(options=options, height=height, key=key)


# ------------------------------------------------------------------ 分项小图
def hbar_simple(labels: list[str], values: list[float], mode: str = "dark",
                height: int = 260, color_map: dict | None = None,
                title: str | None = None) -> go.Figure:
    cmap = color_map or attribute_colors()
    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h",
        marker_color=[cmap.get(l, palette(mode)["brand"]) for l in labels],
        text=[round(float(v), 4) for v in values], textposition="outside",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="")
    return finish(fig, mode, height=height, title=title)


def donut(labels: list[str], values: list[float], mode: str = "dark",
          height: int = 300) -> go.Figure:
    cmap = attribute_colors()
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.58,
        marker=dict(colors=[cmap.get(l, palette(mode)["muted"]) for l in labels]),
        textinfo="label+percent", hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"))
    fig.update_layout(showlegend=False)
    return finish(fig, mode, height=height)
