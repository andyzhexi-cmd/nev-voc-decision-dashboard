"""ui.views.sentiment_explorer — 属性情感分析（Aspect-Based Sentiment Explorer）

版式（遵循 ui/layout.py 的四层结构：区块标题 → 卡片面 → 图表/表格 → 口径脚注）
    顶部   section_header + 全局筛选摘要 callout + KPI 行（属性覆盖 / 正负比 /
           最满意与最不满意属性 / 双模型一致率）
    总览   属性总览表面板 → 主栏正负量与情感均值图 + 边栏雷达 → 品牌×属性热力图面板
    链路   平台→属性→极性→重要性档位桑基面板 + 平台→属性→极性旭日面板（layout.split("even")）
    下钻   控件行（even3）→ 月度趋势 / 情感分布 / 高频词 三面板（even3）
           → 词云面板 → 关键词高亮评论面板（分页与排序）
    竞品   车型明细面板 + 排行面板 → 品牌×属性双矩阵面板（main_rail）

栅格只用 GRIDS 内的具名比例；图表高度只取 layout.H 四档（150/260/340/440）；
宽度统一 width="stretch"（st.plotly_chart 在 1.50 无 width 参数，默认即占满父宽）。

数据全部来自 state.store 与 services.features 契约，视图内不读任何 CSV：
    store.theme / store.filters / store.filters_json / store.data_version /
    store.features_frame（kind 级 st.cache_data 缓存）
    features.attr_overview / brand_attr_matrix / radar_series / sankey_data /
    sunburst_data / trend_series / model_ranking / kpi_summary / top_words /
    comment_page / comment_index / apply_filters / Filters
"""
from __future__ import annotations

import html
import re

import pandas as pd
import streamlit as st

from services import features as F
from state import store
from ui import layout
from ui.components import (
    callout,
    empty_state,
    heatmap,
    hbar_simple,
    kpi_row,
    legend_html,
    radar,
    rank_bars,
    sankey,
    section_header,
    sentiment_bars_by_attr,
    sentiment_hist,
    status_chip,
    sunburst,
    trend_line,
)
from ui.theme import attribute_colors, palette, tokens

# 评论明细表的列名中文化（仅影响展示，不改动契约返回值）
_COLUMN_LABEL = {
    "comment_id": "评论ID",
    "platform": "平台",
    "brand": "品牌",
    "model": "车型",
    "comment_date": "日期",
    "_attr_ground_truth": "属性",
    "final_sentiment": "情感值",
}

_POLARITY_VALUE = {"全部": None, "正面": "positive", "负面": "negative"}
_POLARITY_TEXT = {"positive": "正面", "negative": "负面"}


# ================================================================== 缓存封装
def _cached(kind: str, fjson: str, version: int, default):
    """走 store.features_frame（kind + 筛选 + 数据版本 为缓存键）。"""
    try:
        return store.features_frame(kind, fjson, version)
    except Exception:
        return default


def _rebuild(fjson: str) -> F.Filters:
    import json

    d = json.loads(fjson or "{}")
    return F.Filters(**d) if d else F.Filters()


@st.cache_data(show_spinner=False, ttl=600, max_entries=8)
def _top_words(attr: str | None, polarity: str | None, n: int,
               fjson: str, version: int) -> pd.DataFrame:
    """高频词（按 属性/极性/筛选 缓存，避免每次交互重复分词统计）。"""
    return F.top_words(attr, polarity, n, _rebuild(fjson))


@st.cache_data(show_spinner=False, ttl=600, max_entries=8)
def _comment_page(fjson: str, page: int, size: int, sort_by: str,
                  version: int) -> tuple[pd.DataFrame, int]:
    """评论分页（缓存排序与切片结果）。"""
    return F.comment_page(_rebuild(fjson), page, size, sort_by)


@st.cache_data(show_spinner=False, ttl=600, max_entries=8)
def _hist_frame(attr: str | None, fjson: str, version: int) -> pd.DataFrame:
    """情感分布所需数据：comment_index（lru_cache 单例）只读过滤 + 列选取 + copy。

    绝不原地修改 lru_cache 返回的 DataFrame。
    """
    idx = F.comment_index()
    if idx.empty:
        return pd.DataFrame(columns=["final_sentiment"])
    f = _rebuild(fjson)
    if attr:
        f = F.Filters(**{**f.__dict__, "attrs": [attr]})
    sub = F.apply_filters(idx, f)
    if sub is None or sub.empty or "final_sentiment" not in sub.columns:
        return pd.DataFrame(columns=["final_sentiment"])
    return sub[["final_sentiment"]].copy()


def _snap_height(raw: int) -> int:
    """把动态计算出的高度吸附到 layout.H 四档（150/260/340/440）。"""
    ladder = [layout.height(k) for k in ("xs", "s", "m", "l")]
    return min(ladder, key=lambda h: (abs(h - raw), h))


# ================================================================== 文案与工具
def _filter_summary(f: F.Filters) -> str:
    brands = "、".join(f.brand) if f.brand else "全部"
    models = "、".join(f.model) if f.model else "全部"
    polar = _POLARITY_TEXT.get(f.polarity or "", "全部")
    attrs = "、".join(f.attrs) if f.attrs else "全部"
    span = f"{f.date_range[0]} ~ {f.date_range[1]}" if f.date_range else "全时段"
    return (f"当前筛选：品牌 {brands} / 车型 {models} / 极性 {polar} / "
            f"属性 {attrs} / 时间 {span}")


def _kpi_cards(ov: pd.DataFrame, k: dict) -> list[dict]:
    """顶部 KPI 行：属性覆盖、正负比、最满意/最不满意属性、双模型一致率。"""
    covered = ov[ov["评论数"] > 0] if not ov.empty else ov
    n_cover = int(len(covered))
    n_total_attr = int(len(ov)) if not ov.empty else 0

    pos = int(k.get("pos", 0))
    neg = int(k.get("neg", 0))
    ratio = f"{pos / neg:.2f}" if neg else "—"
    keep = float(k.get("keep_rate", 0.0) or 0.0)

    best = covered.loc[covered["情感均值"].idxmax()] if n_cover else None
    worst = covered.loc[covered["情感均值"].idxmin()] if n_cover else None

    def side(row, empty_label: str) -> tuple[str, str, str]:
        if row is None:
            return empty_label, "—", "当前筛选下无覆盖"
        return (str(row["属性"]), f"{float(row['情感均值']):+.3f}",
                f"覆盖 {int(row['评论数']):,} 条评论")

    best_attr, best_val, best_hint = side(best, "—")
    worst_attr, worst_val, worst_hint = side(worst, "—")

    return [
        {"label": "属性覆盖数", "value": f"{n_cover}/{n_total_attr}",
         "hint": f"筛选后有效评论 {int(k.get('n_kept', 0)):,} 条"},
        {"label": "正负比（正:负）", "value": ratio,
         "delta": f"正向占比 {k.get('pos_ratio', 0)}%",
         "hint": f"正面 {pos:,} · 负面 {neg:,} 条",
         "tone": "up" if pos >= neg else "down"},
        {"label": "最满意属性", "value": best_attr, "delta": best_val,
         "hint": best_hint, "tone": "up"},
        {"label": "最不满意属性", "value": worst_attr, "delta": worst_val,
         "hint": worst_hint, "tone": "down"},
        {"label": "双模型一致率", "value": f"{keep:.1f}%",
         "hint": f"VADER × 朴素贝叶斯一致保留 {int(k.get('n_kept', 0)):,} 条"},
    ]


def _mark_style(mode: str) -> str:
    """高亮底色取自主题令牌 palette.<mode>.warning（视图内不硬编码颜色）。"""
    pal = tokens().get("palette", {}) or {}
    key = "light" if mode == "light" else "dark"
    sources = [pal.get(key) or {}, pal.get("dark") or {}, attribute_colors()]
    for src in sources:
        raw = src.get("warning") or src.get("舒适性")
        value = str(raw or "").lstrip("#")
        if len(value) != 6:
            continue
        try:
            r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            continue
        return (f"background:rgba({r},{g},{b},.30);color:inherit;"
                f"padding:0 2px;border-radius:3px;")
    # 令牌缺失时仅保留内边距，底色交给浏览器默认高亮样式
    return "color:inherit;padding:0 2px;border-radius:3px;"


def _highlight(text: str, words: list[str], mode: str = "dark") -> str:
    """把关键词在评论正文中标出（先按词切分，再逐段转义，杜绝 HTML 注入）。"""
    body = "" if text is None else str(text)
    if not words:
        return html.escape(body)
    style = _mark_style(mode)
    pat = re.compile("(" + "|".join(re.escape(w) for w in words) + ")")
    parts = pat.split(body)
    out = []
    for i, part in enumerate(parts):
        esc = html.escape(part)
        if i % 2:
            esc = f'<mark style="{style}">{esc}</mark>'
        out.append(esc)
    return "".join(out)


def _comment_block(row: pd.Series, words: list[str], mode: str = "dark") -> str:
    meta_bits = []
    for col, label in (("comment_id", "ID"), ("platform", "平台"),
                       ("brand", "品牌"), ("model", "车型")):
        if col in row and pd.notna(row[col]):
            meta_bits.append(f"{label} {html.escape(str(row[col]))}")
    if "comment_date" in row and pd.notna(row["comment_date"]):
        try:
            meta_bits.append(pd.to_datetime(row["comment_date"]).strftime("%Y-%m-%d"))
        except Exception:
            meta_bits.append(html.escape(str(row["comment_date"])))
    if "_attr_ground_truth" in row and pd.notna(row["_attr_ground_truth"]):
        meta_bits.append(f"属性 {html.escape(str(row['_attr_ground_truth']))}")
    if "final_sentiment" in row and pd.notna(row["final_sentiment"]):
        meta_bits.append(f"情感值 {float(row['final_sentiment']):+.3f}")
    meta = " · ".join(meta_bits) if meta_bits else "评论"
    text = _highlight(row.get("comment_text", ""), words, mode)
    # 卡片面用主题 .ds-card（自带边框/圆角/16px 内边距），行内只保留 8pt 节奏与令牌色
    return (
        '<div class="ds-card" style="margin-bottom:8px;">'
        f'<div style="font-size:11.5px;color:var(--muted);">{meta}</div>'
        f'<div style="font-size:13.5px;line-height:1.7;margin-top:4px;'
        f'color:var(--text);word-break:break-all;">{text}</div></div>'
    )


# ================================================================== 词云
def _cjk_font() -> str | None:
    """寻找可用的中文字体文件（词云必须有真实字体，否则降级）。"""
    import os

    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    for root in ("/System/Library/Fonts", "/Library/Fonts", "/usr/share/fonts"):
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                low = name.lower()
                if any(k in low for k in ("pingfang", "heiti", "songti",
                                          "wqy", "notosanscjk", "sourcehans")):
                    return os.path.join(dirpath, name)
    return None


@st.cache_data(show_spinner=False, ttl=600, max_entries=4)
def _wordcloud_image(freq_items: tuple, background: str):
    """词频 → 词云位图（PIL Image）。字体与主题面板底色作为缓存键的一部分。"""
    if not freq_items:
        return None
    font = _cjk_font()
    if not font:
        return None
    try:
        from wordcloud import WordCloud

        cols = list(attribute_colors().values())
        if not cols:
            return None

        def color_func(word, font_size, position, orientation,
                       random_state=None, **kwargs):
            return cols[sum(ord(c) for c in str(word)) % len(cols)]

        wc = WordCloud(
            font_path=font, width=960, height=layout.height("l"),
            background_color=background, max_words=80,
            min_font_size=10, max_font_size=170,
            prefer_horizontal=0.9, margin=6, color_func=color_func,
        )
        wc.generate_from_frequencies(dict(freq_items))
        return wc.to_image()
    except Exception:
        return None


def _render_wordcloud(freq: dict, mode: str) -> bool:
    """渲染词云；环境不具备条件时给出中文降级说明并返回 False。"""
    import importlib.util

    try:
        has_module = importlib.util.find_spec("wordcloud") is not None
    except Exception:
        has_module = False
    if not has_module:
        st.caption("当前环境未安装 wordcloud，已降级为下方词频条形图。")
        return False
    if not _cjk_font():
        st.caption("当前环境未找到中文字体文件，已降级为下方词频条形图。")
        return False
    img = _wordcloud_image(tuple(freq.items()), palette(mode)["panel"])
    if img is None:
        st.caption("词云渲染失败，已降级为下方词频条形图。")
        return False
    st.image(img, caption="词云 · 字号与词频成正比，颜色对应属性色标",
             width="stretch")
    return True


# ================================================================== 四个标签页
def _tab_overview(mode: str, ov: pd.DataFrame, fjson: str,
                  version: int, attrs: list[str]) -> None:
    section_header("属性总览", subtitle="六大属性的评论量、情感均值与论文基准对照",
                   tag="TABLE")

    with layout.panel("属性总览明细",
                      "口径：仅统计双模型一致的评论；Δvs论文 = 当前筛选情感均值 − 论文基准情感。"):
        st.dataframe(ov, hide_index=True, width="stretch")

    left, right = layout.split("main_rail")
    with left:
        with layout.panel("各属性正/负面评论量与情感均值",
                          "堆叠柱为正面/负面评论量（左轴），折线为情感均值（右轴，-1 ~ 1）。"):
            st.plotly_chart(
                sentiment_bars_by_attr(ov, mode=mode, height=layout.height("m")),
                key="abse_bars")
            st.markdown(legend_html(), unsafe_allow_html=True)
    with right:
        with layout.panel("品牌属性雷达",
                          "各品牌在六大属性上的情感均值归一化到 40 ~ 100，悬停可见原始均值。"):
            series = _cached("radar", fjson, version, [])
            if series:
                st.plotly_chart(
                    radar(series, attrs, mode=mode, height=layout.height("m")),
                    key="abse_radar")
                st.markdown(legend_html(), unsafe_allow_html=True)
            else:
                st.info("当前筛选下没有可绘制的品牌雷达数据。")

    with layout.panel("品牌 × 属性情感热力图",
                      "数值为情感均值（VADER，-1 ~ 1），行=品牌、列=属性；"
                      "由红到绿表示口碑由差到好，0 附近为中性。"):
        pv_mean, _pv_cnt = _cached("brand_attr", fjson, version,
                                   (pd.DataFrame(), pd.DataFrame()))
        if pv_mean is None or pv_mean.empty:
            st.info("当前筛选下没有品牌 × 属性数据。")
        else:
            st.plotly_chart(
                heatmap(pv_mean, mode=mode, height=layout.height("m"),
                        diverging=True, fmt=".3f"),
                key="abse_heatmap")


def _tab_flow(mode: str, fjson: str, version: int) -> None:
    section_header("评论链路", subtitle="平台 → 属性 → 极性 → 重要性档位的流向结构",
                   tag="FLOW")

    left, right = layout.split("even")
    with left:
        with layout.panel(
                "流向桑基",
                "口径：连线宽度 = 评论条数。从左到右依次为：平台 → 属性 → 属性×极性 → "
                "重要性档位（按论文 P(x) 排序：Top2=高、中段=中、末位=低）。"):
            sankey(_cached("sankey", fjson, version, {"nodes": [], "links": []}),
                   mode=mode, key="sankey1",
                   height=f'{layout.height("l")}px')
    with right:
        with layout.panel(
                "层级旭日",
                "口径：扇区面积 = 评论条数。内环为平台，中环为属性，外环为情感极性；"
                "面积与筛选后的评论数成正比。"):
            sunburst(_cached("sunburst", fjson, version,
                             {"name": "评论", "children": []}),
                     mode=mode, key="sunburst1",
                     height=f'{layout.height("l")}px')

    st.markdown(legend_html(), unsafe_allow_html=True)
    st.caption("属性色标与全站图表一致；两图均基于当前全局筛选口径。")


def _tab_drill(mode: str, f: F.Filters, fjson: str, version: int,
               attrs: list[str]) -> None:
    section_header("下钻分析", subtitle="月度趋势 · 情感分布 · 高频词与关键词高亮评论",
                   tag="DRILL")
    colors = attribute_colors()

    # ---- 控件：属性（带色标）+ 极性
    c1, c2, c3 = layout.split("even3")
    with c1:
        attr = st.selectbox("属性", attrs, key="abse_attr")
    with c2:
        default_pol = {"positive": 1, "negative": 2}.get(f.polarity or "", 0)
        pol_text = st.radio("极性", ["全部", "正面", "负面"], index=default_pol,
                            horizontal=True, key="abse_polarity")
    with c3:
        st.markdown(legend_html(), unsafe_allow_html=True)
        st.caption(f"属性色标：{attr} = {colors.get(attr, '—')}；"
                   "极性选择会覆盖全局极性筛选。")
    pol = _POLARITY_VALUE.get(pol_text)
    tw = _top_words(attr, pol, 12, fjson, version)          # Top12：条形图 + 高亮词共用
    words = sorted(tw["词"].tolist(), key=len, reverse=True) if not tw.empty else []

    # ---- 左：月度趋势  中：情感分布  右：高频词（三面板等高，卡片底边对齐）
    lc, mc, rc = layout.split("even3")
    with lc:
        with layout.panel("月度情感趋势",
                          "按月聚合的情感均值，每条曲线对应一个属性；悬停可见当月评论数。"):
            trend = _cached("trend", fjson, version,
                            pd.DataFrame(columns=["月份", "属性", "情感均值", "评论数"]))
            if trend is None or trend.empty:
                st.info("当前筛选下没有可绘制的月度数据。")
            else:
                st.plotly_chart(trend_line(trend, mode=mode,
                                           height=layout.height("m")),
                                key="abse_trend")
                st.markdown(legend_html(), unsafe_allow_html=True)
    with mc:
        with layout.panel("情感值分布",
                          f"「{attr}」的 VADER 情感值直方图（-1 ~ 1）；"
                          "仅随全局筛选与所选属性变化，不随上方极性开关截断。"):
            hist = _hist_frame(attr, fjson, version)
            if hist.empty:
                st.info("当前筛选下没有可绘制的情感分布。")
            else:
                st.plotly_chart(sentiment_hist(hist, mode=mode,
                                               height=layout.height("m")),
                                key="abse_hist")
    with rc:
        with layout.panel("高频词 Top12",
                          f"「{attr}」·{pol_text}评论中出现次数最多的中文词"
                          "（2-6 字，已去停用词）。"):
            if tw.empty:
                st.info("当前筛选下没有可统计的分词结果。")
            else:
                st.plotly_chart(
                    hbar_simple(tw["词"].tolist(), tw["次数"].tolist(), mode=mode,
                                height=layout.height("m"), title=None),
                    key="abse_topwords")

    # ---- 词云（无 wordcloud / 无中文字体时降级为词频条形图）
    freq_df = _top_words(attr, pol, 60, fjson, version)
    freq = ({str(w): int(c) for w, c in zip(freq_df["词"], freq_df["次数"])}
            if not freq_df.empty else {})
    with layout.panel("关键词词云", f"「{attr}」·{pol_text}评论的高频词可视化"):
        if not freq:
            st.info("当前筛选下没有可分词的评论文本，请调整属性或极性。")
        elif not _render_wordcloud(freq, mode):
            top = freq_df.head(20)
            st.plotly_chart(
                hbar_simple(top["词"].tolist(), top["次数"].tolist(), mode=mode,
                            height=layout.height("m"),
                            title="词频 Top20（词云降级视图）"),
                key="abse_cloud_fallback")
            st.caption("降级视图与右栏词频口径一致，仅展示次数最多的 20 个词。")

    # ---- 关键词高亮评论（分页）
    f_attr = F.Filters(**{**f.__dict__, "attrs": [attr], "polarity": pol})
    f_attr_json = store.filters_json(f_attr)
    with layout.panel("关键词高亮评论",
                      f"正文中标出高频词，支持分页与排序；口径：「{attr}」·{pol_text}"):
        s1, s2, s3 = layout.split("even3")
        with s1:
            sort_by = st.radio("排序方式", ["情感升序", "情感降序", "时间倒序"],
                               horizontal=True, key="abse_sort")
        with s2:
            size = st.selectbox("每页条数", [10, 20, 50], index=0,
                                key="abse_page_size")
        with s3:
            page_input = st.number_input("页码（从 0 开始）", min_value=0,
                                         max_value=9999, value=0, step=1,
                                         key="abse_page")

        df0, total = _comment_page(f_attr_json, 0, size, sort_by, version)
        n_pages = max((total + size - 1) // size, 1)
        page = min(int(page_input or 0), n_pages - 1)
        page_df = df0 if page == 0 else _comment_page(f_attr_json, page, size,
                                                      sort_by, version)[0]
        st.caption(f"共 {total} 条 · 每页 {size} 条 · 第 {page + 1}/{n_pages} 页"
                   f"（口径：「{attr}」·{pol_text}）")

        if page_df is None or page_df.empty:
            st.info("当前条件下没有匹配的评论，可调整属性、极性或页码。")
            return

        show_cols = [c for c in ("comment_id", "platform", "brand", "model",
                                 "comment_date", "_attr_ground_truth",
                                 "final_sentiment") if c in page_df.columns]
        table = page_df[show_cols].rename(columns=_COLUMN_LABEL).copy()
        st.dataframe(table, hide_index=True, width="stretch")

        st.caption(f"正文高亮关键词（Top{len(words)}）：" +
                   ("、".join(words) if words else "无"))
        for _, row in page_df.iterrows():
            st.markdown(_comment_block(row, words, mode), unsafe_allow_html=True)


def _tab_rival(mode: str, fjson: str, version: int) -> None:
    section_header("车型口碑榜", subtitle="按情感均值降序，覆盖当前筛选下的全部车型",
                   tag="RANK")
    rk = _cached("ranking", fjson, version, pd.DataFrame())
    if rk is None or rk.empty:
        st.info("当前筛选下没有车型数据。")
    else:
        shown = rk.rename(columns={"brand": "品牌", "model": "车型"})
        with layout.panel("车型口碑明细",
                          "含评论数与正向率等完整指标，与下方排行同口径。"):
            st.dataframe(shown, hide_index=True, width="stretch")

        disp = rk.copy()
        disp["车型"] = disp["brand"].astype(str) + " " + disp["model"].astype(str)
        bars = disp.sort_values("情感均值", ascending=False)
        height = _snap_height(max(layout.height("s"), 30 * len(bars) + 90))
        with layout.panel("车型情感均值排行",
                          "条形越长表示口碑越好（VADER，-1 ~ 1）；完整指标见上方明细表。"):
            st.plotly_chart(
                rank_bars(bars, x="情感均值", y="车型", mode=mode, height=height,
                          title=None),
                key="abse_rank_bars")

    # ---- 品牌 × 属性：情感均值 / 评论数 双矩阵，指标切换
    section_header("品牌 × 属性对比", subtitle="情感均值与评论数双矩阵对照", tag="MATRIX")
    mean_m, cnt_m = _cached("brand_attr", fjson, version,
                            (pd.DataFrame(), pd.DataFrame()))
    if (mean_m is None or mean_m.empty) and (cnt_m is None or cnt_m.empty):
        st.info("当前筛选下没有品牌 × 属性数据。")
        return
    mean_m = mean_m if mean_m is not None else pd.DataFrame()
    cnt_m = cnt_m if cnt_m is not None else pd.DataFrame()

    metric = st.radio("指标", ["情感均值", "评论数"], horizontal=True,
                      key="abse_metric")
    is_mean = metric == "情感均值"
    main = mean_m if is_mean else cnt_m
    side = cnt_m if is_mean else mean_m
    side_name = "评论数" if is_mean else "情感均值"

    lc, rc = layout.split("main_rail")
    with lc:
        with layout.panel(f"品牌 × 属性 · {metric}",
                          "数值为情感均值（VADER，-1 ~ 1），行=品牌、列=属性；颜色由红到绿表示口碑由差到好。"
                          if is_mean else
                          "数值为筛选后的评论条数，行=品牌、列=属性；颜色越深表示样本量越充足。"):
            if main.empty:
                st.info(f"当前筛选下没有「{metric}」矩阵数据。")
            else:
                st.plotly_chart(
                    heatmap(main, mode=mode, height=layout.height("m"),
                            diverging=is_mean,
                            fmt=".3f" if is_mean else ".0f",
                            title=f"品牌 × 属性 · {metric}"),
                    key="abse_matrix_heat")
    with rc:
        with layout.panel(f"{side_name}矩阵（明细）",
                          f"与左侧「{metric}」矩阵同口径，便于逐格核对。"):
            if side.empty:
                st.info(f"当前筛选下没有「{side_name}」矩阵数据。")
            else:
                st.dataframe(side, hide_index=True, width="stretch")
    st.markdown(legend_html(), unsafe_allow_html=True)
    st.caption("属性色标与全站图表一致。")


# ================================================================== 视图入口
def render() -> None:
    """属性情感分析视图（无参数、无返回值）。"""
    MODE = store.theme()
    f = store.filters()
    fjson = store.filters_json(f)
    version = store.data_version()

    layout.page_head(
        "属性情感分析",
        "以六大属性为轴，串联评论量、情感极性、流向链路与竞品对照")
    callout(_filter_summary(f))
    status_chip("VADER × 朴素贝叶斯 · 双模型一致口径", "ok")

    kpi = _cached("kpi", fjson, version, {}) or {}
    ov = _cached("overview", fjson, version, pd.DataFrame())
    if (ov is None or ov.empty) or int(kpi.get("n_kept", 0) or 0) <= 0:
        empty_state("当前筛选下没有可用评论",
                    "请在左侧调整品牌 / 车型 / 时间 / 极性筛选，"
                    "或到「数据管理」页重新运行数据流水线。")
        return

    kpi_row(_kpi_cards(ov, kpi))

    attrs = ov["属性"].tolist() if "属性" in ov.columns else list(attribute_colors())

    tab_overview, tab_flow, tab_drill, tab_rival = st.tabs(
        ["总览", "链路", "下钻", "竞品"])
    with tab_overview:
        _tab_overview(MODE, ov, fjson, version, attrs)
    with tab_flow:
        _tab_flow(MODE, fjson, version)
    with tab_drill:
        _tab_drill(MODE, f, fjson, version, attrs)
    with tab_rival:
        _tab_rival(MODE, fjson, version)
