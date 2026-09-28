"""services.features — 视图层数据契约（所有聚合/整形在这里完成）

所有函数返回"可直接喂给 Plotly/ECharts 的纯 dict / DataFrame"，
不含任何 Streamlit 调用，便于单测与复用。

过滤器约定（Filters）：
    brand: list[str] | None      品牌筛选
    model: list[str] | None      车型筛选
    attrs: list[str] | None      属性筛选
    polarity: str | None         'positive' | 'negative' | None(全部)
    date_range: tuple[str,str] | None   ('2024-01-01','2024-12-31')
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from services.data_store import ATTRS, FEATURES_DIR, PATHS, ROOT, store

POLARITY_LABEL = {"positive": "正面", "negative": "负面", "neutral": "中性"}


@dataclass
class Filters:
    brand: list[str] | None = None
    model: list[str] | None = None
    attrs: list[str] | None = None
    polarity: str | None = None
    date_range: tuple[str, str] | None = None
    only_kept: bool = True          # 只统计双模型一致的评论

    def key(self) -> str:
        return f"{self.brand}|{self.model}|{self.attrs}|{self.polarity}|{self.date_range}|{self.only_kept}"


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df
    if f.only_kept and "dual_agree" in out.columns:
        out = out[out["dual_agree"].astype(bool)]
    if f.brand and "brand" in out.columns:
        out = out[out["brand"].isin(f.brand)]
    if f.model and "model" in out.columns:
        out = out[out["model"].isin(f.model)]
    if f.attrs and "_attr_ground_truth" in out.columns:
        out = out[out["_attr_ground_truth"].isin(f.attrs)]
    if f.polarity and "final_sentiment" in out.columns:
        if f.polarity == "positive":
            out = out[out["final_sentiment"] > 0]
        elif f.polarity == "negative":
            out = out[out["final_sentiment"] < 0]
    if f.date_range and "comment_date" in out.columns:
        lo, hi = pd.to_datetime(f.date_range[0]), pd.to_datetime(f.date_range[1])
        out = out[(out["comment_date"] >= lo) & (out["comment_date"] <= hi)]
    return out


# ---------------------------------------------------------------- 评论索引
@lru_cache(maxsize=4)
def comment_index() -> pd.DataFrame:
    """轻量评论索引：comment_id / 维度 / 属性 / 情感 / 分词，供下钻与词频。"""
    pq = FEATURES_DIR / "comment_index.parquet"
    src = PATHS["sentiment"]
    if pq.exists() and src.exists() and pq.stat().st_mtime >= src.stat().st_mtime:
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    cols = ["comment_id", "platform", "brand", "model", "comment_date", "comment_text",
            "_attr_ground_truth", "tokens", "vader_compound", "nb_pos_proba",
            "dual_agree", "final_sentiment"]
    df = store().sentiment()
    if df.empty:
        return pd.DataFrame()
    # tokens / 文本列不在裁剪集里 → 补读一次
    try:
        extra = pd.read_csv(src, usecols=["comment_id", "tokens"],
                            dtype={"tokens": "string"}, keep_default_na=False)
        df = df.merge(extra, on="comment_id", how="left")
    except Exception:
        df["tokens"] = ""
    if "tokens" not in df.columns:
        df["tokens"] = ""
    df["polarity"] = np.where(df["final_sentiment"] > 0, "positive",
                              np.where(df["final_sentiment"] < 0, "negative", "neutral"))
    df["comment_date"] = pd.to_datetime(df["comment_date"], errors="coerce")
    keep = ["comment_id", "platform", "brand", "model", "comment_date", "comment_text",
            "_attr_ground_truth", "tokens", "vader_compound", "nb_pos_proba",
            "dual_agree", "final_sentiment", "polarity"]
    df = df[[c for c in keep if c in df.columns]]
    try:
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pq, index=False)
    except Exception:
        pass
    return df


# ---------------------------------------------------------------- KPI
def kpi_summary(f: Filters | None = None) -> dict:
    f = f or Filters()
    raw = store().raw()
    idx = comment_index()
    kept = apply_filters(idx, f) if not idx.empty else pd.DataFrame()
    total_raw = len(raw)
    noise = int((raw.get("_attr_ground_truth", pd.Series(dtype=str)) == "噪声").sum()) if total_raw else 0
    mean_sent = float(kept["final_sentiment"].mean()) if len(kept) else 0.0
    pos = int((kept["final_sentiment"] > 0).sum()) if len(kept) else 0
    neg = int((kept["final_sentiment"] < 0).sum()) if len(kept) else 0
    fresh = store().freshness()
    from core.algorithm.models import baselines
    paper_mean = list(baselines()["sentiment"]["paper_mean"].values())
    return {
        "n_raw": total_raw,
        "n_noise": noise,
        "n_kept": int(len(kept)),
        "n_dropped": int(total_raw - noise - len(kept)) if total_raw else 0,
        "keep_rate": round(len(kept) / max(total_raw - noise, 1) * 100, 1) if total_raw else 0.0,
        "n_platforms": int(raw["platform"].nunique()) if total_raw else 0,
        "n_brands": int(raw["brand"].nunique()) if total_raw else 0,
        "n_models": int(raw["model"].nunique()) if total_raw else 0,
        "mean_sentiment": round(mean_sent, 4),
        "paper_mean_sentiment": round(float(np.mean(paper_mean)), 4),
        "pos": pos, "neg": neg,
        "pos_ratio": round(pos / max(pos + neg, 1) * 100, 1),
        "freshness": fresh,
    }


# ---------------------------------------------------------------- 属性总览
def attr_overview(f: Filters | None = None) -> pd.DataFrame:
    from core.algorithm.models import baselines
    b = baselines()
    f = f or Filters()
    idx = comment_index()
    if idx.empty:
        return pd.DataFrame(columns=["属性", "评论数", "占比", "情感均值", "情感标准差",
                                     "正向", "负向", "正负比", "论文基准情感", "Δvs论文"])
    df = apply_filters(idx, f)
    rows = []
    total = max(len(df), 1)
    paper = b["sentiment"]["paper_mean"]
    for a in ATTRS:
        sub = df[df["_attr_ground_truth"] == a]
        n = len(sub)
        s = sub["final_sentiment"] if n else pd.Series(dtype=float)
        pos = int((s > 0).sum()) if n else 0
        neg = int((s < 0).sum()) if n else 0
        mean = float(s.mean()) if n else 0.0
        base = float(paper.get(a, 0.0))
        rows.append({
            "属性": a, "评论数": n, "占比": round(n / total * 100, 1),
            "情感均值": round(mean, 4), "情感标准差": round(float(s.std()) if n else 0.0, 4),
            "正向": pos, "负向": neg,
            "正负比": round(pos / max(neg, 1), 2),
            "论文基准情感": base, "Δvs论文": round(mean - base, 4),
        })
    return pd.DataFrame(rows)


def brand_attr_matrix(f: Filters | None = None, metric: str = "mean") -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (情感均值矩阵, 评论数矩阵)，index=brand，columns=属性。"""
    f = f or Filters()
    idx = comment_index()
    if idx.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = apply_filters(idx, f)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    pv_mean = df.pivot_table(index="brand", columns="_attr_ground_truth",
                             values="final_sentiment", aggfunc="mean").reindex(columns=ATTRS)
    pv_cnt = df.pivot_table(index="brand", columns="_attr_ground_truth",
                            values="comment_id", aggfunc="count").reindex(columns=ATTRS)
    return pv_mean.fillna(0).round(4), pv_cnt.fillna(0).astype(int)


def radar_series(f: Filters | None = None) -> list[dict]:
    """雷达图：每个品牌一条 6 维曲线（各属性情感均值归一化到 0-100）。"""
    pv_mean, pv_cnt = brand_attr_matrix(f)
    out = []
    if pv_mean.empty:
        return out
    all_vals = pv_mean.values
    lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    span = (hi - lo) or 1.0
    for brand, row in pv_mean.iterrows():
        norm = [round(float((v - lo) / span * 60 + 40), 1) if np.isfinite(v) else 0 for v in row.values]
        out.append({"name": brand, "values": norm,
                    "raw": [round(float(v), 4) if np.isfinite(v) else None for v in row.values],
                    "counts": [int(pv_cnt.loc[brand, a]) if a in pv_cnt.columns else 0 for a in ATTRS]})
    return out


# ---------------------------------------------------------------- 链路桑基
def sankey_data(f: Filters | None = None) -> dict:
    """平台 → 属性 → 情感极性 → 重要性档位（P(x) 分档）。

    重要性档位来自 core.algorithm 基准 P(x)：Top2=高、中段=中、末位=低。
    """
    from core.algorithm.models import baselines
    b = baselines()
    f = f or Filters()
    idx = comment_index()
    if idx.empty:
        return {"nodes": [], "links": []}
    df = apply_filters(idx, f)
    if df.empty:
        return {"nodes": [], "links": []}

    px = dict(zip(b["meta"]["attributes"], b["vikor"]["Pxi"]))
    order = sorted(px, key=lambda k: -px[k])
    tier = {a: ("高" if i < 2 else "中" if i < 4 else "低") for i, a in enumerate(order)}

    nodes: list[dict] = []
    index: dict[str, int] = {}

    def node(name: str, group: str) -> int:
        if name not in index:
            index[name] = len(nodes)
            nodes.append({"name": name, "group": group})
        return index[name]

    links: dict[tuple[int, int], float] = {}
    for plat, sub in df.groupby("platform"):
        i1 = node(str(plat), "平台")
        for a, s2 in sub.groupby("_attr_ground_truth"):
            i2 = node(str(a), "属性")
            links[(i1, i2)] = links.get((i1, i2), 0) + len(s2)
            for pol, s3 in s2.groupby("polarity"):
                label = POLARITY_LABEL.get(str(pol), str(pol))
                i3 = node(f"{a}·{label}", "属性情感")
                links[(i2, i3)] = links.get((i2, i3), 0) + len(s3)
                i4 = node(f"{a}·重要性{tier.get(a, '中')}", "重要性档位")
                links[(i3, i4)] = links.get((i3, i4), 0) + len(s3)
    return {"nodes": nodes,
            "links": [{"source": s, "target": t, "value": round(v, 0)} for (s, t), v in links.items()]}


def sunburst_data(f: Filters | None = None) -> dict:
    """旭日图：平台 → 属性 → 极性（ECharts sunburst 格式）。"""
    f = f or Filters()
    idx = comment_index()
    if idx.empty:
        return {"name": "评论", "children": []}
    df = apply_filters(idx, f)
    children = []
    for plat, s1 in df.groupby("platform"):
        p_children = []
        for a, s2 in s1.groupby("_attr_ground_truth"):
            pol = s2["polarity"].value_counts().to_dict()
            p_children.append({"name": str(a), "value": int(len(s2)),
                               "children": [{"name": POLARITY_LABEL.get(k, k), "value": int(v)}
                                            for k, v in pol.items()]})
        children.append({"name": str(plat), "value": int(len(s1)), "children": p_children})
    return {"name": "评论", "children": children}


# ---------------------------------------------------------------- 下钻与词频
_WORD_RE = re.compile(r"^[一-鿿]{2,6}$")


def top_words(attr: str | None = None, polarity: str | None = None, n: int = 20,
              f: Filters | None = None) -> pd.DataFrame:
    f = f or Filters()
    if attr:
        f = Filters(**{**f.__dict__, "attrs": [attr]})
    if polarity:
        f = Filters(**{**f.__dict__, "polarity": polarity})
    idx = comment_index()
    if idx.empty:
        return pd.DataFrame(columns=["词", "次数"])
    df = apply_filters(idx, f)
    if df.empty:
        return pd.DataFrame(columns=["词", "次数"])
    counter: dict[str, int] = {}
    for tokens in df["tokens"].head(40000).fillna(""):
        for w in str(tokens).split():
            if _WORD_RE.match(w):
                counter[w] = counter.get(w, 0) + 1
    top = sorted(counter.items(), key=lambda kv: -kv[1])[:n]
    return pd.DataFrame(top, columns=["词", "次数"])


def comment_page(f: Filters | None = None, page: int = 0, size: int = 50,
                 sort_by: str = "情感升序") -> tuple[pd.DataFrame, int]:
    """评论下钻分页：返回 (当前页 DataFrame, 总数)。"""
    f = f or Filters()
    idx = comment_index()
    if idx.empty:
        return pd.DataFrame(), 0
    df = apply_filters(idx, f).copy()
    if df.empty:
        return pd.DataFrame(), 0
    if sort_by == "情感升序":
        df = df.sort_values("final_sentiment")
    elif sort_by == "情感降序":
        df = df.sort_values("final_sentiment", ascending=False)
    elif sort_by == "时间倒序":
        df = df.sort_values("comment_date", ascending=False)
    total = len(df)
    start = page * size
    cols = ["comment_id", "platform", "brand", "model", "comment_date",
            "comment_text", "_attr_ground_truth", "final_sentiment", "polarity"]
    cols = [c for c in cols if c in df.columns]
    return df.iloc[start:start + size][cols].reset_index(drop=True), total


def trend_series(f: Filters | None = None) -> pd.DataFrame:
    """月度情感趋势：DataFrame[月份, 属性, 情感均值, 评论数]。"""
    f = f or Filters()
    idx = comment_index()
    if idx.empty:
        return pd.DataFrame(columns=["月份", "属性", "情感均值", "评论数"])
    df = apply_filters(idx, f)
    if df.empty:
        return pd.DataFrame(columns=["月份", "属性", "情感均值", "评论数"])
    df = df.copy()
    df["月份"] = df["comment_date"].dt.to_period("M").astype(str)
    g = df.groupby(["月份", "_attr_ground_truth"])["final_sentiment"].agg(["mean", "count"]).reset_index()
    g.columns = ["月份", "属性", "情感均值", "评论数"]
    return g


# ---------------------------------------------------------------- 竞品/车型
def model_ranking(f: Filters | None = None) -> pd.DataFrame:
    """车型口碑榜：车型 × 情感均值/评论数/正向率。"""
    f = f or Filters()
    idx = comment_index()
    if idx.empty:
        return pd.DataFrame()
    df = apply_filters(idx, f)
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(["brand", "model"]).agg(
        评论数=("comment_id", "count"),
        情感均值=("final_sentiment", "mean"),
        正向率=("final_sentiment", lambda s: float((s > 0).mean())),
    ).reset_index()
    g["情感均值"] = g["情感均值"].round(4)
    g["正向率"] = (g["正向率"] * 100).round(1)
    return g.sort_values("情感均值", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- 叙事卡片
def insight_cards(f: Filters | None = None) -> dict:
    """由数据自动生成的优势/短板叙事（替代旧版写死文案）。"""
    from core.algorithm.models import baselines
    b = baselines()
    ov = attr_overview(f)
    px = dict(zip(b["meta"]["attributes"], b["vikor"]["Pxi"]))
    if ov.empty:
        return {"strengths": [], "weaknesses": [], "headline": "暂无数据"}
    ov = ov.copy()
    ov["重要度"] = ov["属性"].map(px)
    # 优势 = 用户满意度最高（情感均值降序，重要度并列加权）
    # 短板 = 重要度高但满意度低：缺口 = 重要度 × (最高情感 − 本属性情感)
    max_sent = float(ov["情感均值"].max())
    ov["score_up"] = ov["情感均值"] + 0.05 * ov["重要度"]
    ov["score_down"] = ov["重要度"] * (max_sent - ov["情感均值"])
    strengths = ov.sort_values("score_up", ascending=False).head(3)
    weaknesses = ov.sort_values("score_down", ascending=False).head(3)

    def card(r, kind):
        if kind == "up":
            body = (f"情感均值 {r['情感均值']:+.3f}（论文基准 {r['论文基准情感']:+.3f}），"
                    f"覆盖 {r['评论数']:,} 条评论，正负比 {r['正负比']}。"
                    f"{'口碑领先，建议保持投入并作为传播抓手。' if r['情感均值'] >= r['论文基准情感'] else '仍高于竞品基准但优势收窄，需持续关注。'}")
            tone = "positive"
        else:
            body = (f"情感均值 {r['情感均值']:+.3f}，重要度 P(x)={r['重要度']:.4f}，"
                    f"覆盖 {r['评论数']:,} 条评论，负向 {r['负向']:,} 条。"
                    f"高重要度低满意度，列为改款车型重点攻关项。")
            tone = "negative"
        return {"属性": r["属性"], "title": f"{r['属性']}", "body": body, "tone": tone,
                "value": float(r["情感均值"]), "n": int(r["评论数"])}

    top = ov.sort_values("重要度", ascending=False).iloc[0]
    return {
        "headline": f"本次分析覆盖 {int(ov['评论数'].sum()):,} 条有效评论；"
                    f"最重要属性为「{top['属性']}」（P(x)={top['重要度']:.4f}）",
        "strengths": [card(r, "up") for _, r in strengths.iterrows()],
        "weaknesses": [card(r, "down") for _, r in weaknesses.iterrows()],
    }
