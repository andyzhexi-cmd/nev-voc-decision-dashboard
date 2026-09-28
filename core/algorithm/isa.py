"""core.algorithm.isa — 重要性-满意度分析（ISA）四象限（论文 §5.6，表5.20/图5.10）

双口径：
  * paper      — 论文表5.20 问卷均值
  * figure     — 图5.10 口径（续航满意度=2.8，与表5.20 的 4.1 冲突）
  * sentiment  — 由平台实测评论情感值换算到李克特 1-5 量纲
"""
from __future__ import annotations

import numpy as np

from core.algorithm.models import baselines

QUADRANTS = ["保持区", "机会区", "低优先级区", "改进区"]


def sentiment_to_likert(values: dict[str, float], lo: float = -1.0, hi: float = 1.0) -> dict[str, float]:
    """VADER compound ∈ [-1,1] → 李克特 1..5。"""
    out = {}
    for k, v in values.items():
        t = (float(np.clip(v, lo, hi)) - lo) / (hi - lo)
        out[k] = round(1.0 + 4.0 * t, 3)
    return out


def isa_quadrants(
    importance: dict[str, float] | None = None,
    satisfaction: dict[str, float] | None = None,
    source: str = "paper",
    importance_mean: float | None = None,
    satisfaction_mean: float | None = None,
) -> dict:
    """划分四象限，返回 {records, means, source, counts}。"""
    b = baselines()
    cfg = b["isa"]
    attrs = list(b["meta"]["attributes"])

    if importance is None:
        importance = dict(cfg["importance"])
    if satisfaction is None:
        satisfaction = dict(cfg["satisfaction_figure510"] if source == "figure" else cfg["satisfaction"])
    if source == "paper":
        satisfaction = dict(cfg["satisfaction"])

    imp_mean = float(np.mean(list(importance.values()))) if importance_mean is None else float(importance_mean)
    sat_mean = float(np.mean(list(satisfaction.values()))) if satisfaction_mean is None else float(satisfaction_mean)
    # 论文口径优先使用表5.20 的综合均值
    if source in ("paper", "figure") and importance_mean is None:
        imp_mean = float(cfg["importance_mean"])
    if source == "paper" and satisfaction_mean is None:
        sat_mean = float(cfg["satisfaction_mean"])
    if source == "figure" and satisfaction_mean is None:
        sat_mean = float(np.mean(list(satisfaction.values())))

    records = []
    for a in attrs:
        imp = float(importance.get(a, imp_mean))
        sat = float(satisfaction.get(a, sat_mean))
        high_i, high_s = imp >= imp_mean, sat >= sat_mean
        quad = ("保持区" if high_i and high_s else
                "机会区" if (not high_i) and high_s else
                "低优先级区" if (not high_i) and (not high_s) else
                "改进区")
        records.append({"属性": a, "重要性": round(imp, 3), "满意度": round(sat, 3),
                        "象限": quad,
                        "Δ重要性": round(imp - imp_mean, 3),
                        "Δ满意度": round(sat - sat_mean, 3)})
    counts = {q: sum(1 for r in records if r["象限"] == q) for q in QUADRANTS}
    return {"records": records, "importance_mean": imp_mean, "satisfaction_mean": sat_mean,
            "source": source, "counts": counts,
            "importance_std": float(np.std(list(importance.values()))),
            "satisfaction_std": float(np.std(list(satisfaction.values())))}


def quadrant_advice(record: dict) -> dict:
    """象限 → 改进建议（用于报告与推演卡片）。"""
    q = record["象限"]
    a = record["属性"]
    advice = {
        "保持区": {"tone": "positive", "title": f"{a} · 保持区",
                 "body": "高重要性 + 高满意度：用户高度关注且体验达标，应保持技术领先并持续投入，作为口碑护城河。"},
        "机会区": {"tone": "info", "title": f"{a} · 机会区",
                 "body": "低重要性 + 高满意度：体验超出预期的差异化亮点，可作为营销传播抓手放大声量。"},
        "低优先级区": {"tone": "neutral", "title": f"{a} · 低优先级区",
                 "body": "低重要性 + 低满意度：投入产出比低，建议维持现状，把资源投向改进区。"},
        "改进区": {"tone": "negative", "title": f"{a} · 改进区",
                 "body": "高重要性 + 低满意度：核心痛点，改进优先级最高，建议列入改款车型的重点攻关项。"},
    }
    return advice[q]
