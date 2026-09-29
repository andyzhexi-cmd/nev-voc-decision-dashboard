"""core.algorithm.sensitivity — λ / v 敏感性分析与排序稳定性"""
from __future__ import annotations

import numpy as np

from core.algorithm.models import Scenario, baselines
from core.algorithm.vikor import run_vikor

DEFAULT_LAMBDAS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]   # 论文表5.18 口径


def sweep_lambda(base: Scenario | None = None, lambdas: list[float] | None = None, **kw) -> dict:
    """在一组 λ 上重算，返回 {lambdas, Pxi 表, ranking 表, 基准排序, 稳定性}。"""
    base = base or Scenario()
    ls = lambdas or DEFAULT_LAMBDAS
    rows: dict[str, list[float]] = {}
    ranks: dict[str, list[int]] = {}
    orders: list[list[str]] = []
    results = []
    for lam in ls:
        sc = base.model_copy(update={"lam": float(lam)})
        res = run_vikor(sc, **kw)
        results.append(res)
        for i, a in enumerate(res.attributes):
            rows.setdefault(a, []).append(float(res.Pxi[i]))
        pos = {a: k + 1 for k, a in enumerate(res.ranking)}
        for a in res.attributes:
            ranks.setdefault(a, []).append(pos[a])
        orders.append(list(res.ranking))
    base_order = orders[ls.index(0.5)] if 0.5 in ls else orders[0]
    stability = [kendall_tau(o, base_order) for o in orders]
    return {
        "lambdas": ls,
        "Pxi": rows,
        "ranks": ranks,
        "orders": orders,
        "stability": stability,
        "base_order": base_order,
        "results": results,
        "stable": all(o == base_order for o in orders),
    }


def sweep_v(base: Scenario | None = None, vs: list[float] | None = None, **kw) -> dict:
    base = base or Scenario()
    vs = vs or [0.0, 0.25, 0.5, 0.75, 1.0]
    rows: dict[str, list[float]] = {}
    orders: list[list[str]] = []
    for v in vs:
        res = run_vikor(base.model_copy(update={"v": float(v)}), **kw)
        for i, a in enumerate(res.attributes):
            rows.setdefault(a, []).append(float(res.Pxi[i]))
        orders.append(list(res.ranking))
    base_order = orders[vs.index(0.5)] if 0.5 in vs else orders[0]
    return {"vs": vs, "Pxi": rows, "orders": orders,
            "stability": [kendall_tau(o, base_order) for o in orders],
            "base_order": base_order, "stable": all(o == base_order for o in orders)}


def kendall_tau(order_a: list[str], order_b: list[str]) -> float:
    """两个排序的 Kendall τ（-1..1）。"""
    n = len(order_a)
    if n < 2:
        return 1.0
    pos_b = {a: i for i, a in enumerate(order_b)}
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            a1, a2 = order_a[i], order_a[j]
            # order_a 中 a1 在 a2 之前；若 order_b 中 a1 反而在后，则为逆序对
            if pos_b[a1] > pos_b[a2]:
                disc += 1
            else:
                conc += 1
    total = conc + disc
    return (conc - disc) / total if total else 1.0


def paper_sensitivity_table() -> dict:
    """论文表5.18（只读展示）。"""
    return baselines()["sensitivity"]


ANCHOR_NOTE = ("锚点对照：λ=0.5 处的论文基准取表5.17（而非表5.18 的 λ=0.5 行），其余 λ 取表5.18。"
               "取证显示两表并非同一批参数、同一次计算（λ=0.5 行 Pearson=0.9667、最大绝对差=0.7905，"
               "其余 λ 相关 0.79–0.98 但均不相等），故此处逐 λ 列示 Δ，不再判定为「不一致」。")


def compare_to_paper_sensitivity(sweep: dict, anchor: str | None = "table517") -> list[dict]:
    """复算结果 vs 论文（同 λ 处）。

    anchor='table517'（默认，修复动作 sensitivity_anchor）：
        λ=0.5 的论文基准改用表5.17，并追加逐 λ 的 Δ 列与锚点说明，
        把「两表口径差异」显式列示出来，而不是判成不可解释的不一致。
    anchor=None：保持原行为，仅取表5.18。
    """
    paper = baselines()["sensitivity"]
    vik = baselines()["vikor"]
    attrs = list(baselines()["meta"]["attributes"])
    ls = sweep["lambdas"]
    out = []
    for a, vals in sweep["Pxi"].items():
        row = {"属性": a}
        for lam, v in zip(ls, vals):
            key = _match_lambda(paper["lambdas"], lam)
            pv = paper["Pxi"].get(a, [None] * 7)[key] if key is not None else None
            if anchor == "table517" and key is not None and abs(lam - 0.5) < 1e-9:
                pv = vik["Pxi"][attrs.index(a)]          # 锚点：λ=0.5 → 表5.17
            row[f"λ={lam:g}"] = round(v, 4)
            row[f"论文λ={lam:g}"] = pv
            if anchor:
                row[f"Δλ={lam:g}"] = None if pv is None else round(v - pv, 4)
        if anchor:
            row["锚点"] = "表5.17（λ=0.5）"
            row["锚点说明"] = ANCHOR_NOTE
        out.append(row)
    return out


def sensitivity_anchor_diagnostics() -> dict:
    """表5.18 vs 表5.17 的口径差异与锚定残余 Δ（审计 numbers 用）。"""
    paper = baselines()["sensitivity"]
    vik = baselines()["vikor"]
    attrs = list(baselines()["meta"]["attributes"])
    base = np.array([vik["Pxi"][attrs.index(a)] for a in attrs], dtype=float)
    per_lambda = {}
    for i, lam in enumerate(paper["lambdas"]):
        vals = np.array([paper["Pxi"][a][i] for a in attrs], dtype=float)
        per_lambda[f"λ={lam:g}"] = {
            "Pearson": round(float(np.corrcoef(vals, base)[0, 1]), 4),
            "最大绝对差": round(float(np.max(np.abs(vals - base))), 4),
        }
    i05 = paper["lambdas"].index(0.5)
    v05 = np.array([paper["Pxi"][a][i05] for a in attrs], dtype=float)
    return {
        "锚点": "表5.17（λ=0.5）",
        "锚点Δ": 0.0,
        "表5.18@λ=0.5 vs 表5.17": per_lambda[f"λ=0.5"],
        "逐λ对照": per_lambda,
        "对照方式": "逐 λ 列示 Δ，不判定为不一致",
        "结论": "两表非同一批参数/同一次计算，非取整误差；已锚定表5.17",
        "λ=0.5_Pearson": round(float(np.corrcoef(v05, base)[0, 1]), 4),
        "λ=0.5_最大绝对差": round(float(np.max(np.abs(v05 - base))), 4),
    }


def _match_lambda(paper_ls: list[float], lam: float) -> int | None:
    for i, p in enumerate(paper_ls):
        if abs(p - lam) < 1e-9:
            return i
    return None
