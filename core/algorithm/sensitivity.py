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


def compare_to_paper_sensitivity(sweep: dict) -> list[dict]:
    """复算结果 vs 论文表5.18（同 λ 处）。"""
    paper = baselines()["sensitivity"]
    ls = sweep["lambdas"]
    out = []
    for a, vals in sweep["Pxi"].items():
        row = {"属性": a}
        for lam, v in zip(ls, vals):
            key = _match_lambda(paper["lambdas"], lam)
            pv = paper["Pxi"].get(a, [None] * 7)[key] if key is not None else None
            row[f"λ={lam:g}"] = round(v, 4)
            row[f"论文λ={lam:g}"] = pv
        out.append(row)
    return out


def _match_lambda(paper_ls: list[float], lam: float) -> int | None:
    for i, p in enumerate(paper_ls):
        if abs(p - lam) < 1e-9:
            return i
    return None
