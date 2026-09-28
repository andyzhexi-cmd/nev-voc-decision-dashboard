"""core.algorithm.benchmarks — 方法对比（论文表5.19：TOPSIS / 前景理论 / 传统 VIKOR）

三种方法均作用在同一规范化决策矩阵与同一组权重上，保证可比性。
返回"排序（优→劣 / 重要性高→低）"用于与论文表5.19 对照。
"""
from __future__ import annotations

import numpy as np

from core.algorithm.models import Scenario, baselines
from core.algorithm.plts import build_decision_matrix, normalize_columns
from core.algorithm.vikor import _ideal_solutions, _resolve_weights, run_vikor
from core.algorithm.weights import expand_to_criteria


def _base(scenario: Scenario | None, **kw):
    b = baselines()
    sc = scenario or Scenario()
    _w3, w12, _meta = _resolve_weights(sc, kw.get("ahp_matrix"), kw.get("dematel_z"))
    F, F_lo, F_hi, _ = build_decision_matrix(kw.get("plts_matrix"), completion=sc.prob_completion)
    Fn, _, _ = normalize_columns(F, F_lo, F_hi)
    f_star, f_minus, scope = _ideal_solutions(sc, Fn, F_lo, F_hi)
    attrs = list(b["meta"]["attributes"])
    return sc, attrs, Fn, w12, f_star, f_minus, scope


def topsis(scenario: Scenario | None = None, **kw) -> dict:
    """TOPSIS：贴近度 Cᵢ = S⁻/(S⁺+S⁻)，越大越优。"""
    sc, attrs, Fn, w, fs, fm, scope = _base(scenario, **kw)
    if scope == "criterion":
        fp, fmn = fs, fm
    else:
        fp, fmn = fs.max(), fm.min()
    # 全部为效益型指标
    d_pos = np.sqrt((w * (Fn - fp) ** 2).sum(axis=1))
    d_neg = np.sqrt((w * (Fn - fmn) ** 2).sum(axis=1))
    closeness = d_neg / (d_pos + d_neg + 1e-12)
    order = list(np.argsort(-closeness))
    return {"name": "TOPSIS", "scores": closeness,
            "ranking": [attrs[i] for i in order]}


def prospect_theory(scenario: Scenario | None = None, alpha: float = 0.88,
                    lam_loss: float = 2.25, **kw) -> dict:
    """简化前景理论：参考点=各指标列均值，价值函数
    v(x)= x^α（收益），v(x)= -λ·(-x)^β（损失），收益分数越大越优。
    """
    sc, attrs, Fn, w, fs, fm, scope = _base(scenario, **kw)
    ref = Fn.mean(axis=0)
    delta = Fn - ref
    v = np.where(delta >= 0, np.power(np.clip(delta, 0, None), alpha),
                 -lam_loss * np.power(np.clip(-delta, 0, None), alpha))
    score = (w * v).sum(axis=1)
    order = list(np.argsort(-score))
    return {"name": "前景理论", "scores": score, "ranking": [attrs[i] for i in order]}


def traditional_vikor(scenario: Scenario | None = None, **kw) -> dict:
    """传统 VIKOR：shortfall 口径 + 按指标列理想解（不引入情感）。"""
    sc = (scenario or Scenario()).model_copy(
        update={"direction": "shortfall", "ideal_strategy": "criterion"})
    res = run_vikor(sc, **kw)
    order = list(np.argsort(res.Q))
    return {"name": "传统VIKOR", "scores": res.Q,
            "ranking": [res.attributes[i] for i in order], "result": res}


def paper_vikor(scenario: Scenario | None = None, **kw) -> dict:
    """融合情感值的 PLTS-VIKOR（本系统主方法，按 P(x) 降序）。"""
    res = run_vikor(scenario or Scenario(), **kw)
    order = list(np.argsort(-res.Pxi))
    return {"name": "PLTS-VIKOR", "scores": res.Pxi,
            "ranking": [res.attributes[i] for i in order], "result": res}


def compare_methods(scenario: Scenario | None = None, **kw) -> dict:
    """四方法并排对比 + 与论文表5.19 对照。"""
    b = baselines()
    methods = [paper_vikor(scenario, **kw), traditional_vikor(scenario, **kw),
               topsis(scenario, **kw), prospect_theory(scenario, **kw)]
    paper = b["method_comparison"]
    rows = []
    for m in methods:
        key = {"PLTS-VIKOR": "PLTS_VIKOR", "传统VIKOR": "传统VIKOR",
               "TOPSIS": "TOPSIS", "前景理论": "前景理论"}[m["name"]]
        rows.append({
            "方法": m["name"],
            "排序": " > ".join(m["ranking"]),
            "论文排序": " > ".join(paper.get(key, [])),
            "与论文一致": list(m["ranking"]) == list(paper.get(key, [])),
        })
    top1_match = sum(1 for r in rows if r["排序"].split(" > ")[0] == r["论文排序"].split(" > ")[0])
    return {"methods": methods, "rows": rows,
            "top1_agreement": f"{top1_match}/{len(rows)}",
            "paper": paper}
