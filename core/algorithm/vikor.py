"""core.algorithm.vikor — 融合情感值的 PLTS-VIKOR（论文 §4.3，式 4.8–4.13）

计算链路
--------
  PLTS 决策矩阵(表5.13) → 期望得分 + 概率补全区间
  → 式(4.8) 列规范化 F̄
  → 式(4.7) 综合权重 w（一级 3 维 → 二级 12 维）
  → 理想解（按指标列 或 按属性情感值，式/表 5.14-5.15）
  → 式(4.9)(4.10) 群体效用 S / 个体遗憾 R（含区间）
  → 式(4.11) 折衷评价值 Q + 正态区间 Q′
  → 式(4.12) 折衷可能度矩阵 P（正态分布比较）
  → 式(4.13) 总体优势度 P(xᵢ) 与排序

效用方向 direction
------------------
  attainment（论文口径）：S 越大表示群体效用越大，Q 越大越"优/重要"，
                          P(x) 降序即属性重要性排序（论文表5.17 结论方向）。
  shortfall（传统 VIKOR）：S 越小越优，Q 越小越优。
两种口径在同一引擎中并存，UI 明示当前方向，避免口径混淆。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from core.algorithm.ahp import ahp_weights
from core.algorithm.dematel import dematel_weights
from core.algorithm.models import Scenario, VIKORResult, baselines
from core.algorithm.plts import build_decision_matrix, normalize_columns
from core.algorithm.weights import combine_weights, expand_to_criteria

GROUPS = ["C1", "C2", "C3"]


def _resolve_weights(scenario: Scenario, ahp_matrix, dematel_z):
    """返回 (w3, w12, meta)。"""
    b = baselines()
    wA = wD = None
    sub = None
    meta: dict = {"source": "paper"}

    if scenario.weight_overrides_3 is not None:
        w3 = np.array(scenario.weight_overrides_3, dtype=float)
        w3 = w3 / w3.sum()
        meta["source"] = "override"
        wA = wD = None
    elif scenario.use_paper_weights:
        w3, winfo = combine_weights(scenario.lam)          # 论文表5.8 + 表5.11
        wA, wD = winfo["wA"], winfo["wD"]
    else:
        ahp = ahp_weights(ahp_matrix, use_paper=False) if ahp_matrix is not None else ahp_weights(use_paper=False)
        dem = dematel_weights(dematel_z, use_paper=False) if dematel_z is not None else dematel_weights(use_paper=False)
        w3, winfo = combine_weights(scenario.lam, ahp.weights, dem.weights_first)
        wA, wD = winfo["wA"], winfo["wD"]
        sub = dem.weights_sub
        meta = {"source": "live", "ahp_CR": ahp.CR, "ahp_consistent": ahp.consistent}

    if ahp_matrix is not None and wA is not None:
        ahp = ahp_weights(ahp_matrix, use_paper=False)
        meta["ahp_CR"] = ahp.CR
        meta["ahp_consistent"] = ahp.consistent

    w12 = expand_to_criteria(w3, sub, mode=scenario.weight_mode)
    meta.update({"w3": w3, "wA": wA, "wD": wD, "w12": w12})
    return w3, w12, meta


def _ideal_solutions(scenario: Scenario, Fn: np.ndarray, F_lo: np.ndarray, F_hi: np.ndarray):
    """返回 (f_star, f_minus, scope)；scope = 'criterion'(12) 或 'attribute'(6)。"""
    b = baselines()
    if scenario.ideal_strategy == "criterion":
        return Fn.max(axis=0), Fn.min(axis=0), "criterion"

    # 按属性情感值（论文表5.14/5.15 口径）
    if scenario.sentiment_ideals is not None:
        fs = np.array(scenario.sentiment_ideals[0], dtype=float)
        fm = np.array(scenario.sentiment_ideals[1], dtype=float)
        shift = float(fs.max())                 # 正则化：全体 + max(f*)
        if fs.min() + shift < 0:
            shift = abs(float(fm.min())) if fm.min() < 0 else 0.0
        return fs + shift, fm + shift, "attribute"
    reg = b["ideal_solutions"]["regularized"]
    return np.array(reg["f_star"], dtype=float), np.array(reg["f_minus"], dtype=float), "attribute"


def _attainment_matrix(F: np.ndarray, f_star, f_minus, scope: str, direction: str) -> np.ndarray:
    if scope == "criterion":
        denom = f_star - f_minus
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
        a = (F - f_minus) / denom
    else:
        denom = (f_star - f_minus)[:, None]
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
        a = (F - f_minus[:, None]) / denom
    a = np.clip(a, 0.0, 1.0)
    return 1.0 - a if direction == "shortfall" else a


def _normal_q(S, R, v, direction, ref=None):
    """式(4.11) 折衷评价值。

    ref=(S_lo_bound, S_hi_bound, R_lo_bound, R_hi_bound) 用于让区间上下界
    与点估计共用同一套极值基准，保证 Q_lo ≤ Q ≤ Q_hi 成立。
    """
    if direction == "attainment":
        if ref is None:
            ref = (S.min(), S.max(), R.min(), R.max())
        S_lo_b, S_hi_b, R_lo_b, R_hi_b = ref
        ds = (S_hi_b - S_lo_b) or 1.0
        dr = (R_hi_b - R_lo_b) or 1.0
        Q = v * (S - S_lo_b) / ds + (1 - v) * (R - R_lo_b) / dr
    else:
        if ref is None:
            ref = (S.min(), S.max(), R.min(), R.max())
        S_lo_b, S_hi_b, R_lo_b, R_hi_b = ref
        ds = (S_hi_b - S_lo_b) or 1.0
        dr = (R_hi_b - R_lo_b) or 1.0
        Q = v * (S - S_lo_b) / ds + (1 - v) * (R - R_lo_b) / dr
    return np.clip(Q, 0.0, 1.0)


def run_vikor(
    scenario: Scenario | None = None,
    *,
    ahp_matrix=None,
    dematel_z=None,
    plts_matrix=None,
    sigma_floor: float = 0.06,
    possible_degree: str = "normal",
) -> VIKORResult:
    """执行完整 PLTS-VIKOR 计算，返回全部中间矩阵。"""
    b = baselines()
    sc = scenario or Scenario()
    attrs = list(b["meta"]["attributes"])
    crit = list(b["meta"]["second_level"])

    # 1) 权重
    w3, w12, wmeta = _resolve_weights(sc, ahp_matrix, dematel_z)

    # 2) PLTS → 决策矩阵 + 规范化
    F, F_lo, F_hi, pdiag = build_decision_matrix(plts_matrix, completion=sc.prob_completion)
    Fn, Fn_lo, Fn_hi = normalize_columns(F, F_lo, F_hi)

    # 3) 理想解
    f_star, f_minus, scope = _ideal_solutions(sc, Fn, F_lo, F_hi)

    # 4) 归一化效用矩阵（含区间）
    A = _attainment_matrix(Fn, f_star, f_minus, scope, sc.direction)
    A_lo = _attainment_matrix(Fn_lo, f_star, f_minus, scope, sc.direction)   # 得分下界 → 效用下界
    A_hi = _attainment_matrix(Fn_hi, f_star, f_minus, scope, sc.direction)

    # 5) S / R（式4.9 4.10）
    w = w12[None, :]
    S = (A * w).sum(axis=1)
    R = (A * w).max(axis=1)
    S_lo = (A_lo * w).sum(axis=1)
    S_hi = (A_hi * w).sum(axis=1)
    R_lo = (A_lo * w).max(axis=1)
    R_hi = (A_hi * w).max(axis=1)

    # 6) Q（式4.11）+ 正态区间 Q′（点估计与区间共用极值基准）
    ref = (float(S.min()), float(S.max()), float(R.min()), float(R.max()))
    Q = _normal_q(S, R, sc.v, sc.direction, ref)
    Q_lo = _normal_q(S_lo, R_lo, sc.v, sc.direction, ref)
    Q_hi = _normal_q(S_hi, R_hi, sc.v, sc.direction, ref)
    Q_prime = 0.5 * (Q_lo + Q_hi)
    sigma = np.maximum((Q_hi - Q_lo) / (2 * 1.96), sigma_floor)

    # 7) 可能度矩阵（式4.12）与总体优势度（式4.13）
    if possible_degree == "crisp":
        P = np.where(Q[:, None] >= Q[None, :], 1.0, 0.0)
        np.fill_diagonal(P, 0.5)
    else:
        denom = np.sqrt(sigma[:, None] ** 2 + sigma[None, :] ** 2)
        denom = np.where(denom < 1e-12, 1e-12, denom)
        P = norm.cdf((Q[:, None] - Q[None, :]) / denom)
        np.fill_diagonal(P, 0.5)
    Pxi = P.sum(axis=1)

    order_imp = list(np.argsort(-Pxi))
    ranking = [attrs[i] for i in order_imp]
    q_order = list(np.argsort(Q)) if sc.direction == "shortfall" else list(np.argsort(-Q))
    Q_order = [attrs[i] for i in q_order]

    return VIKORResult(
        attributes=attrs, criteria=crit, scenario=sc,
        F_point=F, F_lo=F_lo, F_hi=F_hi, F_norm=Fn,
        prob_sum_deviation=pdiag["n_incomplete"],
        w3=w3, w12=w12, f_star=f_star, f_minus=f_minus,
        attainment=A, S=S, R=R, Q=Q, Q_prime=Q_prime, sigma=sigma,
        S_lo=S_lo, S_hi=S_hi, R_lo=R_lo, R_hi=R_hi, Q_lo=Q_lo, Q_hi=Q_hi,
        P=P, Pxi=Pxi, ranking=ranking, Q_order=Q_order,
        diagnostics={
            "ideal_scope": scope,
            "weight_source": wmeta.get("source"),
            "ahp_CR": wmeta.get("ahp_CR"),
            "ahp_consistent": wmeta.get("ahp_consistent"),
            "prob": pdiag,
            "sigma_floor": sigma_floor,
            "possible_degree": possible_degree,
            "direction": sc.direction,
            "wA": None if wmeta.get("wA") is None else np.round(wmeta["wA"], 4).tolist(),
            "wD": None if wmeta.get("wD") is None else np.round(wmeta["wD"], 4).tolist(),
        },
    )


def paper_reference() -> VIKORResult:
    """论文表5.16/5.17 基准（只读展示用，不参与计算）。"""
    b = baselines()
    attrs = list(b["meta"]["attributes"])
    v = b["vikor"]
    from core.algorithm.models import VIKORResult as _V  # noqa
    return _V(
        attributes=attrs, criteria=list(b["meta"]["second_level"]), scenario=Scenario(),
        F_point=np.zeros((6, 12)), F_lo=np.zeros((6, 12)), F_hi=np.zeros((6, 12)),
        F_norm=np.zeros((6, 12)), prob_sum_deviation=0,
        w3=np.array(b["combined_weights"]["w"]), w12=np.zeros(12),
        f_star=np.zeros(6), f_minus=np.zeros(6), attainment=np.zeros((6, 12)),
        S=np.array(v["S"]), R=np.array(v["R"]), Q=np.array(v["Q"]),
        Q_prime=np.array(v["Q_prime"]), sigma=np.zeros(6),
        S_lo=np.array(v["S"]), S_hi=np.array(v["S"]), R_lo=np.array(v["R"]),
        R_hi=np.array(v["R"]), Q_lo=np.array(v["Q"]), Q_hi=np.array(v["Q"]),
        P=np.zeros((6, 6)), Pxi=np.array(v["Pxi"]),
        ranking=list(v["ranking"]), Q_order=list(v["ranking"]),
        diagnostics={"source": "paper_table_5_16_5_17"},
    )
