"""core.algorithm.weights — AHP/DEMATEL 综合权重融合与指标展开（式4.7）"""
from __future__ import annotations

import numpy as np

from core.algorithm.models import baselines

GROUPS = ["C1", "C2", "C3"]


def combine_weights(
    lam: float = 0.5,
    wA: np.ndarray | list[float] | None = None,
    wD: np.ndarray | list[float] | None = None,
) -> tuple[np.ndarray, dict]:
    """式(4.7)：w = λ·wD + (1−λ)·wA。

    λ=0 仅 AHP 主观权重；λ=1 仅 DEMATEL 关联权重。
    """
    b = baselines()
    a = np.array(wA if wA is not None else b["combined_weights"]["wA"], dtype=float)
    d = np.array(wD if wD is not None else b["dematel"]["first_weights"], dtype=float)
    if not (0.0 <= lam <= 1.0):
        raise ValueError("λ ∈ [0,1]")
    w = lam * d + (1.0 - lam) * a
    w = w / w.sum()
    return w, {"wA": a, "wD": d, "lam": lam, "paper_w": np.array(b["combined_weights"]["w"], dtype=float)}


def expand_to_criteria(
    w3: np.ndarray,
    sub_weights: np.ndarray | None = None,
    mode: str = "dematel_within",
) -> np.ndarray:
    """一级权重 → 12 个二级指标权重（Σ=1）。

    mode:
      dematel_within — 组内按 DEMATEL 二级权重份额分配（默认，信息量最大）
      equal_within   — 组内均分
      pure_sub       — 忽略一级权重，直接用 DEMATEL 12 维权重
      first_only     — 同组同权（等价 equal_within，仅命名不同）
    """
    b = baselines()
    crit = b["meta"]["second_level"]
    group = b["meta"]["group_of"]
    sub = np.array(sub_weights if sub_weights is not None
                   else [b["dematel"]["sub_weights"][c] for c in crit], dtype=float)

    if mode == "pure_sub":
        return sub / sub.sum()

    w = np.zeros(len(crit))
    for gi, g in enumerate(GROUPS):
        idx = [k for k, c in enumerate(crit) if group[c] == g]
        wg = float(w3[gi])
        if mode in ("equal_within", "first_only"):
            w[idx] = wg / len(idx)
        else:
            s = sub[idx].sum()
            w[idx] = wg * (sub[idx] / s if s > 0 else 1.0 / len(idx))
    total = w.sum()
    if total > 0:
        w = w / total
    return w


def weight_records(w3: np.ndarray, w12: np.ndarray) -> list[dict]:
    b = baselines()
    crit = b["meta"]["second_level"]
    group = b["meta"]["group_of"]
    return [{
        "指标": c,
        "名称": b["meta"]["second_level_names"][c],
        "一级": group[c],
        "一级权重": float(w3[GROUPS.index(group[c])]),
        "二级权重": float(w12[i]),
    } for i, c in enumerate(crit)]
