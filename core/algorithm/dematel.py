"""core.algorithm.dematel — DEMATEL（论文 §4.2.2，表5.9-5.11）"""
from __future__ import annotations

import numpy as np

from core.algorithm.models import DEMATELResult, baselines

LANG = {"l0": 0, "l1": 1, "l2": 2, "l3": 3, "l4": 4, "l5": 5}


def language_to_matrix(z_raw: list[list[str]] | None = None) -> np.ndarray:
    """语言术语矩阵 l0..l5 → 数值矩阵。"""
    b = baselines()
    raw = z_raw or b["dematel"]["Z"]
    return np.array([[LANG[str(v)] for v in row] for row in raw], dtype=float)


def dematel_weights(
    z: np.ndarray | list[list[str]] | None = None,
    norm: str = "max_sum",
    use_paper: bool = False,
) -> DEMATELResult:
    """DEMATEL 计算。

    norm:
      max_sum  — z̄ = z / max_j(Σ_i z_ij)（论文式 4.4 的常见读法）
      max_all  — 除以整个矩阵行/列和的最大值
    """
    b = baselines()
    Z = z if isinstance(z, np.ndarray) else language_to_matrix(z)
    Z = np.asarray(Z, dtype=float)

    if norm == "max_all":
        denom = max(Z.sum(axis=0).max(), Z.sum(axis=1).max())
    else:
        denom = Z.sum(axis=0).max()
    if denom == 0:
        denom = 1.0
    Zb = Z / denom

    n = Zb.shape[0]
    T = Zb @ np.linalg.inv(np.eye(n) - Zb)
    SR = T.sum(axis=1)
    SC = T.sum(axis=0)
    centrality = SR + SC
    causality = SR - SC
    W = np.sqrt(centrality ** 2 + causality ** 2)
    weights_sub = W / W.sum()

    # 一级聚合：二级权重组内求和
    group = b["meta"]["group_of"]
    crit = b["meta"]["second_level"]
    first = np.array([sum(weights_sub[k] for k, c in enumerate(crit) if group[c] == g)
                      for g in ["C1", "C2", "C3"]])
    first = first / first.sum()

    paper_sub = np.array([b["dematel"]["sub_weights"][c] for c in crit], dtype=float)
    paper_first = np.array(b["dematel"]["first_weights"], dtype=float)

    w_sub = paper_sub if use_paper else weights_sub
    w_first = paper_first if use_paper else first
    # 即使采用论文权重，聚合口径仍由论文权重自身给出
    dev = float(np.max(np.abs(w_sub - paper_sub)))

    return DEMATELResult(
        weights_sub=w_sub, weights_first=w_first, SR=SR, SC=SC,
        centrality=centrality, causality=causality, T=T,
        paper_sub=paper_sub, paper_first=paper_first, deviation=dev,
    )


def causal_chart_data(res: DEMATELResult, criteria: list[str] | None = None) -> list[dict]:
    """因果图（中心度 D+R vs 原因度 D−R）数据。"""
    b = baselines()
    names = criteria or b["meta"]["second_level"]
    return [{
        "指标": names[i],
        "名称": b["meta"]["second_level_names"].get(names[i], names[i]),
        "中心度": float(res.centrality[i]),
        "原因度": float(res.causality[i]),
        "权重": float(res.weights_sub[i]),
        "类型": "原因型" if res.causality[i] >= 0 else "结果型",
    } for i in range(len(names))]
