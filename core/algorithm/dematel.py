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
    aggregation: str = "top_down",
    sub_weights: np.ndarray | list | None = None,
) -> DEMATELResult:
    """DEMATEL 计算。

    norm:
      max_sum  — z̄ = z / max_j(Σ_i z_ij)（论文式 4.4 的常见读法）
      max_all  — 除以整个矩阵行/列和的最大值

    aggregation（一级权重口径，审计修复动作 dematel_bottom_up）:
      top_down — 自顶向下：由 T 矩阵中心度→原因度合成二级权重，再按组求和
      bottom_up— 自底向上（论文口径）：表5.10 二级权重按组求和归一，
                 实测可精确复现表5.11 (0.516, 0.204, 0.280)
    sub_weights — bottom_up 指定二级权重来源；默认 use_paper 时取表5.10，否则取本次计算值
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

    # 一级聚合：二级权重组内求和（自顶向下 = 用本次计算出的二级权重）
    group = b["meta"]["group_of"]
    crit = b["meta"]["second_level"]

    def _group_sum(sub_vec: np.ndarray) -> np.ndarray:
        g = np.array([sum(sub_vec[k] for k, c in enumerate(crit) if group[c] == gid)
                      for gid in ["C1", "C2", "C3"]])
        total = g.sum()
        return g / total if total else g

    first_top = _group_sum(weights_sub)

    paper_sub = np.array([b["dematel"]["sub_weights"][c] for c in crit], dtype=float)
    paper_first = np.array(b["dematel"]["first_weights"], dtype=float)

    w_sub = paper_sub if use_paper else weights_sub

    if aggregation == "bottom_up":
        if sub_weights is not None:
            src = np.asarray(sub_weights, dtype=float)
            if src.sum():
                src = src / src.sum()
        else:
            src = paper_sub if use_paper else weights_sub
        w_first = _group_sum(src)          # 自底向上：二级权重按组求和归一
    else:
        w_first = paper_first if use_paper else first_top

    # 即使采用论文权重，聚合口径仍由论文权重自身给出
    dev = float(np.max(np.abs(w_sub - paper_sub)))

    return DEMATELResult(
        weights_sub=w_sub, weights_first=w_first, SR=SR, SC=SC,
        centrality=centrality, causality=causality, T=T,
        paper_sub=paper_sub, paper_first=paper_first, deviation=dev,
        aggregation=aggregation,
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
