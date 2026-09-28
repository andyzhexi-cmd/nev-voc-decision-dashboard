"""core.algorithm.ahp — 层次分析法（论文 §4.2.1，表5.7/5.8）"""
from __future__ import annotations

import numpy as np

from core.algorithm.models import AHPResult, baselines


def ahp_weights(matrix: np.ndarray | list | None = None, use_paper: bool = False) -> AHPResult:
    """方根法求权重 + 一致性检验。

    Parameters
    ----------
    matrix : 判断矩阵（默认取论文表5.7）
    use_paper : True 时直接返回论文表5.8 权重（校准模式），一致性指标仍按矩阵计算
    """
    b = baselines()
    A = np.array(matrix if matrix is not None else b["ahp"]["matrix"], dtype=float)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("判断矩阵必须为方阵")
    if np.any(A <= 0):
        raise ValueError("判断矩阵元素必须为正")

    row_geo = np.prod(A, axis=1) ** (1.0 / n)
    w = row_geo / row_geo.sum()
    Aw = A @ w
    lambda_max = float(np.sum(Aw / (n * w)))
    CI = float((lambda_max - n) / (n - 1))
    RI = float(b["ahp"]["RI"].get(str(n), b["ahp"]["RI"].get(n, 1.24)))
    CR = float(CI / RI) if RI > 0 else 0.0

    paper = np.array(b["ahp"]["weights"], dtype=float)
    if use_paper:
        w_out = paper
    else:
        w_out = w
    dev = float(np.max(np.abs(w_out - paper))) if len(paper) == len(w_out) else None

    return AHPResult(
        weights=w_out, lambda_max=lambda_max, CI=CI, CR=CR, RI=RI, n=n,
        consistent=CR < 0.1, paper_weights=paper, deviation=dev,
    )


def geomean_method(matrix: np.ndarray) -> np.ndarray:
    A = np.asarray(matrix, dtype=float)
    n = A.shape[0]
    g = np.prod(A, axis=1) ** (1.0 / n)
    return g / g.sum()


def eigenvector_method(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    A = np.asarray(matrix, dtype=float)
    vals, vecs = np.linalg.eig(A)
    i = int(np.argmax(vals.real))
    w = np.abs(vecs[:, i].real)
    w = w / w.sum()
    return w, float(vals[i].real)
