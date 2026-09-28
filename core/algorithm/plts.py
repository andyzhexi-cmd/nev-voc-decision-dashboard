"""core.algorithm.plts — 概率语言术语集（PLTS）

论文 §4.1 / §5.5：L = {l0..l4}，映射为李克特分值 β(l0..l4) = 1..5。
概率补全（prob completion）：论文表5.13 中部分格子 Σp < 1（如 0.9/0.95），
此时未分配概率视为不确定质量：
  * interval 模式：下界把余量记在最低术语、上界记在最高术语 → 区间 [E⁻, E⁺]
  * normalize 模式：按 Σp 归一化为标准 PLTS → 点估计
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.algorithm.models import baselines

DEFAULT_SCALE = {"l0": 1.0, "l1": 2.0, "l2": 3.0, "l3": 4.0, "l4": 5.0}


@dataclass(frozen=True)
class PLTS:
    """一个概率语言术语项，如 {l3^0.2, l4^0.8}。"""
    terms: tuple[str, ...]
    probs: tuple[float, ...]
    scale: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self):
        if len(self.terms) != len(self.probs):
            raise ValueError("terms 与 probs 长度必须一致")
        if any(p < -1e-9 for p in self.probs):
            raise ValueError("概率不能为负")
        if self.scale is None:
            object.__setattr__(self, "scale", DEFAULT_SCALE)

    @property
    def prob_sum(self) -> float:
        return float(sum(self.probs))

    @property
    def values(self) -> np.ndarray:
        return np.array([self.scale[t] for t in self.terms], dtype=float)

    @property
    def base_expectation(self) -> float:
        return float(np.dot(self.probs, self.values))

    def expectation(self, completion: str = "interval") -> tuple[float, float, float]:
        """返回 (点估计, 下界, 上界)。"""
        v = self.values
        e = self.base_expectation
        missing = max(0.0, 1.0 - self.prob_sum)
        if completion == "normalize" or missing <= 1e-9 or self.prob_sum <= 0:
            p = np.array(self.probs, dtype=float)
            if completion == "normalize" and self.prob_sum > 0:
                e = float(np.dot(p / self.prob_sum, v))
            return e, e, e
        lo = e + missing * v.min()
        hi = e + missing * v.max()
        return (lo + hi) / 2.0, lo, hi

    def to_dict(self) -> dict:
        return {"terms": list(self.terms), "probs": list(self.probs)}

    @classmethod
    def from_cell(cls, cell: dict, scale: dict[str, float] | None = None) -> "PLTS":
        return cls(tuple(cell["terms"]), tuple(float(p) for p in cell["probs"]), scale or DEFAULT_SCALE)


def build_decision_matrix(
    matrix_cfg: dict | None = None,
    completion: str = "interval",
    scale: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """构造 6×12 决策矩阵。

    Returns
    -------
    F_point, F_lo, F_hi : np.ndarray (6, 12)
    diagnostics : {"n_incomplete": 概率不全的格子数, "min_prob_sum": ..., "cells": ...}
    """
    b = baselines()
    cfg = matrix_cfg or b["plts_decision_matrix"]
    attrs = [f"A{i}" for i in range(1, 7)]
    crit = b["meta"]["second_level"]
    sc = scale or {t: float(v) for t, v in zip(b["linguistic_scale"]["terms"], b["linguistic_scale"]["values"])}

    n = len(attrs)
    F_point = np.zeros((n, len(crit)))
    F_lo = np.zeros((n, len(crit)))
    F_hi = np.zeros((n, len(crit)))
    n_incomplete = 0
    min_sum = 1.0
    cells = {}
    for j, c in enumerate(crit):
        for i, a in enumerate(attrs):
            plts = PLTS.from_cell(cfg[c][a], sc)
            e, lo, hi = plts.expectation(completion)
            F_point[i, j], F_lo[i, j], F_hi[i, j] = e, lo, hi
            s = plts.prob_sum
            min_sum = min(min_sum, s)
            if s < 1 - 1e-9:
                n_incomplete += 1
            cells[(c, a)] = plts
    diag = {"n_incomplete": n_incomplete, "min_prob_sum": round(min_sum, 4), "n_cells": n * len(crit)}
    return F_point, F_lo, F_hi, diag


def normalize_columns(F: np.ndarray, F_lo: np.ndarray | None = None, F_hi: np.ndarray | None = None):
    """式(4.8) 列向量规范化：f̄_ij = f_ij / sqrt(Σ_i f_ij²)。

    区间上下界沿用点估计的列范数（保证区间与点估计同尺度、保持序关系）。
    """
    denom = np.linalg.norm(F, axis=0)
    denom[denom == 0] = 1.0
    Fn = F / denom
    Flo = F_lo / denom if F_lo is not None else None
    Fhi = F_hi / denom if F_hi is not None else None
    return Fn, Flo, Fhi


def matrix_summary(F: np.ndarray, attributes: list[str], criteria: list[str]) -> list[dict]:
    """决策矩阵 → UI 可直接渲染的长表记录。"""
    out = []
    for i, a in enumerate(attributes):
        for j, c in enumerate(criteria):
            out.append({"属性": a, "指标": c, "得分": float(F[i, j])})
    return out
