"""core.algorithm.validate — 三张可编辑矩阵的校验与修复建议（纯函数，零 Streamlit）

给「决策矩阵可编辑化」用：
  * AHP 判断矩阵：形状 / 对角线 / 互反性 / 标度合法性 / CR 一致性判定 + 逐格修复提示
  * DEMATEL 语言矩阵：形状 / 对角线 / 值域(0..5) / 非法语言项
  * PLTS 决策矩阵：单元格合法性（terms 与 probs 等长、概率和、项名范围、缺项）
  * 通用：按权重反推完全一致性矩阵、两矩阵差异摘要（用于"原值 vs 当前值"对照）

所有函数只读不写，返回可直接渲染的 dict；不抛异常，错误以字段返回。
"""
from __future__ import annotations

from math import isclose

import numpy as np

#: AHP 1–9 标度及其倒数（论文表5.7 使用的合法取值）
AHP_SCALE = tuple(sorted({v for v in [1, 2, 3, 4, 5, 6, 7, 8, 9]
                          if True} | {1 / v for v in [2, 3, 4, 5, 6, 7, 8, 9]}))
#: PLTS 语言项（正文定义 l0..l4，论文表5.9 出现 l5，故校验范围放宽到 l0..l5）
PLTS_TERMS = tuple(f"l{i}" for i in range(6))
DEMATEL_VALUE_MAX = 5


def ahp_diagnostics(matrix) -> dict:
    """AHP 判断矩阵体检。

    返回字段：
      errors   结构性硬错误（形状 / 非数值 / 对角线 / 互反性）→ 必须修复才能计算
      warnings 软提醒（标度越界、CR≥0.1 不一致）→ 可计算，但需要用户知情
      hints    一键修复建议
    ok=True 仅表示"结构合法可计算"，是否一致看 consistent。
    """
    out = {"ok": False, "errors": [], "warnings": [], "hints": [],
           "cr": None, "lambda_max": None, "weights": [], "consistent": False}
    try:
        m = np.array(matrix, dtype=float)
    except Exception as exc:
        out["errors"].append(f"矩阵无法转成数值：{exc}")
        return out
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        out["errors"].append(f"必须是方阵，当前形状 {m.shape}")
        return out
    n = m.shape[0]
    if n < 3:
        out["errors"].append(f"至少 3 阶，当前 {n} 阶")
        return out
    if not np.isfinite(m).all():
        out["errors"].append("存在空值或非数值（请输入 1/9 ~ 9 的数字）")
        return out

    # 对角线
    if not np.allclose(np.diag(m), 1.0, atol=1e-9):
        bad = [i for i in range(n) if not isclose(float(m[i, i]), 1.0, abs_tol=1e-9)]
        out["errors"].append(f"对角线必须为 1（第 {bad[0] + 1} 行为 {m[bad[0], bad[0]]:g}）")

    # 互反性
    recip_bad = []
    for i in range(n):
        for j in range(i + 1, n):
            if m[j, i] == 0 or not isclose(float(m[i, j]) * float(m[j, i]), 1.0, rel_tol=1e-6, abs_tol=1e-9):
                recip_bad.append((i, j))
    if recip_bad:
        i, j = recip_bad[0]
        out["errors"].append(
            f"互反性被破坏：a[{i + 1},{j + 1}]={m[i, j]:g} 与 a[{j + 1},{i + 1}]={m[j, i]:g} 之积应为 1；"
            f"共 {len(recip_bad)} 处")
        out["hints"].append("填写上三角后系统会自动回填 1/a（点「同步互反值」）")

    # 标度合法性（软规则：反推的一致性矩阵会出现非 1~9 的比值，故只作提醒）
    scale_bad = [(i, j, float(m[i, j])) for i in range(n) for j in range(n)
                 if not any(isclose(float(m[i, j]), s, rel_tol=1e-6, abs_tol=1e-9) for s in AHP_SCALE)]
    if scale_bad:
        i, j, v = scale_bad[0]
        out["warnings"].append(
            f"a[{i + 1},{j + 1}]={v:g} 不在 Saaty 1~9 标度内（共 {len(scale_bad)} 处）"
            f"——按权重反推的矩阵本就会出现这类比值，可计算")

    if out["errors"]:
        return out

    from core.algorithm.ahp import ahp_weights
    res = ahp_weights(m.tolist(), use_paper=False)
    out.update(ok=True, cr=float(res.CR), lambda_max=float(res.lambda_max),
               weights=[float(x) for x in res.weights], consistent=bool(res.consistent))
    if not res.consistent:
        out["warnings"].append(
            f"一致性比率 CR={res.CR:.4f} ≥ 0.1，判断矩阵不一致（论文可接受阈值 0.1）")
        out["hints"].append("点击「按权重反推一致性矩阵」可得到 CR=0 的可复现版本")
    return out


def dematel_diagnostics(matrix) -> dict:
    """DEMATEL 直接关系矩阵（数值 0..5，对角线 0）体检。"""
    out = {"ok": False, "errors": [], "hints": [], "n": 0}
    try:
        z = np.array(matrix, dtype=float)
    except Exception as exc:
        out["errors"].append(f"矩阵无法转成数值：{exc}")
        return out
    if z.ndim != 2 or z.shape[0] != z.shape[1]:
        out["errors"].append(f"必须是方阵，当前形状 {z.shape}")
        return out
    out["n"] = int(z.shape[0])
    if not np.isfinite(z).all():
        out["errors"].append("存在空值或非数值（语言项应落在 l0 ~ l5 / 0 ~ 5）")
        return out
    nz = np.diag(z)
    if not np.allclose(nz, 0, atol=1e-9):
        idx = int(np.argmax(np.abs(nz)))
        out["errors"].append(f"对角线应为 0（自身不影响自身），当前 a[{idx + 1},{idx + 1}]={nz[idx]:g}")
    if z.min() < 0:
        out["errors"].append(f"存在负值 {z.min():g}，语言标度最小为 0")
    if z.max() > DEMATEL_VALUE_MAX:
        out["errors"].append(
            f"存在超过语言标度上限的值 {z.max():g}（最大 {DEMATEL_VALUE_MAX}，对应 l5）")
    if np.allclose(z, 0):
        out["errors"].append("全零矩阵无法计算 T = Z̄(I−Z̄)⁻¹")
    out["ok"] = not out["errors"]
    if out["ok"] and float(np.abs(z - z.T).max()) > 1e-9:
        out["hints"].append("该矩阵非对称：DEMATEL 允许有向影响，属正常情况")
    return out


def plts_diagnostics(matrix: dict) -> dict:
    """PLTS 决策矩阵体检：{指标: {A1: {terms, probs}, ...}}。"""
    out = {"ok": True, "cells": 0, "errors": [], "warnings": [], "bad_cells": []}
    min_sum = 1.0
    incomplete = 0
    invalid_terms = 0
    negative_prob = 0
    for ind, row in (matrix or {}).items():
        if not isinstance(row, dict):
            out["errors"].append(f"指标 {ind} 的行不是对象")
            continue
        for attr, cell in row.items():
            out["cells"] += 1
            terms = list((cell or {}).get("terms") or [])
            probs = list((cell or {}).get("probs") or [])
            if len(terms) != len(probs):
                out["errors"].append(f"{ind}/{attr}：terms({len(terms)}) 与 probs({len(probs)}) 长度不一致")
                continue
            if not terms:
                out["errors"].append(f"{ind}/{attr}：为空单元格")
                continue
            if any(t not in PLTS_TERMS for t in terms):
                invalid_terms += 1
                out["bad_cells"].append(f"{ind}/{attr} 含非法语言项")
            if any(float(p) < 0 or float(p) > 1 for p in probs):
                negative_prob += 1
                out["bad_cells"].append(f"{ind}/{attr} 概率越界 (0,1)")
            s = float(sum(probs))
            min_sum = min(min_sum, s)
            if s < 1 - 1e-9:
                incomplete += 1
    if out["errors"]:
        out["ok"] = False
    if invalid_terms:
        out["errors"].append(f"{invalid_terms} 个单元格含非法语言项（合法：{'/'.join(PLTS_TERMS)}）")
        out["ok"] = False
    if negative_prob:
        out["errors"].append(f"{negative_prob} 个单元格概率不在 [0,1] 内")
        out["ok"] = False
    if incomplete:
        out["warnings"].append(
            f"{incomplete}/{out['cells']} 个单元格概率和 <1（最小 {min_sum:.3g}），"
            f"式(4.12) 前需按「区间补全」或「归一化」处理")
    out["min_prob_sum"] = min_sum
    out["incomplete_cells"] = incomplete
    out["bad_cells"] = out["bad_cells"][:20]
    return out


def consistent_matrix_from_weights(weights) -> list[list[float]]:
    """按权重反推完全一致性矩阵 a_ij = w_i / w_j（方根法可精确还原 weights，CR=0）。"""
    w = np.array([float(x) for x in weights], dtype=float)
    if w.sum() <= 0 or (w <= 0).any():
        raise ValueError("权重必须为正数且和大于 0")
    w = w / w.sum()
    return (np.outer(w, w) / np.outer(w, w).diagonal()).tolist()


def mirror_upper_tri(matrix) -> list[list[float]]:
    """把上三角同步到下三角（a_ji = 1/a_ij），对角线置 1。"""
    m = np.array(matrix, dtype=float).copy()
    n = m.shape[0]
    for i in range(n):
        m[i, i] = 1.0
        for j in range(i + 1, n):
            v = float(m[i, j])
            if v <= 0:
                v = 1.0
            m[i, j] = v
            m[j, i] = 1.0 / v
    return m.tolist()


def matrix_diff(current, reference) -> dict:
    """原值 vs 当前值差异摘要（用于对照热力图与"已修改 N 格"提示）。"""
    a = np.array(current, dtype=float)
    b = np.array(reference, dtype=float)
    if a.shape != b.shape:
        return {"comparable": False, "shape_current": list(a.shape),
                "shape_reference": list(b.shape), "n_changed": 0, "max_abs": None,
                "positions": []}
    d = a - b
    pos = np.argwhere(np.abs(d) > 1e-9)
    return {
        "comparable": True, "shape_current": list(a.shape), "shape_reference": list(b.shape),
        "n_changed": int(len(pos)),
        "max_abs": float(np.abs(d).max()) if d.size else 0.0,
        "positions": [(int(i), int(j), float(d[i, j])) for i, j in pos[:50]],
    }
