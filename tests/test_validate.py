"""矩阵校验与修复建议（core/algorithm/validate.py）单测。"""
from __future__ import annotations

import pytest

from core.algorithm.validate import (
    ahp_diagnostics,
    consistent_matrix_from_weights,
    dematel_diagnostics,
    matrix_diff,
    mirror_upper_tri,
    plts_diagnostics,
)

PAPER_AHP = [[1, 5, 4], [0.2, 1, 1 / 7], [0.25, 7, 1]]


# ---------------------------------------------------------------- AHP
def test_paper_matrix_structurally_ok_but_inconsistent():
    d = ahp_diagnostics(PAPER_AHP)
    assert d["ok"] is True                      # 结构合法 → 可计算
    assert d["consistent"] is False             # 但 CR≥0.1
    assert d["errors"] == []                    # 不一致不算硬错误
    assert any("CR=" in w for w in d["warnings"])
    assert any("反推" in h for h in d["hints"])
    assert d["cr"] == pytest.approx(0.3259, abs=1e-3)
    assert sum(d["weights"]) == pytest.approx(1.0, abs=1e-9)


def test_reconstructed_consistent_matrix_passes():
    target = [0.434, 0.187, 0.379]
    m = consistent_matrix_from_weights(target)
    d = ahp_diagnostics(m)
    assert d["ok"] and d["consistent"]
    assert d["errors"] == []
    assert d["cr"] == pytest.approx(0.0, abs=1e-9)
    assert d["weights"] == pytest.approx(target, abs=1e-9)   # 方根法精确还原论文权重


def test_non_reciprocal_is_hard_error():
    bad = [[1, 5, 4], [0.2, 1, 3], [0.25, 7, 1]]           # 5×0.2=1 但 4 与 3 不互反
    d = ahp_diagnostics(bad)
    assert d["ok"] is False
    assert any("互反" in e for e in d["errors"])


def test_diagonal_and_shape_errors():
    d_diag = ahp_diagnostics([[2, 5, 4], [0.2, 1, 3], [0.25, 7, 1]])
    assert any("对角线" in e for e in d_diag["errors"])
    d_shape = ahp_diagnostics([[1, 5], [0.2, 1]])
    assert any("方阵" in e or "3 阶" in e for e in d_shape["errors"])
    d_nan = ahp_diagnostics([[1, 5, "x"], [0.2, 1, 3], [0.25, 7, 1]])
    assert any("非数值" in e or "无法转成数值" in e for e in d_nan["errors"])


def test_mirror_upper_tri_restores_reciprocity():
    m = mirror_upper_tri([[1, 5, 4], [0.1, 1, 3], [0.3333333, 0.5, 1]])
    d = ahp_diagnostics(m)
    assert all(abs(m[i][i] - 1) < 1e-12 for i in range(3))
    assert abs(m[0][1] * m[1][0] - 1) < 1e-9
    assert "互反" not in "".join(d["errors"])


# ---------------------------------------------------------------- DEMATEL
def test_dematel_valid_and_invalid():
    ok = [[0, 2, 1], [3, 0, 4], [2, 1, 0]]
    assert dematel_diagnostics(ok)["ok"] is True
    bad_diag = [[1, 2, 1], [3, 0, 4], [2, 1, 0]]
    assert any("对角线" in e for e in dematel_diagnostics(bad_diag)["errors"])
    bad_range = [[0, 9, 1], [3, 0, 4], [2, 1, 0]]
    assert any("语言标度上限" in e for e in dematel_diagnostics(bad_range)["errors"])
    assert dematel_diagnostics([[0, 0, 0], [0, 0, 0], [0, 0, 0]])["ok"] is False


# ---------------------------------------------------------------- PLTS
def test_plts_incomplete_and_invalid_terms():
    d = plts_diagnostics({
        "c11": {
            "A1": {"terms": ["l2", "l3"], "probs": [0.7, 0.3]},   # 完整
            "A2": {"terms": ["l3"], "probs": [0.9]},              # 概率和 <1
            "A3": {"terms": ["l9"], "probs": [1.0]},              # 非法语言项
            "A4": {"terms": ["l1", "l2"], "probs": [0.4]},        # 长度不一致
        }
    })
    assert d["ok"] is False
    assert d["cells"] == 4
    assert d["incomplete_cells"] == 1
    assert d["min_prob_sum"] == pytest.approx(0.9)
    assert any("长度不一致" in e for e in d["errors"])
    assert any("非法语言项" in e for e in d["errors"])
    assert any("概率和 <1" in w for w in d["warnings"])


def test_plts_full_matrix_only_warns():
    cell = {"terms": ["l3"], "probs": [1.0]}
    d = plts_diagnostics({f"c{i}": {f"A{j}": dict(cell) for j in range(1, 7)} for i in range(1, 13)})
    assert d["ok"] is True
    assert d["cells"] == 72
    assert d["incomplete_cells"] == 0
    assert d["warnings"] == []


# ---------------------------------------------------------------- 差异
def test_matrix_diff_counts_changes():
    cur = [[1, 5], [0.5, 1]]
    ref = [[1, 4], [0.25, 1]]
    d = matrix_diff(cur, ref)
    assert d["comparable"] and d["n_changed"] == 2
    assert d["max_abs"] == pytest.approx(1.0)      # 5 − 4 = 1
    assert (0, 1, 1.0) in d["positions"]
    assert (1, 0, 0.25) in d["positions"]          # 0.5 − 0.25


def test_matrix_diff_shape_mismatch():
    d = matrix_diff([[1, 2]], [[1, 2, 3], [4, 5, 6]])
    assert d["comparable"] is False
    assert d["shape_reference"] == [2, 3]
