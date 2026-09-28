"""PLTS 概率语言术语集单测"""
import numpy as np
import pytest

from core.algorithm.plts import PLTS, build_decision_matrix, normalize_columns, DEFAULT_SCALE


def test_complete_cell_point_interval():
    p = PLTS(("l3", "l4"), (0.2, 0.8))
    e, lo, hi = p.expectation("interval")
    assert p.prob_sum == pytest.approx(1.0)
    assert e == lo == hi == pytest.approx(0.2 * 4 + 0.8 * 5)


def test_incomplete_cell_interval():
    p = PLTS(("l3", "l4"), (0.75, 0.2))          # Σp = 0.95（论文表5.13 原文）
    e, lo, hi = p.expectation("interval")
    assert p.prob_sum < 1
    assert lo < e < hi
    assert lo == pytest.approx(0.75 * 4 + 0.2 * 5 + 0.05 * 4)
    assert hi == pytest.approx(0.75 * 4 + 0.2 * 5 + 0.05 * 5)


def test_normalize_completion():
    p = PLTS(("l3", "l4"), (0.75, 0.2))
    e, lo, hi = p.expectation("normalize")
    assert e == lo == hi == pytest.approx((0.75 * 4 + 0.2 * 5) / 0.95)


def test_negative_probability_rejected():
    with pytest.raises(ValueError):
        PLTS(("l3",), (-0.1,))


def test_decision_matrix_shape_and_diagnostics():
    F, Flo, Fhi, diag = build_decision_matrix()
    assert F.shape == (6, 12)
    assert Flo.shape == Fhi.shape == (6, 12)
    assert np.all(Flo <= F + 1e-9) and np.all(F <= Fhi + 1e-9)
    # 论文表5.13 中 Σp<1 的格子
    assert diag["n_incomplete"] == 16
    assert diag["min_prob_sum"] == pytest.approx(0.9)
    assert diag["n_cells"] == 72


def test_normalize_columns_unit_norm():
    F, Flo, Fhi, _ = build_decision_matrix()
    Fn, _, _ = normalize_columns(F, Flo, Fhi)
    norms = np.linalg.norm(Fn, axis=0)
    assert np.allclose(norms, 1.0)
    # 规范化保持列内序关系
    assert np.all(np.argsort(F, axis=0) == np.argsort(Fn, axis=0))


def test_scale_mapping():
    assert DEFAULT_SCALE == {"l0": 1.0, "l1": 2.0, "l2": 3.0, "l3": 4.0, "l4": 5.0}
