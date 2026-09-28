"""AHP / DEMATEL / 权重融合单测（论文表5.7-5.12 口径）"""
import numpy as np
import pytest

from core.algorithm.ahp import ahp_weights, eigenvector_method, geomean_method
from core.algorithm.dematel import dematel_weights, language_to_matrix
from core.algorithm.models import baselines
from core.algorithm.weights import combine_weights, expand_to_criteria

B = baselines()


def test_ahp_computed_weights_differ_from_paper():
    """文档化事实：论文表5.7 矩阵推不出表5.8 权重（审计项 AHP_WEIGHT_MISMATCH）。"""
    res = ahp_weights(use_paper=False)
    paper = np.array(B["ahp"]["weights"])
    assert not np.allclose(res.weights, paper, atol=0.02)
    assert res.CR == pytest.approx(0.3259, abs=1e-3)
    assert res.consistent is False          # CR > 0.1
    assert res.deviation == pytest.approx(float(np.max(np.abs(res.weights - paper))), abs=1e-9)


def test_ahp_methods_agree():
    M = np.array(B["ahp"]["matrix"])
    w_geo = geomean_method(M)
    w_eig, lmax = eigenvector_method(M)
    assert np.allclose(w_geo, w_eig, atol=1e-9)
    res = ahp_weights()
    assert res.lambda_max == pytest.approx(lmax, abs=1e-9)
    assert res.weights.sum() == pytest.approx(1.0)


def test_ahp_use_paper_returns_baseline():
    res = ahp_weights(use_paper=True)
    assert np.allclose(res.weights, B["ahp"]["weights"], atol=1e-9)


def test_dematel_matrix_and_T_property():
    Z = language_to_matrix()
    assert Z.shape == (12, 12)
    assert Z.max() == 5                       # 论文表5.9 含 l5
    res = dematel_weights(use_paper=False)
    # T = Z̄ (I − Z̄)^{-1}  ⇒  T (I − Z̄) = Z̄
    Zb = Z / Z.sum(axis=0).max()
    assert np.allclose(res.T @ (np.eye(12) - Zb), Zb, atol=1e-9)
    assert res.weights_sub.sum() == pytest.approx(1.0)
    assert res.weights_first.sum() == pytest.approx(1.0)


def test_dematel_first_level_differs_from_paper():
    res = dematel_weights(use_paper=False)
    assert not np.allclose(res.weights_first, B["dematel"]["first_weights"], atol=0.02)
    assert res.deviation == pytest.approx(float(np.max(np.abs(res.weights_sub - res.paper_sub))), abs=1e-9)


def test_combine_weights_reproduces_table_5_12():
    w, info = combine_weights(0.5)
    assert np.allclose(w, B["combined_weights"]["w"], atol=2e-3)
    assert info["lam"] == 0.5
    # 边界：λ=0 纯 AHP，λ=1 纯 DEMATEL
    w0, _ = combine_weights(0.0)
    w1, _ = combine_weights(1.0)
    assert np.allclose(w0, B["combined_weights"]["wA"], atol=1e-9)
    assert np.allclose(w1, B["dematel"]["first_weights"], atol=1e-9)


def test_combine_weights_validates_lambda():
    with pytest.raises(ValueError):
        combine_weights(1.5)


def test_expand_to_criteria_modes_sum_to_one():
    w3, _ = combine_weights(0.5)
    for mode in ("dematel_within", "equal_within", "pure_sub", "first_only"):
        w12 = expand_to_criteria(w3, mode=mode)
        assert w12.shape == (12,)
        assert w12.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(w12 >= 0)


def test_expand_respects_group_shares():
    w3, _ = combine_weights(0.5)
    w12 = expand_to_criteria(w3, mode="dematel_within")
    group = B["meta"]["group_of"]
    crit = B["meta"]["second_level"]
    c1 = sum(w12[k] for k, c in enumerate(crit) if group[c] == "C1")
    assert c1 == pytest.approx(w3[0], abs=1e-9)
