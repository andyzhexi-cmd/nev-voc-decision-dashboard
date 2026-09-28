"""PLTS-VIKOR 主链路单测（式4.8-4.13）"""
import numpy as np
import pytest

from core.algorithm import Scenario, run_vikor
from core.algorithm.models import baselines
from core.algorithm.vikor import paper_reference

B = baselines()


def test_shapes_and_ranges():
    r = run_vikor()
    assert r.F_point.shape == (6, 12)
    assert r.F_norm.shape == (6, 12)
    assert r.attainment.shape == (6, 12)
    assert r.w12.shape == (12,) and r.w12.sum() == pytest.approx(1.0)
    assert r.w3.shape == (3,) and r.w3.sum() == pytest.approx(1.0)
    for arr in (r.S, r.R, r.Q, r.Q_prime, r.Pxi):
        assert arr.shape == (6,)
    assert np.all(r.attainment >= 0) and np.all(r.attainment <= 1)
    assert np.all(r.S >= 0) and np.all(r.S <= 1 + 1e-9)
    assert np.all(r.Q >= 0) and np.all(r.Q <= 1 + 1e-9)


def test_possible_degree_matrix():
    r = run_vikor()
    assert r.P.shape == (6, 6)
    assert np.allclose(np.diag(r.P), 0.5)
    assert np.all(r.P >= 0) and np.all(r.P <= 1)
    assert np.allclose(r.Pxi, r.P.sum(axis=1))
    # P(x) 越大 → 重要性越高，排序为降序
    order = list(np.argsort(-r.Pxi))
    assert [r.attributes[i] for i in order] == r.ranking
    assert sorted(r.ranking) == sorted(r.attributes)


def test_intervals_bracket_point_estimate():
    r = run_vikor()
    assert np.all(r.S_lo <= r.S + 1e-9) and np.all(r.S <= r.S_hi + 1e-9)
    assert np.all(r.Q_lo <= r.Q + 1e-9) and np.all(r.Q <= r.Q_hi + 1e-9)
    assert np.all(r.sigma >= r.diagnostics["sigma_floor"] - 1e-12)
    assert r.prob_sum_deviation == 16          # 论文表5.13 中 16 格 Σp<1


def test_lambda_changes_result():
    r0 = run_vikor(Scenario(lam=0.0))
    r1 = run_vikor(Scenario(lam=1.0))
    assert not np.allclose(r0.w3, r1.w3)
    assert not np.allclose(r0.Pxi, r1.Pxi)


def test_v_changes_Q():
    r0 = run_vikor(Scenario(v=0.0))
    r1 = run_vikor(Scenario(v=1.0))
    assert not np.allclose(r0.Q, r1.Q)


def test_direction_semantics():
    att = run_vikor(Scenario(direction="attainment"))
    sho = run_vikor(Scenario(direction="shortfall"))
    # shortfall：Q 越小越优；attainment：Q 越大越优；两者排序相反
    # shortfall 口径下 a' = 1 − a，且 Σw=1 ⇒ S' = 1 − S（口径可互证）
    assert np.allclose(sho.S, 1.0 - att.S, atol=1e-9)
    assert att.Q_order == att.ranking
    assert sorted(sho.Q_order) == sorted(att.attributes)


def test_ideal_strategy_sentiment():
    r = run_vikor(Scenario(ideal_strategy="sentiment"))
    assert r.diagnostics["ideal_scope"] == "attribute"
    assert np.allclose(r.f_star, B["ideal_solutions"]["regularized"]["f_star"], atol=1e-9)
    assert np.allclose(r.f_minus, B["ideal_solutions"]["regularized"]["f_minus"], atol=1e-9)
    # 表5.15 = 表5.14 + max(f*)：可复现的论文正则化口径
    raw = B["ideal_solutions"]["raw"]
    shift = max(raw["f_star"])
    assert np.allclose(np.array(raw["f_star"]) + shift, r.f_star, atol=1e-3)


def test_weight_overrides():
    r = run_vikor(Scenario(weight_overrides_3=(0.6, 0.2, 0.2)))
    assert r.w3.sum() == pytest.approx(1.0)
    assert r.w3[0] == pytest.approx(0.6)
    assert r.diagnostics["weight_source"] == "override"


def test_live_weights_mode():
    r = run_vikor(Scenario(use_paper_weights=False))
    assert r.diagnostics["weight_source"] == "live"
    assert r.diagnostics["ahp_consistent"] is False    # CR≈0.326 未通过


def test_paper_reference_matches_baseline():
    ref = paper_reference()
    assert np.allclose(ref.S, B["vikor"]["S"])
    assert np.allclose(ref.R, B["vikor"]["R"])
    assert np.allclose(ref.Q, B["vikor"]["Q"])
    assert np.allclose(ref.Pxi, B["vikor"]["Pxi"])
    assert ref.ranking == list(B["vikor"]["ranking"])


def test_q_direction_convention_documented():
    """attainment 口径下，Q 最小的属性其群体效用最低（论文解释方向）。"""
    r = run_vikor(Scenario(direction="attainment"))
    i_min, i_max = int(np.argmin(r.Q)), int(np.argmax(r.Q))
    assert r.S[i_min] == pytest.approx(r.S.min())
    assert r.S[i_max] == pytest.approx(r.S.max())
