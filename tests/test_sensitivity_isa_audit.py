"""敏感性 / 方法对比 / ISA / 审计 单测"""
import numpy as np
import pytest

from core.algorithm import Scenario, isa_quadrants, run_audit, sweep_lambda, sweep_v
from core.algorithm.audit import audit_summary
from core.algorithm.benchmarks import compare_methods
from core.algorithm.isa import quadrant_advice, sentiment_to_likert
from core.algorithm.models import baselines
from core.algorithm.sensitivity import DEFAULT_LAMBDAS, kendall_tau, compare_to_paper_sensitivity

B = baselines()


# ---------------- sensitivity ----------------
def test_sweep_lambda_structure():
    s = sweep_lambda()
    assert s["lambdas"] == DEFAULT_LAMBDAS
    assert set(s["Pxi"].keys()) == set(B["meta"]["attributes"])
    assert all(len(v) == len(DEFAULT_LAMBDAS) for v in s["Pxi"].values())
    assert len(s["orders"]) == len(DEFAULT_LAMBDAS)
    assert all(len(o) == 6 for o in s["orders"])
    assert len(s["stability"]) == len(DEFAULT_LAMBDAS)


def test_kendall_tau_bounds():
    order = ["a", "b", "c", "d", "e", "f"]
    assert kendall_tau(order, order) == pytest.approx(1.0)
    assert kendall_tau(order, order[::-1]) == pytest.approx(-1.0)
    assert -1.0 <= kendall_tau(order, ["b", "a", "c", "d", "e", "f"]) <= 1.0


def test_sweep_v_structure():
    s = sweep_v()
    assert s["vs"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert len(s["orders"]) == 5


def test_compare_to_paper_sensitivity_has_paper_columns():
    rows = compare_to_paper_sensitivity(sweep_lambda())
    assert len(rows) == 6
    assert "论文λ=0.5" in rows[0]
    assert rows[0]["论文λ=0.5"] is not None


# ---------------- benchmarks ----------------
def test_compare_methods_four_rankings():
    cm = compare_methods()
    assert len(cm["methods"]) == 4
    names = {m["name"] for m in cm["methods"]}
    assert names == {"PLTS-VIKOR", "传统VIKOR", "TOPSIS", "前景理论"}
    for m in cm["methods"]:
        assert sorted(m["ranking"]) == sorted(B["meta"]["attributes"])
    assert "/" in cm["top1_agreement"]
    assert len(cm["rows"]) == 4


def test_topsis_scores_monotone_with_ranking():
    cm = compare_methods()
    top = next(m for m in cm["methods"] if m["name"] == "TOPSIS")
    scores = {a: float(s) for a, s in zip(B["meta"]["attributes"], top["scores"])}
    vals = [scores[a] for a in top["ranking"]]
    assert all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1))


# ---------------- ISA ----------------
def test_isa_figure_source_puts_range_in_improvement():
    res = isa_quadrants(source="figure")
    quad = {r["属性"]: r["象限"] for r in res["records"]}
    assert quad["续航"] == "改进区"          # 图5.10 口径
    assert quad["性价比"] == "保持区"
    assert res["counts"]["改进区"] == 1


def test_isa_paper_source_uses_table_5_20():
    res = isa_quadrants(source="paper")
    sat = {r["属性"]: r["满意度"] for r in res["records"]}
    assert sat["续航"] == pytest.approx(B["isa"]["satisfaction"]["续航"])
    assert res["satisfaction_mean"] == pytest.approx(B["isa"]["satisfaction_mean"])


def test_isa_quadrants_partition():
    res = isa_quadrants(source="figure")
    assert sum(res["counts"].values()) == 6
    for r in res["records"]:
        assert r["象限"] in {"保持区", "机会区", "低优先级区", "改进区"}


def test_advice_and_likert():
    res = isa_quadrants(source="figure")
    adv = quadrant_advice(res["records"][0])
    assert adv["title"] and adv["body"]
    assert sentiment_to_likert({"外观": -1.0, "内饰": 0.0, "空间": 1.0}) == {
        "外观": 1.0, "内饰": 3.0, "空间": 5.0}


# ---------------- audit ----------------
def test_audit_contains_documented_findings():
    findings = run_audit()
    ids = {f.id for f in findings}
    for expected in ["AHP_WEIGHT_MISMATCH", "DEMATEL_WEIGHT_MISMATCH", "DEMATEL_L5",
                     "Q_NOT_DERIVABLE_FROM_S_R", "SENSITIVITY_VS_RESULT",
                     "ISA_SATISFACTION_CONFLICT", "COMBINED_WEIGHT_OK",
                     "LIVE_VS_PAPER", "PLTS_PROB_SUM"]:
        assert expected in ids
    assert len(findings) >= 9


def test_audit_statuses_are_meaningful():
    findings = run_audit()
    by_id = {f.id: f for f in findings}
    # 整改后：四项可修复偏差 → 已定位并修复（resolved），默认不出现无法解释的 fail
    for fid in ["Q_NOT_DERIVABLE_FROM_S_R", "AHP_WEIGHT_MISMATCH",
                "DEMATEL_WEIGHT_MISMATCH", "SENSITIVITY_VS_RESULT"]:
        assert by_id[fid].status == "resolved"
    # 可复现项必须标绿
    assert by_id["COMBINED_WEIGHT_OK"].status == "pass"
    assert all(f.status != "fail" for f in findings)
    # 每条 finding 都带数字
    assert all(isinstance(f.numbers, dict) and f.numbers for f in findings)


def test_audit_summary_counts():
    s = audit_summary(run_audit())
    assert s["total"] == s["pass"] + s["resolved"] + s["warn"] + s["fail"]
    assert s["unexplained"] == 0
    assert s["resolved"] >= 4
    assert s["high"] >= 3


def test_audit_accepts_custom_scenario():
    findings = run_audit(Scenario(lam=0.2, ideal_strategy="sentiment"))
    assert len(findings) >= 9
