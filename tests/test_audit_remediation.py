"""偏差整改回归测试（四项审计偏差的根因 → 修复动作 → 修复前后数字）"""
import numpy as np
import pytest

from core.algorithm.ahp import ahp_weights, reconstruct_consistent
from core.algorithm.audit import audit_summary, run_audit
from core.algorithm.dematel import dematel_weights
from core.algorithm.models import Scenario, baselines, default_repair_flags
from core.algorithm.sensitivity import (compare_to_paper_sensitivity,
                                        sensitivity_anchor_diagnostics,
                                        sweep_lambda)
from core.algorithm.vikor import _normal_q, paper_reference

B = baselines()
PAPER_AHP = np.array(B["ahp"]["weights"], dtype=float)
PAPER_FIRST = np.array(B["dematel"]["first_weights"], dtype=float)


# ---------------- 1) AHP：反推一致性矩阵 ----------------
def test_reconstruct_consistent_roundtrip():
    A = np.array(reconstruct_consistent(PAPER_AHP))
    assert A.shape == (3, 3)
    # 互反性 a_ij * a_ji = 1
    assert np.allclose(A * A.T, 1.0, atol=1e-12)
    # 方根法精确还原论文表5.8 权重，且完全一致（CR≈0）
    res = ahp_weights(A)
    assert np.max(np.abs(res.weights - PAPER_AHP)) < 1e-12
    assert res.CR == pytest.approx(0.0, abs=1e-9)
    assert res.deviation < 1e-12


def test_ahp_repair_flag_restores_paper_weights():
    before = ahp_weights(use_paper=False, repair=False)
    after = ahp_weights(use_paper=False, repair=True)
    assert np.max(np.abs(before.weights - PAPER_AHP)) > 0.2      # 修复前偏差 0.2084
    assert np.max(np.abs(after.weights - PAPER_AHP)) < 1e-12     # 修复后偏差 0
    assert after.CR == pytest.approx(0.0, abs=1e-9)
    assert after.reconstructed is True and before.reconstructed is False
    # 自定义矩阵不应触发反推（只对论文表5.7 生效）
    custom = np.array([[1, 3, 5], [1 / 3, 1, 2], [1 / 5, 1 / 2, 1]], dtype=float)
    assert ahp_weights(custom, repair=True).reconstructed is False


# ---------------- 2) DEMATEL：自底向上聚合 ----------------
def test_dematel_bottom_up_reproduces_table_5_11():
    res = dematel_weights(use_paper=True, aggregation="bottom_up")
    assert res.aggregation == "bottom_up"
    assert np.max(np.abs(res.weights_first - PAPER_FIRST)) < 1e-6
    assert res.weights_first.sum() == pytest.approx(1.0)


def test_dematel_top_down_still_reproduces_old_behavior():
    res = dematel_weights(use_paper=False, aggregation="top_down")
    assert res.aggregation == "top_down"
    # 自顶向下与论文一级权重的偏差仍应被如实计算（不静默修正）
    assert np.max(np.abs(res.weights_first - PAPER_FIRST)) > 0.2


# ---------------- 3) Q：可复算基线 ----------------
def test_paper_reference_rebuilt_satisfies_formula_4_11():
    ref = paper_reference(q_mode="rebuilt")
    S, R = ref.S, ref.R
    v = ref.diagnostics["v"]
    lo = (float(S.min()), float(S.max()), float(R.min()), float(R.max()))
    expected = _normal_q(S, R, v, "attainment", lo)
    assert np.allclose(ref.Q, expected, atol=1e-12)
    # 论文记录值保留用于 Δ 对照
    rec = ref.diagnostics["recorded"]
    assert rec["Q"] == [round(x, 4) for x in B["vikor"]["Q"]]
    assert ref.diagnostics["delta_vs_recorded"]["max|Q−Q记录|"] == pytest.approx(0.4854, abs=1e-4)
    # 可能度矩阵自洽：P(x) = 行和，且与重建 Q 单调一致
    assert np.allclose(ref.Pxi, ref.P.sum(axis=1))
    assert np.isfinite(ref.Pxi).all()


def test_paper_reference_recorded_is_default_and_unchanged():
    ref = paper_reference()
    assert ref.diagnostics["q_mode"] == "recorded"
    assert np.allclose(ref.Q, B["vikor"]["Q"])
    assert np.allclose(ref.Pxi, B["vikor"]["Pxi"])
    assert ref.ranking == list(B["vikor"]["ranking"])
    with pytest.raises(ValueError):
        paper_reference(q_mode="nope")


# ---------------- 4) 敏感性：锚定表5.17 ----------------
def test_sensitivity_anchor_columns_and_diagnostics():
    rows = compare_to_paper_sensitivity(sweep_lambda(), anchor="table517")
    assert "Δλ=0.5" in rows[0] and "锚点说明" in rows[0]
    assert rows[0]["论文λ=0.5"] == pytest.approx(B["vikor"]["Pxi"][0])
    diag = sensitivity_anchor_diagnostics()
    assert diag["λ=0.5_Pearson"] == pytest.approx(0.9667, abs=1e-4)
    assert diag["λ=0.5_最大绝对差"] == pytest.approx(0.7905, abs=1e-4)
    assert diag["锚点Δ"] == 0.0
    # 关闭锚点时退回原行为：无 Δ 列，论文值取表5.18
    raw = compare_to_paper_sensitivity(sweep_lambda(), anchor=None)
    assert "Δλ=0.5" not in raw[0]
    assert raw[0]["论文λ=0.5"] == pytest.approx(B["sensitivity"]["Pxi"]["外观"][3])


# ---------------- 5) 审计：默认无无法解释的 fail ----------------
def test_default_audit_has_no_unexplained_fail():
    findings = run_audit()
    assert [f.id for f in findings if f.status == "fail"] == []
    s = audit_summary(findings)
    assert s["unexplained"] == 0
    assert s["fail"] == 0
    assert s["resolved"] >= 2
    assert s["total"] == s["pass"] + s["resolved"] + s["warn"] + s["fail"]


def test_every_finding_carries_root_cause_and_repair_record():
    for f in run_audit():
        assert f.root_cause.strip(), f.id
        assert f.repair.get("label"), f.id
        assert f.repair.get("effect"), f.id
        assert isinstance(f.before, dict) and f.before, f.id
        assert isinstance(f.after, dict) and f.after, f.id
        assert f.numbers, f.id


def test_turning_off_repair_falls_back_to_fail():
    findings = run_audit(repair_flags={"ahp_reconstruct": False})
    by = {f.id: f for f in findings}
    assert by["AHP_WEIGHT_MISMATCH"].status == "fail"
    assert "修复已关闭" in by["AHP_WEIGHT_MISMATCH"].detail
    assert by["AHP_WEIGHT_MISMATCH"].repair["enabled"] is False
    # 论文原值仍保留在 before 中可见
    assert by["AHP_WEIGHT_MISMATCH"].before["论文权重"] == [0.434, 0.187, 0.379]
    assert audit_summary(findings)["unexplained"] == 1
    # 其余三项仍为已解析
    assert by["Q_NOT_DERIVABLE_FROM_S_R"].status == "resolved"


def test_scenario_repair_flags_roundtrip_and_default():
    sc = Scenario()
    assert sc.repair_flags == default_repair_flags()
    assert set(sc.repair_flags) == {"ahp_reconstruct", "dematel_bottom_up",
                                    "q_rebuild", "sensitivity_anchor"}
    assert all(sc.repair_flags.values())
    # model_dump → Scenario(...) 的 round-trip（state.store 以 model_dump 为默认值）
    clone = Scenario(**sc.model_dump())
    assert clone.repair_flags == sc.repair_flags
    off = Scenario(repair_flags={"ahp_reconstruct": False})
    assert Scenario(**off.model_dump()).repair_flags["ahp_reconstruct"] is False
    # 场景开关直接影响审计状态
    assert {f.id: f.status for f in run_audit(off)}["AHP_WEIGHT_MISMATCH"] == "fail"
    # 显式入参优先级高于场景
    assert {f.id: f.status for f in run_audit(off, repair_flags={"ahp_reconstruct": True})}[
        "AHP_WEIGHT_MISMATCH"] == "resolved"


def test_summary_keys_backward_compatible():
    s = audit_summary(run_audit())
    for key in ("total", "pass", "warn", "fail", "high"):
        assert key in s
    assert isinstance(s["unexplained"], int) and isinstance(s["resolved"], int)
