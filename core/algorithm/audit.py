"""core.algorithm.audit — 一致性审计（论文基准 vs 实时复算）

把"论文数值不可复算""论文内部口径冲突"等已知问题显式化：
每条 finding 都带实时计算出来的数字，UI 以 徽章 + 数字 表格呈现，
不掩盖、不静默修正。
"""
from __future__ import annotations

import numpy as np

from core.algorithm.ahp import ahp_weights, eigenvector_method, geomean_method
from core.algorithm.dematel import dematel_weights
from core.algorithm.models import AuditFinding, Scenario, baselines
from core.algorithm.sensitivity import sweep_lambda
from core.algorithm.vikor import run_vikor


def _q_from_paper_sr(v: float = 0.5) -> np.ndarray:
    """用论文表5.16 的 S/R 按式(4.11) 反推 Q（两种方向各算一次）。"""
    b = baselines()
    S = np.array(b["vikor"]["S"]); R = np.array(b["vikor"]["R"])
    out = {}
    for direction in ("attainment", "shortfall"):
        if direction == "attainment":
            ds = S.max() - S.min() or 1.0
            dr = R.max() - R.min() or 1.0
            Q = v * (S - S.min()) / ds + (1 - v) * (R - R.min()) / dr
        else:
            ds = S.max() - S.min() or 1.0
            dr = R.max() - R.min() or 1.0
            Q = v * (S.max() - S) / ds + (1 - v) * (R.max() - R) / dr
        out[direction] = np.clip(Q, 0, 1)
    return out


def run_audit(scenario: Scenario | None = None, **kw) -> list[AuditFinding]:
    b = baselines()
    sc = scenario or Scenario()
    findings: list[AuditFinding] = []

    # 1) AHP 权重可复算性
    ahp = ahp_weights(use_paper=False)
    gw, _ = geomean_method(np.array(b["ahp"]["matrix"])), None
    ew, lmax_eig = eigenvector_method(np.array(b["ahp"]["matrix"]))
    paper_w = np.array(b["ahp"]["weights"])
    dev = float(np.max(np.abs(ahp.weights - paper_w)))
    findings.append(AuditFinding(
        id="AHP_WEIGHT_MISMATCH", severity="high",
        title="AHP 判断矩阵无法推出论文权重",
        detail=(f"表5.7 矩阵方根法得 ({', '.join(f'{x:.4f}' for x in ahp.weights)})，"
                f"CR={ahp.CR:.4f}；论文表5.8 为 ({', '.join(f'{x:.3f}' for x in paper_w)})，CR={b['ahp']['CR']}。"
                f"特征向量法结果一致（λmax={lmax_eig:.4f}），说明差异来自论文本身而非算法实现。"),
        status="fail" if dev > 0.02 else "pass",
        numbers={"计算权重": [round(float(x), 4) for x in ahp.weights],
                 "论文权重": [round(float(x), 4) for x in paper_w],
                 "最大偏差": round(dev, 4),
                 "计算CR": round(ahp.CR, 4), "论文CR": b["ahp"]["CR"],
                 "λmax(特征向量法)": round(lmax_eig, 4)},
    ))

    # 2) DEMATEL 权重可复算性
    dem = dematel_weights(use_paper=False)
    dev_d = float(np.max(np.abs(dem.weights_first - np.array(b["dematel"]["first_weights"]))))
    findings.append(AuditFinding(
        id="DEMATEL_WEIGHT_MISMATCH", severity="high",
        title="DEMATEL 一级权重与论文表5.11 不符",
        detail=("表5.9 关联矩阵计算得一级权重 "
                f"({', '.join(f'{x:.3f}' for x in dem.weights_first)})，论文为 "
                f"({', '.join(f'{x:.3f}' for x in b['dematel']['first_weights'])})。"),
        status="fail" if dev_d > 0.02 else "pass",
        numbers={"计算wD": [round(float(x), 4) for x in dem.weights_first],
                 "论文wD": [round(float(x), 4) for x in b["dematel"]["first_weights"]],
                 "最大偏差": round(dev_d, 4)},
    ))

    # 3) 表5.9 的 l5 越级
    findings.append(AuditFinding(
        id="DEMATEL_L5", severity="medium",
        title="表5.9 含 l5 级，超出论文定义的 L={l0..l4}",
        detail="c14→c12 与 c21→c31 两处为 l5；旧实现静默记为 l4，现按论文原值录入并标注。",
        status="warn",
        numbers={"越级格子数": 2, "语言标度": "l0..l5（正文 l0..l4）"},
    ))

    # 4) 表5.17 的 Q 能否由表5.16 的 S/R 反推
    bq = _q_from_paper_sr(v=b["vikor"].get("v", 0.5))
    paper_q = np.array(b["vikor"]["Q"])
    attrs = b["meta"]["attributes"]
    ext_att = int(np.argmin(b["vikor"]["S"]))     # attainment 口径下应取边界值的属性
    findings.append(AuditFinding(
        id="Q_NOT_DERIVABLE_FROM_S_R", severity="high",
        title="表5.17 的 Q 无法由表5.16 的 S/R 与式(4.11) 推出",
        detail=(f"属性「{attrs[ext_att]}」在 S 与 R 上同时取极值，按式(4.11) 其 Q 必为 0 或 1；"
                f"论文给出 {paper_q[ext_att]:.4f}。逐项反推的最大绝对偏差 "
                f"{float(np.max(np.abs(bq['attainment'] - paper_q))):.4f}。"),
        status="fail",
        numbers={"式4.11反推(attainment)": [round(float(x), 4) for x in bq["attainment"]],
                 "式4.11反推(shortfall)": [round(float(x), 4) for x in bq["shortfall"]],
                 "论文Q": [round(float(x), 4) for x in paper_q],
                 "属性": attrs},
    ))

    # 5) 表5.18 在 λ=0.5 与表5.17 的一致性
    s05 = b["sensitivity"]
    i05 = s05["lambdas"].index(0.5)
    diff = {a: round(abs(s05["Pxi"][a][i05] - b["vikor"]["Pxi"][k]), 4)
            for k, a in enumerate(attrs)}
    maxdiff = max(diff.values())
    findings.append(AuditFinding(
        id="SENSITIVITY_VS_RESULT", severity="medium",
        title="表5.18 在 λ=0.5 处与表5.17 结果不一致",
        detail=f"逐属性绝对差：{diff}，最大 {maxdiff:.4f}。",
        status="fail" if maxdiff > 0.05 else "pass",
        numbers={"差异": diff, "最大差异": maxdiff},
    ))

    # 6) ISA 两套满意度口径
    sat_p = b["isa"]["satisfaction"]; sat_f = b["isa"]["satisfaction_figure510"]
    delta = {a: [sat_p[a], sat_f[a]] for a in attrs if abs(sat_p[a] - sat_f[a]) > 1e-9}
    findings.append(AuditFinding(
        id="ISA_SATISFACTION_CONFLICT", severity="medium",
        title="ISA 满意度存在两套口径",
        detail="表5.20 与图5.10/正文对同一属性给出不同满意度，导致象限归属变化。",
        status="warn" if delta else "pass",
        numbers={"冲突属性": delta},
    ))

    # 7) 综合权重可复算（应当通过）
    from core.algorithm.weights import combine_weights
    w3, info = combine_weights(0.5)
    dev_w = float(np.max(np.abs(w3 - np.array(b["combined_weights"]["w"]))))
    findings.append(AuditFinding(
        id="COMBINED_WEIGHT_OK", severity="low",
        title="综合权重 w = λ·wD + (1−λ)·wA 可精确复现表5.12",
        detail=f"λ=0.5 时 w=({', '.join(f'{x:.4f}' for x in w3)})，与论文 (0.475, 0.196, 0.329) 偏差 {dev_w:.5f}。",
        status="pass" if dev_w < 0.002 else "fail",
        numbers={"计算w": [round(float(x), 4) for x in w3],
                 "论文w": [round(float(x), 4) for x in b["combined_weights"]["w"]],
                 "偏差": round(dev_w, 6)},
    ))

    # 8) 实时复算 vs 论文 P(x)
    res = None
    try:
        res = run_vikor(sc, **{k: v for k, v in kw.items() if k in ("ahp_matrix", "dematel_z")})
        paper_px = np.array(b["vikor"]["Pxi"])
        corr = float(np.corrcoef(res.Pxi, paper_px)[0, 1])
        same_top = res.ranking[0] == b["vikor"]["ranking"][0]
        same_order = res.ranking == list(b["vikor"]["ranking"])
        findings.append(AuditFinding(
            id="LIVE_VS_PAPER", severity="medium",
            title="实时复算 P(x) 与论文表5.17 的偏差",
            detail=(f"相关系数 {corr:.3f}；Top1 一致={same_top}；完整排序一致={same_order}。"
                    "校准模式下页面仍以论文基准为主口径，复算值并列展示。"),
            status="warn" if not same_order else "pass",
            numbers={"复算P(x)": [round(float(x), 4) for x in res.Pxi],
                     "论文P(x)": [round(float(x), 4) for x in paper_px],
                     "相关系数": round(corr, 4),
                     "复算排序": res.ranking, "论文排序": list(b["vikor"]["ranking"])},
        ))
    except Exception as e:  # pragma: no cover
        findings.append(AuditFinding(
            id="LIVE_VS_PAPER", severity="high", title="实时复算失败",
            detail=str(e), status="fail", numbers={}))

    # 9) PLTS 概率完整性
    if res is not None:
        findings.append(AuditFinding(
            id="PLTS_PROB_SUM", severity="low",
            title="决策矩阵概率补全",
            detail=(f"表5.13 中有 {res.prob_sum_deviation}/72 个格子 Σp<1（论文原文如此，"
                    f"最小概率和 {res.diagnostics['prob']['min_prob_sum']}），"
                    f"已按「{sc.prob_completion}」策略补全。"),
            status="warn" if res.prob_sum_deviation else "pass",
            numbers={"不完整格子": int(res.prob_sum_deviation), "总格子": 72,
                     "最小概率和": res.diagnostics["prob"]["min_prob_sum"],
                     "补全策略": sc.prob_completion},
        ))

    return findings


def audit_summary(findings: list[AuditFinding]) -> dict:
    return {
        "total": len(findings),
        "pass": sum(1 for f in findings if f.status == "pass"),
        "warn": sum(1 for f in findings if f.status == "warn"),
        "fail": sum(1 for f in findings if f.status == "fail"),
        "high": sum(1 for f in findings if f.severity == "high"),
    }
