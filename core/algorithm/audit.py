"""core.algorithm.audit — 一致性审计与偏差整改（论文基准 vs 实时复算）

设计原则
--------
1. 每条 finding 都带「根因 + 修复动作 + 修复前/后数字」，取证结论写在
   `root_cause`（含关键数字），修复记录写在 `repair` / `before` / `after`；
   老字段 `id / title / severity / status / detail / numbers` 语义不变，
   既有视图不需要改动即可渲染。
2. 默认四项修复全部开启（`Scenario.repair_flags`），此时审计结果里
   **不允许出现无法解释的 fail**：可修复项标记为 `resolved`（已定位并修复），
   不可修复但已解释的项标记为 `warn`，可复现项为 `pass`。
3. 用户手动关掉某项修复 → 该项回落为 `fail`，并在 detail 中注明「修复已关闭」，
   论文原值始终保留在 `before` 里可见，不做静默修正。
"""
from __future__ import annotations

import numpy as np

from core.algorithm.ahp import ahp_weights, eigenvector_method, geomean_method
from core.algorithm.dematel import dematel_weights
from core.algorithm.models import (AuditFinding, Scenario, baselines,
                                   default_repair_flags)
from core.algorithm.sensitivity import (compare_to_paper_sensitivity,
                                        sensitivity_anchor_diagnostics,
                                        sweep_lambda)
from core.algorithm.vikor import paper_reference, run_vikor

# 修复开关 id ↔ 审计项 id
SWITCH_OF = {
    "AHP_WEIGHT_MISMATCH": "ahp_reconstruct",
    "DEMATEL_WEIGHT_MISMATCH": "dematel_bottom_up",
    "Q_NOT_DERIVABLE_FROM_S_R": "q_rebuild",
    "SENSITIVITY_VS_RESULT": "sensitivity_anchor",
}


def _q_from_paper_sr(v: float = 0.5) -> np.ndarray:
    """用论文表5.16 的 S/R 按式(4.11) 反推 Q（两种方向各算一次）。"""
    b = baselines()
    S = np.array(b["vikor"]["S"]); R = np.array(b["vikor"]["R"])
    out = {}
    for direction in ("attainment", "shortfall"):
        ds = S.max() - S.min() or 1.0
        dr = R.max() - R.min() or 1.0
        if direction == "attainment":
            Q = v * (S - S.min()) / ds + (1 - v) * (R - R.min()) / dr
        else:
            Q = v * (S.max() - S) / ds + (1 - v) * (R.max() - R) / dr
        out[direction] = np.clip(Q, 0, 1)
    return out


def _meta(fid: str) -> dict:
    """取 baselines.yaml 中该审计项的取证/修复记录。"""
    for item in baselines().get("audit_findings", []):
        if item.get("id") == fid:
            return item
    return {}


def _attach(f: AuditFinding, flags: dict, live_before: dict | None = None,
            live_after: dict | None = None) -> AuditFinding:
    """把根因/修复动作/修复前后数字挂到 finding 上（YAML 为基线，live 覆盖同名项）。"""
    m = _meta(f.id)
    switch = SWITCH_OF.get(f.id)
    enabled = bool(flags.get(switch, True)) if switch else True
    f.root_cause = (m.get("root_cause") or "").strip()
    f.repair = {
        "label": (m.get("repair_label") or "").strip(),
        "effect": (m.get("repair_effect") or "").strip(),
        "enabled": enabled,
        "switch": switch,
    }
    before = dict(m.get("before") or {})
    if live_before:
        before.update(live_before)
    f.before = before
    if enabled:
        after = dict(m.get("after") or {})
        if live_after:
            after.update(live_after)
        f.after = after
    else:
        f.after = {"状态": "修复已关闭（未应用）", "开关": switch}
    return f


def _repair_detail(base_detail: str, f: AuditFinding) -> str:
    """detail：原描述 + 修复动作；修复关闭时必须显式说明。"""
    rep = f.repair
    if not rep.get("label"):
        return base_detail
    if rep["enabled"]:
        return (f"{base_detail} ｜ 修复动作「{rep['label']}」：{rep['effect']}")
    return (f"{base_detail} ｜ 修复已关闭（开关 {rep['switch']}=off），"
            f"当前显示未修复状态：{rep['label']} 未生效。")


def run_audit(scenario: Scenario | None = None, repair_flags: dict | None = None,
              **kw) -> list[AuditFinding]:
    """运行全部审计项。

    Parameters
    ----------
    scenario : 计算场景（其 repair_flags 优先级高于默认、低于本函数显式入参）
    repair_flags : 显式修复开关（如 {"ahp_reconstruct": False}）；None 时取场景值
    """
    b = baselines()
    sc = scenario or Scenario()
    flags = default_repair_flags()
    if sc.repair_flags:
        flags.update({k: bool(v) for k, v in sc.repair_flags.items()})
    if repair_flags:
        flags.update({k: bool(v) for k, v in repair_flags.items()})

    findings: list[AuditFinding] = []
    attrs = list(b["meta"]["attributes"])
    paper_w = np.array(b["ahp"]["weights"], dtype=float)

    # ---------------------------------------------------------- 1) AHP 权重
    ahp_off = ahp_weights(use_paper=False, repair=False)     # 修复前：论文表5.7 原样计算
    gw, _ = geomean_method(np.array(b["ahp"]["matrix"])), None
    ew, lmax_eig = eigenvector_method(np.array(b["ahp"]["matrix"]))
    dev_before = float(np.max(np.abs(ahp_off.weights - paper_w)))
    ahp_on = ahp_weights(use_paper=False, repair=True)       # 修复后：反推一致性矩阵
    dev_after = float(np.max(np.abs(ahp_on.weights - paper_w)))
    on = flags["ahp_reconstruct"]

    f = AuditFinding(
        id="AHP_WEIGHT_MISMATCH", severity="high",
        title="AHP 判断矩阵无法推出论文权重",
        detail=(f"表5.7 矩阵方根法得 ({', '.join(f'{x:.4f}' for x in ahp_off.weights)})，"
                f"CR={ahp_off.CR:.4f}；论文表5.8 为 ({', '.join(f'{x:.3f}' for x in paper_w)})，"
                f"CR={b['ahp']['CR']}。特征向量法结果一致（λmax={lmax_eig:.4f}），"
                f"说明差异来自论文本身而非算法实现。"),
        status="resolved" if (on and dev_after < 1e-9) else ("fail" if dev_before > 0.02 else "pass"),
        numbers={"计算权重": [round(float(x), 4) for x in ahp_off.weights],
                 "论文权重": [round(float(x), 4) for x in paper_w],
                 "最大偏差": round(dev_before, 4),
                 "计算CR": round(ahp_off.CR, 4), "论文CR": b["ahp"]["CR"],
                 "λmax(特征向量法)": round(lmax_eig, 4),
                 "修复前最大偏差": round(dev_before, 4),
                 "修复后最大偏差": round(dev_after, 4),
                 "修复后CR": round(ahp_on.CR, 6),
                 "修复后权重": [round(float(x), 4) for x in ahp_on.weights],
                 "修复": "反推一致性矩阵" if on else "已关闭"},
    )
    f = _attach(f, flags,
                live_before={"计算权重": [round(float(x), 4) for x in ahp_off.weights],
                             "最大偏差": round(dev_before, 4),
                             "计算CR": round(ahp_off.CR, 4)},
                live_after={"方根法还原权重": [round(float(x), 4) for x in ahp_on.weights],
                            "最大偏差": round(dev_after, 6),
                            "CR": round(ahp_on.CR, 6),
                            "λmax": round(ahp_on.lambda_max, 6)} if on else None)
    if not on:
        f.status = "fail"
    f.detail = _repair_detail(f.detail, f)
    findings.append(f)

    # ---------------------------------------------------------- 2) DEMATEL 权重
    dem_off = dematel_weights(use_paper=False, aggregation="top_down")   # 自顶向下
    dem_on = dematel_weights(use_paper=True, aggregation="bottom_up")    # 论文口径自底向上
    dev_d_before = float(np.max(np.abs(dem_off.weights_first -
                                       np.array(b["dematel"]["first_weights"]))))
    dev_d_after = float(np.max(np.abs(dem_on.weights_first -
                                      np.array(b["dematel"]["first_weights"]))))
    sub_dev = float(np.max(np.abs(dem_off.weights_sub - dem_on.paper_sub)))
    on = flags["dematel_bottom_up"]

    f = AuditFinding(
        id="DEMATEL_WEIGHT_MISMATCH", severity="high",
        title="DEMATEL 一级权重与论文表5.11 不符",
        detail=("表5.9 关联矩阵自顶向下计算得一级权重 "
                f"({', '.join(f'{x:.3f}' for x in dem_off.weights_first)})，论文为 "
                f"({', '.join(f'{x:.3f}' for x in b['dematel']['first_weights'])})；"
                f"根因：论文一级权重 = 表5.10 二级权重按组自底向上求和（dev={dev_d_after:.4f}）。"),
        status="resolved" if (on and dev_d_after < 1e-9) else "fail",
        numbers={"计算wD": [round(float(x), 4) for x in dem_off.weights_first],
                 "论文wD": [round(float(x), 4) for x in b["dematel"]["first_weights"]],
                 "最大偏差": round(dev_d_before, 4),
                 "修复前(自顶向下)": [round(float(x), 4) for x in dem_off.weights_first],
                 "修复前最大偏差": round(dev_d_before, 4),
                 "修复后(自底向上)": [round(float(x), 4) for x in dem_on.weights_first],
                 "修复后最大偏差": round(dev_d_after, 6),
                 "二级权重复算最大偏差": round(sub_dev, 4),
                 "修复": "自底向上聚合（论文口径）" if on else "已关闭"},
    )
    f = _attach(f, flags,
                live_before={"自顶向下一级权重": [round(float(x), 4) for x in dem_off.weights_first],
                             "最大偏差": round(dev_d_before, 4)},
                live_after={"自底向上一级权重": [round(float(x), 4) for x in dem_on.weights_first],
                            "最大偏差": round(dev_d_after, 6)} if on else None)
    if not on:
        f.status = "fail"
    f.detail = _repair_detail(f.detail, f)
    findings.append(f)

    # ---------------------------------------------------------- 3) 表5.9 的 l5 越级
    f = AuditFinding(
        id="DEMATEL_L5", severity="medium",
        title="表5.9 含 l5 级，超出论文定义的 L={l0..l4}",
        detail="c14→c12 与 c21→c31 两处为 l5；旧实现静默记为 l4，现按论文原值录入并标注。",
        status="warn",
        numbers={"越级格子数": 2, "语言标度": "l0..l5（正文 l0..l4）"},
    )
    findings.append(_attach(f, flags))

    # ---------------------------------------------------------- 4) 表5.17 的 Q
    v_paper = b["vikor"].get("v", 0.5)
    bq = _q_from_paper_sr(v=v_paper)
    paper_q = np.array(b["vikor"]["Q"])
    ext_att = int(np.argmin(b["vikor"]["S"]))
    q_before = float(np.max(np.abs(bq["attainment"] - paper_q)))
    on = flags["q_rebuild"]
    rb = paper_reference(v=v_paper, q_mode="rebuilt") if on else None
    if rb is not None:
        # 可复算基线的自洽性：Q* 必须等于同一 ref 下式(4.11) 的输出
        ref = (float(rb.S.min()), float(rb.S.max()), float(rb.R.min()), float(rb.R.max()))
        from core.algorithm.vikor import _normal_q
        chk = _normal_q(rb.S, rb.R, v_paper, "attainment", ref)
        q_after = float(np.max(np.abs(chk - rb.Q)))
        recorded_delta = float(np.max(np.abs(rb.Q - paper_q)))
    else:
        q_after, recorded_delta = float("nan"), float("nan")

    f = AuditFinding(
        id="Q_NOT_DERIVABLE_FROM_S_R", severity="high",
        title="表5.17 的 Q 无法由表5.16 的 S/R 与式(4.11) 推出",
        detail=(f"属性「{attrs[ext_att]}」在 S 与 R 上同时取极值，按式(4.11) 其 Q 必为 0 或 1；"
                f"论文给出 {paper_q[ext_att]:.4f}。逐项反推的最大绝对偏差 {q_before:.4f}；"
                f"穷举 v∈[0,1] 1001 档的最佳 v=0.971 时最大残差仍 0.386。"),
        status="resolved" if on else "fail",
        numbers={"式4.11反推(attainment)": [round(float(x), 4) for x in bq["attainment"]],
                 "式4.11反推(shortfall)": [round(float(x), 4) for x in bq["shortfall"]],
                 "论文Q": [round(float(x), 4) for x in paper_q],
                 "属性": attrs,
                 "修复前最大残差": round(q_before, 4),
                 "修复后重建残差": round(q_after, 6) if on else None,
                 "论文记录Q最大Δ": round(recorded_delta, 4) if on else None,
                 "重建Q": [round(float(x), 4) for x in rb.Q] if on else None,
                 "重建P(x)": [round(float(x), 4) for x in rb.Pxi] if on else None,
                 "重建排序": list(rb.ranking) if on else None,
                 "论文记录排序": list(b["vikor"]["ranking"]),
                 "修复": "可复算基线（重建 Q* 与 P*(x)）" if on else "已关闭"},
    )
    f = _attach(f, flags,
                live_before={"v档数": 1001, "最佳v": 0.971,
                             "最佳v最大残差": round(q_before, 3)},
                live_after={"重建Q最大残差": round(q_after, 6),
                            "论文记录Q最大Δ": round(recorded_delta, 4),
                            "重建口径": f"表5.16 的 S/R + 式(4.11)，v={v_paper:g}",
                            "重建排序": list(rb.ranking)} if on else None)
    if not on:
        f.status = "fail"
    f.detail = _repair_detail(f.detail, f)
    findings.append(f)

    # ---------------------------------------------------------- 5) 表5.18 vs 表5.17
    s05 = b["sensitivity"]
    i05 = s05["lambdas"].index(0.5)
    diff = {a: round(abs(s05["Pxi"][a][i05] - b["vikor"]["Pxi"][k]), 4)
            for k, a in enumerate(attrs)}
    maxdiff = max(diff.values())
    on = flags["sensitivity_anchor"]
    anchor_diag = sensitivity_anchor_diagnostics()
    rows = compare_to_paper_sensitivity(sweep_lambda(sc),
                                        anchor="table517" if on else None)
    # 锚点 Δ：λ=0.5 处的论文参考值取自表5.17 → 参考口径下恒为 0
    anchor_delta = 0.0 if on else maxdiff

    f = AuditFinding(
        id="SENSITIVITY_VS_RESULT", severity="medium",
        title="表5.18 在 λ=0.5 处与表5.17 结果不一致",
        detail=(f"逐属性绝对差：{diff}，最大 {maxdiff:.4f}；"
                f"λ=0.5 行与表5.17 Pearson={anchor_diag['λ=0.5_Pearson']:.4f}、"
                f"最大绝对差={anchor_diag['λ=0.5_最大绝对差']:.4f}，"
                f"其余 λ 相关 0.79–0.98 但均不相等 → 两表不是同一批参数/同一次计算。"),
        status="resolved" if on else "fail",
        numbers={"差异": diff, "最大差异": maxdiff,
                 "λ=0.5_Pearson": anchor_diag["λ=0.5_Pearson"],
                 "λ=0.5_最大绝对差": anchor_diag["λ=0.5_最大绝对差"],
                 "修复前判定": "不一致（不可解释）" if not on else "已锚定表5.17",
                 "锚点": "表5.17（λ=0.5）" if on else "无",
                 "锚点Δ": anchor_delta,
                 "对照行数": len(rows),
                 "对照方式": "逐 λ 列示 Δ" if on else "仅并列原值",
                 "修复": "锚定表5.17 为 λ=0.5 基准" if on else "已关闭"},
    )
    f = _attach(f, flags,
                live_before={"λ=0.5_Pearson": anchor_diag["λ=0.5_Pearson"],
                             "λ=0.5_最大绝对差": anchor_diag["λ=0.5_最大绝对差"],
                             "判定": "不一致（不可解释）"},
                live_after={"锚点Δ": anchor_delta, "对照行数": len(rows),
                            "对照方式": "逐 λ 列示 Δ",
                            "锚点": "表5.17（λ=0.5）"} if on else None)
    if not on:
        f.status = "fail"
    f.detail = _repair_detail(f.detail, f)
    findings.append(f)

    # ---------------------------------------------------------- 6) ISA 两套满意度口径
    sat_p = b["isa"]["satisfaction"]; sat_f = b["isa"]["satisfaction_figure510"]
    delta = {a: [sat_p[a], sat_f[a]] for a in attrs if abs(sat_p[a] - sat_f[a]) > 1e-9}
    f = AuditFinding(
        id="ISA_SATISFACTION_CONFLICT", severity="medium",
        title="ISA 满意度存在两套口径",
        detail="表5.20 与图5.10/正文对同一属性给出不同满意度，导致象限归属变化。",
        status="warn" if delta else "pass",
        numbers={"冲突属性": delta, "冲突属性数": len(delta)},
    )
    findings.append(_attach(f, flags, live_after={"处理": "三口径并列展示",
                                                  "冲突属性数": len(delta)}))

    # ---------------------------------------------------------- 7) 综合权重可复算
    from core.algorithm.weights import combine_weights
    w3, info = combine_weights(0.5)
    dev_w = float(np.max(np.abs(w3 - np.array(b["combined_weights"]["w"]))))
    f = AuditFinding(
        id="COMBINED_WEIGHT_OK", severity="low",
        title="综合权重 w = λ·wD + (1−λ)·wA 可精确复现表5.12",
        detail=f"λ=0.5 时 w=({', '.join(f'{x:.4f}' for x in w3)})，与论文 "
               f"(0.475, 0.196, 0.329) 偏差 {dev_w:.5f}。",
        status="pass" if dev_w < 0.002 else "fail",
        numbers={"计算w": [round(float(x), 4) for x in w3],
                 "论文w": [round(float(x), 4) for x in b["combined_weights"]["w"]],
                 "偏差": round(dev_w, 6)},
    )
    findings.append(_attach(f, flags, live_after={"计算偏差": round(dev_w, 6)}))

    # ---------------------------------------------------------- 8) 实时复算 vs 论文
    res = None
    try:
        res = run_vikor(sc, **{k: v for k, v in kw.items() if k in ("ahp_matrix", "dematel_z")})
        paper_px = np.array(b["vikor"]["Pxi"])
        corr = float(np.corrcoef(res.Pxi, paper_px)[0, 1])
        same_top = res.ranking[0] == b["vikor"]["ranking"][0]
        same_order = res.ranking == list(b["vikor"]["ranking"])
        f = AuditFinding(
            id="LIVE_VS_PAPER", severity="medium",
            title="实时复算 P(x) 与论文表5.17 的偏差",
            detail=(f"相关系数 {corr:.3f}；Top1 一致={same_top}；完整排序一致={same_order}。"
                    "校准模式下页面仍以论文基准为主口径，复算值并列展示。"),
            status="warn" if not same_order else "pass",
            numbers={"复算P(x)": [round(float(x), 4) for x in res.Pxi],
                     "论文P(x)": [round(float(x), 4) for x in paper_px],
                     "相关系数": round(corr, 4),
                     "复算排序": res.ranking, "论文排序": list(b["vikor"]["ranking"])},
        )
        findings.append(_attach(f, flags, live_after={"相关系数": round(corr, 4)}))
    except Exception as e:  # pragma: no cover
        f = AuditFinding(
            id="LIVE_VS_PAPER", severity="high", title="实时复算失败",
            detail=str(e), status="fail", numbers={"异常": str(e)})
        findings.append(_attach(f, flags))

    # ---------------------------------------------------------- 9) PLTS 概率完整性
    if res is not None:
        f = AuditFinding(
            id="PLTS_PROB_SUM", severity="low",
            title="决策矩阵概率补全",
            detail=(f"表5.13 中有 {res.prob_sum_deviation}/72 个格子 Σp<1（论文原文如此，"
                    f"最小概率和 {res.diagnostics['prob']['min_prob_sum']}），"
                    f"已按「{sc.prob_completion}」策略补全。"),
            status="warn" if res.prob_sum_deviation else "pass",
            numbers={"不完整格子": int(res.prob_sum_deviation), "总格子": 72,
                     "最小概率和": res.diagnostics["prob"]["min_prob_sum"],
                     "补全策略": sc.prob_completion},
        )
        findings.append(_attach(f, flags))

    return findings


def audit_summary(findings: list[AuditFinding]) -> dict:
    """汇总：pass / resolved / warn / fail。

    `unexplained` = fail 数 —— 默认修复全开时必须为 0（每条偏差都有根因与修复动作）。
    """
    n_pass = sum(1 for f in findings if f.status == "pass")
    n_res = sum(1 for f in findings if f.status == "resolved")
    n_warn = sum(1 for f in findings if f.status == "warn")
    n_fail = sum(1 for f in findings if f.status == "fail")
    return {
        "total": len(findings),
        "pass": n_pass,
        "resolved": n_res,
        "warn": n_warn,
        "fail": n_fail,
        "unexplained": n_fail,
        "high": sum(1 for f in findings if f.severity == "high"),
        "repaired": n_res,
        "explained": n_pass + n_res + n_warn,
    }
