"""矩阵编辑器（ui/views/simulator.py）交互测试：AppTest 双主题渲染 + 编辑/预设/撤销/导入。"""
from __future__ import annotations

import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

from core.algorithm.validate import ahp_diagnostics

VIEW = "PLTS-VIKOR 模拟器"


def _run(theme: str | None = None) -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=180)
    at.session_state["dsh.view"] = VIEW
    if theme:
        at.session_state["dsh.theme"] = theme
    at.run()
    return at


def _render_error(at: AppTest):
    """at.session_state.get(...) 会抛 AttributeError，这里按约定用 try/except。"""
    try:
        return at.session_state["dsh.render_error"]
    except Exception:
        return None


def _overrides(at: AppTest) -> dict:
    try:
        return dict(at.session_state["dsh.overrides"])
    except Exception:
        return {}


def _scenario(at: AppTest) -> dict:
    return dict(at.session_state["dsh.scenario"])


def _clean(at: AppTest, theme: str = "dark") -> None:
    assert len(at.exception) == 0, f"[{theme}] 异常：{[str(e)[:300] for e in at.exception]}"
    assert not _render_error(at), f"[{theme}] 视图渲染失败：{str(_render_error(at))[:300]}"


def _settle(at: AppTest) -> AppTest:
    """再空跑一轮：AppTest 在「标签页内按钮触发 st.rerun」的那一轮会把该页元素
    合并出重复副本（stale widget 查不到、后续点击失效），下一轮即恢复干净树。
    这只是 AppTest 的树合并伪影，浏览器端与真实页面行为不受影响。"""
    at.run()
    _clean(at)
    return at


# ---------------------------------------------------------------- 双主题渲染
def test_renders_clean_in_both_themes():
    for theme in ("dark", "light"):
        at = _run(theme)
        _clean(at, theme)
        md = len(at.get("markdown"))
        pc = len(at.get("plotly_chart"))
        ar = len(at.get("arrow_data_frame"))
        print(f"[{theme}] markdown={md} plotly_chart={pc} arrow_data_frame={ar} "
              f"buttons={len(at.get('button'))} checkbox={len(at.get('checkbox'))} "
              f"number_input={len(at.get('number_input'))} text_area={len(at.get('text_area'))} "
              f"download_button={len(at.get('download_button'))}")
        assert md >= 50 and pc >= 6 and ar >= 10, f"[{theme}] 元素过少"


def test_default_scenario_stays_paper_calibrated():
    """默认（不点「应用参数」）场景必须仍是论文校准 + 论文权重，扩项不得改变默认行为。"""
    at = _run()
    _clean(at)
    sc = _scenario(at)
    assert sc["mode"] == "calibrated"
    assert sc["use_paper_weights"] is True
    assert _overrides(at) == {}


# ---------------------------------------------------------------- AHP 上三角编辑
def test_ahp_upper_triangle_edit_apply_and_mirror():
    at = _run()
    _clean(at)
    # 只改上三角 a12 = 3：草稿阶段不写覆盖
    at.number_input(key="mx_ahp_01").set_value(3.0).run()
    _clean(at)
    assert "ahp_matrix" not in _overrides(at)
    # 应用：写覆盖并自动回填下三角（成功提示只在本轮可见，先断言再结算伪影）
    at.button(key="apply_ahp").click().run()
    _clean(at)
    m = _overrides(at).get("ahp_matrix")
    assert m is not None
    assert m[0][1] == 3.0
    assert m[1][0] * m[0][1] == 1.0            # 互反回填 1/a
    assert all(abs(m[i][i] - 1.0) < 1e-12 for i in range(3))   # 对角线锁定 1
    d = ahp_diagnostics(m)
    assert d["ok"] and d["errors"] == []
    ok_msgs = [s.value for s in at.get("success")]
    assert any("仅在线复算口径参与计算" in s for s in ok_msgs), ok_msgs
    _settle(at)
    # 一键切到在线复算口径
    at.button(key="ahp_go_live").click().run()
    _clean(at)
    sc = _scenario(at)
    assert sc["mode"] == "live" and sc["use_paper_weights"] is False


# ---------------------------------------------------------------- w3 归一化
def test_w3_normalize_and_delta_summary():
    at = _run()
    _clean(at)
    at.slider(key="mx_w3_0").set_value(0.60).run()
    _clean(at)
    chips = [m.value for m in at.get("markdown")]
    assert any("未归一化" in c for c in chips), "Σw≠1 时应提示未归一化"
    at.button(key="mx_w3_normalize").click().run()
    _clean(at)
    w = _overrides(at).get("w3")
    assert w is not None and abs(sum(w) - 1.0) < 5e-6     # 6 位小数舍入容差
    caps = [c.value for c in at.get("caption")]
    assert any("原值 vs 当前：已修改" in c for c in caps), caps[:6]


# ---------------------------------------------------------------- 预设 / 撤销 / 重做
def test_presets_undo_redo_and_clear():
    at = _run()
    _clean(at)
    # 均匀权重
    at.button(key="mx_preset_uniform").click().run()
    _clean(at)
    assert _overrides(at).get("w3") == [0.333333, 0.333333, 0.333333]
    # 一致性重建：CR=0，方根法还原论文表5.8 权重
    at.button(key="mx_preset_consistent").click().run()
    _clean(at)
    d = ahp_diagnostics(_overrides(at)["ahp_matrix"])
    assert d["ok"] and d["cr"] is not None and abs(d["cr"]) < 1e-9
    # 撤销两步
    at.button(key="mx_undo").click().run()
    _clean(at)
    assert "ahp_matrix" not in _overrides(at) and "w3" in _overrides(at)
    at.button(key="mx_undo").click().run()
    _clean(at)
    assert _overrides(at) == {}
    # 空栈时撤销按钮禁用，且重做可恢复
    assert at.button(key="mx_undo").disabled is True
    assert at.button(key="mx_redo").disabled is False
    at.button(key="mx_redo").click().run()
    _clean(at)
    assert _overrides(at).get("w3") is not None
    # 论文原值（只清 w3 + ahp_matrix）
    at.button(key="mx_preset_paper").click().run()
    _clean(at)
    assert "w3" not in _overrides(at) and "ahp_matrix" not in _overrides(at)
    # 恢复全部实验修改
    at.button(key="mx_preset_uniform").click().run()
    _clean(at)
    assert _overrides(at), "应存在覆盖"
    at.button(key="mx_clear_all").click().run()
    _clean(at)
    assert _overrides(at) == {}


# ---------------------------------------------------------------- JSON 导入
def test_json_import_validation_and_apply():
    at = _run()
    _clean(at)
    # 1) 非法 JSON → 中文可执行报错，不写覆盖
    at.text_area(key="mx_import_text").set_value("{oops").run()
    at.button(key="mx_import_apply").click().run()
    _clean(at)
    assert len(at.get("error")) >= 1
    assert _overrides(at) == {}
    # 2) 键/形状不合法 → 报错且不写覆盖
    at.text_area(key="mx_import_text").set_value('{"w3": [1, 2], "foo": 1}').run()
    at.button(key="mx_import_apply").click().run()
    _clean(at)
    errs = [e.value for e in at.get("error")]
    assert any("未知键" in e for e in errs) and any("w3 必须是 3 个正数" in e for e in errs)
    assert _overrides(at) == {}
    # 3) 合法导入 → 写覆盖 + 成功提示（成功提示只在本轮可见，先断言再结算伪影）
    at.text_area(key="mx_import_text").set_value(
        '{"w3": [0.5, 0.3, 0.2], "ahp_matrix": [[1, 5, 4], [0.2, 1, 0.25], [0.25, 4, 1]]}').run()
    at.button(key="mx_import_apply").click().run()
    _clean(at)
    ov = _overrides(at)
    assert ov.get("w3") == [0.5, 0.3, 0.2]
    assert ahp_diagnostics(ov["ahp_matrix"])["ok"]
    assert any("已导入 2 项覆盖" in s.value for s in at.get("success"))
    _settle(at)
    # 4) PLTS 结构不匹配 → 报错
    at.text_area(key="mx_import_text").set_value(
        '{"plts_matrix": {"c11": {"A1": {"terms": ["l3"], "probs": [1.0]}}}}').run()
    at.button(key="mx_import_apply").click().run()
    _clean(at)
    assert any("plts_matrix" in e.value for e in at.get("error"))
    assert "plts_matrix" not in _overrides(at)


# ---------------------------------------------------------------- Δ 热力图开关
def test_delta_heatmap_toggles():
    at = _run()
    _clean(at)
    base = len(at.get("plotly_chart"))
    for key in ("mx_delta_w3", "mx_delta_ahp", "mx_delta_dematel", "mx_delta_plts"):
        at.checkbox(key=key).check().run()
        _clean(at)
    assert len(at.get("plotly_chart")) == base + 4, (base, len(at.get("plotly_chart")))
    at.checkbox(key="mx_delta_ahp").uncheck().run()
    _clean(at)
    assert len(at.get("plotly_chart")) == base + 3


# ---------------------------------------------------------------- 静态版式约束
def test_simulator_static_constraints():
    src = Path("ui/views/simulator.py").read_text(encoding="utf-8")
    assert not re.search(r"st\.columns", src), "禁止裸写 st.columns"
    assert "use_container_width" not in src, "已弃用参数"
    assert not re.search(r"#[0-9A-Fa-f]{6}", src), "禁止硬编码十六进制颜色"
    assert 'st.markdown("")' not in src, "禁止空 markdown"
    heights = set(re.findall(r'layout\.height\("([a-z]+)"\)', src))
    assert heights <= {"xs", "s", "m", "l"}, heights
