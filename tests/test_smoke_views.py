"""应用级冒烟：用 streamlit.testing.v1.AppTest 跑六个视图，断言无异常、无视图错误。"""
from streamlit.testing.v1 import AppTest

VIEWS = ["产品导览", "决策总览", "属性情感分析", "PLTS-VIKOR 模拟器",
         "洞察与报告", "数据管理"]


def _run(view: str) -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=180)
    at.session_state["dsh.view"] = view
    at.run()
    return at


def _render_error(at: AppTest):
    try:
        return at.session_state["dsh.render_error"]
    except Exception:
        return None


def test_no_exception_any_view():
    problems = []
    for v in VIEWS:
        try:
            at = _run(v)
            if at.exception:
                problems.append((v, "streamlit 异常: " + str(at.exception)[:700]))
            err = _render_error(at)
            if err:
                problems.append((v, "视图渲染失败: " + str(err)[:700]))
            if len(at.main) < 5:
                problems.append((v, f"视图渲染元素过少：{len(at.main)}（疑似空页面）"))
        except Exception as e:
            problems.append((v, f"AppTest 失败: {type(e).__name__}: {e}"[:700]))
    if problems:
        msg = "\n".join(f"[{v}] {p}" for v, p in problems)
        raise AssertionError("视图渲染异常:\n" + msg)
