"""ui.layout — 版式系统（栅格 · 面板 · 节奏）

设计基准（参考 SAP Fiori 语义页面结构、8pt 间距体系、主流 SaaS 看板的
「页头 → 区块标题 → 卡片面 → 图表/表格」四层结构）：

    页头  page_head()   —— 每页仅一次：标题 + 副题 + 右侧状态条
    区块  section()     —— 区块标题 + 副题 + 色条，间距固定
    面板  panel()       —— 统一卡片（标题栏 + 1px 边框 + 16px 圆角）
    栅格  split()       —— 只允许使用下面 7 组具名比例，杜绝随手写的 [1.37,1]

约定：
  * 视图内禁止裸写 st.columns([a, b])，一律走 split("<具名比例>")
  * 图表高度只能取 H 中的档位，禁止出现 330/357 之类的散值
  * 需要占满父宽的控件统一 width="stretch"（不再用已弃用的 use_container_width）
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

import streamlit as st

# ------------------------------------------------------------------ 栅格
#: 具名比例 → (列比例, gap)。新增比例需同步更新本文档与各视图。
GRIDS: dict[str, tuple[list[float], str]] = {
    # 标题行 / 右侧操作
    "head_action": ([4.0, 1.0], "small"),
    "head_actions": ([4.0, 1.0, 1.0, 1.0], "small"),
    # 主内容 + 右侧结论栏（看板主骨架）
    "main_rail": ([1.6, 1.0], "large"),
    "title_badges": ([1.2, 3.0], "small"),
    # 左控制台 + 右画布（模拟器骨架）
    "control_canvas": ([1.0, 2.6], "large"),
    "rail_canvas_rail": ([0.9, 3.0, 0.9], "large"),
    # 宽控制台 + 画布 + 结论栏（模拟器需要可滚动的参数表单，左栏不能太窄）
    "console_canvas_result": ([1.3, 3.0, 0.9], "large"),
    # 等分
    "even": ([1.0, 1.0], "large"),
    "even3": ([1.0, 1.0, 1.0], "large"),
}


def split(name: str) -> list:
    """按具名比例切列。未知名称直接报错，防止版式回退成随手比例。"""
    if name not in GRIDS:
        raise KeyError(f"未知栅格 {name!r}，可用：{sorted(GRIDS)}")
    ratio, gap = GRIDS[name]
    return st.columns(ratio, gap=gap)


def cols(n: int) -> list:
    """等分 n 列（KPI 行等动态场景的唯一入口，视图与组件都不得自己写 st.columns）。"""
    if n <= 1:
        return [st.container()]
    return st.columns(n)


# ------------------------------------------------------------------ 高度
#: 图表高度档位（px）。只有这四个，保证纵向节奏一致。
H = {"xs": 150, "s": 260, "m": 340, "l": 440}


def height(name: str) -> int:
    if name not in H:
        raise KeyError(f"未知高度档 {name!r}，可用：{sorted(H)}")
    return H[name]


# ------------------------------------------------------------------ 页头
def page_head(title: str, subtitle: str | None = None,
              chips: Sequence[str] = ()) -> None:
    """每页顶部的统一标题区（仅调用一次）。"""
    chip_html = "".join(c if c.startswith("<") else f'<span class="ds-chip chip-muted">{c}</span>'
                        for c in chips)
    st.markdown(
        f'<div class="ds-pagehead"><div class="ph-title">{title}</div>'
        f'{f"<div class=ph-sub>{subtitle}</div>" if subtitle else ""}'
        f'<div class="ph-chips">{chip_html}</div></div>',
        unsafe_allow_html=True)


# ------------------------------------------------------------------ 面板
@contextmanager
def panel(title: str | None = None, subtitle: str | None = None,
          tag: str | None = None, footer: str | None = None,
          ) -> Iterator[None]:
    """统一卡片面：标题栏 + 内容区。所有图表/表格都应放进面板。

    with layout.panel("S-R 效用散点", "复算 vs 论文基准"):
        charts.sr_scatter(...)
    """
    with st.container(border=True):
        if title or tag:
            tag_html = f'<span class="ds-chip chip-brand">{tag}</span>' if tag else ""
            st.markdown(
                f'<div class="ds-panelhead"><div class="pn-title">{title or ""}{tag_html}</div>'
                f'{f"<div class=pn-sub>{subtitle}</div>" if subtitle else ""}</div>',
                unsafe_allow_html=True)
        yield
        if footer:
            st.markdown(f'<div class="ds-panelfoot">{footer}</div>', unsafe_allow_html=True)


def note(text: str, tone: str = "muted") -> str:
    """面板脚注 HTML（供传给 panel(footer=...)）。"""
    return f'<span class="ds-note note-{tone}">{text}</span>'


def stretch(widget) -> None:
    """占满父宽（1.50 起 use_container_width 已弃用）。"""
    try:
        widget(width="stretch")
    except TypeError:
        widget(use_container_width=True)
