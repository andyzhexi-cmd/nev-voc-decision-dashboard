"""ui.views.data_manager — 数据管理（阶段状态 / 上传 / 流水线 / 数据质量 / 字段字典）

区块（自上而下）
  1. 标题区 section_header("数据管理")
  2. 流水线六阶段卡片（就绪徽章 / 行数 / 修改时间 / 相对时间）
  3. 上传评论 CSV（多文件，同名先备份到 data/upload_backups/，写完 bump_data_version）
  4. 流水线运行区（清洗分词 / 属性聚类 / 情感分析，各自独立按钮 + spinner + traceback）
  5. 数据质量卡（kpi_summary）+ 原始表逐列缺失率
  6. 附录：数据字典与双模型一致机制说明

约定：顶层不读大数据；流水线脚本较重，模块一律在点击后延迟 import，
入口函数先 hasattr 判定（main / run），都没有则提示"暂不支持"，不硬调用。
"""
from __future__ import annotations

import io
import shutil
import time
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

from services.data_store import ROOT, store as data_store
from state import store
from ui.components import callout, kpi_row, section_header

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
BACKUP_DIR = ROOT / "data" / "upload_backups"

# 流水线步骤注册表：入口函数按 entries 顺序做 hasattr 探测
STEPS: list[dict] = [
    {"key": "clean", "name": "清洗分词", "module": "services.preprocess.pipeline",
     "entries": ("main", "run"),
     "desc": "三级清洗 + jieba 分词 + 停用词过滤",
     "src": "data/raw/comments_raw.csv", "dst": "data/processed/comments_cleaned.csv"},
    {"key": "cluster", "name": "属性聚类", "module": "services.aspect_mining.run_clustering",
     "entries": ("main", "run"),
     "desc": "TF-IDF + KMeans 六属性聚类（K=6，输出纯度/轮廓诊断）",
     "src": "data/processed/comments_cleaned.csv", "dst": "data/processed/comments_clustered.csv"},
    {"key": "sentiment", "name": "情感分析", "module": "services.sentiment.run_sentiment",
     "entries": ("main", "run"),
     "desc": "VADER + 朴素贝叶斯双模型融合，产出 dual_agree / final_sentiment",
     "src": "data/processed/comments_cleaned.csv", "dst": "data/processed/sentiment_results.csv"},
]

_FLASH_KEY = "dsh.dm.flash"


# ------------------------------------------------------------------ 工具
def _flash(msg: str) -> None:
    st.session_state[_FLASH_KEY] = msg


def _invalidate_caches() -> None:
    """流水线产物变化后失效各级缓存（parquet 中间层 / lru_cache / st.cache_data）。"""
    store.bump_data_version()
    try:
        from services import features as F
        F.comment_index.cache_clear()
    except Exception:
        pass
    try:
        from services import data_store as DS
        DS.reset_cache()
    except Exception:
        pass


def _resolve_entry(module, candidates: tuple[str, ...]):
    """按候选名探测 CLI 入口；返回 (入口名, 可调用对象) 或 (None, None)。"""
    for name in candidates:
        fn = getattr(module, name, None)
        if callable(fn):
            return name, fn
    return None, None


# ------------------------------------------------------------------ 缓存取数
@st.cache_data(show_spinner=False, max_entries=4)
def _quality(filters_json: str, version: int) -> dict:
    """数据质量 KPI（原始 / 噪声 / 双模一致 / 保留率）。"""
    from services.features import Filters, kpi_summary
    import json
    d = json.loads(filters_json or "{}")
    return kpi_summary(Filters(**d) if d else Filters())


@st.cache_data(show_spinner=False, max_entries=4)
def _missing_table(version: int) -> pd.DataFrame:
    """原始表逐列缺失率（object 列的空串也计为缺失）。"""
    raw = data_store().raw()
    cols = ["字段", "缺失数", "缺失率(%)"]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for c in raw.columns:
        s = raw[c]
        miss = s.isna()
        if s.dtype == object:
            miss = miss | s.astype(str).str.strip().isin(["", "nan", "None", "<NA>"])
        n = int(miss.sum())
        rows.append({"字段": str(c).replace("\ufeff", "").strip(),
                     "缺失数": n, "缺失率(%)": round(n / max(len(raw), 1) * 100, 2)})
    return (pd.DataFrame(rows)
            .sort_values("缺失率(%)", ascending=False)
            .reset_index(drop=True))


# ------------------------------------------------------------------ 区块 2：阶段状态
def _stage_section() -> None:
    stages = data_store().stage_statuses()
    fresh = data_store().freshness()

    head = st.columns([4, 1, 1, 1])
    with head[0]:
        st.markdown(
            f'<span class="ds-chip chip-{"ok" if fresh["all_ready"] else "warn"}">'
            f'● {fresh["ready"]}/{fresh["total"]} 阶段就绪</span> '
            f'<span class="ds-chip chip-muted">最新产物 {fresh["newest"]} · {fresh["age"]}</span>',
            unsafe_allow_html=True)
    with head[3]:
        if st.button("刷新状态", use_container_width=True):
            st.rerun()

    cols = st.columns(3)
    for i, s in enumerate(stages):
        with cols[i % 3]:
            chip = ('<span class="ds-chip chip-ok">● 就绪</span>' if s.exists
                    else '<span class="ds-chip chip-bad">○ 缺失</span>')
            mtime = (time.strftime("%m-%d %H:%M", time.localtime(s.mtime))
                     if s.mtime else "—")
            st.markdown(
                f'<div class="ds-card" style="padding:12px 14px;">'
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;gap:8px;">'
                f'<b style="font-size:14px;">{s.name}</b>{chip}</div>'
                f'<div style="font-size:12px;color:var(--muted);margin-top:8px;'
                f'line-height:1.8;">'
                f'行数 <b style="color:var(--text)">{s.rows:,}</b> · '
                f'{mtime}（{s.age}）<br>'
                f'<span style="font-size:11px;">{s.path}</span></div></div>',
                unsafe_allow_html=True)
    if fresh["ready"] < fresh["total"]:
        st.caption("缺失阶段可在下方「流水线运行」区补齐；缺失阶段的下游视图会降级为空态。")


# ------------------------------------------------------------------ 区块 3：上传
def _save_uploads(files: list, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    saved, backed, bad = [], [], []
    for f in files:
        name = Path(getattr(f, "name", "upload.csv")).name        # 防路径穿越
        if not name.lower().endswith(".csv"):
            bad.append(f"{name}（非 CSV，已跳过）")
            continue
        payload = f.getvalue()
        try:
            pd.read_csv(io.BytesIO(payload), nrows=5)
        except Exception as exc:
            bad.append(f"{name}（无法解析：{exc}）")
            continue
        target = dest / name
        if target.exists():                                       # 同名覆盖前备份
            bak = BACKUP_DIR / f"{target.stem}_{time.strftime('%Y%m%d-%H%M%S')}{target.suffix}"
            shutil.copy2(target, bak)
            backed.append(bak.name)
        target.write_bytes(payload)
        saved.append(f"{name}（{len(payload) / 1024:.1f} KB）")

    if saved:
        _invalidate_caches()
        note = f"已保存 {len(saved)} 个文件至 {dest.relative_to(ROOT)}"
        if backed:
            note += f"；同名备份 {len(backed)} 个到 data/upload_backups/"
        if bad:
            note += "；跳过 " + "、".join(bad)
        _flash(note)
        st.rerun()
    for w in bad:
        st.warning(w)


def _upload_section() -> None:
    which = st.radio("落盘目录", ["data/raw（原始评论 · 阶段①输入）",
                                 "data/processed（加工产物）"],
                     horizontal=True)
    dest = RAW_DIR if which.startswith("data/raw") else PROCESSED_DIR
    files = st.file_uploader("上传评论 CSV", type=["csv"], accept_multiple_files=True,
                             help="支持多选；同名文件会先备份到 data/upload_backups/ 再覆盖。"
                                  "落原始评论时建议文件名保持 comments_raw.csv，"
                                  "以被流水线阶段①读取。")
    if files:
        meta = pd.DataFrame([{"文件名": Path(f.name).name,
                              "大小(KB)": round(len(f.getvalue()) / 1024, 1)}
                             for f in files])
        st.dataframe(meta, use_container_width=True, hide_index=True)
        if st.button("保存上传文件", type="primary"):
            _save_uploads(files, dest)


# ------------------------------------------------------------------ 区块 4：流水线
def _run_step(step: dict) -> None:
    import importlib
    try:
        mod = importlib.import_module(step["module"])
    except Exception:
        st.error(f"导入模块失败：{step['module']}")
        with st.expander("查看 traceback", expanded=True):
            st.code(traceback.format_exc())
        return

    entry, fn = _resolve_entry(mod, step["entries"])
    if fn is None:
        st.warning(f"暂不支持：{step['module']} 未提供 {step['entries']} 入口函数，"
                   f"请改用命令行运行该阶段。")
        return

    t0 = time.time()
    with st.spinner(f"{step['name']} 运行中（{entry}()），数据量大时可达分钟级，请勿刷新页面…"):
        try:
            fn()
        except Exception:
            st.error(f"{step['name']} 运行失败")
            with st.expander("查看 traceback", expanded=True):
                st.code(traceback.format_exc())
            return
    _invalidate_caches()
    _flash(f"{step['name']} 完成，用时 {time.time() - t0:.1f} 秒；阶段状态与数据缓存已刷新。")
    st.rerun()


def _pipeline_section() -> None:
    callout("以下脚本较重（分钟级），请逐个点击、按 ②→③→④ 顺序执行；"
            "每个步骤运行前后都会刷新上方阶段状态。")
    for step in STEPS:
        left, right = st.columns([4.2, 1.2])
        with left:
            st.markdown(f"**{step['name']}** —— {step['desc']}")
            st.caption(f"输入 {step['src']} → 输出 {step['dst']} · 入口 {step['module']}")
        with right:
            if st.button(f"运行 {step['name']}", key=f"dm_run_{step['key']}",
                         use_container_width=True):
                _run_step(step)


# ------------------------------------------------------------------ 区块 5：数据质量
def _quality_section() -> None:
    k = _quality(store.filters_json(), store.data_version())
    kpi_row([
        {"label": "原始评论", "value": f"{k['n_raw']:,}",
         "delta": f"平台 {k['n_platforms']} · 品牌 {k['n_brands']}",
         "hint": "data/raw/comments_raw.csv", "tone": "flat"},
        {"label": "噪声评论", "value": f"{k['n_noise']:,}",
         "delta": f"占比 {round(k['n_noise'] / max(k['n_raw'], 1) * 100, 1)}%",
         "hint": "_attr_ground_truth = 噪声，阶段③前剔除", "tone": "flat"},
        {"label": "双模型一致", "value": f"{k['n_kept']:,}",
         "delta": f"剔除 {k['n_dropped']:,} 条",
         "hint": "dual_agree = True 方进入统计", "tone": "flat"},
        {"label": "数据保留率", "value": f"{k['keep_rate']}%",
         "delta": f"情感均值 {k['mean_sentiment']:+.3f}",
         "hint": f"论文基准均值 {k['paper_mean_sentiment']:+.3f}", "tone": "flat"},
    ])

    st.markdown("")
    t1, t2 = st.columns([1, 3])
    with t1:
        st.markdown('<div class="ds-section"><h3>原始表缺失率</h3>'
                    '<div class="rule"></div></div>', unsafe_allow_html=True)
        st.caption("统计口径：data/raw/comments_raw.csv 全量；"
                   "文本列空串亦计为缺失。")
    with t2:
        st.dataframe(_missing_table(store.data_version()),
                     use_container_width=True, hide_index=True)


# ------------------------------------------------------------------ 区块 6：附录
def _dictionary_section() -> None:
    from core.algorithm.models import baselines
    b = baselines()

    rows = [
        {"字段": "comment_id", "含义": "评论唯一编号（主键）", "取值示例": "1, 2, 3 …"},
        {"字段": "platform", "含义": "采集平台（5 个）",
         "取值示例": "汽车之家 / 易车网 / 爱卡汽车 / 太平洋汽车 / 网上车市"},
        {"字段": "brand / model", "含义": "品牌与车型",
         "取值示例": "特斯拉 Model Y / 比亚迪 汉 / 小鹏 G6 …"},
        {"字段": "comment_date", "含义": "评论发布日期", "取值示例": "2024-06-06"},
        {"字段": "comment_text", "含义": "原始评论文本", "取值示例": "整体来说，屏幕清晰流畅…"},
        {"字段": "cleaned_text / tokens", "含义": "三级清洗后的文本与 jieba 分词词串（空格分隔）",
         "取值示例": "屏幕 流畅 音响"},
        {"字段": "_attr_ground_truth", "含义": "属性金标准：六属性之一或「噪声」，用于聚类/分类验收",
         "取值示例": "外观 / 内饰 / 空间 / 续航 / 性价比 / 舒适性 / 噪声"},
        {"字段": "_sentiment_ground_truth", "含义": "情感金标准：1 正面、-1 负面，仅用于指标验收",
         "取值示例": "1 / -1"},
        {"字段": "vader_compound", "含义": "VADER 规则情感值，区间 [-1, 1]，>0 记正面",
         "取值示例": "0.6241"},
        {"字段": "nb_pos_proba", "含义": "朴素贝叶斯正类概率，区间 [0, 1]，>0.5 记正面",
         "取值示例": "0.8735"},
        {"字段": "dual_agree", "含义": "双模型一致标记：两模型极性同为正或同为负",
         "取值示例": "True / False"},
        {"字段": "final_sentiment", "含义": "最终情感值：一致时取 vader_compound，不一致记 0（剔除）",
         "取值示例": "0.6241 / 0.0"},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=360)

    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**六个属性 ground truth**")
        st.markdown(
            "属性金标准列 `_attr_ground_truth` 取值为 **外观 / 内饰 / 空间 / 续航 / 性价比 / 舒适性**"
            "（外加「噪声」水帖）。仿真数据按论文表5.1 的平台×品牌格子与图3.5 关键词分布生成；"
            "真实数据导入后由 TF-IDF + KMeans 聚类（K=6）映射得到，并与该列对照计算纯度。")
        st.caption(f"论文基准：K=6、轮廓系数 {b['clustering']['silhouette']}、"
                   f"纯度 {b['clustering']['purity']}、有效评论 {b['clustering']['n_valid']:,} 条。")
    with c2:
        st.markdown("**双模型一致机制（dual_agree）**")
        m = b["sentiment"]["paper_metrics"]
        st.markdown(
            "同一评论分别由 **VADER 中文规则词典**（vader_compound）与 **朴素贝叶斯分类器**"
            "（nb_pos_proba）给出极性：\n\n"
            "- vader_compound > 0 且 nb_pos_proba > 0.5 → 正面一致；\n"
            "- vader_compound < 0 且 nb_pos_proba < 0.5 → 负面一致；\n\n"
            "两者一致时 `dual_agree=True`，`final_sentiment` 取 vader_compound 连续值；"
            "分歧或中性则 `final_sentiment=0`，不进入下游情感统计（这也是保留率的来源）。")
        st.caption(f"论文融合基准：P={m['fused']['P']:.4f} · R={m['fused']['R']:.4f} · "
                   f"F1={m['fused']['F1']:.4f}（单模型 VADER F1={m['vader']['F1']:.4f}、"
                   f"NB F1={m['nb']['F1']:.4f}）。")


# ------------------------------------------------------------------ 视图
def render() -> None:
    MODE = store.theme()          # 本视图无图表；保留主题读取以与其它视图一致

    msg = st.session_state.pop(_FLASH_KEY, None)
    if msg:
        st.success(msg)

    section_header(
        "数据管理",
        subtitle="评论数据上传 · 六阶段流水线 · 数据质量监控 · 字段字典",
        tag="数据")

    # 1) 阶段状态
    section_header("流水线阶段状态", subtitle="产物是否就绪、行数与新鲜度（相对时间）")
    _stage_section()

    # 2) 上传
    section_header("上传评论 CSV", subtitle="支持多文件；同名覆盖前自动备份到 data/upload_backups/")
    _upload_section()

    # 3) 流水线运行
    section_header("流水线运行", subtitle="清洗分词 → 属性聚类 → 情感分析（脚本较重，逐个执行）")
    _pipeline_section()

    # 4) 数据质量
    section_header("数据质量", subtitle="原始 / 噪声 / 双模型一致 / 保留率 与逐列缺失率")
    _quality_section()

    # 5) 附录
    section_header("附录 · 数据字典", subtitle="字段含义、属性金标准与双模型一致机制")
    _dictionary_section()
