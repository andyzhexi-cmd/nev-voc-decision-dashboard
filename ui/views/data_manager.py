"""ui.views.data_manager — 数据管理（阶段状态 / 上传 / 流水线 / 数据质量 / 字段字典）

版式（ui.layout）：页头 page_head → 区块 section_header → 面板 layout.panel → 内容；
栅格只用具名比例（head_actions / head_action / even3 / even），高度只取 layout.H 档位，
占满宽度一律 width="stretch"，不手工插入空行撑间距。

区块（自上而下）
  页头 layout.page_head（含就绪度徽章）+ 1) 流水线阶段状态（六张等高 ds-kpi 卡）
  2) 导入真实评论（上传 → 编码/分隔符嗅探 → 列映射可调 → 干跑体检 → 确认落盘；
     附标准模板下载与 data/upload_backups/ 备份清单，回滚只给说明不给按钮）
  3) 上传评论 CSV（多文件，同名先备份到 data/upload_backups/，写完 bump_data_version）
  4) 流水线运行区（清洗分词 / 属性聚类 / 情感分析，各自独立按钮 + spinner + traceback）
  5) 数据质量卡（kpi_summary）+ 原始表逐列缺失率
  6) 附录：数据字典与双模型一致机制说明

约定：顶层不读大数据；流水线脚本较重，模块一律在点击后延迟 import，
入口函数先 hasattr 判定（main / run），都没有则提示"暂不支持"，不硬调用。
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import time
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

from services.data_store import ROOT, store as data_store
from state import store
from ui import layout
from ui.components import callout, kpi_row, section_header

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
BACKUP_DIR = ROOT / "data" / "upload_backups"

# ---- 导入真实评论（services.data_ingest.importer 的视图外壳）----
ING_PREFIX = "ing_"                                   # 本区 session_state key 前缀
ING_TYPES = ["csv", "tsv", "txt", "json", "xlsx", "xls"]
ING_COLS = ["platform", "brand", "model", "comment_date", "comment_text"]
ING_LABELS = {
    "platform": "平台 platform",
    "brand": "品牌 brand",
    "model": "车型 model",
    "comment_date": "发布时间 comment_date",
    "comment_text": "评论正文 comment_text（唯一必需）",
}
ING_SKIP = "不导入"
ING_TMP_DIR = ROOT / "data" / "upload_tmp"             # 上传暂存（按内容指纹命名）
ING_TEMPLATE = ROOT / "docs" / "samples" / "comments_template.csv"

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

    with layout.panel("产物就绪度与行数", "六阶段产物是否就绪、行数与新鲜度（相对时间）"):
        head = layout.split("head_actions")
        with head[0]:
            st.caption(f"最新产物 {fresh['newest']} · {fresh['age']}；"
                       f"{'全部阶段就绪' if fresh['all_ready'] else '仍有阶段缺失'}。")
        with head[3]:
            if st.button("刷新状态", width="stretch"):
                st.rerun()

        cols = layout.split("even3")
        for i, s in enumerate(stages):
            with cols[i % 3]:
                chip = ('<span class="ds-chip chip-ok">● 就绪</span>' if s.exists
                        else '<span class="ds-chip chip-bad">○ 缺失</span>')
                mtime = (time.strftime("%m-%d %H:%M", time.localtime(s.mtime))
                         if s.mtime else "—")
                st.markdown(
                    f'<div class="ds-kpi">'
                    f'<div class="k-label">{s.name}</div>'
                    f'<div class="k-value">{f"{s.rows:,} 行" if s.exists else "—"}</div>'
                    f'<div class="k-delta k-flat">{chip}</div>'
                    f'<div class="k-hint">{mtime}（{s.age}）<br>{s.path}</div>'
                    f'</div>',
                    unsafe_allow_html=True)
        if fresh["ready"] < fresh["total"]:
            st.caption("缺失阶段可在下方「流水线运行」区补齐；缺失阶段的下游视图会降级为空态。")


# ------------------------------------------------------------------ 区块 3：导入真实评论
@st.cache_data(show_spinner=False, max_entries=4)
def _ing_probe(path: str) -> dict:
    """读入暂存文件：源列名（文本/原始两套）、嗅探元信息与自动映射。"""
    from services.data_ingest.importer import detect_mapping, load_table
    df, meta = load_table(path)
    auto = detect_mapping(df)
    return {"cols": [str(c) for c in df.columns],
            "orig": {str(c): c for c in df.columns},
            "auto": {k: (None if v is None else str(v)) for k, v in auto.items()},
            "meta": meta}


@st.cache_data(show_spinner=False, max_entries=4)
def _ing_dry_run(path: str, mapping_json: str, dedupe: bool):
    """干跑校验（不落盘）；缓存键 = 暂存路径（含内容指纹）+ 列映射 + 去重开关。"""
    from services.data_ingest.importer import import_comments
    _, rep = import_comments(path, mapping=json.loads(mapping_json),
                             dry_run=True, dedupe=dedupe)
    return rep


def _ing_stage(uploaded) -> tuple[Path | None, str, str]:
    """上传内容暂存到 data/upload_tmp/（按内容指纹命名），返回 (路径, 指纹, 展示文件名)。"""
    payload = uploaded.getvalue()
    name = Path(uploaded.name).name                                    # 防路径穿越
    fp = hashlib.sha1(payload).hexdigest()[:16]
    suffix = Path(name).suffix.lower()
    if suffix not in {f".{t}" for t in ING_TYPES}:
        suffix = ".csv"
    path = ING_TMP_DIR / f"ingest_{fp}{suffix}"
    try:
        if not path.exists():
            ING_TMP_DIR.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        _ing_prune_tmp(path)
    except OSError as exc:
        st.error(f"上传暂存失败（{exc}）：请检查 data/upload_tmp/ 目录可写后重试。")
        return None, fp, name
    return path, fp, name


def _ing_prune_tmp(keep: Path) -> None:
    """暂存目录只保留最近 5 个文件，避免反复上传堆积。"""
    try:
        olds = sorted((p for p in ING_TMP_DIR.iterdir() if p.is_file()),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        for p in olds[5:]:
            if p != keep:
                p.unlink()
    except OSError:
        pass


def _ing_sync_defaults(fp: str, probe: dict) -> None:
    """换文件时按自动嗅探结果回填五个映射默认值（须在 selectbox 实例化之前调用）。"""
    if st.session_state.get(f"{ING_PREFIX}fp") == fp:
        return
    cols = probe["cols"]
    for std in ING_COLS:
        src = probe["auto"].get(std)
        st.session_state[f"{ING_PREFIX}map_{std}"] = src if src in cols else ING_SKIP
    st.session_state[f"{ING_PREFIX}fp"] = fp


def _ing_select(std: str, options: list) -> str | None:
    """单个标准列 → 源列下拉；返回源列名（None = 不导入）。"""
    choice = st.selectbox(ING_LABELS[std], options, key=f"{ING_PREFIX}map_{std}",
                          width="stretch",
                          help="选择源文件里对应的列；选「不导入」表示该字段留空导入。")
    return None if choice == ING_SKIP else choice


def _ing_mapping_widgets(probe: dict) -> dict:
    """五个标准列各一个源列下拉 + 去重勾选；返回 标准列 → 源列（None = 不导入）。"""
    options = [ING_SKIP] + probe["cols"]
    auto = probe["auto"]
    for std in ING_COLS:                            # 兜底：上个文件的选项在新文件里失效时回退
        if st.session_state.get(f"{ING_PREFIX}map_{std}") not in options:
            src = auto.get(std)
            st.session_state[f"{ING_PREFIX}map_{std}"] = src if src in probe["cols"] else ING_SKIP
    mapping: dict = {}
    row1 = layout.split("even3")
    for i, std in enumerate(ING_COLS[:3]):
        with row1[i]:
            mapping[std] = _ing_select(std, options)
    row2 = layout.split("even3")
    for i, std in enumerate(ING_COLS[3:]):
        with row2[i]:
            mapping[std] = _ing_select(std, options)
    with row2[2]:
        st.checkbox("按正文去重", value=False, key=f"{ING_PREFIX}dedupe",
                    help="勾选后才会删除正文完全相同的行；默认只计数。")
        st.caption("重复正文默认不去重：灌水类短评由清洗阶段识别为噪声。")
    # 下拉存的是列名文本，写回源文件的原始列对象，避免列名带空格/大小写差异时取不到值
    return {std: (probe["orig"].get(src, src) if src else None) for std, src in mapping.items()}


def _ing_messages(rep) -> None:
    """逐条渲染校验结果：错误（可执行中文、拒绝落盘）与告警（只提示影响面）。"""
    for e in rep.errors:
        st.error(e)
    for w in rep.warnings:
        st.warning(w)


def _ing_kpis(rep, filename: str, dedupe: bool) -> None:
    """体检卡：文件与嗅探元信息 / 行数三件套 / 时间范围 / 品牌·车型·平台计数。"""
    enc = {"xlsx": "Excel", "": "—"}.get(rep.encoding, rep.encoding)
    sep_raw = str(rep.separator or "").strip("'\"")
    sep = {"": "—", "\t": "制表符", "json": "JSON"}.get(sep_raw, sep_raw)
    dr = rep.date_range
    shown = filename if len(filename) <= 20 else filename[:17] + "…"
    kpi_row([
        {"label": "上传文件", "value": shown,
         "delta": f"编码 {enc} · 分隔符 {sep}",
         "hint": "编码与分隔符自动嗅探（支持中文 GBK / UTF-8）", "tone": "flat"},
        {"label": "原始行数", "value": f"{rep.rows_raw:,}",
         "delta": f"空行 {rep.rows_empty:,}", "hint": "源文件表头以下总行数", "tone": "flat"},
        {"label": "重复正文", "value": f"{rep.rows_duplicate:,}",
         "delta": "已按去重删除" if dedupe else "仅计数 · 不去重",
         "hint": "灌水类短评由清洗阶段识别为噪声", "tone": "flat"},
        {"label": "保留行数", "value": f"{rep.rows_kept:,}",
         "delta": f"较原始 -{max(rep.rows_raw - rep.rows_kept, 0):,}",
         "hint": "标准化后将写入 comments_raw.csv 的条数", "tone": "flat"},
    ])
    kpi_row([
        {"label": "时间范围", "value": f"{dr[0]} ~ {dr[1]}" if dr else "—",
         "delta": f"无法解析 {rep.rows_bad_date:,} 条",
         "hint": "comment_date 统一格式化为 YYYY-MM-DD", "tone": "flat"},
        {"label": "品牌数", "value": f"{len(rep.brands):,}",
         "delta": "、".join(rep.brands[:3]) or "—",
         "hint": "去重枚举 · 影响分组与车型榜", "tone": "flat"},
        {"label": "车型数", "value": f"{len(rep.models):,}",
         "delta": "、".join(rep.models[:3]) or "—",
         "hint": "去重枚举 · 影响车型排名", "tone": "flat"},
        {"label": "平台数", "value": f"{len(rep.platforms):,}",
         "delta": "、".join(rep.platforms[:3]) or "—",
         "hint": "去重枚举 · 影响平台分布", "tone": "flat"},
    ])


def _ing_preview(rep) -> None:
    """预览前 10 行，列顺序强制取标准 schema。"""
    from services.data_ingest.importer import STANDARD_COLUMNS
    sample = rep.sample
    if sample is None or sample.empty:
        st.caption("暂无预览行：清洗后没有可用评论，请回到上方调整列映射或更换源文件。")
        return
    cols = [c for c in STANDARD_COLUMNS if c in sample.columns]
    st.caption(f"预览前 {len(sample)} 行 · 列顺序即标准 schema：{' → '.join(cols)}")
    st.dataframe(sample[cols], width="stretch", height=layout.height("m"), hide_index=True)


def _ing_write(state: dict) -> None:
    """正式导入：重新完整校验一遍，通过才原子落盘；成功后失效全部缓存。"""
    from services.data_ingest.importer import import_comments
    had_raw = (RAW_DIR / "comments_raw.csv").exists()                 # 本次是否会产生备份
    try:
        _, done = import_comments(state["path"], mapping=state["mapping"],
                                  dry_run=False, dedupe=state["dedupe"])
    except Exception:
        st.error("导入失败：已中止，data/raw/comments_raw.csv 未被修改。")
        with st.expander("查看 traceback", expanded=True):
            st.code(traceback.format_exc())
        return
    if not done.ok:                                                    # 校验失败：不落盘，只显示错误
        for e in done.errors:
            st.error(e)
        st.caption("校验未通过，未写入任何文件。")
        return
    _invalidate_caches()                                               # 内含 state.store.bump_data_version()
    backup = "旧文件已备份到 data/upload_backups/" if had_raw else "首次导入，暂无旧文件可备份"
    _flash(f"导入成功：{done.wrote_to} · 保留 {done.rows_kept:,} 条 · {backup}；"
           f"下一步到下方『三阶段执行器』重跑清洗 → 聚类 → 情感，或直接查看新数据。")
    st.rerun()


def _ing_commit(state: dict, rep) -> None:
    """确认导入按钮：校验失败时禁用（绝不落盘），通过则给下一步指路。"""
    if st.button("确认导入到 data/raw/comments_raw.csv", type="primary",
                 key=f"{ING_PREFIX}commit", width="stretch", disabled=not rep.ok):
        _ing_write(state)
    if rep.ok:
        st.caption("下一步：落盘后到下方『三阶段执行器』依次运行 清洗分词 → 属性聚类 → 情感分析，"
                   "或直接查看新数据（阶段状态会自动刷新）。")
    else:
        st.caption("校验未通过：请先修正上方报错（多为列映射问题），此处不会写入任何文件。")


def _ing_backups() -> pd.DataFrame:
    """扫描 data/upload_backups/（可能不存在），取最近 10 个备份文件。"""
    cols = ["文件名", "备份时间", "大小(KB)"]
    try:
        files = sorted((p for p in BACKUP_DIR.iterdir() if p.is_file()),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return pd.DataFrame(columns=cols)
    rows = []
    for p in files[:10]:
        try:
            info = p.stat()
        except OSError:
            continue
        rows.append({"文件名": p.name,
                     "备份时间": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info.st_mtime)),
                     "大小(KB)": round(info.st_size / 1024, 1)})
    return pd.DataFrame(rows, columns=cols)


def _ing_template_panel() -> None:
    """标准模板下载 + 最近备份清单；只给回滚说明与路径，不提供自动回滚按钮。"""
    with layout.panel(
            "标准模板与历史备份",
            "覆盖式导入自动留档到 data/upload_backups/；回滚 = 把对应文件改名回 comments_raw.csv 后重跑流水线",
            footer=layout.note("本区不提供自动回滚按钮：请把上表中的备份改名为 comments_raw.csv "
                               "放回 data/raw/，再重跑三阶段流水线。", "muted")):
        left, right = layout.split("even")
        with left:
            try:
                blob = ING_TEMPLATE.read_bytes()
            except OSError:
                blob = None
            if blob is None:
                st.error("标准模板缺失：docs/samples/comments_template.csv 未找到，请联系维护者补齐。")
            else:
                st.download_button(
                    "下载标准模板 CSV", data=blob, file_name="comments_template.csv",
                    mime="text/csv", key=f"{ING_PREFIX}template", width="stretch",
                    help="含表头与 5 行样例，列顺序 brand, model, platform, comment_date, comment_text")
            st.caption("正文列命名为 评论内容 / comment_text / text / content 任一即可被自动识别；"
                       "平台 / 品牌 / 车型 / 时间四列可缺省，缺列只告警不拦截。")
        with right:
            callout("回滚三步：① 停止流水线与视图写操作；② 把所选备份复制或改名为 comments_raw.csv "
                    "放回 data/raw/；③ 重跑 清洗分词 → 属性聚类 → 情感分析。")
            st.caption(f"备份目录：data/upload_backups/（相对 {ROOT.name}/）")
        backups = _ing_backups()
        if backups.empty:
            st.caption("暂无备份：data/upload_backups/ 尚不存在或为空，首次覆盖式导入后自动生成。")
        else:
            st.caption("最近 10 个备份（按时间倒序）：")
            st.dataframe(backups, width="stretch", height=layout.height("m"), hide_index=True)


def _ing_upload() -> dict | None:
    """上传控件 → 暂存 → 嗅探 → 列映射；未选文件或读取失败返回 None。"""
    uploaded = st.file_uploader(
        "上传评论文件（CSV / TSV / TXT / JSON / XLSX / XLS）",
        type=ING_TYPES, key=f"{ING_PREFIX}uploader",
        help="支持中文 GBK / UTF-8 / Big5 等编码与逗号、制表符、分号、竖线分隔符的自动识别；"
             "文件先暂存到 data/upload_tmp/，确认导入前不写 data/raw/comments_raw.csv。")
    if not uploaded:
        st.caption("选择文件后会先**干跑**一次（只读取与校验，不落盘）；"
                   "还没有现成文件可先下载下方「标准模板 CSV」照着整理。")
        return None
    path, fp, name = _ing_stage(uploaded)
    if path is None:
        return None
    try:
        probe = _ing_probe(str(path))
    except Exception:
        st.error(f"无法解析「{name}」：请确认格式属于 CSV / TSV / TXT / JSON / XLSX / XLS，且文件未损坏。")
        with st.expander("查看 traceback", expanded=False):
            st.code(traceback.format_exc())
        return None
    _ing_sync_defaults(fp, probe)
    return {"path": str(path), "name": name,
            "mapping": _ing_mapping_widgets(probe),
            "dedupe": bool(st.session_state.get(f"{ING_PREFIX}dedupe", False))}


def _ing_safe_dry(state: dict):
    """带异常兜底的干跑：读取/解析失败只报错，不中断整页渲染。"""
    try:
        key = json.dumps(state["mapping"], ensure_ascii=False, sort_keys=True)
        return _ing_dry_run(state["path"], key, state["dedupe"])
    except Exception:
        st.error("干跑校验失败：读取暂存文件出错，请重新上传或改用其它格式。")
        with st.expander("查看 traceback", expanded=True):
            st.code(traceback.format_exc())
        return None


def _import_section() -> None:
    """导入真实评论：上传与列映射（面板一）→ 干跑体检、预览与落盘（面板二）→ 模板与备份。"""
    state = None
    rep = None
    with layout.panel("上传与列映射",
                      "上传 → 自动嗅探编码/分隔符 → 微调列对应；本区只干跑，不写任何文件"):
        state = _ing_upload()
        if state is not None:
            rep = _ing_safe_dry(state)
    if state is not None and rep is not None:
        with layout.panel("干跑体检与预览",
                          "行数 / 重复 / 时间 / 品牌车型平台计数 + 前 10 行预览；校验通过才允许落盘"):
            _ing_kpis(rep, state["name"], state["dedupe"])
            _ing_messages(rep)
            _ing_preview(rep)
            _ing_commit(state, rep)
    _ing_template_panel()


# ------------------------------------------------------------------ 区块 4：上传
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
    with layout.panel("文件预览与落盘",
                      "落盘目录可选 data/raw / data/processed；同名文件先备份再覆盖"):
        _upload_body()


def _upload_body() -> None:
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
        st.dataframe(meta, width="stretch", hide_index=True)
        if st.button("保存上传文件", type="primary", width="stretch"):
            _save_uploads(files, dest)


# ------------------------------------------------------------------ 区块 5：流水线
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
    with layout.panel("三阶段执行器", "清洗分词 → 属性聚类 → 情感分析；脚本较重，逐个执行"):
        callout("以下脚本较重（分钟级），请逐个点击、按 ②→③→④ 顺序执行；"
                "每个步骤运行前后都会刷新上方阶段状态。")
        for step in STEPS:
            left, right = layout.split("head_action")
            with left:
                st.markdown(f"**{step['name']}** —— {step['desc']}")
                st.caption(f"输入 {step['src']} → 输出 {step['dst']} · 入口 {step['module']}")
            with right:
                if st.button(f"运行 {step['name']}", key=f"dm_run_{step['key']}",
                             width="stretch"):
                    _run_step(step)


# ------------------------------------------------------------------ 区块 6：数据质量
def _quality_section() -> None:
    k = _quality(store.filters_json(), store.data_version())
    with layout.panel("四项口径指标",
                      "原始 / 噪声 / 双模型一致 / 保留率，随全局筛选与数据版本重算"):
        _quality_cards(k)
    with layout.panel("原始表缺失率",
                      "统计口径：data/raw/comments_raw.csv 全量；文本列空串亦计为缺失。"):
        st.dataframe(_missing_table(store.data_version()), width="stretch", hide_index=True)


def _quality_cards(k: dict) -> None:
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


# ------------------------------------------------------------------ 区块 7：附录
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
    with layout.panel("列与含义", "原始与加工产物的全部列（按流水线阶段排列）"):
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                     height=layout.height("l"))

    c1, c2 = layout.split("even")
    with c1:
        with layout.panel("六个属性 ground truth", "属性金标准列的取值、来源与验收方式"):
            _gt_card(b)
    with c2:
        with layout.panel("双模型一致机制（dual_agree）", "两模型极性一致方进入情感统计"):
            _fusion_card(b)


def _gt_card(b: dict) -> None:
    st.markdown(
        "属性金标准列 `_attr_ground_truth` 取值为 **外观 / 内饰 / 空间 / 续航 / 性价比 / 舒适性**"
        "（外加「噪声」水帖）。仿真数据按论文表5.1 的平台×品牌格子与图3.5 关键词分布生成；"
        "真实数据导入后由 TF-IDF + KMeans 聚类（K=6）映射得到，并与该列对照计算纯度。")
    st.caption(f"论文基准：K=6、轮廓系数 {b['clustering']['silhouette']}、"
               f"纯度 {b['clustering']['purity']}、有效评论 {b['clustering']['n_valid']:,} 条。")


def _fusion_card(b: dict) -> None:
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

    fresh = data_store().freshness()
    layout.page_head(
        "数据管理",
        "真实数据导入 · 评论文件上传 · 六阶段流水线 · 数据质量监控 · 字段字典",
        chips=[
            f'<span class="ds-chip chip-{"ok" if fresh["all_ready"] else "warn"}">'
            f'● {fresh["ready"]}/{fresh["total"]} 阶段就绪</span>',
            f'<span class="ds-chip chip-muted">最新产物 {fresh["newest"]} · {fresh["age"]}</span>',
        ])

    # 1) 阶段状态
    section_header("流水线阶段状态", subtitle="产物是否就绪、行数与新鲜度（相对时间）")
    _stage_section()

    # 2) 导入真实评论（干跑 → 校验 → 落盘 → 指路重跑流水线）
    section_header("导入真实评论",
                   subtitle="导入 → 校验 → 落盘 → 重跑流水线；原始文件先备份再覆盖")
    _import_section()

    # 3) 通用上传落盘
    section_header("上传评论 CSV", subtitle="支持多文件；同名覆盖前自动备份到 data/upload_backups/")
    _upload_section()

    # 4) 流水线运行
    section_header("流水线运行", subtitle="清洗分词 → 属性聚类 → 情感分析（脚本较重，逐个执行）")
    _pipeline_section()

    # 5) 数据质量
    section_header("数据质量", subtitle="原始 / 噪声 / 双模型一致 / 保留率 与逐列缺失率")
    _quality_section()

    # 6) 附录
    section_header("附录 · 数据字典", subtitle="字段含义、属性金标准与双模型一致机制")
    _dictionary_section()
