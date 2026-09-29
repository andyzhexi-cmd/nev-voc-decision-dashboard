"""
真实评论数据导入适配器（阶段 1 · 生产接入）
=================================================
用途：把八爪鱼采集器 / 平台导出 / 手工整理的 CSV、TSV、Excel、JSON 转换为
与仿真数据**完全一致**的标准 schema，后续流水线与算法统一读这个 schema。

标准 schema（顺序即写入 `data/raw/comments_raw.csv` 的顺序）：
    comment_id, platform, brand, model, comment_date, comment_text,
    _attr_ground_truth, _sentiment_ground_truth

设计要点：
  * 编码嗅探：utf-8-sig → utf-8 → gb18030 → gbk → latin-1（国内平台导出多为 GBK）
  * 分隔符嗅探：逗号 / 制表符 / 分号 / 竖线（csv.Sniffer + 兜底）
  * 列名别名：中英文混写、大小写不敏感，见 COLUMN_ALIASES；可用 mapping 显式覆盖
  * 校验：`comment_text` 为必需列，缺失给出可执行的中文报错；其余列缺失只给警告
  * 干跑：`dry_run=True` 只出报告不落盘，供 UI 预览
  * 落盘：同名旧文件先备份到 `data/upload_backups/`，再原子替换
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

STANDARD_COLUMNS = ["comment_id", "platform", "brand", "model",
                    "comment_date", "comment_text",
                    "_attr_ground_truth", "_sentiment_ground_truth"]
REQUIRED = ["comment_text"]
OPTIONAL = ["platform", "brand", "model", "comment_date"]

COLUMN_ALIASES = {
    "platform":     ["platform", "来源", "网站", "source", "website", "平台",
                     "发布平台", "media"],
    "brand":        ["brand", "品牌", "car_brand", "车系", "厂商"],
    "model":        ["model", "车型", "car_model", "车款", "车系型号"],
    "comment_date": ["comment_date", "时间", "日期", "date", "publish_time",
                     "发布时间", "评论时间", "发布日期", "created_at"],
    "comment_text": ["comment_text", "评论", "内容", "text", "content", "comment",
                     "评论内容", "正文", "comment_content", "body"],
}

ENCODINGS = ["utf-8-sig", "utf-8", "gb18030", "gbk", "big5", "latin-1"]
SEPARATORS = [",", "\t", ";", "|"]

DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%Y年%m月%d日", "%Y.%m.%d",
                "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M"]


class ImportValidationError(ValueError):
    """导入校验失败：报错信息是给用户看的、可执行的中文提示。"""


@dataclass
class ImportReport:
    """一次导入的体检报告（可直接渲染到 UI）。"""
    filename: str = ""
    encoding: str = ""
    separator: str = ""
    columns_raw: list = field(default_factory=list)
    column_mapping: dict = field(default_factory=dict)   # 标准列 -> 源列（None=缺失）
    rows_raw: int = 0
    rows_empty: int = 0
    rows_duplicate: int = 0
    rows_kept: int = 0
    rows_bad_date: int = 0
    missing_required: list = field(default_factory=list)
    missing_optional: list = field(default_factory=list)
    date_range: tuple | None = None
    brands: list = field(default_factory=list)
    models: list = field(default_factory=list)
    platforms: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    sample: pd.DataFrame | None = None
    wrote_to: str | None = None

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("sample", None)
        d["date_range"] = tuple(self.date_range) if self.date_range else None
        return d


# ------------------------------------------------------------------ 读取与嗅探
def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def sniff_encoding(blob: bytes) -> str:
    for enc in ENCODINGS:
        try:
            blob.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    return "latin-1"


def sniff_separator(text: str, sample_lines: int = 50) -> str:
    head = "\n".join(text.splitlines()[:sample_lines])
    try:
        dialect = csv_sniffer().sniff(head, delimiters="".join(SEPARATORS))
        return dialect.delimiter
    except Exception:
        counts = {s: head.count(s) for s in SEPARATORS}
        return max(counts, key=counts.get) or ","


def csv_sniffer():
    import csv
    return csv.Sniffer()


def load_table(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """读入任意支持的文件，返回 (DataFrame, 嗅探元信息)。"""
    path = Path(path)
    if not path.exists():
        raise ImportValidationError(f"文件不存在：{path}")
    meta = {"filename": path.name, "encoding": "", "separator": ""}
    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xls", ".xlsm"):
        df = pd.read_excel(path)
        meta.update(encoding="xlsx", separator="")
        return df, meta
    if suffix == ".json":
        df = pd.read_json(path, encoding="utf-8")
        meta.update(encoding="utf-8", separator="json")
        return df, meta

    blob = _read_bytes(path)
    enc = sniff_encoding(blob)
    text = blob.decode(enc, errors="replace")
    if suffix in (".csv", ".tsv", ".txt") or not suffix:
        sep = "\t" if suffix == ".tsv" else sniff_separator(text)
        df = pd.read_csv(io.StringIO(text), sep=sep)
        meta.update(encoding=enc, separator=repr(sep))
        return df, meta
    # 未知扩展名：按文本尝试
    sep = sniff_separator(text)
    df = pd.read_csv(io.StringIO(text), sep=sep)
    meta.update(encoding=enc, separator=repr(sep))
    return df, meta


# ------------------------------------------------------------------ 列映射
def detect_mapping(df: pd.DataFrame) -> dict:
    """标准列 -> 源列名（未命中为 None）。大小写、空格、中英文别名均不敏感。"""
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    mapping = {}
    for std, aliases in COLUMN_ALIASES.items():
        hit = None
        for a in aliases:
            if a.lower() in lower_map:
                hit = lower_map[a.lower()]
                break
        mapping[std] = hit
    return mapping


def validate_mapping(df: pd.DataFrame, mapping: dict) -> tuple[list, list]:
    """返回 (errors, warnings)。errors 非空则拒绝导入。"""
    errors, warnings = [], []
    missing_required = [c for c in REQUIRED if not mapping.get(c)]
    if missing_required:
        errors.append(
            f"缺少必需列「评论内容」：已识别的列有 {list(df.columns)}；"
            f"请把评论正文列命名为 评论内容 / comment_text / text / content 之一，"
            f"或在上方的列映射里手动指定。")
    for c in OPTIONAL:
        if not mapping.get(c):
            warnings.append(f"未识别到「{c}」列，对应字段将以空值导入（"
                            f"{'品牌/车型会影响分组与排名' if c in ('brand', 'model') else '不影响算法主链路'}）。")
    return errors, warnings


# ------------------------------------------------------------------ 标准化
def _apply_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for std, src in mapping.items():
        out[std] = df[src].values if src else ""
    return out


def _parse_dates(series: pd.Series) -> tuple[pd.Series, int]:
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    if parsed.isna().mean() > 0.5:          # 整体失败时逐格式重试
        for fmt in DATE_FORMATS:
            retry = pd.to_datetime(series, errors="coerce", format=fmt)
            if retry.notna().mean() > parsed.notna().mean():
                parsed = retry
    bad = int(parsed.isna().sum())
    return parsed, bad


def standardize(df_raw: pd.DataFrame, mapping: dict, *, dedupe: bool = False,
                min_len: int = 1) -> tuple[pd.DataFrame, ImportReport]:
    """按映射产出标准 schema，并完成清洗与统计。"""
    rep = ImportReport(rows_raw=int(len(df_raw)))
    df = _apply_mapping(df_raw, mapping)

    df["comment_text"] = (df["comment_text"].astype(str).str.strip()
                          .replace({"nan": "", "None": ""}))
    empty_mask = df["comment_text"].str.len() < max(1, min_len)
    rep.rows_empty = int(empty_mask.sum())
    df = df[~empty_mask]

    rep.rows_duplicate = int(df.duplicated(subset=["comment_text"]).sum())
    if dedupe and rep.rows_duplicate:
        df = df.drop_duplicates(subset=["comment_text"])

    df["comment_date"], rep.rows_bad_date = _parse_dates(df["comment_date"])
    df["comment_date"] = df["comment_date"].dt.strftime("%Y-%m-%d").fillna("")

    for col in OPTIONAL:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().fillna("").replace({"nan": ""})

    df["_attr_ground_truth"] = ""
    df["_sentiment_ground_truth"] = 0
    df = df.reset_index(drop=True)
    df.insert(0, "comment_id", range(1, len(df) + 1))
    df = df[STANDARD_COLUMNS]

    rep.rows_kept = int(len(df))
    rep.date_range = _range(df["comment_date"])
    rep.brands = sorted({v for v in df["brand"].tolist() if v})[:200]
    rep.models = sorted({v for v in df["model"].tolist() if v})[:500]
    rep.platforms = sorted({v for v in df["platform"].tolist() if v})[:50]
    rep.sample = df.head(10)
    return df, rep


def _range(series: pd.Series):
    vals = sorted({v for v in series.tolist() if v})
    return (vals[0], vals[-1]) if vals else None


# ------------------------------------------------------------------ 落盘
def write_raw(df: pd.DataFrame, dest: str | Path | None = None) -> Path:
    """原子写入 data/raw/，同名旧文件先备份。"""
    from services.data_store import PATHS
    dest = Path(dest) if dest else PATHS["raw"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        backup_dir = dest.parent.parent / "upload_backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.joinpath(f"{dest.stem}_{stamp}{dest.suffix}").write_bytes(dest.read_bytes())
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(dest)
    return dest


def import_comments(path: str | Path, mapping: dict | None = None, *,
                    dry_run: bool = False, dedupe: bool = False) -> tuple[pd.DataFrame, ImportReport]:
    """导入真实评论：读取 → 映射 → 校验 → 标准化 →（可选）落盘。"""
    raw, meta = load_table(path)
    mapping = dict(mapping) if mapping else detect_mapping(raw)
    errors, warnings = validate_mapping(raw, mapping)
    # 列错位检测：pandas 只在"某行字段数 > 表头数"时把首列吞成索引，出现 object 索引
    # 或 Unnamed 列基本可以断定正文里的逗号/换行没被引号包裹。
    if str(getattr(raw.index, "dtype", "")) == "object" or \
            any(str(c).startswith("Unnamed") for c in raw.columns):
        errors.append(
            "检测到列错位（出现多余的索引列或 Unnamed 列）：正文中的逗号/换行未被正确引用。"
            "请在 Excel 中另存为「CSV UTF-8（逗号分隔）」，或直接上传 .xlsx 文件。")

    df, rep = standardize(raw, mapping, dedupe=dedupe)
    rep.__dict__.update(meta)
    rep.column_mapping = dict(mapping)
    rep.missing_required = [c for c in REQUIRED if not mapping.get(c)]
    rep.missing_optional = [c for c in OPTIONAL if not mapping.get(c)]
    rep.errors = list(errors)
    rep.warnings = list(warnings)

    if rep.rows_raw and rep.rows_empty / max(1, rep.rows_raw) > 0.5 and not rep.errors:
        rep.errors.append(
            "超过 50% 的行「评论内容」为空，通常是正文里的逗号/换行导致列错位："
            "请用带引号的 CSV（Excel 另存为「CSV UTF-8」格式）或直接上传 .xlsx。")
    if rep.rows_kept == 0 and not rep.errors:
        rep.errors.append("清洗后没有可用评论（正文全为空），请检查源文件。")
    if rep.rows_duplicate and dedupe:
        rep.warnings.append(f"按正文去重删除了 {rep.rows_duplicate} 条重复评论。")
    elif rep.rows_duplicate:
        rep.warnings.append(
            f"检测到 {rep.rows_duplicate} 条正文完全相同的评论（默认不去重；"
            f"灌水类短评会在清洗阶段被识别为噪声）。")
    if rep.rows_bad_date:
        rep.warnings.append(f"{rep.rows_bad_date} 条发布时间无法解析，已留空（不影响算法主链路）。")

    if rep.errors:
        return df, rep

    if not dry_run:
        rep.wrote_to = str(write_raw(df))
    return df, rep


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        frame, report = import_comments(sys.argv[1], dry_run=("--dry-run" in sys.argv))
        print(f"编码 {report.encoding} 分隔符 {report.separator}")
        print(f"原始 {report.rows_raw} → 保留 {report.rows_kept}（空 {report.rows_empty} / 重复 {report.rows_duplicate}）")
        print("映射:", report.column_mapping)
        print("告警:", *report.warnings, sep="\n  - ")
        if report.errors:
            print("错误:", *report.errors, sep="\n  - ")
    else:
        print("用法: python importer.py <comments.csv|.xlsx> [--dry-run]")
