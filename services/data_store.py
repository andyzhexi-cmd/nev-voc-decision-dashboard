"""services.data_store — 统一数据访问层

职责：
  * 统一路径注册表（RAW / PROCESSED / FEATURES / RESULTS）
  * CSV → parquet 中间层（首次转换，后续直读 parquet，实测 17MB CSV → 约 4MB）
  * 列裁剪读取（sentiment 表只读需要的 12 列）
  * 流水线产物状态（阶段/行数/修改时间），供 UI 徽章与数据新鲜度使用

本模块不依赖 Streamlit；UI 侧用 st.cache_data 包一层即可。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PATHS = {
    "raw": ROOT / "data" / "raw" / "comments_raw.csv",
    "cleaned": ROOT / "data" / "processed" / "comments_cleaned.csv",
    "clustered": ROOT / "data" / "processed" / "comments_clustered.csv",
    "sentiment": ROOT / "data" / "processed" / "sentiment_results.csv",
    "attr_sentiment": ROOT / "outputs" / "results" / "attr_sentiment.csv",
    "vikor": ROOT / "outputs" / "results" / "vikor_results.csv",
    "isa": ROOT / "outputs" / "results" / "isa_results.csv",
    "figures": ROOT / "outputs" / "figures",
    "reports": ROOT / "outputs" / "reports",
    "features": ROOT / "data" / "features",
}
FEATURES_DIR: Path = PATHS["features"]

# sentiment 表：只读这些列（其余文本重复列一律裁掉）
SENTIMENT_USECOLS = [
    "comment_id", "platform", "brand", "model", "comment_date", "comment_text",
    "_attr_ground_truth", "_sentiment_ground_truth",
    "vader_compound", "nb_pos_proba", "dual_agree", "final_sentiment",
]

ATTRS = ["外观", "内饰", "空间", "续航", "性价比", "舒适性"]
BASE_COLS = ["comment_id", "platform", "brand", "model", "comment_date", "comment_text"]


@dataclass
class StageStatus:
    name: str
    exists: bool
    path: str
    rows: int = 0
    mtime: float = 0.0
    age: str = ""

    def as_dict(self) -> dict:
        return {"阶段": self.name, "就绪": self.exists, "行数": self.rows,
                "修改时间": time.strftime("%m-%d %H:%M", time.localtime(self.mtime)) if self.mtime else "—",
                "路径": self.path}


def _age(ts: float) -> str:
    if not ts:
        return "—"
    d = time.time() - ts
    if d < 60:
        return f"{int(d)} 秒前"
    if d < 3600:
        return f"{int(d // 60)} 分钟前"
    if d < 86400:
        return f"{int(d // 3600)} 小时前"
    return f"{int(d // 86400)} 天前"


def _read(path: Path, usecols: list[str] | None = None, parquet_fallback: bool = True) -> pd.DataFrame:
    """读取表格；优先 parquet 缓存（列裁剪 + 类型优化）。

    部署约定：CSV 源文件可以不入库（体积大），只要 `data/features/<name>.parquet` 存在，
    即使对应 CSV 缺失也能正常读取；两者都在时取更新的那个。
    """
    pq = FEATURES_DIR / (path.stem + ".parquet")
    fresh_pq = pq.exists() and (not path.exists() or pq.stat().st_mtime >= path.stat().st_mtime)
    if parquet_fallback and fresh_pq:
        try:
            return pd.read_parquet(pq, columns=usecols)
        except Exception:
            pass
    if not path.exists():
        return pd.DataFrame()
    kw = {"usecols": usecols} if usecols else {}
    try:
        df = pd.read_csv(path, **kw)
    except ValueError:
        df = pd.read_csv(path)          # 列不齐时退回全量读
        if usecols:
            df = df[[c for c in usecols if c in df.columns]]
    if parquet_fallback:
        try:
            FEATURES_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(pq, index=False)
        except Exception:
            pass
    return df


class DataStore:
    """数据访问单例（进程内共享）。"""

    # ---------------- tables ----------------
    def raw(self) -> pd.DataFrame:
        return _read(PATHS["raw"])

    def cleaned(self) -> pd.DataFrame:
        return _read(PATHS["cleaned"])

    def clustered(self) -> pd.DataFrame:
        return _read(PATHS["clustered"])

    def sentiment(self) -> pd.DataFrame:
        df = _read(PATHS["sentiment"], usecols=SENTIMENT_USECOLS)
        if df.empty:
            return df
        for c in ("comment_date",):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        return df

    def sentiment_kept(self) -> pd.DataFrame:
        """双模型一致、参与下游统计的评论。"""
        df = self.sentiment()
        if df.empty:
            return df
        return df[df["dual_agree"].astype(bool)].copy()

    def attr_sentiment(self) -> pd.DataFrame:
        """各属性情感汇总（outputs/results/attr_sentiment.csv 或实时计算）。"""
        p = PATHS["attr_sentiment"]
        if p.exists():
            try:
                df = pd.read_csv(p)
                df.columns = ["属性", "mean_sentiment", "n_reviews", "std"][: len(df.columns)]
                return df
            except Exception:
                pass
        kept = self.sentiment_kept()
        if kept.empty:
            return pd.DataFrame(columns=["属性", "mean_sentiment", "n_reviews", "std"])
        g = kept.groupby("_attr_ground_truth")["final_sentiment"].agg(["mean", "count", "std"])
        g.columns = ["mean_sentiment", "n_reviews", "std"]
        return g.reset_index().rename(columns={"_attr_ground_truth": "属性"})

    # ---------------- status ----------------
    def stage_statuses(self) -> list[StageStatus]:
        spec = [("① 原始评论", "raw"), ("② 清洗分词", "cleaned"),
                ("③ 属性聚类", "clustered"), ("④ 情感分析", "sentiment"),
                ("⑤ 决策结果", "vikor"), ("⑥ ISA 矩阵", "isa")]
        out = []
        for name, key in spec:
            p: Path = PATHS[key]
            exists = p.exists()
            rows = 0
            if exists:
                try:
                    rows = sum(1 for _ in open(p, encoding="utf-8")) - 1
                except Exception:
                    rows = 0
            out.append(StageStatus(name=name, exists=exists, path=str(p.relative_to(ROOT)),
                                   rows=rows, mtime=p.stat().st_mtime if exists else 0.0,
                                   age=_age(p.stat().st_mtime) if exists else "—"))
        return out

    def freshness(self) -> dict:
        st = self.stage_statuses()
        ready = [s for s in st if s.exists]
        newest = max((s.mtime for s in ready), default=0.0)
        return {
            "ready": len(ready), "total": len(st),
            "all_ready": len(ready) == len(st),
            "newest": time.strftime("%m-%d %H:%M", time.localtime(newest)) if newest else "—",
            "age": _age(newest),
            "stages": [s.as_dict() for s in st],
        }

    def has(self, key: str) -> bool:
        return PATHS[key].exists()

    def missing_stages(self) -> list[str]:
        return [s.name for s in self.stage_statuses() if not s.exists]


@lru_cache(maxsize=1)
def store() -> DataStore:
    return DataStore()


def reset_cache() -> None:
    store.cache_clear()
    for fn in (raw_df, cleaned_df, clustered_df, sentiment_df):  # noqa: F821
        fn.cache_clear()


# ---------------- 便捷缓存函数（供 st.cache_data 再包一层） ----------------
@lru_cache(maxsize=8)
def raw_df() -> pd.DataFrame:
    return store().raw()


@lru_cache(maxsize=8)
def cleaned_df() -> pd.DataFrame:
    return store().cleaned()


@lru_cache(maxsize=8)
def clustered_df() -> pd.DataFrame:
    return store().clustered()


@lru_cache(maxsize=8)
def sentiment_df() -> pd.DataFrame:
    return store().sentiment()
