"""部署数据包维护脚本：检查 / 重建 / 剔除中间 CSV。

背景（详见 部署/03-数据与产物体积.md）
------------------------------------
产品运行时的取数顺序是：**先看 data/features/<表名>.parquet，再看同名 CSV**，
两者都在时取更新的那份；只要 parquet 在，对应 CSV 缺失也能正常工作。
所以公有云部署只需要提交：

    data/features/*.parquet      （5 个文件，约 4 MB）
    data/raw/comments_raw.csv    （原始评论，约 5.6 MB，数据管理页的质量体检要用）

而 data/processed/*.csv（约 38 MB）不用入库。

用法（在 产品实现/ 目录执行）：
    ./vene/bin/python 部署/prepare_public_data.py                 # 检查现状（默认）
    ./vene/bin/python 部署/prepare_public_data.py --build         # 从 processed CSV 重建 parquet
    ./vene/bin/python 部署/prepare_public_data.py --prune-csv --yes   # 删除 processed CSV，模拟云端布局
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FEATURES = ROOT / "data" / "features"
RAW_CSV = ROOT / "data" / "raw" / "comments_raw.csv"
PROCESSED = ROOT / "data" / "processed"
PACKAGE = [
    FEATURES / "comments_raw.parquet",
    FEATURES / "comments_cleaned.parquet",
    FEATURES / "comments_clustered.parquet",
    FEATURES / "sentiment_results.parquet",
    FEATURES / "comment_index.parquet",
]


def _mb(p: pathlib.Path) -> float:
    return p.stat().st_size / 1024 / 1024 if p.exists() else 0.0


def _rows(p: pathlib.Path) -> str:
    try:
        if p.suffix == ".parquet":
            import pyarrow.parquet as pq
            return f"{pq.ParquetFile(p).metadata.num_rows:,}"
        return f"{sum(1 for _ in p.open('rb')) - 1:,}"
    except Exception:
        return "—"


def check() -> int:
    print("=" * 74)
    print("部署数据包现状")
    print("=" * 74)
    total = 0.0
    for p in [RAW_CSV, *PACKAGE]:
        ok = p.exists()
        total += _mb(p)
        rel = p.relative_to(ROOT)
        print(f"  {'✓' if ok else '✗'} {str(rel):<42} {_mb(p):6.2f} MB  行数 {_rows(p) if ok else '—'}")
    print(f"  —— 合计 {total:.2f} MB（仓库里 data/processed 的 CSV 不入库，约 "
          f"{sum(_mb(f) for f in PROCESSED.glob('*.csv')):.1f} MB 已被排除）")
    missing = [p for p in [RAW_CSV, *PACKAGE] if not p.exists()]
    if missing:
        print("\n缺少文件，可执行：./vene/bin/python 部署/prepare_public_data.py --build")
        return 1
    print("\n结论：部署数据包齐全 ✓（可直接推送/构建镜像）")
    return 0


def build() -> int:
    from services.data_store import store
    from services.features import comment_index

    ds = store()
    print("从 data/ 重新生成 parquet 缓存……")
    for name in ("raw", "cleaned", "clustered", "sentiment"):
        df = getattr(ds, name)()
        print(f"  {name:<10} rows={len(df):>6}  cols={len(df.columns)}")
    idx = comment_index()
    print(f"  {'index':<10} rows={len(idx):>6}  cols={len(idx.columns)}")
    print("\n生成后：")
    return check()


def prune_csv(assume_yes: bool) -> int:
    missing = [p for p in PACKAGE if not p.exists()]
    if missing:
        print("parquet 不齐全，先执行 --build：", [str(p.relative_to(ROOT)) for p in missing])
        return 1
    victims = sorted(PROCESSED.glob("*.csv"))
    if not victims:
        print("data/processed/ 下没有 CSV，无需处理")
        return 0
    size = sum(_mb(v) for v in victims)
    print("将删除（均为 gitignore 的中间产物，可由 data/raw 重跑流水线再生）：")
    for v in victims:
        print(f"  - {v.relative_to(ROOT)}  {_mb(v):.2f} MB")
    print(f"合计 {size:.2f} MB")
    if not assume_yes:
        print("\n这是破坏性操作：确认后重跑并加 --yes")
        return 2
    for v in victims:
        v.unlink()
    print("\n已删除。恢复方式：把 data/raw/comments_raw.csv 放回 data/raw/，"
          "在「数据管理 → 三阶段执行器」重跑清洗分词 → 属性聚类 → 情感分析。")
    return check()


def main() -> int:
    ap = argparse.ArgumentParser(description="部署数据包检查 / 重建 / 剔除中间 CSV")
    ap.add_argument("--build", action="store_true", help="从 processed CSV 重建 parquet 缓存")
    ap.add_argument("--prune-csv", action="store_true", help="删除 data/processed/*.csv（需 --yes）")
    ap.add_argument("--yes", action="store_true", help="确认破坏性操作")
    args = ap.parse_args()
    if args.prune_csv:
        return prune_csv(args.yes)
    if args.build:
        return build()
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
