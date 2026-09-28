"""
真实评论数据导入适配器（阶段1预留）
=================================================
用途：八爪鱼采集器/手工整理导出的 CSV、Excel 经此模块转换为
与仿真数据完全一致的 schema，后续所有算法模块统一读这个 schema。

支持字段（列名不区分大小写，可别名）：
  - platform   平台        别名: 来源/网站/source/platform
  - brand      品牌        别名: 品牌/brand/car_brand
  - model      车型        别名: 车型/model/car_model
  - comment_date 发布时间  别名: 时间/日期/date/publish_time
  - comment_text 评论内容  别名: 评论/内容/text/content/comment

真实数据没有 _attr_ground_truth / _sentiment_ground_truth，这两列留空，
算法模块据此自动判断"这是真实数据，不做 ground truth 对照"。
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

COLUMN_ALIASES = {
    "platform":     ["platform", "来源", "网站", "source", "website", "平台"],
    "brand":        ["brand", "品牌", "car_brand"],
    "model":        ["model", "车型", "car_model"],
    "comment_date": ["comment_date", "时间", "日期", "date", "publish_time", "发布时间"],
    "comment_text": ["comment_text", "评论", "内容", "text", "content", "comment", "评论内容"],
}


def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """把任意别名映射到标准列名。"""
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    out = pd.DataFrame()
    for std, aliases in COLUMN_ALIASES.items():
        matched = None
        for a in aliases:
            if a.lower() in lower_map:
                matched = lower_map[a.lower()]
                break
        out[std] = df[matched] if matched else ""
    return out


def import_comments(path: str | Path) -> pd.DataFrame:
    """导入 CSV / Excel，返回标准 schema 的 DataFrame。"""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df = _resolve_columns(df)

    # 清洗：去空评论、去重
    df["comment_text"] = df["comment_text"].astype(str).str.strip()
    df = df[df["comment_text"].str.len() > 0]
    df = df.drop_duplicates(subset=["comment_text"]).reset_index(drop=True)

    # 真实数据无 ground truth 列
    df["_attr_ground_truth"] = ""
    df["_sentiment_ground_truth"] = 0
    df.insert(0, "comment_id", range(1, len(df) + 1))

    print(f"已导入真实数据: {path.name} -> {len(df)} 条")
    return df


if __name__ == "__main__":
    # 自测：python importer.py <文件路径>
    import sys
    if len(sys.argv) > 1:
        result = import_comments(sys.argv[1])
        print(result.head())
    else:
        print("用法: python importer.py <comments.csv 或 .xlsx>")
