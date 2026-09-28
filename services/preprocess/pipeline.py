"""
预处理主流程（对应论文第三章 3.1.2 / 5.2.2）
=================================================
流程：
  原始评论 → 三级清洗 → jieba 分词 → 停用词过滤 → 清洗后语料

输出：data/processed/comments_cleaned.csv
  字段：comment_id, platform, brand, model, comment_date,
        raw_text, cleaned_text, tokens,
        _attr_ground_truth, _sentiment_ground_truth
"""
from __future__ import annotations
from pathlib import Path
import re

import pandas as pd
import jieba

ROOT = Path(__file__).resolve().parents[1]

from services.preprocess.clean import clean_text
from services.preprocess.stopwords import load_stopwords

RAW = ROOT / "data" / "raw" / "comments_raw.csv"
OUT = ROOT / "data" / "processed" / "comments_cleaned.csv"


def tokenize(text: str, stopwords: set[str]) -> str:
    """jieba 分词 + 停用词过滤，返回空格分隔的词串。
    额外过滤：纯标点、纯数字、单字无意义词。"""
    if not text:
        return ""
    words = jieba.lcut(text)
    kept = []
    for w in words:
        w = w.strip()
        if not w:
            continue
        if w in stopwords:
            continue
        # 纯标点过滤
        if re.fullmatch(r"[，。！？；：、""''（）《》—…·～,.!?;:\-\s]+", w):
            continue
        # 纯数字/英文短串过滤
        if re.fullmatch(r"[a-zA-Z0-9]+", w) and len(w) < 4:
            continue
        kept.append(w)
    return " ".join(kept)


def run():
    print("加载原始数据...")
    df = pd.read_csv(RAW)
    n_raw = len(df)
    print(f"  原始: {n_raw} 条")

    print("加载停用词...")
    stopwords = load_stopwords()
    print(f"  停用词: {len(stopwords)} 个")

    print("三级清洗...")
    df["cleaned_text"] = [clean_text(t) for t in df["comment_text"]]

    # 清洗后为空的评论剔除（噪声/纯符号/无意义）
    before = len(df)
    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)
    after = len(df)
    print(f"  清洗剔空: {before} → {after} 条（剔除 {before-after} 条）")

    print("jieba 分词 + 停用词过滤...")
    df["tokens"] = [tokenize(t, stopwords) for t in df["cleaned_text"]]

    # tokens 为空的也剔除（分词后无有效词）
    before = len(df)
    df = df[df["tokens"].str.len() > 0].reset_index(drop=True)
    after = len(df)
    print(f"  分词剔空: {before} → {after} 条（剔除 {before-after} 条）")

    # 保留需要的列
    out_cols = [
        "comment_id", "platform", "brand", "model", "comment_date",
        "comment_text", "cleaned_text", "tokens",
    ]
    for c in ["_attr_ground_truth", "_sentiment_ground_truth"]:
        if c in df.columns:
            out_cols.append(c)

    out = df[out_cols]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {OUT}  ({len(out)} 条)")

    # 对照报告
    print("\n" + "=" * 60)
    print("预处理对照")
    print("=" * 60)
    print(f"原始评论:        {n_raw:>6} 条")
    print(f"清洗+分词后:     {len(out):>6} 条")
    print(f"剔除比例:        {(n_raw-len(out))/n_raw*100:.1f}%")
    print(f"论文口径:        原始 50520 → 有效 45836（剔除 9.3%）")
    print(f"停用词数量:      {len(stopwords)} 个（论文口径 3257）")

    # 抽样对比
    print("\n【清洗前后对比样例 3 条】")
    sample = out.sample(min(3, len(out)), random_state=42)
    for _, r in sample.iterrows():
        print(f"\n  原文: {r['comment_text'][:60]}")
        print(f"  清洗: {r['cleaned_text'][:60]}")
        print(f"  分词: {r['tokens'][:60]}")

    # 高频词 Top10（论文图3.5对照：空间16385/座椅9761/内饰7849/舒服7682/外观6273...）
    print("\n【分词后高频词 Top 10】")
    from collections import Counter
    all_words = " ".join(out["tokens"]).split()
    top = Counter(all_words).most_common(10)
    for w, c in top:
        print(f"  {w:<6} {c:>6}")
    print("=" * 60)


if __name__ == "__main__":
    run()
