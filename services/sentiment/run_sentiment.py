"""
阶段 4：双模型情感分析主流程（论文 3.3.3 节）
=================================================
流程：
  1. 加载清洗后数据（含 ground truth）
  2. VADER 中文情感分析 → compound ∈ [-1,1]
  3. 朴素贝叶斯训练 + 预测 → P(正面) ∈ [0,1]
  4. 双模型融合：
     - VADER > 0 且 NB > 0.5 → 正面，保留
     - VADER < 0 且 NB < 0.5 → 负面，保留
     - 其他（不一致或中性）→ 剔除
  5. 保留的评论，最终情感值 = VADER compound
  6. 与 ground truth 对比，算 Precision/Recall/F1
  7. 按属性分组，算各属性平均情感值（论文表5.5满意度基准）
  8. 输出 results/sentiment_results.csv
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from services.sentiment.vader_cn import analyze as vader_analyze
from services.sentiment.nb_model import train_and_evaluate, predict_proba
from services.preprocess.stopwords import load_stopwords

PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "data" / "processed" / "sentiment_results.csv"
FIG_DIR = ROOT / "outputs" / "figures"
RESULT_DIR = ROOT / "outputs" / "results"


def parse_tokens(tokens_str) -> list[str]:
    """把 CSV 里存的 tokens 字符串还原成 list。"""
    if isinstance(tokens_str, list):
        return tokens_str
    if isinstance(tokens_str, str):
        try:
            import ast
            # 安全解析：CSV 中存的是 Python 列表字面量，禁用 eval()
            return ast.literal_eval(tokens_str)
        except Exception:
            return tokens_str.split()
    return []


def main():
    print("=" * 64)
    print("阶段 4：双模型情感分析（VADER + 朴素贝叶斯）")
    print("=" * 64)

    # 1. 加载数据
    print("\n[1/7] 加载清洗后数据...")
    df = pd.read_csv(PROCESSED / "comments_cleaned.csv")
    print(f"  总评论数: {len(df)}")
    # 过滤噪声评论
    df = df[df["_attr_ground_truth"] != "噪声"].copy()
    print(f"  剔除噪声后: {len(df)}")

    # 解析 tokens
    df["tokens_list"] = df["tokens"].apply(parse_tokens)

    # 2. VADER 情感分析
    print("\n[2/7] VADER 中文情感分析...")
    df["vader_compound"] = df["cleaned_text"].apply(vader_analyze)
    df["vader_label"] = np.where(df["vader_compound"] > 0, 1,
                                  np.where(df["vader_compound"] < 0, -1, 0))
    print(f"  VADER 正面: {(df['vader_label']==1).sum()}")
    print(f"  VADER 负面: {(df['vader_label']==-1).sum()}")
    print(f"  VADER 中性: {(df['vader_label']==0).sum()}")

    # VADER 单独评估（对非中性评论）
    vader_valid = df[df["vader_label"] != 0]
    if len(vader_valid) > 0:
        vader_pred = vader_valid["vader_label"].values
        vader_true = vader_valid["_sentiment_ground_truth"].values
        from sklearn.metrics import precision_score, recall_score, f1_score
        # 二分类：1 vs -1
        vp = (vader_pred == 1).astype(int)
        vt = (vader_true == 1).astype(int)
        print(f"  VADER 单独指标: P={precision_score(vt,vp):.4f}  R={recall_score(vt,vp):.4f}  F1={f1_score(vt,vp):.4f}")

    # 3. 朴素贝叶斯训练
    print("\n[3/7] 朴素贝叶斯训练（3000特征词）...")
    clf, vectorizer, nb_metrics = train_and_evaluate(df, text_col="tokens", label_col="_sentiment_ground_truth")
    print(f"  NB 训练集: {nb_metrics['n_train']} 条")
    print(f"  NB 测试集: {nb_metrics['n_test']} 条")
    print(f"  NB 特征数: {nb_metrics['n_features']}")
    print(f"  NB 单独指标: P={nb_metrics['precision']:.4f}  R={nb_metrics['recall']:.4f}  F1={nb_metrics['f1']:.4f}")
    print(f"  论文基准: VADER P=86.89% R=91.64% F1=89.20%;  NB P=86.24% R=90.36% F1=88.25%")

    # 4. 朴素贝叶斯预测全量
    print("\n[4/7] 朴素贝叶斯全量预测...")
    df["nb_pos_proba"] = predict_proba(clf, vectorizer, df["tokens_list"].tolist())
    df["nb_label"] = np.where(df["nb_pos_proba"] > 0.5, 1, -1)

    # 5. 双模型融合
    print("\n[5/7] 双模型融合（论文 3.3.3 规则）...")
    # 保留两模型判定一致的评论
    # VADER > 0 且 NB > 0.5 → 正面
    # VADER < 0 且 NB < 0.5 → 负面
    agree_pos = (df["vader_compound"] > 0) & (df["nb_pos_proba"] > 0.5)
    agree_neg = (df["vader_compound"] < 0) & (df["nb_pos_proba"] < 0.5)
    df["dual_agree"] = agree_pos | agree_neg
    df["final_sentiment"] = 0.0  # 默认中性/剔除
    df.loc[agree_pos, "final_sentiment"] = df.loc[agree_pos, "vader_compound"]
    df.loc[agree_neg, "final_sentiment"] = df.loc[agree_neg, "vader_compound"]

    n_total = len(df)
    n_kept = df["dual_agree"].sum()
    n_dropped = n_total - n_kept
    print(f"  总评论: {n_total}")
    print(f"  双模型一致（保留）: {n_kept} ({n_kept/n_total*100:.1f}%)")
    print(f"  不一致/中性（剔除）: {n_dropped} ({n_dropped/n_total*100:.1f}%)")

    # 6. 融合后指标
    print("\n[6/7] 融合后指标评估...")
    kept = df[df["dual_agree"]].copy()
    if len(kept) > 0:
        y_true = (kept["_sentiment_ground_truth"] == 1).astype(int)
        y_pred = (kept["final_sentiment"] > 0).astype(int)
        from sklearn.metrics import precision_score, recall_score, f1_score
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f = f1_score(y_true, y_pred)
        print(f"  双模型融合: P={p:.4f}  R={r:.4f}  F1={f:.4f}")
        print(f"  论文基准:   P=90.56% R=93.25% F1=91.89%")

    # 7. 按属性分组算平均情感值
    print("\n[7/7] 各属性平均情感值（论文表5.5满意度基准）...")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    attr_sent = kept.groupby("_attr_ground_truth")["final_sentiment"].agg(["mean", "count", "std"])
    attr_sent.columns = ["mean_sentiment", "n_reviews", "std"]
    # 论文基准
    paper_sent = {
        "性价比": 0.8239,
        "舒适性": 0.6082,
        "外观": 0.4164,
        "空间": 0.3996,
        "内饰": 0.3943,
        "续航": 0.2858,
    }
    print(f"  {'属性':<6} {'本产品':>8} {'论文':>8} {'评论数':>8}")
    for attr in ["外观", "内饰", "空间", "续航", "性价比", "舒适性"]:
        if attr in attr_sent.index:
            ours = attr_sent.loc[attr, "mean_sentiment"]
            n = int(attr_sent.loc[attr, "n_reviews"])
            ref = paper_sent.get(attr, "-")
            print(f"  {attr:<6} {ours:>8.4f} {ref:>8} {n:>8}")

    # 保存结果
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"\n  已保存: {OUTPUT}")

    # 保存属性情感值表
    attr_sent.to_csv(RESULT_DIR / "attr_sentiment.csv", encoding="utf-8-sig")
    print(f"  已保存: {RESULT_DIR / 'attr_sentiment.csv'}")

    print("\n" + "=" * 64)
    print("阶段 4 完成")
    print("=" * 64)


if __name__ == "__main__":
    main()
