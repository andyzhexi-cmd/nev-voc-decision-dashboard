"""
朴素贝叶斯情感分类器（论文 3.3.2 节）
=================================================
- 输入：分词后的 tokens（list[str]）
- 输出：正面概率 p ∈ [0, 1]
- p > 0.5 → 正面，p < 0.5 → 负面

论文参数：特征词数 3000，MultinomialNB
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def train_and_evaluate(
    df: pd.DataFrame,
    text_col: str = "tokens",
    label_col: str = "_sentiment_ground_truth",
    max_features: int = 3000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[MultinomialNB, TfidfVectorizer, dict]:
    """
    训练朴素贝叶斯情感分类器并评估。

    label_col 的值应为 1（正面）或 -1（负面），噪声评论（label=0）剔除。
    """
    # 只保留有情感标签的评论（剔除噪声）
    sub = df[df[label_col].isin([1, -1])].copy()
    # tokens 列已经是空格分隔的字符串，直接用
    sub["text"] = sub[text_col].astype(str)
    y = (sub[label_col] == 1).astype(int).values  # 1=正面, 0=负面

    # TF-IDF 向量化
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(sub["text"])

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 训练朴素贝叶斯
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # 评估
    y_pred = clf.predict(X_test)
    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "n_features": X_train.shape[1],
    }

    return clf, vectorizer, metrics


def predict_proba(clf: MultinomialNB, vectorizer: TfidfVectorizer, tokens_list: list[list[str]]) -> np.ndarray:
    """批量预测正面概率。"""
    texts = [" ".join(t) for t in tokens_list]
    X = vectorizer.transform(texts)
    proba = clf.predict_proba(X)[:, 1]  # P(正面)
    return proba
