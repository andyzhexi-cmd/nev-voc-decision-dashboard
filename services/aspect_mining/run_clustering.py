"""
阶段3：特征属性识别（论文 3.2 / 5.3）
=================================================
步骤：
  1. TF-IDF 向量化，统计词频，取 Top400 高频词
  2. 剔除不含 Top400 词的评论（论文：50520 → 45836）
  3. K=1..10 跑 K-means，记录 SSE（肘部法则）和轮廓系数
  4. 确定最优 K（论文：K=6，轮廓系数 0.598）
  5. K=6 聚类，每簇取 TF-IDF 最高的前 10 个词命名主题
  6. 计算聚类纯度（论文：0.853）
  7. 输出图：肘部+轮廓系数图、簇分布图
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["PingFang HK", "PingFang SC", "Heiti TC", "STHeiti"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]

CLEAN = ROOT / "data" / "processed" / "comments_cleaned.csv"
FIG_DIR = ROOT / "outputs" / "figures"
OUT_DIR = ROOT / "data" / "processed"

# 论文基准
PAPER = {
    "n_raw": 50520,
    "n_valid": 45836,
    "top_n_words": 400,
    "best_k": 6,
    "best_silhouette": 0.598,
    "purity": 0.853,
    "cluster_sizes": {"外观": 11147, "内饰": 9722, "空间": 7875,
                      "续航": 6101, "性价比": 5697, "舒适性": 5292},
}


def step1_load():
    print("=" * 64)
    print("步骤 1：加载清洗后数据")
    print("=" * 64)
    df = pd.read_csv(CLEAN)
    print(f"  清洗后评论数: {len(df)}")
    print(f"  字段: {list(df.columns)}")
    return df


def step2_tfidf(df):
    print("\n" + "=" * 64)
    print("步骤 2：TF-IDF 向量化")
    print("=" * 64)
    # 用 tokens 列（已分词去停用词）
    texts = df["tokens"].fillna("").tolist()
    vectorizer = TfidfVectorizer(max_features=5000)  # 先取5000候选词
    X = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    print(f"  向量维度: {X.shape}（{len(df)} 条评论 × {len(vocab)} 个词）")
    return X, vectorizer, vocab


def step3_top_words(df, vocab, X):
    print("\n" + "=" * 64)
    print(f"步骤 3：统计词频，取 Top{PAPER['top_n_words']} 高频词")
    print("=" * 64)
    # 词频 = X 中非零元素个数（即有多少条评论包含该词）
    word_count = np.array(X.sum(axis=0)).flatten()  # 用TF-IDF总分排序更合理
    # 实际词频：统计每个词在多少条评论中出现
    doc_freq = np.array(X.getnnz(axis=0)).flatten()

    word_df = pd.DataFrame({
        "word": vocab,
        "doc_freq": doc_freq,
        "tfidf_sum": word_count,
    }).sort_values("doc_freq", ascending=False)

    top_words = word_df.head(PAPER["top_n_words"])["word"].tolist()
    print(f"  Top {PAPER['top_n_words']} 高频词（前15个）:")
    for i, row in word_df.head(15).iterrows():
        print(f"    {row['word']:<8} 出现 {int(row['doc_freq']):>5} 条评论")

    # 论文图3.5对照：空间16385/座椅9761/内饰7849/舒服7682/外观6273/价格4776/耗电3456/优惠2474/科技2238/宽敞1529
    print(f"\n  论文图3.5 Top10 词频对照:")
    paper_top10 = ["空间", "座椅", "内饰", "舒服", "外观", "价格", "耗电", "优惠", "科技", "宽敞"]
    for w in paper_top10:
        row = word_df[word_df["word"] == w]
        if len(row):
            print(f"    {w:<6} 本文档 {int(row['doc_freq'].values[0]):>5} 条  |  论文")
        else:
            print(f"    {w:<6} 未出现  |  论文")

    return top_words, word_df


def step4_filter(df, top_words):
    print("\n" + "=" * 64)
    print(f"步骤 4：剔除不含 Top{PAPER['top_n_words']} 词的评论")
    print("=" * 64)
    top_set = set(top_words)
    # 判断每条评论的 tokens 是否包含至少一个 top 词
    has_top = df["tokens"].fillna("").apply(
        lambda t: any(w in top_set for w in t.split())
    )
    n_before = len(df)
    df_valid = df[has_top].reset_index(drop=True)
    n_after = len(df_valid)
    print(f"  剔除前: {n_before} 条")
    print(f"  剔除后: {n_after} 条")
    print(f"  剔除: {n_before - n_after} 条（{(n_before-n_after)/n_before*100:.1f}%）")
    print(f"  论文口径: {PAPER['n_raw']} → {PAPER['n_valid']}（剔除 {(PAPER['n_raw']-PAPER['n_valid'])/PAPER['n_raw']*100:.1f}%）")
    return df_valid


def step5_k_scan(X_valid, df_valid):
    print("\n" + "=" * 64)
    print("步骤 5：K=1..10 扫描（肘部法则 + 轮廓系数）")
    print("=" * 64)
    sse_list = []
    sil_list = []
    k_range = range(1, 11)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_valid)
        sse = km.inertia_
        sse_list.append(sse)
        if k == 1:
            sil_list.append(np.nan)  # 轮廓系数需要至少2类
            print(f"  K={k}: SSE={sse:.2f}  轮廓=N/A")
        else:
            sil = silhouette_score(X_valid, labels, sample_size=min(2000, len(df_valid)), random_state=42)
            sil_list.append(sil)
            print(f"  K={k}: SSE={sse:.2f}  轮廓系数={sil:.3f}")

    print(f"\n  论文图5.6 对照:")
    print(f"    论文 K=6 轮廓系数 = {PAPER['best_silhouette']}")
    best_k_idx = int(np.nanargmax(sil_list))
    best_k = list(k_range)[best_k_idx]
    best_sil = sil_list[best_k_idx]
    print(f"    本文档最优 K={best_k}  轮廓系数={best_sil:.3f}")

    return sse_list, sil_list, list(k_range)


def step6_plot_elbow(k_range, sse_list, sil_list):
    print("\n" + "=" * 64)
    print("步骤 6：绘制肘部法则 + 轮廓系数图（论文图5.6）")
    print("=" * 64)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    color1 = "#d97706"
    ax1.set_xlabel("K 值")
    ax1.set_ylabel("SSE（误差平方和）", color=color1)
    ax1.plot(k_range, sse_list, "o-", color=color1, label="肘部法则 SSE")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "#2563eb"
    ax2.set_ylabel("轮廓系数", color=color2)
    ax2.plot(k_range, sil_list, "s-", color=color2, label="轮廓系数")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("肘部法则与轮廓系数确定最优 K 值（论文图5.6）")
    fig.tight_layout()
    path = FIG_DIR / "01_elbow_silhouette.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {path}")


def step7_final_cluster(X_valid, df_valid, vectorizer, k=6):
    print("\n" + "=" * 64)
    print(f"步骤 7：K={k} 最终聚类 + 每簇 Top10 关键词")
    print("=" * 64)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_valid)
    df_valid = df_valid.copy()
    df_valid["cluster"] = labels

    # 每簇 TF-IDF 平均权重，取 Top10
    centroids = km.cluster_centers_  # shape (k, n_features)
    vocab = vectorizer.get_feature_names_out()
    cluster_keywords = {}
    for i in range(k):
        top_idx = centroids[i].argsort()[-10:][::-1]
        words = [vocab[j] for j in top_idx]
        cluster_keywords[f"Cluster_{i+1}"] = words
        size = (labels == i).sum()
        print(f"  Cluster_{i+1}（{size} 条）: {', '.join(words)}")

    # 论文表5.4 对照
    print(f"\n  论文表5.4 各簇主题词对照:")
    paper_clusters = {
        "A1(外观)": "外观 耐看 车身 颜色 造型 大气 大灯 简约 颜值 轮毂",
        "A2(内饰)": "内饰 方向盘 屏幕 座椅 真皮 包裹 音响 空调 环保 材质",
        "A3(空间)": "宽敞 空间 紧凑 前排 后排 够用 后备箱 腿部 舒服 感受",
        "A4(续航)": "续航 电耗 充电 电池 公里 能耗 充电快 稳定 电能 充电桩",
        "A5(性价比)": "价格 便宜 实用 家用 够用 耐用 省钱 满意 有点贵 接受",
        "A6(舒适性)": "舒服 隔音 噪音 满足 体验 方便 加热 调节 科技 平稳",
    }
    for k_name, words in paper_clusters.items():
        print(f"    {k_name}: {words}")

    return df_valid, cluster_keywords, km


def step8_purity(df_valid):
    print("\n" + "=" * 64)
    print("步骤 8：聚类纯度（论文 0.853）")
    print("=" * 64)
    if "_attr_ground_truth" not in df_valid.columns:
        print("  （真实数据无 ground truth，跳过纯度计算）")
        return None

    # 纯度 = 每簇中最多真实类别的数量 / 总数
    ct = pd.crosstab(df_valid["cluster"], df_valid["_attr_ground_truth"])
    print(f"  簇 × 真实属性 交叉表:")
    print(ct.to_string())

    purity = ct.max(axis=1).sum() / ct.values.sum()
    print(f"\n  聚类纯度 = {purity:.3f}")
    print(f"  论文纯度 = {PAPER['purity']}")

    # 簇大小分布
    print(f"\n  各簇大小:")
    size_dist = df_valid.groupby("cluster").size()
    for c, n in size_dist.items():
        print(f"    Cluster_{c+1}: {n} 条（{n/len(df_valid)*100:.2f}%）")

    return purity


def step9_plot_dist(df_valid):
    print("\n" + "=" * 64)
    print("步骤 9：簇分布柱状图（论文图5.7）")
    print("=" * 64)
    size_dist = df_valid.groupby("cluster").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [f"Cluster_{i+1}" for i in size_dist.index]
    pct = size_dist.values / size_dist.values.sum() * 100
    ax.barh(names, pct, color="#2563eb")
    ax.set_xlabel("占比（%）")
    ax.set_title("聚类类别分布（论文图5.7）")
    for i, (n, p) in enumerate(zip(size_dist.values, pct)):
        ax.text(p + 0.3, i, f"{p:.2f}%", va="center", fontsize=9)
    plt.tight_layout()
    path = FIG_DIR / "02_cluster_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {path}")


def main():
    df = step1_load()
    X, vectorizer, vocab = step2_tfidf(df)
    top_words, word_df = step3_top_words(df, vocab, X)
    df_valid = step4_filter(df, top_words)

    # 对过滤后的子集重新做 TF-IDF（聚类用）
    texts_valid = df_valid["tokens"].fillna("").tolist()
    X_valid = vectorizer.transform(texts_valid)

    sse_list, sil_list, k_range = step5_k_scan(X_valid, df_valid)
    step6_plot_elbow(k_range, sse_list, sil_list)

    df_valid, cluster_keywords, km = step7_final_cluster(X_valid, df_valid, vectorizer, k=6)
    purity = step8_purity(df_valid)
    step9_plot_dist(df_valid)

    # 保存聚类结果
    out_path = OUT_DIR / "comments_clustered.csv"
    df_valid.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  聚类结果已保存: {out_path}")

    print("\n" + "=" * 64)
    print("阶段 3 完成")
    print("=" * 64)


if __name__ == "__main__":
    main()
