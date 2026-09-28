"""
阶段 6：ISA（Importance-Satisfaction Analysis）四象限矩阵
=================================================
论文 5.6 节，表5.20：
  重要性均值 3.55（σ=0.466）
  满意度均值 3.43（σ=0.338）

象限划分：
  第一象限（保持区）：高重要性 + 高满意度
  第二象限（机会区）：低重要性 + 高满意度
  第三象限（低优先级区）：低重要性 + 低满意度
  第四象限（改进区）：高重要性 + 低满意度
"""
from __future__ import annotations
from pathlib import Path
import sys

import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["PingFang HK", "PingFang SC", "Heiti TC"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "outputs" / "figures"
RESULT_DIR = ROOT / "outputs" / "results"

# 论文表5.20 ISA数据
ISA_DATA = pd.DataFrame({
    "属性": ["外观", "内饰", "空间", "续航", "性价比", "舒适性"],
    "重要性": [3.0, 3.2, 3.3, 4.3, 4.0, 3.5],
    "满意度": [3.1, 3.4, 3.5, 2.8, 4.2, 3.3],
})

IMPORTANCE_MEAN = 3.55
IMPORTANCE_STD = 0.466
SATISFACTION_MEAN = 3.43
SATISFACTION_STD = 0.338


def classify_quadrant(row: pd.Series) -> str:
    """根据重要性和满意度均值划分象限。"""
    high_imp = row["重要性"] >= IMPORTANCE_MEAN
    high_sat = row["满意度"] >= SATISFACTION_MEAN

    if high_imp and high_sat:
        return "保持区"
    elif not high_imp and high_sat:
        return "机会区"
    elif not high_imp and not high_sat:
        return "低优先级区"
    else:
        return "改进区"


def plot_isa(df: pd.DataFrame, out_path: Path):
    """绘制ISA四象限矩阵图（论文图5.10）。"""
    fig, ax = plt.subplots(figsize=(9, 7))

    # 象限背景色
    ax.axhspan(SATISFACTION_MEAN, 4.5, xmin=0, xmax=1, alpha=0.08, color="green")
    ax.axhspan(2.5, SATISFACTION_MEAN, xmin=0, xmax=1, alpha=0.05, color="red")

    # 分隔线
    ax.axvline(IMPORTANCE_MEAN, color="gray", linestyle="--", linewidth=1)
    ax.axhline(SATISFACTION_MEAN, color="gray", linestyle="--", linewidth=1)

    # 象限标签
    ax.text(3.0, 4.25, "机会区", fontsize=13, color="gray", ha="center")
    ax.text(4.05, 4.25, "保持区", fontsize=13, color="gray", ha="center")
    ax.text(3.0, 2.65, "低优先级区", fontsize=13, color="gray", ha="center")
    ax.text(4.05, 2.65, "改进区", fontsize=13, color="gray", ha="center")

    # 颜色映射
    color_map = {
        "保持区": "#2ecc71",
        "机会区": "#3498db",
        "低优先级区": "#95a5a6",
        "改进区": "#e74c3c",
    }

    # 画散点
    for _, row in df.iterrows():
        q = row["象限"]
        ax.scatter(row["重要性"], row["满意度"], s=200, c=color_map[q], zorder=5, edgecolors="white", linewidth=1.5)
        ax.annotate(f"{row['属性']}\n({row['重要性']:.1f},{row['满意度']:.1f})",
                    (row["重要性"], row["满意度"]),
                    textcoords="offset points", xytext=(12, 5), fontsize=10)

    ax.set_xlabel("重要性（Importance）", fontsize=12)
    ax.set_ylabel("满意度（Satisfaction）", fontsize=12)
    ax.set_title("ISA 四象限矩阵（重要性-满意度分析）", fontsize=14, fontweight="bold")
    ax.set_xlim(2.5, 4.5)
    ax.set_ylim(2.5, 4.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {out_path}")


def main():
    print("=" * 64)
    print("阶段 6：ISA 四象限矩阵分析")
    print("=" * 64)

    df = ISA_DATA.copy()
    df["象限"] = df.apply(classify_quadrant, axis=1)

    print(f"\n  重要性均值: {IMPORTANCE_MEAN}（σ={IMPORTANCE_STD}）")
    print(f"  满意度均值: {SATISFACTION_MEAN}（σ={SATISFACTION_STD}）")
    print(f"\n  {'属性':<6} {'重要性':>6} {'满意度':>6} {'象限':>10}")
    print("  " + "-" * 36)
    for _, row in df.iterrows():
        print(f"  {row['属性']:<6} {row['重要性']:>6.1f} {row['满意度']:>6.1f} {row['象限']:>10}")

    # 论文对照
    print(f"\n  论文图5.10对照:")
    print(f"    保持区: 性价比 (4.0, 4.0)")
    print(f"    改进区: 续航 (4.3, 2.8)")
    print(f"    机会区: 外观(3.0,3.5), 空间(3.3,3.5)")
    print(f"    低优先级区: 内饰 (3.2, 3.2)")

    # 画图
    print(f"\n[绘图] ISA四象限图...")
    plot_isa(df, FIG_DIR / "03_isa_matrix.png")

    # 保存数据
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULT_DIR / "isa_results.csv", index=False, encoding="utf-8-sig")
    print(f"  已保存: {RESULT_DIR / 'isa_results.csv'}")

    print("\n" + "=" * 64)
    print("阶段 6 完成")
    print("=" * 64)


if __name__ == "__main__":
    main()
