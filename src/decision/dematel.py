"""
DEMATEL 决策试验与评价实验室法权重计算（论文 4.2.2 节，表5.9-5.11）
=================================================
专家组给出二级指标间直接关联矩阵 Z（12×12）
语言术语：l0=无影响(0), l1=影响低(1), l2=影响较低(2), l3=影响适中(3), l4=影响高(4)

论文结果：
  二级指标权重（表5.10）
  一级指标权重（表5.11）：wD = (0.516, 0.204, 0.280)
"""
from __future__ import annotations
import numpy as np


# 语言术语映射
L_MAP = {"l0": 0, "l1": 1, "l2": 2, "l3": 3, "l4": 4}

# 二级指标名
SUB_INDICATORS = [
    "c11", "c12", "c13", "c14",  # 期望值
    "c21", "c22", "c23", "c24",  # 敏感值
    "c31", "c32", "c33", "c34",  # 吸引值
]

# 表5.9 专家组直接关联矩阵（l0~l4）
# 行=影响方，列=受影响方
Z_RAW = [
    # c11  c12  c13  c14  c21  c22  c23  c24  c31  c32  c33  c34
    ["l0","l1","l1","l1","l3","l4","l4","l4","l3","l4","l4","l3"],  # c11
    ["l0","l0","l0","l0","l1","l2","l2","l3","l1","l1","l2","l3"],  # c12
    ["l1","l2","l0","l3","l1","l1","l1","l2","l1","l2","l2","l3"],  # c13
    ["l1","l4","l1","l0","l2","l1","l1","l4","l1","l2","l3","l4"],  # c14
    ["l0","l0","l1","l1","l0","l3","l3","l4","l4","l2","l2","l4"],  # c21
    ["l0","l1","l0","l0","l3","l0","l4","l0","l0","l1","l3","l2"],  # c22
    ["l0","l2","l0","l0","l1","l3","l0","l2","l1","l2","l2","l1"],  # c23
    ["l0","l3","l0","l0","l4","l3","l3","l0","l4","l3","l3","l3"],  # c24
    ["l0","l1","l0","l0","l1","l2","l2","l2","l0","l3","l3","l2"],  # c31
    ["l1","l1","l0","l0","l1","l1","l2","l4","l4","l0","l3","l3"],  # c32
    ["l0","l1","l0","l0","l1","l1","l1","l2","l2","l0","l0","l4"],  # c33
    ["l0","l1","l0","l0","l1","l1","l1","l2","l2","l4","l0","l0"],  # c34
]


def dematel_weights(z_matrix: list[list[str]] = Z_RAW) -> dict:
    """
    DEMATEL 计算：
      1. 语言矩阵 → 数值矩阵 Z
      2. 标准化：Z̄ = Z / max(sum(Z))
      3. 综合影响矩阵 T = Z̄ × (I - Z̄)^(-1)
      4. 中心度 SR+SC、原因度 SR-SC
      5. 重要度 Wj = sqrt((SR+SC)^2 + (SR-SC)^2)
      6. 归一化
    """
    # 转数值矩阵
    Z = np.array([[L_MAP[v] for v in row] for row in z_matrix], dtype=float)

    # 标准化
    col_sums = Z.sum(axis=0)
    max_sum = col_sums.max()
    Z_bar = Z / max_sum

    # 综合影响矩阵 T = Z_bar * (I - Z_bar)^(-1)
    n = Z_bar.shape[0]
    I = np.eye(n)
    T = Z_bar @ np.linalg.inv(I - Z_bar)

    # 行和 SR（影响度）、列和 SC（被影响度）
    SR = T.sum(axis=1)  # 行和
    SC = T.sum(axis=0)  # 列和

    # 中心度、原因度
    centrality = SR + SC
    causality = SR - SC

    # 重要度
    W = np.sqrt(centrality**2 + causality**2)

    # 归一化
    weights = W / W.sum()

    return {
        "weights_sub": weights,
        "SR": SR,
        "SC": SC,
        "centrality": centrality,
        "causality": causality,
        "W": W,
    }


def aggregate_to_first_level(weights_sub: np.ndarray) -> dict:
    """把 12 个二级指标权重聚合到 3 个一级指标。"""
    wC1 = weights_sub[0:4].sum()
    wC2 = weights_sub[4:8].sum()
    wC3 = weights_sub[8:12].sum()
    return {
        "期望值C1": wC1,
        "敏感值C2": wC2,
        "吸引值C3": wC3,
    }


# 论文表5.10 给出的二级指标权重（校准用）
PAPER_SUB_WEIGHTS = {
    "c11": 0.189, "c12": 0.067, "c13": 0.153, "c14": 0.107,
    "c21": 0.051, "c22": 0.106, "c23": 0.016, "c24": 0.031,
    "c31": 0.077, "c32": 0.085, "c33": 0.039, "c34": 0.079,
}

# 论文表5.11 一级指标权重
PAPER_FIRST_WEIGHTS = {
    "期望值C1": 0.516,
    "敏感值C2": 0.204,
    "吸引值C3": 0.280,
}


if __name__ == "__main__":
    result = dematel_weights()
    print("=" * 60)
    print("DEMATEL 权重计算（论文表5.9-5.11）")
    print("=" * 60)
    print("\n二级指标权重（计算值 vs 论文表5.10）：")
    for i, name in enumerate(SUB_INDICATORS):
        paper = PAPER_SUB_WEIGHTS[name]
        calc = result["weights_sub"][i]
        print(f"  {name}: 计算 {calc:.4f}  论文 {paper:.3f}")

    first = aggregate_to_first_level(result["weights_sub"])
    print("\n一级指标权重（计算值 vs 论文表5.11）：")
    for k in ["期望值C1", "敏感值C2", "吸引值C3"]:
        print(f"  {k}: 计算 {first[k]:.4f}  论文 {PAPER_FIRST_WEIGHTS[k]:.3f}")
