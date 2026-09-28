"""
AHP 层次分析法权重计算（论文 4.2.1 节，表5.7-5.8）
=================================================
一级指标：期望值 C1、敏感值 C2、吸引值 C3
判断矩阵（表5.7）：
     C1   C2   C3
C1    1    5    4
C2   1/5   1   1/7
C3   1/4   7    1

论文结果：wA = (0.434, 0.187, 0.379)，CR = 0.0846
"""
from __future__ import annotations
import numpy as np


# 表5.7 非专家组一级指标 1-9 标度打分表
AHP_MATRIX = np.array([
    [1.0,   5.0,   4.0],
    [1/5.0, 1.0,  1/7.0],
    [1/4.0, 7.0,   1.0],
])

INDICATORS = ["期望值C1", "敏感值C2", "吸引值C3"]

# 随机一致性指标 RI（n=3 时 RI=0.52，论文表4.2）
RI_TABLE = {1: 0, 2: 0, 3: 0.52, 4: 0.89, 5: 1.12, 6: 1.24}


def ahp_weights(matrix: np.ndarray = AHP_MATRIX) -> dict:
    """
    用方根法计算 AHP 权重。

    返回：
      weights: 归一化权重向量
      lambda_max: 最大特征值
      CI: 一致性指标
      CR: 一致性比率
      n: 阶数
    """
    n = matrix.shape[0]

    # 步骤1：逐行求几何平均
    row_geo = np.prod(matrix, axis=1) ** (1.0 / n)

    # 步骤2：归一化得到权重
    weights = row_geo / row_geo.sum()

    # 步骤3：计算最大特征值
    Aw = matrix @ weights
    lambda_max = np.sum(Aw / (n * weights))

    # 步骤4：一致性检验
    CI = (lambda_max - n) / (n - 1)
    RI = RI_TABLE.get(n, 1.24)  # 默认 n=6
    CR = CI / RI if RI > 0 else 0.0

    return {
        "weights": weights,
        "lambda_max": lambda_max,
        "CI": CI,
        "CR": CR,
        "n": n,
        "RI": RI,
        "consistent": CR < 0.1,
    }


if __name__ == "__main__":
    result = ahp_weights()
    print("=" * 50)
    print("AHP 权重计算（论文表5.7-5.8）")
    print("=" * 50)
    print(f"判断矩阵：")
    for i, row in enumerate(AHP_MATRIX):
        print(f"  {INDICATORS[i]:<8} {row}")
    print()
    print(f"权重 wA（方根法计算）:")
    for i, ind in enumerate(INDICATORS):
        print(f"  {ind}: {result['weights'][i]:.4f}")
    print(f"  论文表5.8: (0.434, 0.187, 0.379)")
    print(f"  λmax = {result['lambda_max']:.4f}")
    print(f"  CR   = {result['CR']:.4f}  (论文 0.0846)")
    print()
    print("  注：论文最终权重采用专家校准结果，本系统直接录入论文权重。")
