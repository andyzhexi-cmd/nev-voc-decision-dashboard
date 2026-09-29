"""core.algorithm.models — 数据模型与配置加载（pydantic v2）"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]
BASELINES_PATH = ROOT / "config" / "baselines.yaml"


# ---------------------------------------------------------------- config
@lru_cache(maxsize=1)
def load_baselines() -> dict:
    """加载论文基准配置（进程内缓存，配置改了需重启/清缓存）。"""
    with open(BASELINES_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def baselines() -> dict:
    return load_baselines()


# ---------------------------------------------------------------- 修复开关
REPAIR_FLAGS: tuple[str, ...] = (
    "ahp_reconstruct",       # AHP：按论文权重反推一致性矩阵
    "dematel_bottom_up",     # DEMATEL：一级权重自底向上聚合（论文口径）
    "q_rebuild",             # Q：由表5.16 的 S/R 按式(4.11) 重建可复算基线
    "sensitivity_anchor",    # 敏感性：锚定表5.17 为 λ=0.5 对照基准
)


def default_repair_flags() -> dict:
    """默认四项修复全开（审计默认不允许出现无法解释的 fail）。"""
    return {k: True for k in REPAIR_FLAGS}


class _Arr(BaseModel):
    """带 numpy 支持的基类。"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


# ---------------------------------------------------------------- AHP / DEMATEL
class AHPResult(_Arr):
    weights: np.ndarray                 # 归一化权重
    lambda_max: float
    CI: float
    CR: float
    RI: float
    n: int
    consistent: bool
    paper_weights: np.ndarray | None = None    # 论文表5.8
    deviation: float | None = None             # 与论文最大绝对偏差
    reconstructed: bool = False                # 是否使用了「按权重反推的一致性矩阵」


class DEMATELResult(_Arr):
    weights_sub: np.ndarray             # 12 个二级指标权重
    weights_first: np.ndarray           # 3 个一级指标权重
    SR: np.ndarray                      # 影响度（行和）
    SC: np.ndarray                      # 被影响度（列和）
    centrality: np.ndarray
    causality: np.ndarray
    T: np.ndarray                       # 综合影响矩阵
    paper_sub: np.ndarray | None = None
    paper_first: np.ndarray | None = None
    deviation: float | None = None
    aggregation: str = "top_down"       # 一级权重聚合口径：top_down / bottom_up


# ---------------------------------------------------------------- 场景参数
class Scenario(_Arr):
    """一次决策计算的全部输入参数。"""
    mode: Literal["calibrated", "live"] = "calibrated"   # 论文校准 / 在线复算
    lam: float = 0.5                                     # 式(4.7) 融合系数
    v: float = 0.5                                       # 式(4.11) 决策偏好
    ideal_strategy: Literal["criterion", "sentiment"] = "criterion"
    weight_mode: Literal["dematel_within", "pure_sub", "equal_within", "first_only"] = "dematel_within"
    prob_completion: Literal["interval", "normalize"] = "interval"
    direction: Literal["attainment", "shortfall"] = "attainment"   # 效用方向
    use_paper_weights: bool = True                      # 校准模式下采用论文表5.8/5.11 权重
    weight_overrides_3: tuple[float, float, float] | None = None   # 用户直接改一级权重
    sentiment_ideals: tuple[list[float], list[float]] | None = None  # (f*, f-) 来自实测情感
    repair_flags: dict[str, bool] = Field(default_factory=default_repair_flags)  # 四项偏差修复开关

    @property
    def key(self) -> str:
        return (f"{self.mode}|λ={self.lam:.3f}|v={self.v:.3f}|{self.ideal_strategy}|"
                f"{self.weight_mode}|{self.prob_completion}|{self.direction}|"
                f"{self.use_paper_weights}|{self.weight_overrides_3}|{self.sentiment_ideals}|"
                f"{tuple(sorted(self.repair_flags.items()))}")


# ---------------------------------------------------------------- VIKOR 结果
class VIKORResult(_Arr):
    """PLTS-VIKOR 全链路结果（含全部中间矩阵）。"""
    attributes: list[str]
    criteria: list[str]
    scenario: Scenario

    # PLTS 层
    F_point: np.ndarray                 # 6×12 期望得分（点估计）
    F_lo: np.ndarray                    # 概率补全下界
    F_hi: np.ndarray                    # 概率补全上界
    F_norm: np.ndarray                  # 式(4.8) 规范化矩阵
    prob_sum_deviation: float           # Σp<1 的格子数

    # 权重层
    w3: np.ndarray                      # 一级综合权重
    w12: np.ndarray                     # 12 指标权重（Σ=1）

    # 理想解
    f_star: np.ndarray
    f_minus: np.ndarray

    # VIKOR 层
    attainment: np.ndarray              # 6×12 归一化效用矩阵
    S: np.ndarray
    R: np.ndarray
    Q: np.ndarray
    Q_prime: np.ndarray                 # 正态区间数
    sigma: np.ndarray                   # 正态区间标准差
    S_lo: np.ndarray
    S_hi: np.ndarray
    R_lo: np.ndarray
    R_hi: np.ndarray
    Q_lo: np.ndarray
    Q_hi: np.ndarray

    # 可能度与排序
    P: np.ndarray                       # 6×6 可能度矩阵
    Pxi: np.ndarray
    ranking: list[str]                  # P(x) 降序（重要性由高到低）
    Q_order: list[str]                  # Q 升序（传统 VIKOR 优劣）

    # 诊断
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    # ---- 便捷视图 ----
    def as_records(self) -> list[dict]:
        out = []
        for i, a in enumerate(self.attributes):
            out.append({
                "属性": a, "S_i": float(self.S[i]), "R_i": float(self.R[i]),
                "Q_i": float(self.Q[i]), "Q'_i": float(self.Q_prime[i]),
                "P(x_i)": float(self.Pxi[i]),
            })
        return out

    def rank_pairs(self) -> list[tuple[str, float]]:
        order = np.argsort(-self.Pxi)
        return [(self.attributes[i], float(self.Pxi[i])) for i in order]


# ---------------------------------------------------------------- 审计
class AuditFinding(_Arr):
    id: str
    severity: Literal["high", "medium", "low"]
    title: str
    detail: str
    status: Literal["pass", "warn", "fail", "info", "resolved"] = "info"
    numbers: dict[str, Any] = Field(default_factory=dict)
    # --- 整改四件套（修复记录；老字段语义不变，视图可继续只读 status/detail/numbers）---
    root_cause: str = ""                                    # 取证结论（含关键数字）
    repair: dict[str, Any] = Field(default_factory=dict)     # {label, effect, enabled}
    before: dict[str, Any] = Field(default_factory=dict)     # 修复前偏差数字
    after: dict[str, Any] = Field(default_factory=dict)      # 修复后偏差数字
