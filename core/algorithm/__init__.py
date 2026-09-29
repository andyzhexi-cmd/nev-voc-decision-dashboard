"""core.algorithm — PLTS-VIKOR 多属性决策纯算法层

设计约束：
  * 零 Streamlit 依赖、零文件系统写入（只读 config/baselines.yaml）
  * 纯函数 + pydantic 结果模型，100% 可单元测试
  * 所有中间矩阵随结果返回，支撑 UI 的"可解释性"与"审计"能力
"""
from core.algorithm.models import (  # noqa: F401
    AHPResult,
    DEMATELResult,
    AuditFinding,
    REPAIR_FLAGS,
    Scenario,
    VIKORResult,
    default_repair_flags,
    load_baselines,
    baselines,
)
from core.algorithm.plts import PLTS, build_decision_matrix, normalize_columns  # noqa: F401
from core.algorithm.ahp import ahp_weights, reconstruct_consistent  # noqa: F401
from core.algorithm.dematel import dematel_weights  # noqa: F401
from core.algorithm.weights import combine_weights, expand_to_criteria  # noqa: F401
from core.algorithm.vikor import paper_reference, run_vikor  # noqa: F401
from core.algorithm.sensitivity import (  # noqa: F401
    compare_to_paper_sensitivity,
    kendall_tau,
    sweep_lambda,
    sweep_v,
)
from core.algorithm.benchmarks import compare_methods  # noqa: F401
from core.algorithm.isa import isa_quadrants  # noqa: F401
from core.algorithm.audit import audit_summary, run_audit  # noqa: F401
