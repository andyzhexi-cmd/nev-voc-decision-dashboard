"""
中文 VADER 风格情感分析器
=================================================
模拟论文 3.3.1 节 VADER 情感词典：
  - 输出情感复合分数 compound ∈ [-1, 1]
  - compound > 0 → 正面
  - compound < 0 → 负面
  - compound = 0 → 中性

实现：基于中文情感词典的加权词袋模型
  - 正面词 / 负面词词典
  - 否定词反转（不、没、无）
  - 程度副词加权（很、非常、太、超）
"""
from __future__ import annotations
import re
from pathlib import Path

import jieba

# ---------------- 情感词典 ----------------
POSITIVE_WORDS = {
    "好", "满意", "喜欢", "棒", "优秀", "好看", "舒服", "方便", "划算",
    "便宜", "省钱", "安静", "平稳", "宽敞", "大气", "漂亮", "精致",
    "流畅", "稳定", "扎实", "值", "棒", "赞", "不错", "强", "给力",
    "实用", "耐用", "优惠", "给力", "舒服", "静谧", "舒适", "安静",
    "满意", "喜欢", "值得", "良心", "丰富", "友好", "到位", "干净",
    "均匀", "清晰", "漂亮", "帅气", "年轻", "运动", "协调", "有档次",
    "扎实", "省心", "便宜", "划算", "省钱", "耐用", "宽敞", "够用",
    "充裕", "灵活", "纯平", "低", "快", "稳", "高", "准", "亮",
}

NEGATIVE_WORDS = {
    "差", "失望", "讨厌", "贵", "麻烦", "颠簸", "吵", "硬", "小", "慢",
    "卡", "脏", "丑", "薄", "乱", "缺", "减", "虚标", "衰减", "高",
    "窄", "挤", "顶头", "不够", "不足", "差劲", "粗糙", "普通",
    "老气", "小气", "单薄", "简陋", "塑料感", "卡顿", "慢半拍",
    "一般", "差", "难闻", "臭", "硬", "累", "难受", "不舒服",
    "贵", "不值", "少", "高", "快", "凶", "吵", "响", "颠簸",
    "挤", "顶", "局促", "小", "少", "高", "快", "高", "大",
}

NEGATION_WORDS = {"不", "没", "无", "非", "未", "别", "莫"}

INTENSIFIERS = {
    "很": 1.5, "非常": 2.0, "太": 1.8, "超": 1.6, "特别": 1.7,
    "真": 1.3, "好": 1.2, "挺": 1.2, "比较": 1.1, "有点": 0.8,
    "稍微": 0.7, "略": 0.8, "巨": 2.0, "贼": 1.8,
}


def analyze(text: str) -> float:
    """
    分析单条评论的情感，返回 compound ∈ [-1, 1]。

    算法：
      1. jieba 分词
      2. 遍历每个词：
         - 正面词：+1 * 强度系数
         - 负面词：-1 * 强度系数
         - 前面是否定词：符号反转
         - 前面是否程度副词：乘以强度
      3. 归一化：(正分 + 负分) / sqrt(词数)，裁剪到 [-1, 1]
    """
    if not text or not text.strip():
        return 0.0

    tokens = list(jieba.cut(text))
    if not tokens:
        return 0.0

    score = 0.0
    n_pos = 0
    n_neg = 0

    for i, tok in enumerate(tokens):
        # 检查前面是否有否定词
        negated = False
        intensifier = 1.0
        if i > 0:
            prev = tokens[i - 1]
            if prev in NEGATION_WORDS:
                negated = True
            if prev in INTENSIFIERS:
                intensifier = INTENSIFIERS[prev]

        if tok in POSITIVE_WORDS:
            s = intensifier
            if negated:
                s = -s
            score += s
            n_pos += 1
        elif tok in NEGATIVE_WORDS:
            s = -intensifier
            if negated:
                s = -s
            score += s
            n_neg += 1

    # 归一化到 [-1, 1]
    total = len(tokens)
    if total == 0:
        return 0.0

    compound = score / (total ** 0.5)
    # 裁剪到 [-1, 1]
    compound = max(-1.0, min(1.0, compound))
    return compound


def analyze_batch(texts: list[str]) -> list[float]:
    """批量分析。"""
    return [analyze(t) for t in texts]
