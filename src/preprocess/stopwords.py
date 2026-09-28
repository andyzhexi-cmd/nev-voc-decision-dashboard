"""
停用词加载（论文：哈工大 + 百度停用词表整合，3257 个）
"""
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[2] / "assets"
COMBINED = ASSETS / "combined_stopwords.txt"
BUSINESS = ASSETS / "business_stopwords.txt"


def load_stopwords() -> set[str]:
    """加载停用词集合：基础表 + 业务停用词（仿真模板词/通用情感语气词）。"""
    words = set()
    for path in [COMBINED, BUSINESS]:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    words.add(w)
    if not words:
        raise FileNotFoundError(f"未找到停用词表: {COMBINED}")
    return words


if __name__ == "__main__":
    sw = load_stopwords()
    print(f"停用词总数: {len(sw)}")
    print(f"示例: {list(sw)[:20]}")
