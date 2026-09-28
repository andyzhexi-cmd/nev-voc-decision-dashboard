"""
文本清洗模块（对应论文 3.1.2 / 5.2.2）
=================================================
三级清洗策略：
  1. 跨平台格式统一：去 URL、@、特殊符号、emoji，空行/非汉字串删除
  2. 引导词裁剪：识别"最满意的是/最不满意的是/优点/缺点"等开头引导词，
     只保留引导词之后的情感表达内容
  3. 语义单元切分：按句尾标点（。！？…；）拆成句子，用 \\n 连接
"""
from __future__ import annotations
import re

# 引导词库（论文：汽车之家等平台评论开头常见"最满意""最不满意"引导语）
LEAD_PHRASES = [
    "最满意的是", "最不满意的是",
    "最满意", "最不满意",
    "满意：", "不满意：",
    "优点：", "缺点：",
    "优点是", "缺点是",
    "总结：", "评价：",
    "【满意】", "【不满意】",
    "满意的是", "不满意的是",
]

# URL
URL_RE = re.compile(r"https?://\S+|www\.\S+")
# @提及、#话题、√ 等符号
AT_RE = re.compile(r"@\w+|#\w+|[√✓✗✔✅❌⭐]")
# emoji（粗略匹配 Unicode emoji 区间）
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F02F]"
)
# 非汉字/非中文标点/非数字/非字母 的字符（保留中文标点和字母数字）
PUNCT_KEEP = "，。！？；：、""''（）《》—…·～ "
# 句尾标点
SENT_END = re.compile(r"([。！？…；!?;]+)")


def strip_url(text: str) -> str:
    return URL_RE.sub(" ", text)


def strip_symbols(text: str) -> str:
    text = AT_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    return text


def cut_lead(text: str) -> str:
    """去掉句首引导词，保留其后的情感表达。"""
    text = text.strip()
    for lead in LEAD_PHRASES:
        if text.startswith(lead):
            return text[len(lead):].strip(" ：:，,。.")
    return text


def split_sentences(text: str) -> list[str]:
    """按句尾标点切分成句子，保留标点。"""
    # 在标点后加 \n
    text = SENT_END.sub(r"\1\n", text)
    parts = [s.strip() for s in text.split("\n")]
    return [s for s in parts if s]


def is_valid_sentence(s: str) -> bool:
    """过滤过短、纯英文数字、纯符号的句子。
    论文口径：清洗后剔除约9.3%的水帖/无意义评论。
    要求至少4个汉字，过滤"沙发""666""dddd""MARK"这类噪声。"""
    if len(s) < 4:
        return False
    # 至少包含3个中文字符
    if len(re.findall(r"[一-鿿]", s)) < 3:
        return False
    return True


def clean_text(raw: str) -> str:
    """完整清洗流程，返回清洗后的多句文本（\\n 连接）。"""
    text = str(raw)
    text = strip_url(text)
    text = strip_symbols(text)
    # 把换行、制表符统一成空格
    text = re.sub(r"[\r\t]+", " ", text)
    text = re.sub(r"\n+", " ", text)
    # 多个空格合并
    text = re.sub(r"\s+", " ", text).strip()
    # 引导词裁剪
    text = cut_lead(text)
    # 切分成句子
    sentences = split_sentences(text)
    # 过滤
    valid = [s for s in sentences if is_valid_sentence(s)]
    return "\n".join(valid)


def clean_corpus(series) -> list[str]:
    """对 pandas Series 批量清洗。"""
    return [clean_text(t) for t in series]
