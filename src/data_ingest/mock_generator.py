"""
仿真评论数据生成器（路线A）
=================================================
严格复现论文分布：
  - 表5.1：5平台 × 5品牌 原始评论量
  - 表5.3：K=6 聚类后各簇（属性）评论数
  - 表5.5：各属性 正向/负向 评论数
  - 论文口径：约9.3% 噪声评论（不含高频词的水帖/广告/纯表情），
              清洗后有效评论约 45836 条

输出：data/raw/comments_raw.csv
  列：comment_id, platform, brand, model, comment_date, comment_text,
      _attr_ground_truth, _sentiment_ground_truth
  （_ 前缀两列是仿真数据的"标准答案"，真实数据没有；
     用于阶段3聚类纯度、阶段4情感准确率的对照验证）
"""
from __future__ import annotations
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "settings.yaml"
OUTPUT_PATH = ROOT / "data" / "raw" / "comments_raw.csv"


# ---------------- 文本模板库 ----------------
# 每条模板用 {brand}/{model} 做占位符，模拟车主真实口语

POS_TEMPLATES = {
    "外观": [
        "{model}的外观真好看，车身线条很流畅",
        "大灯设计很有辨识度，颜值在线",
        "停路边回头率高，轮毂造型也漂亮",
        "车身颜色很正，造型简约大气",
        "外观耐看，越看越喜欢",
        "前脸设计有科技感，车身比例协调",
        "外观是我选它的最大原因，太帅了",
        "车漆质感不错，LED大灯晚上很亮",
        "整体造型年轻运动，符合我的审美",
        "尾部设计层次感强，辨识度高",
    ],
    "内饰": [
        "内饰用料扎实，方向盘握感舒服",
        "屏幕清晰流畅，音响效果不错",
        "座椅真皮包裹感好，几乎没异味",
        "内饰环保，新车提回来没什么味道",
        "中控台设计简约，材质显档次",
        "空调制冷快，出风口设计合理",
        "内饰做工精细，接缝均匀",
        "座椅调节方便，包裹性好",
        "氛围灯加分，晚上内饰很有感觉",
        "扶手箱空间实用，材质手感不错",
    ],
    "空间": [
        "前排空间宽敞，腿部活动自如",
        "后排坐三个人不挤，后备箱够用",
        "空间比同级别大，家用很合适",
        "紧凑型车但乘坐空间体验不错",
        "后排腿部空间充裕，地台几乎纯平",
        "后备箱放两个行李箱没问题",
        "储物格多，杯架设计实用",
        "头部空间也不错，180cm坐进去不压抑",
        "空间灵活，后排座椅能放倒",
        "一家人出行空间完全够用",
    ],
    "续航": [
        "续航扎实，市区通勤一周充一次就行",
        "充电速度快，半小时能从30到80",
        "实际续航和表显差不多，能耗控制得好",
        "高速续航也没打折太多，满意",
        "电池稳定，开了一年续航没怎么掉",
        "充电桩多的地方很方便，日常通勤无焦虑",
        "电耗低，市区百公里才13度电",
        "续航达成率高，冬天也能有八折",
        "快充接口友好，充电方便",
        "满电跑长途中途充一次就够",
    ],
    "性价比": [
        "这个价格买到这样的配置很值，性价比高",
        "便宜又实用，用着省钱，家用很划算",
        "优惠后价格便宜，性价比没得说",
        "保养便宜，用车成本低，省钱省心",
        "同价位里配置最良心，价格很划算",
        "降价后入手的，便宜了两万，买值了",
        "耐用省心，后期没什么开销，省钱",
        "送的权益多，充电免费，性价比高",
        "价格亲民，配置齐全，便宜大碗",
        "性价比高，便宜又好开，值得入手",
        "优惠力度大，价格便宜，划算",
        "省钱又耐用，性价比没话说",
    ],
    "舒适性": [
        "座椅舒服，隔音效果比预期好，噪音小",
        "底盘滤震不错，过减速带很平稳，悬挂软",
        "高速噪音小，座舱静谧，长途坐着不累",
        "悬挂调校偏舒适，减震到位，家人坐着不晕",
        "NVH做得好，关窗世界都安静，隔音棒",
        "底盘滤震干净，细碎颠簸过滤得好，平稳",
        "隔音出色，风噪胎噪都很小，静谧性好",
        "座椅舒服，悬挂软，减震到位，乘坐舒适",
        "噪音控制优秀，高速行驶很安静，隔音好",
        "底盘质感扎实，滤震干净，悬挂舒适",
        "静谧性好，放音乐很享受，噪音小",
        "座椅舒服，底盘稳，过弯不侧倾，悬挂舒服",
    ],
}

NEG_TEMPLATES = {
    "外观": [
        "外观中规中矩，颜色选择太少",
        "造型不太合我审美，大灯有点小气",
        "车漆偏薄，小石子就崩掉漆",
        "外观普通，停在街上没什么辨识度",
        "轮毂样式老气，不太喜欢",
    ],
    "内饰": [
        "内饰塑料感强，和宣传不符",
        "屏幕偶尔卡顿，反应慢半拍",
        "音响效果一般，还不如手机外放",
        "新车味道大，晒了两周还是有味",
        "内饰接缝不均，做工糙了点",
    ],
    "空间": [
        "后排头部空间局促，180以上顶头",
        "后备箱太小，婴儿车都放不下",
        "后排中间地台隆起高，坐中间难受",
        "储物空间少，手机都没地方放",
    ],
    "续航": [
        "冬天续航直接打五折，电耗太高了",
        "跑长途充电要排队，续航焦虑严重",
        "续航虚标，实际能跑的比表显少一截",
        "电池衰减快，开了一年明显变短",
        "夏天开空调续航掉得飞快",
        "充电桩太少，出远门不方便",
        "快充速度一般，充一次要等很久",
        "续航达成率低，高速只能跑六成",
    ],
    "性价比": [
        "价格太贵了，同价位竞品便宜两万",
        "优惠力度小，等了半年不降价，不值",
        "保养不便宜，售后收费高，省钱无门",
        "配置同价位偏少，很多要加钱选装，贵",
        "价格虚高，优惠少，性价比低",
    ],
    "舒适性": [
        "高速风噪胎噪大，隔音差，耳朵难受",
        "底盘太硬，过减速带颠簸明显，悬挂不舒服",
        "座椅偏短，久坐腿累，滤震差",
        "隔音一般，发动机介入时很吵，噪音大",
        "悬挂调校偏运动，底盘硬，坐着不舒服",
    ],
}

# 噪声评论：不含任何属性高频词，模拟水帖/广告/无意义内容
NOISE_TEMPLATES = [
    "沙发",
    "前排围观",
    "楼主说得对",
    "dddd",
    "哈哈哈",
    "说得好",
    "学习了",
    "路过看看",
    "MARK",
    "666",
    "同感同感",
    "顶一下",
    "不错不错",
    "沙发是我的",
    "先码后看",
    "已下单",
    "求楼主回复",
    "我也想知道",
]

OPENERS = [
    "",
    "说实话，",
    "讲真，",
    "个人感觉，",
    "整体来说，",
    "提车两周了，",
    "用了一个月，",
    "试驾完就定了，",
    "开了几千公里，",
    "家里第二台车，",
]

# 口语化语气尾巴
TAILS = ["。", "！", "，很满意", "，挺失望的", "", "，推荐", "，不太推荐"]


# ---------------- 工具函数 ----------------

def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def allocate(total: int, weights: list[float]) -> list[int]:
    """最大余数法：把 total 按 weights 比例分配成整数列表，和恰好为 total。"""
    total_w = sum(weights)
    raw = [w / total_w * total for w in weights]
    floors = [int(x) for x in raw]
    remainder = total - sum(floors)
    order = sorted(range(len(weights)), key=lambda i: raw[i] - floors[i], reverse=True)
    for k in range(remainder):
        floors[order[k]] += 1
    return floors


def random_date(rng: random.Random) -> str:
    start = date(2024, 1, 1)
    end = date(2025, 2, 28)
    delta = (end - start).days
    d = start + timedelta(days=rng.randint(0, delta))
    return d.isoformat()


def build_text(attr: str, sentiment: int, brand: str, model: str, rng: random.Random) -> str:
    opener = rng.choice(OPENERS)
    tail = rng.choice(TAILS)
    if sentiment == 1:
        tpl = rng.choice(POS_TEMPLATES[attr])
    else:
        tpl = rng.choice(NEG_TEMPLATES[attr])
    body = tpl.format(brand=brand, model=model)
    return f"{opener}{body}{tail}"


# ---------------- 主流程 ----------------

def generate() -> pd.DataFrame:
    cfg = load_config()
    seed = cfg["project"]["seed"]
    rng = random.Random(seed)

    attrs = list(cfg["cluster_counts"].keys())          # 外观/内饰/空间/续航/性价比/舒适性
    attr_weights = [cfg["cluster_counts"][a] for a in attrs]
    noise_ratio = cfg["noise_ratio"]

    # 各属性正负评论数 -> 正负权重
    sent_ratio = {}
    for a in attrs:
        p = cfg["sentiment_counts"][a]["positive"]
        n = cfg["sentiment_counts"][a]["negative"]
        sent_ratio[a] = [p, n]

    rows: list[dict] = []

    for platform, brands in cfg["platform_brand_counts"].items():
        for brand, n_cell in brands.items():
            models = cfg["brand_models"][brand]
            # 本格子：噪声数 + 有效评论数
            n_noise = round(n_cell * noise_ratio)
            n_valid = n_cell - n_noise

            # 噪声评论
            for _ in range(n_noise):
                rows.append({
                    "platform": platform,
                    "brand": brand,
                    "model": rng.choice(models),
                    "comment_date": random_date(rng),
                    "comment_text": rng.choice(NOISE_TEMPLATES),
                    "_attr_ground_truth": "噪声",
                    "_sentiment_ground_truth": 0,
                })

            # 有效评论：按 cluster 权重分配到各属性
            attr_alloc = allocate(n_valid, attr_weights)
            for attr, n_attr in zip(attrs, attr_alloc):
                # 该属性内按正负比例分配
                pos_n, neg_n = allocate(n_attr, sent_ratio[attr])
                for _ in range(pos_n):
                    rows.append({
                        "platform": platform,
                        "brand": brand,
                        "model": rng.choice(models),
                        "comment_date": random_date(rng),
                        "comment_text": build_text(attr, 1, brand, rng.choice(models), rng),
                        "_attr_ground_truth": attr,
                        "_sentiment_ground_truth": 1,
                    })
                for _ in range(neg_n):
                    rows.append({
                        "platform": platform,
                        "brand": brand,
                        "model": rng.choice(models),
                        "comment_date": random_date(rng),
                        "comment_text": build_text(attr, -1, brand, rng.choice(models), rng),
                        "_attr_ground_truth": attr,
                        "_sentiment_ground_truth": -1,
                    })

    df = pd.DataFrame(rows)
    # 全表打乱，模拟真实采集顺序
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "comment_id", range(1, len(df) + 1))
    return df


def report(df: pd.DataFrame, cfg: dict) -> None:
    """打印与论文数值的对照。"""
    print("=" * 64)
    print("仿真评论数据生成完成")
    print("=" * 64)
    print(f"总条数: {len(df)}  (论文表5.1明细合计 51224；正文口径 50520)")
    print()

    print("【平台分布】 vs 论文表5.1行合计:")
    plat_cnt = df.groupby("platform").size()
    for p in plat_cnt.index:
        print(f"  {p:<8} {plat_cnt[p]:>6} 条")
    print()

    print("【品牌分布】 vs 论文表5.1列合计:")
    brand_cnt = df.groupby("brand").size()
    for b in brand_cnt.index:
        print(f"  {b:<5} {brand_cnt[b]:>6} 条")
    print()

    print("【属性分布】 vs 论文表5.3:")
    attr_cnt = df[df["_attr_ground_truth"] != "噪声"].groupby("_attr_ground_truth").size()
    for a, n in cfg["cluster_counts"].items():
        actual = attr_cnt.get(a, 0)
        print(f"  {a:<5} 生成 {actual:>5} 条 / 论文 {n:>5} 条")
    noise_n = (df["_attr_ground_truth"] == "噪声").sum()
    print(f"  噪声  生成 {noise_n:>5} 条（占比 {noise_n/len(df)*100:.1f}%）")
    print()

    print("【各属性 正/负 】 vs 论文表5.5:")
    for a in cfg["cluster_counts"]:
        sub = df[df["_attr_ground_truth"] == a]
        pos = (sub["_sentiment_ground_truth"] == 1).sum()
        neg = (sub["_sentiment_ground_truth"] == -1).sum()
        p_p = cfg["sentiment_counts"][a]["positive"]
        p_n = cfg["sentiment_counts"][a]["negative"]
        print(f"  {a:<5} 正 {pos:>5}/{p_p:<5}  负 {neg:>4}/{p_n:<4}")
    print()

    # 抽样展示
    print("【随机样例 5 条】")
    for _, r in df.sample(5, random_state=cfg["project"]["seed"]).iterrows():
        print(f"  [{r['platform']}/{r['brand']}/{r['_attr_ground_truth']}/"
              f"{'正' if r['_sentiment_ground_truth']==1 else '负' if r['_sentiment_ground_truth']==-1 else '噪声'}] "
              f"{r['comment_text']}")
    print("=" * 64)


def main():
    cfg = load_config()
    df = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"已保存: {OUTPUT_PATH}  ({len(df)} 行)\n")
    report(df, cfg)


if __name__ == "__main__":
    main()
