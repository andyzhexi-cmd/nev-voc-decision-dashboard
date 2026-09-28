"""services 包：应用服务层

  data_ingest/   导入与合成数据
  preprocess/    清洗分词
  aspect_mining/ 属性聚类
  sentiment/     情感分析（VADER + 朴素贝叶斯）
  data_store.py  统一数据访问（CSV → parquet 中间层）
  features.py    视图数据契约（聚合 / 整形）
  report.py      报告导出（Excel / Markdown / PDF）
"""
import os
from pathlib import Path

# 保证 matplotlib 缓存落在工作区内（不改动系统配置）
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
