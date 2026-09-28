#!/bin/bash
# 智评车行 · Mac 一键启动脚本
# 双击即可运行；首次运行会自动生成仿真数据（若尚不存在），然后打开看板
cd "$(dirname "$0")"

PY="./vene/bin/python"
STREAMLIT="./vene/bin/streamlit"

if [ ! -f "data/raw/comments_raw.csv" ]; then
  echo "首次启动：正在生成仿真评论数据..."
  $PY -m src.data_ingest.mock_generator
fi

echo "正在启动智评车行看板（浏览器将自动打开）..."
exec $STREAMLIT run app.py
