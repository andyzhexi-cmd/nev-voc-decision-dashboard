# 智评车行 · 决策看板 — 生产镜像
#
# 构建（在 产品实现/ 目录执行，上下文必须是仓库根）：
#   docker build -f 部署/Dockerfile -t zhiping-nexv:1.0 .
# 运行：
#   docker run --rm -p 8501:8501 zhiping-nexv:1.0
#
# 说明：
#   * 基础镜像固定 python:3.11-slim —— 需求清单里的 scipy 1.13.1 在 3.13 上没有预编译轮子，
#     3.11/3.12 才是「全部依赖都有 wheel」的安全区间；
#   * 装上 fonts-noto-cjk，词云与 matplotlib 才能渲染中文（与 packages.txt 保持一致）；
#   * MPLCONFIGDIR 指向可写目录，避免非 root 用户下 matplotlib 字体缓存报错。
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/mplconfig \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-noto-cjk \
 && rm -rf /var/lib/apt/lists/*

# 依赖单独一层：改代码不会导致重新装依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 运行期可写目录（报告导出、上传暂存、备份、字体缓存）
RUN mkdir -p outputs/reports outputs/results data/upload_tmp data/upload_backups /tmp/mplconfig

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health').status==200 else 1)"

CMD ["python", "-m", "streamlit", "run", "app.py"]
