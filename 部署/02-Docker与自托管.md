# 路线 B / C · Docker 与自托管

适用于：要 7×24 常驻、要接内网数据源、要自定义域名与鉴权，或想放到 Render / Railway / Fly.io。

---

## 1. 本地先跑一遍生产镜像

```bash
cd 产品实现

# 构建（上下文是仓库根，Dockerfile 在 部署/ 下）
docker build -f 部署/Dockerfile -t zhiping-nexv:1.0 .

# 运行
docker run --rm -p 8501:8501 --name zhiping-nexv zhiping-nexv:1.0

# 健康检查
curl -s http://127.0.0.1:8501/_stcore/health     # 期望：ok
```

或一条命令（compose 已写好 context 指向上一级）：

```bash
docker compose -f 部署/docker-compose.yml up --build
# 稍后 Ctrl+C 停止；加 -d 后台运行
```

镜像要点（见 [`Dockerfile`](Dockerfile)）：

- 基础镜像 `python:3.11-slim`（与 `runtime.txt`、`.streamlit` 策略一致，避开 3.13 的 scipy 坑）；
- 装 `fonts-noto-cjk`，中文词云/图表格子不再变方框；
- `MPLCONFIGDIR=/tmp/mplconfig`，避免非 root 用户下 matplotlib 字体缓存不可写；
- 依赖单独一层，改代码不会触发重装依赖；
- `HEALTHCHECK` 打 `/_stcore/health`，PaaS 可直接用这个端点做存活探针。

镜像体积参考：约 1.2–1.6 GB（pandas / scipy / scikit-learn / matplotlib 决定的）。

## 2. Render（最接近「免费常驻」的 Docker 托管）

1. <https://render.com> → New → **Web Service** → 连接 GitHub 仓库。
2. Runtime 选 **Docker**；Dockerfile Path 填 `部署/Dockerfile`；Docker Build Context Directory 填 `.`。
3. Instance Type 选 Free（会休眠）或 Starter（常驻）。
4. Health Check Path：`/_stcore/health`。
5. 部署完成后在 **Settings → Custom Domain** 绑定域名。

> 注意：免费实例的磁盘同样是临时的；`data/upload_*`、`outputs/reports/` 在重启后清空。
> 需要持久化时挂 **Persistent Disk**（付费），或把上传/导出改接对象存储。

## 3. Railway / Fly.io

Railway：New Project → Deploy from GitHub repo → 自动识别 `部署/Dockerfile`（若没识别，在
Settings 里把 Dockerfile Path 指到 `部署/Dockerfile`）→ 生成公网域名。端口用环境变量 `PORT` 时，
把启动命令改为 `python -m streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`。

Fly.io：

```bash
fly launch --dockerfile 部署/Dockerfile --internal-port 8501 --no-deploy
# 按提示生成 fly.toml，确认/补充：
#   [http_service]
#     internal_port = 8501
#     force_https = true
#     [[http_service.checks]]
#       path = "/_stcore/health"
fly deploy
```

## 4. 自有服务器 + 反向代理（企业内网/公网皆可）

```bash
# 服务器上（已装 Docker）
git clone <仓库地址> app && cd app/产品实现 2>/dev/null || cd app
docker build -f 部署/Dockerfile -t zhiping-nexv:1.0 .
docker run -d --restart unless-stopped -p 127.0.0.1:8501:8501 --name zhiping-nexv zhiping-nexv:1.0
```

Nginx 反代 + 基础鉴权（WebSocket 必须放行，否则页面一直转圈）：

```nginx
server {
  listen 443 ssl;
  server_name nev.example.com;
  # ssl_certificate ...; ssl_certificate_key ...;

  location / {
    auth_basic "restricted";
    auth_basic_user_file /etc/nginx/.htpasswd;   # htpasswd -c /etc/nginx/.htpasswd viewer

    proxy_pass http://127.0.0.1:8501;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_read_timeout 3600s;
  }
}
```

## C. Hugging Face Spaces

适合「发给客户一个链接就能点开、免登录」。Spaces 的 Streamlit SDK 要求
`app.py`、`requirements.txt` 位于**空间仓库根**，并用 README 的 YAML 头声明 SDK：

```markdown
---
title: 智评车行决策看板
emoji: 🚗
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.50.0
app_file: app.py
pinned: false
---
```

步骤：

1. <https://huggingface.co/new-space> → SDK 选 **Streamlit** → 创建。
2. 用一个「空间专用」仓库推送文件（不要改动主仓库结构）：

```bash
cd 产品实现
git remote add hf https://huggingface.co/spaces/<用户名>/<空间名>
# 只推运行必需内容（app.py / ui / core / services / state / config / data 包 / requirements.txt /
# packages.txt / .streamlit/config.toml / README）——用 git subtree 或单独 clone 后复制均可：
git push hf main            # 若直接推主仓库，注意 HF 会忽略 runtime.txt、使用 Docker 模板的 Python 版本
```

3. 构建日志在 Space 的 **Logs** 标签页；`Restart this Space` 可重跑。
4. Spaces 免费档同样会休眠（默认 48 小时无访问），磁盘临时。
