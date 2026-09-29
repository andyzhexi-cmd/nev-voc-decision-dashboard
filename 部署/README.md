# 部署指南 · 智评车行决策看板

本目录把 `产品实现/` 这套 Streamlit 应用发布到公网所需的全部资产集中在一起：
平台要读的配置、Docker 镜像、数据包维护脚本、部署前自检与逐条操作手册。

三条路线，按推荐顺序：

| 路线 | 适合场景 | 成本 | 公网地址形态 | 手册 |
|---|---|---|---|---|
| **A. Streamlit Community Cloud**（GitHub 直连） | 客户试用、评审演示、内部体验；改动即部署 | 免费 | `https://<应用名>.streamlit.app` | [01-Streamlit云部署.md](01-Streamlit云部署.md) |
| **B. Docker 自托管**（Render / Railway / Fly.io / 自有服务器） | 要 7×24 常驻、要接内网数据库、要自定义域名与鉴权 | 有免费额度 | 自定域名 | [02-Docker与自托管.md](02-Docker与自托管.md) |
| **C. Hugging Face Spaces**（Streamlit SDK） | 想让外部免注册直接点开 | 免费 | `https://huggingface.co/spaces/<用户>/<空间>` | [02-Docker与自托管.md](02-Docker与自托管.md#c-hugging-face-spaces) |

---

## 0. 一分钟自检（推送前必跑）

在 `产品实现/` 目录执行：

```bash
./vene/bin/python 部署/check_deploy_ready.py
```

它检查四件事，任一不通过都会以非 0 退出：

1. 平台要读的文件是否就位（入口、依赖、apt 依赖、Python 版本、Streamlit 配置）；
2. **部署数据包**是否齐全、是否真的会被 git 提交（防止 `.gitignore` 误伤导致云端无数据）；
3. 代码里 import 的第三方包是否都在 `requirements.txt` 里（漏包 = 云端构建成功、一打开就报错）；
4. 即将推送的体积（GitHub 单文件 100 MB、仓库 1 GB 是硬上限）。

数据包的体检/重建：

```bash
./vene/bin/python 部署/prepare_public_data.py           # 看体积、行数、缺失项
./vene/bin/python 部署/prepare_public_data.py --build   # 从 processed CSV 重建 parquet 缓存
```

当前实测（本地全量数据）：

```
data/raw/comments_raw.csv              5.58 MB   51,224 行
data/features/comments_raw.parquet     0.61 MB   51,224 行
data/features/comments_cleaned.parquet 0.83 MB   48,583 行
data/features/comments_clustered.parquet 0.85 MB 48,583 行
data/features/sentiment_results.parquet 0.71 MB  46,459 行
data/features/comment_index.parquet    0.72 MB   46,459 行
—— 部署数据包合计 9.31 MB；git 已跟踪文件合计约 9.9 MB
```

---

## 1. 仓库根上有哪些「平台要读的文件」

Streamlit Community Cloud 只认**仓库根**或**入口文件同目录**下的这些文件；
本仓库的根就是 `产品实现/`，因此它们必须留在根目录（不要挪进 `部署/`）：

| 文件 | 作用 | 缺失后果 |
|---|---|---|
| `app.py` | 应用入口 | 平台找不到主文件 |
| `requirements.txt` | Python 依赖（已钉版本） | 无依赖或版本漂移 |
| `packages.txt` | apt 依赖：只写包名（当前 `fonts-noto-cjk`），**不能有注释/空行/中文** | Cloud 的 apt 步骤报 `E: Unsupported file / given on commandline` 并中断构建 |
| `runtime.txt` | `python-3.11`（平台支持时生效；Cloud 也可在 UI 里选） | 用 3.13 时 `scipy==1.13.1` 没有预编译轮子，构建失败 |
| `.streamlit/config.toml` | 项目级配置（顶栏保留侧栏开关、关闭 Deploy 按钮相关项） | 顶栏/侧栏体验退化 |
| `data/features/*.parquet` + `data/raw/comments_raw.csv` | 部署数据包（详见下文） | 页面能打开但所有指标为空 |

`部署/` 里放的是「不参与运行时」的资产：Dockerfile、compose、云端精简依赖、密钥模板、两个脚本和这本手册。

---

## 2. 最快路径：GitHub → Streamlit Community Cloud

### 2.1 把代码推到 GitHub

```bash
cd 产品实现
./vene/bin/python 部署/check_deploy_ready.py     # 先过自检
git add -A
git commit -m "release: 首个可部署版本"

# 在 GitHub 网页端新建一个空仓库（不要勾选 README/.gitignore，避免首推冲突），然后：
git remote add origin git@github.com:<你的用户名>/<仓库名>.git   # 用 SSH
# 或： git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git branch -M main
git push -u origin main
```

> HTTPS 方式推送时会要求密码，**必须用 Personal Access Token（PAT）**：
> GitHub → Settings → Developer settings → Personal access tokens → Fine-grained token，
> 勾选该仓库的 `Contents: Read and write`，用户名填 GitHub 用户名、密码填这个 token。

### 2.2 在 Streamlit Cloud 建应用

1. 打开 <https://share.streamlit.io>，用 GitHub 账号登录并授权仓库（私有仓库也可授权）。
2. 右上角 **Create app** → **Deploy a public app from GitHub**。
3. 填三项：
   - Repository：`<你的用户名>/<仓库名>`
   - Branch：`main`
   - **Main file path：`app.py`**（本仓库根就是 `产品实现/`，所以是 `app.py` 而不是 `产品实现/app.py`）
4. 点 **Advanced settings** → **Python version** 选 **3.11**（推荐；3.12 也可，**不要选 3.13**）。
   Secrets 留空（本产品不需要密钥，模板见 [`secrets.toml.example`](secrets.toml.example)）。
5. **Deploy**。首次构建 3–8 分钟：装 apt 字体 → 装 Python 依赖 → 启动。
6. 构建日志出现 `You can now view your Streamlit app` 即成功，地址形如
   `https://<应用名>.streamlit.app`，可在 **Settings → General → Custom subdomain** 改名。

### 2.3 之后怎么更新

`git push` 后 Cloud 会自动拉取并重建（几秒到几十秒）。想手动触发：
应用右下角 **⋮ → Rerun**，或在 <https://share.streamlit.io> 应用卡片上 **Reboot**。

---

## 3. 数据从哪来（关键，别跳过）

产品运行时的取数顺序在 `services/data_store.py::_read` 里：**先看 `data/features/<表名>.parquet`，
再看同名 CSV，两者都在时取更新的那份；只要 parquet 在，对应 CSV 缺失也能正常工作**。

因此云端只需要 commit：

- `data/features/*.parquet`（5 个文件，≈ 4 MB，列裁剪后的中间层）；
- `data/raw/comments_raw.csv`（≈ 5.6 MB，数据管理页的质量体检、字段字典、导入体验要用）。

**`data/processed/*.csv`（≈ 38 MB）不入库**——云端不需要它们，理由与验证方式见
[03-数据与产物体积.md](03-数据与产物体积.md)。
如果将来数据量变大，两个方向：`部署/prepare_public_data.py --build` 后提交抽样数据，
或改用 Git LFS（> 100 MB 单文件时）。

---

## 4. 已知限制与对策（企业交付前请确认）

| 限制 | 表现 | 对策 |
|---|---|---|
| 云端文件系统是**临时的** | 「数据管理 → 导入真实评论」落盘的文件、导出到 `outputs/reports/` 的报告，在应用重启/休眠后消失 | 演示用；生产接数据库/对象存储（`部署/secrets.toml.example` 预留了配置位） |
| 免费档会**休眠** | 15 分钟无访问后休眠，下次打开需等 30–60 秒 | 用 uptime 探针定时唤醒，或走 Docker 路线常驻 |
| 资源上限（免费档约 1 vCPU / 1 GB 内存） | 全量重跑流水线（jieba + KMeans + TF-IDF）可能被 OOM 打断 | 数据包已提交，云端**不需要**重跑流水线；重跑请在本地完成后再推送 |
| 应用默认**公开可访问** | 任何人拿到链接都能看 | 免费档支持在 **Settings → Sharing** 里限制为指定邮箱；更严格控制请走 Docker + 反向代理鉴权 |
| 中文字体 | Linux 默认无 CJK 字体 | 已用 `packages.txt` 安装 `fonts-noto-cjk`；PDF 报告还有 reportlab 内置 `STSong-Light` 兜底 |
| Python 3.13 | `scipy==1.13.1` 无 cp313 wheel | 固定 3.11 / 3.12 |

---

## 5. 本目录文件索引

| 文件 | 用途 |
|---|---|
| [`01-Streamlit云部署.md`](01-Streamlit云部署.md) | 路线 A 逐步操作 + 构建失败排查表 |
| [`02-Docker与自托管.md`](02-Docker与自托管.md) | 路线 B/C：镜像、compose、Render/Railway/Fly/HF Spaces |
| [`03-数据与产物体积.md`](03-数据与产物体积.md) | 数据包策略、体积对照、如何验证「没有 CSV 也能跑」 |
| `Dockerfile` | 生产镜像（python:3.11-slim + 中文字体 + 健康检查） |
| `docker-compose.yml` | 本地一键起同款镜像 |
| `requirements-cloud.txt` | 云端精简依赖（去掉 pytest/watchdog，显式补 pydantic） |
| `secrets.toml.example` | 密钥模板（当前为空，接企业数据源时再填） |
| `check_deploy_ready.py` | 部署前自检（文件 / 数据包 / import 覆盖 / 体积） |
| `prepare_public_data.py` | 数据包检查、重建、剔除中间 CSV |
| `../.github/workflows/ci.yml` | 推送即跑单测与语法检查，防止把坏版本推上线 |
