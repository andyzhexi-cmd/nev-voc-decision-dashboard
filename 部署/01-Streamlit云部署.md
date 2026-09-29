# 路线 A · Streamlit Community Cloud（GitHub 直连）

最省事的公网发布方式：代码在 GitHub，云端拉取、构建、托管，`git push` 即更新。
参考官方文档：[App dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)、
[Deploy your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)、
[Status and limitations](https://docs.streamlit.io/deploy/streamlit-community-cloud/status)。

---

## 1. 前置条件

- 一个 GitHub 账号；一个空仓库（**不要**勾选自动生成 README / .gitignore，避免首推冲突）。
- 本仓库（`产品实现/`）已在 `main` 分支上提交完毕。
- 自检通过：

```bash
cd 产品实现
./vene/bin/python 部署/check_deploy_ready.py     # 期望结尾：全部通过，可以推送并部署 ✓
```

## 2. 推送代码

```bash
cd 产品实现

git add -A
git commit -m "release: 首个可部署版本"

# SSH（推荐，需先在 GitHub 配好公钥）
git remote add origin git@github.com:<用户名>/<仓库名>.git
# 或 HTTPS
# git remote add origin https://github.com/<用户名>/<仓库名>.git

git branch -M main
git push -u origin main
```

HTTPS 推送的密码位置要填 **Personal Access Token（PAT）**，不是 GitHub 登录密码：
GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens →
Repository access 选该仓库 → Permissions 勾 `Contents: Read and write`。

私有仓库：Cloud 首次连接时会要求授权该仓库，`Settings → GitHub` 里可随时追加。

## 3. 创建应用

1. 打开 <https://share.streamlit.io> → GitHub 登录 → 授权。
2. **Create app** → **Deploy a public app from GitHub**。
3. 填写：
   - **Repository**：`<用户名>/<仓库名>`
   - **Branch**：`main`
   - **Main file path**：`app.py`
     （本仓库根目录就是 `产品实现/`，所以填 `app.py`；若你把整个工作区推上去，则填 `产品实现/app.py`）
4. **Advanced settings**：
   - **Python version**：**3.11**（推荐）或 3.12 —— 不要选 3.13（见下方排查表第 1 条）
   - **Secrets**：留空（模板见 `secrets.toml.example`）
5. **Deploy**，等构建日志出现 `You can now view your Streamlit app in your browser`。
6. 拿到地址 `https://<随机名>.streamlit.app`；改短名：
   **⋮ → Settings → General → Custom subdomain**。

## 4. 上线后验收清单

```bash
# 1) 健康检查（返回 ok）
curl -s https://<应用名>.streamlit.app/_stcore/health

# 2) 六个视图逐个点开，确认无红色报错
#    产品导览 / 决策总览 / 属性情感分析 / PLTS-VIKOR 模拟器 / 洞察与报告 / 数据管理

# 3) 关键指标应与本地一致（数据包已入库，云端会算出同样的数）
#    评论规模 51,224 · 有效 34,426 · 情感均值 +0.463 · 审计 9 项（通过 1 / 已解析 4 / 待确认 4 / 未解释 0）

# 4) 导览页「3 分钟上手路线」5 张卡都能跳转；洞察与报告能导出 Excel / Markdown / PDF
```

## 5. 构建/运行失败排查表

| 现象 | 原因 | 处理 |
|---|---|---|
| 日志停在 `Apt dependencies were installed from .../packages.txt`，接着 `E: Unsupported file / given on commandline` → `installer returned a non-zero exit code` | **`packages.txt` 里写了注释/空行/中文**：Cloud 把整份文件当包名列表交给 apt，非包名行直接把它喂崩（本仓库实测踩过） | `packages.txt` **只保留一行一个纯包名**（如 `fonts-noto-cjk`），把说明写到文档里；改完推送或点 Reboot |
| 构建日志 `Failed building wheel for scipy` / `meson` 报错 | 平台默认 Python 3.13，而 `scipy==1.13.1` 没有 cp313 预编译轮子 | Advanced settings 把 Python 改成 **3.11**（或 3.12）后 Reboot |
| 页面打开但所有数字为空、KPI 显示 0 | 部署数据包没推上去（被 `.gitignore` 挡了，或用了 `git add` 漏加） | 跑 `部署/check_deploy_ready.py` 看第 2 节；缺就 `git add data/features data/raw/comments_raw.csv` 后推送 |
| `ModuleNotFoundError: No module named 'streamlit_echarts'` 之类 | `requirements.txt` 缺包 | 跑 `部署/check_deploy_ready.py` 第 3 节（import 覆盖检查）并按提示补 |
| 词云 / 聚类图中文变方框 | 容器没有中文字体 | 确认根目录 `packages.txt` 含 `fonts-noto-cjk`；改完 Reboot |
| 构建成功但启动即退出 | 依赖版本冲突（例如本地装过别的 pandas） | 云端只用 `requirements.txt`；保持它与本机 venv 一致，别在 Cloud 里手工 pip |
| 报 `Main file does not exist` | Main file path 填错 | 填 `app.py`（仓库根即 `产品实现/`） |
| 上传大文件到「数据管理」失败 | 默认上传上限 200 MB，且云端磁盘是临时的 | 演示可用；生产走 Docker + 持久卷或对象存储 |
| 打开很慢（30–60 秒） | 免费档休眠后被唤醒 | 正常现象；要常驻请走 [Docker 路线](02-Docker与自托管.md) |

## 6. 更新与回滚

```bash
# 更新：改完本地确认（自检 + pytest）再推
./vene/bin/python scripts_syntax_check.py && ./vene/bin/python -m pytest tests/ -q
git add -A && git commit -m "feat: ..." && git push
# 平台会自动重新部署；也可在 share.streamlit.io 卡片上点 ⋮ → Reboot
```

回滚：`git revert <commit>` 或 `git reset --hard <上一个可用 commit>` 后 `git push --force-with-lease`，
平台会按新的 HEAD 重新构建。
