"""部署前自检（只依赖标准库 + git 命令，任何 Python 3.9+ 都能直接跑）。

检查四件事：
  1. 平台要读的文件是否就位（入口、依赖、apt 依赖、Python 版本、Streamlit 配置）；
  2. 云端要用的「部署数据包」是否齐全、是否真的会被 git 提交（gitignore 不误伤）；
  3. 代码里 import 的第三方包是否都在 requirements.txt 里（漏包 = 云端构建后运行时报错）；
  4. 即将推送的体积概况（GitHub 单文件 100MB / 仓库 1GB 是硬上限）。

用法：
    ./vene/bin/python 部署/check_deploy_ready.py          # 在 产品实现/ 目录执行
退出码 0 = 全部通过；1 = 有阻断项。
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# 入口文件 → 平台要求的位置
REQUIRED_FILES = {
    "app.py": "Streamlit 入口（应用主文件）",
    "requirements.txt": "Python 依赖（Community Cloud / Docker 都会读）",
    "packages.txt": "apt 依赖（中文字体），Community Cloud 在仓库根自动识别",
    "runtime.txt": "Python 版本提示（平台支持时生效，Community Cloud 也可在 UI 里选）",
    ".streamlit/config.toml": "项目级 Streamlit 配置（顶栏开关等）",
    "部署/Dockerfile": "自托管镜像定义",
    "部署/docker-compose.yml": "本地一键起生产镜像",
}

# 部署数据包：云端必须存在，且必须被 git 跟踪
DATA_PACKAGE = [
    "data/raw/comments_raw.csv",
    "data/features/comments_raw.parquet",
    "data/features/comments_cleaned.parquet",
    "data/features/comments_clustered.parquet",
    "data/features/sentiment_results.parquet",
    "data/features/comment_index.parquet",
]

# import 名 → pip 包名（不一致的少数几个）
IMPORT_ALIASES = {
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "PIL": "Pillow",
    "streamlit_echarts": "streamlit-echarts",
    "streamlit_option_menu": "streamlit-option-menu",
    "streamlit_shadcn_ui": "streamlit-shadcn-ui",
}

SKIP_DIRS = {"vene", "_shots", "legacy", "__pycache__", ".git", "build", "dist", ".pytest_cache"}
# 本仓库内部模块，不需要出现在 requirements.txt
INTERNAL_ROOTS = {"core", "ui", "services", "state", "config", "tests", "部署"}

# 标准库：Python 3.10+ 用 sys.stdlib_module_names，3.9 走这份兜底清单
_STDLIB_FALLBACK = {
    "abc", "argparse", "array", "ast", "asyncio", "atexit", "base64", "binascii", "bisect",
    "builtins", "calendar", "codecs", "collections", "concurrent", "contextlib", "copy", "csv",
    "ctypes", "dataclasses", "datetime", "decimal", "difflib", "dis", "email", "enum",
    "functools", "gc", "getpass", "glob", "graphlib", "gzip", "hashlib", "heapq", "hmac",
    "html", "http", "importlib", "inspect", "io", "itertools", "json", "keyword", "locale",
    "logging", "marshal", "math", "mimetypes", "multiprocessing", "numbers", "operator", "os",
    "pathlib", "pdb", "pickle", "platform", "pprint", "profile", "pty", "queue", "random", "re",
    "reprlib", "secrets", "shlex", "shutil", "signal", "site", "socket", "sqlite3", "statistics",
    "string", "struct", "subprocess", "sys", "sysconfig", "tarfile", "tempfile", "textwrap",
    "threading", "time", "timeit", "token", "tokenize", "traceback", "types", "typing",
    "unicodedata", "unittest", "urllib", "uuid", "venv", "warnings", "weakref", "webbrowser",
    "xml", "zipfile", "zlib", "__future__",
}
STDLIB = set(getattr(sys, "stdlib_module_names", ())) or _STDLIB_FALLBACK


def _norm(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "")


def _tracked(path: str) -> bool:
    r = subprocess.run(["git", "ls-files", "--error-unmatch", path],
                       cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0


def _size_mb(p: pathlib.Path) -> float:
    return p.stat().st_size / 1024 / 1024 if p.exists() else 0.0


def _declared_packages() -> set[str]:
    names: set[str] = set()
    req = ROOT / "requirements.txt"
    if not req.exists():
        return names
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        for sep in ("==", ">=", "<=", "~=", ">", "<", "["):
            if sep in line:
                line = line.split(sep, 1)[0]
        names.add(_norm(line.strip()))
    return names


def _code_imports() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for p in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module.split(".")[0]]
            for m in mods:
                if m in INTERNAL_ROOTS or m in STDLIB or m.startswith("_"):
                    continue
                found.setdefault(m, []).append(str(p.relative_to(ROOT)))
    return found


def main() -> int:
    fails: list[str] = []
    warns: list[str] = []
    infos: list[str] = []

    print("=" * 78)
    print("智评车行 · 部署前自检")
    print(f"仓库根：{ROOT}")
    print("=" * 78)

    # 1. 平台文件
    print("\n[1/4] 平台要读的文件")
    for rel, why in REQUIRED_FILES.items():
        ok = (ROOT / rel).exists()
        print(f"  {'✓' if ok else '✗'} {rel:<28} {why}")
        if not ok:
            fails.append(f"缺少 {rel}（{why}）")

    # 2. 部署数据包
    print("\n[2/4] 部署数据包（云端唯一数据来源）")
    total = 0.0
    for rel in DATA_PACKAGE:
        p = ROOT / rel
        exists, tracked = p.exists(), _tracked(rel)
        size = _size_mb(p)
        total += size
        mark = "✓" if (exists and tracked) else "✗"
        print(f"  {mark} {rel:<42} {size:6.2f} MB   git 跟踪：{'是' if tracked else '否'}")
        if not exists:
            fails.append(f"缺少数据文件 {rel}（运行 部署/prepare_public_data.py --build 生成）")
        elif not tracked:
            fails.append(f"{rel} 存在但被 .gitignore 排除，推送后云端拿不到")
    print(f"  —— 部署数据包合计 {total:.2f} MB（不含代码与依赖）")
    if total > 50:
        warns.append(f"部署数据包 {total:.1f} MB 偏大，可考虑抽样后重跑流水线（见 部署/03-数据与产物体积.md）")

    excluded = ROOT / "data" / "processed"
    if excluded.exists() and any(excluded.glob("*.csv")):
        infos.append("data/processed/*.csv 存在但不入库（云端不需要：读取优先走 parquet）")

    # 3. import 覆盖检查
    print("\n[3/4] 代码 import 与 requirements.txt 对照")
    declared = _declared_packages()
    missing: list[tuple[str, str, str]] = []
    for mod, where in sorted(_code_imports().items()):
        dist = IMPORT_ALIASES.get(mod, mod)
        if _norm(dist) not in declared:
            missing.append((mod, dist, where[0]))
    if missing:
        for mod, dist, sample in missing:
            print(f"  ✗ import {mod:<20} → 需要 pip 包 {dist:<20}（例：{sample}）")
            fails.append(f"requirements.txt 缺少 {dist}（代码里 import {mod}）")
    else:
        print(f"  ✓ 全部第三方 import 都已在 requirements.txt 声明（共 {len(declared)} 个包）")

    # 4. 体积与路径
    print("\n[4/4] 即将推送的体积与可移植性")
    tracked_files = subprocess.run(["git", "ls-files"], cwd=ROOT,
                                   capture_output=True, text=True).stdout.split()
    sized = sorted(((f, _size_mb(ROOT / f)) for f in tracked_files if (ROOT / f).exists()),
                   key=lambda x: -x[1])[:5]
    for f, mb in sized:
        print(f"  {mb:7.2f} MB  {f}")
    biggest = sized[0][1] if sized else 0.0
    if biggest > 95:
        fails.append(f"单文件 {sized[0][0]} 达 {biggest:.1f} MB，超过 GitHub 100MB 上限")
    repo_mb = sum(mb for _, mb in ((f, _size_mb(ROOT / f)) for f in tracked_files))
    print(f"  —— git 已跟踪文件合计 {repo_mb:.1f} MB（GitHub 建议 < 1000 MB）")

    mac_paths = subprocess.run(
        ["grep", "-rln", "/System/Library/Fonts", "--include=*.py", "ui", "services", "core", "state"],
        cwd=ROOT, capture_output=True, text=True).stdout.split()
    if mac_paths:
        infos.append("以下文件探测了 macOS 字体路径（均带 Linux / reportlab 兜底，不影响云端）："
                     + "、".join(mac_paths))

    print("\n" + "=" * 78)
    for i in infos:
        print(f"[提示] {i}")
    for w in warns:
        print(f"[警告] {w}")
    for f in fails:
        print(f"[阻断] {f}")
    print("结论：" + ("全部通过，可以推送并部署 ✓" if not fails else f"发现 {len(fails)} 个阻断项，请先修复 ✗"))
    print("=" * 78)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
