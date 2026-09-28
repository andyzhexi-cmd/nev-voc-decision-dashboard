"""语法检查（不写 __pycache__，适配受限沙箱）"""
import sys
from pathlib import Path
bad = []
for p in sorted(Path(".").rglob("*.py")):
    if any(seg in p.parts for seg in ("vene", "__pycache__", "legacy")):
        continue
    try:
        compile(p.read_text(encoding="utf-8"), str(p), "exec")
    except SyntaxError as e:
        bad.append(f"{p}:{e.lineno} {e.msg}")
if bad:
    print("SYNTAX ERRORS:"); [print(" ", x) for x in bad]; sys.exit(1)
print("all syntax ok")
