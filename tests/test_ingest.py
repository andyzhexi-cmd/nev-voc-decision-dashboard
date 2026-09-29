"""真实数据接入（services/data_ingest/importer.py）单测。"""
from __future__ import annotations

import pandas as pd
import pytest

from services.data_ingest import importer as imp


def _csv(tmp_path, name: str, text: str, encoding: str = "utf-8"):
    p = tmp_path / name
    p.write_text(text, encoding=encoding)
    return p


# ------------------------------------------------------------------ 读取与映射
def test_load_utf8_sig_and_detect_mapping(tmp_path):
    p = _csv(tmp_path, "std.csv",
             "﻿comment_id,platform,brand,model,comment_date,comment_text\n"
             "1,汽车之家,特斯拉,Model Y,2024-06-06,底盘不错\n")
    df, meta = imp.load_table(p)
    assert meta["encoding"].startswith("utf-8")
    mapping = imp.detect_mapping(df)
    assert mapping["comment_text"] == "comment_text"
    assert mapping["brand"] == "brand"


def test_gbk_alias_mapping_and_dates(tmp_path):
    p = _csv(tmp_path, "baidu.csv",
             "来源,车系,车型,评论时间,评论内容\n"
             '百度有驾,比亚迪,汉,2024/03/05,"底盘扎实，空间也大"\n'
             "百度有驾,特斯拉,Model 3,2024-03-06,续航扎实\n",
             encoding="gbk")
    df, rep = imp.import_comments(p, dry_run=True)
    assert rep.ok, rep.errors
    assert rep.encoding in ("gbk", "gb18030")
    assert rep.column_mapping["brand"] == "车系"
    assert rep.brands == ["比亚迪", "特斯拉"]
    assert rep.date_range == ("2024-03-05", "2024-03-06")
    assert list(df.columns) == imp.STANDARD_COLUMNS
    assert df.loc[0, "comment_text"] == "底盘扎实，空间也大"   # 逗号未被错切


def test_missing_required_column_reports_actionable_error(tmp_path):
    p = _csv(tmp_path, "bad.csv", "a,b\n1,2\n")
    _, rep = imp.import_comments(p, dry_run=True)
    assert not rep.ok
    assert "评论内容" in rep.errors[0]
    assert "comment_text" in rep.errors[0]      # 给出可执行的列名建议


def test_column_shift_detected(tmp_path):
    """正文含 ASCII 逗号但没加引号 → 首列被吞成索引，必须报错而不是静默错位。"""
    p = _csv(tmp_path, "shift.csv",
             "brand,model,comment_text\n比亚迪,汉,很好,很值\n特斯拉,Model3,不错,推荐\n")
    _, rep = imp.import_comments(p, dry_run=True)
    assert not rep.ok
    assert any("列错位" in e for e in rep.errors)


def test_empty_text_ratio_guard(tmp_path):
    rows = "\n".join(f"特斯拉,Model3,{t}" for t in ["", "", "有内容"])
    p = _csv(tmp_path, "empty.csv", "brand,model,comment_text\n" + rows + "\n")
    _, rep = imp.import_comments(p, dry_run=True)
    assert not rep.ok
    assert any("列错位" in e or "没有可用评论" in e for e in rep.errors)


# ------------------------------------------------------------------ 清洗策略
def test_dedupe_off_by_default_and_warning(tmp_path):
    p = _csv(tmp_path, "dup.csv",
             "brand,comment_text\n比亚迪,底盘扎实\n比亚迪,底盘扎实\n特斯拉,续航扎实\n")
    df, rep = imp.import_comments(p, dry_run=True)
    assert rep.ok
    assert rep.rows_kept == 3 and rep.rows_duplicate == 1
    assert any("完全相同" in w for w in rep.warnings)


def test_dedupe_on_drops_duplicates(tmp_path):
    p = _csv(tmp_path, "dup2.csv",
             "brand,comment_text\n比亚迪,底盘扎实\n比亚迪,底盘扎实\n特斯拉,续航扎实\n")
    df, rep = imp.import_comments(p, dry_run=True, dedupe=True)
    assert rep.rows_kept == 2
    assert any("去重" in w for w in rep.warnings)


def test_dry_run_does_not_write(tmp_path, monkeypatch):
    p = _csv(tmp_path, "a.csv", "brand,comment_text\n比亚迪,底盘不错\n")
    dest = tmp_path / "out.csv"
    monkeypatch.setattr(imp, "write_raw",
                        lambda df, dest=None: pytest.fail("dry_run 不应落盘"))
    _, rep = imp.import_comments(p, dry_run=True)
    assert rep.ok and rep.wrote_to is None
    assert not dest.exists()


def test_write_raw_backs_up_existing(tmp_path):
    dest = tmp_path / "raw" / "comments_raw.csv"
    dest.parent.mkdir(parents=True)
    dest.write_text("old", encoding="utf-8")
    df = pd.DataFrame({"comment_text": ["新车"]})
    out = imp.write_raw(df, dest)
    assert out.read_text(encoding="utf-8-sig").startswith("comment_text")
    backups = list((tmp_path / "upload_backups").glob("comments_raw_*.csv"))
    assert backups and backups[0].read_text(encoding="utf-8") == "old"


def test_standardize_reports_without_ground_truth(tmp_path):
    p = _csv(tmp_path, "real.csv", "comment_text\n真实评论一条\n")
    df, rep = imp.import_comments(p, dry_run=True)
    assert rep.ok
    assert df["_attr_ground_truth"].tolist() == [""]
    assert df["_sentiment_ground_truth"].tolist() == [0]
    assert df["comment_id"].tolist() == [1]
