"""
Basic integration tests for NowEDA.
Run with:  python -m pytest tests/ -v
"""
import os
import pandas as pd
import pytest
import noweda


SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "..", "examples", "sample.csv")


# ---------------------------------------------------------------------------
# read()
# ---------------------------------------------------------------------------

def test_read_csv_returns_dataframe():
    df = noweda.read(SAMPLE_CSV)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_read_unsupported_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        noweda.read("data.parquet")


# ---------------------------------------------------------------------------
# Accessor registration
# ---------------------------------------------------------------------------

def test_accessor_registered():
    df = noweda.read(SAMPLE_CSV)
    assert hasattr(df, "noweda")


# ---------------------------------------------------------------------------
# report()
# ---------------------------------------------------------------------------

def test_report_keys():
    df = noweda.read(SAMPLE_CSV)
    report = df.noweda.report()
    assert "results" in report
    assert "scores" in report
    assert "insights" in report


def test_report_results_has_plugins():
    df = noweda.read(SAMPLE_CSV)
    results = df.noweda.summary()
    for key in ("schema", "stats", "missing", "duplicates", "outliers"):
        assert key in results, f"Missing plugin result: {key}"


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------

def test_scores_structure():
    df = noweda.read(SAMPLE_CSV)
    scores = df.noweda.score()
    assert "data_quality" in scores
    assert "risk" in scores
    assert "model_readiness" in scores
    assert 0 <= scores["data_quality"] <= 100
    assert 0 <= scores["model_readiness"] <= 100
    assert scores["risk"] >= 0


# ---------------------------------------------------------------------------
# insights()
# ---------------------------------------------------------------------------

def test_insights_is_list():
    df = noweda.read(SAMPLE_CSV)
    insights = df.noweda.insights()
    assert isinstance(insights, list)
    assert len(insights) > 0


def test_insights_pii_detected():
    df = noweda.read(SAMPLE_CSV)
    insights = df.noweda.insights()
    pii_insights = [i for i in insights if "PII" in i or "email" in i.lower()]
    assert len(pii_insights) > 0, "Expected PII insight for email column"


# ---------------------------------------------------------------------------
# Lazy evaluation (cache)
# ---------------------------------------------------------------------------

def test_report_cached():
    df = noweda.read(SAMPLE_CSV)
    report1 = df.noweda.report()
    report2 = df.noweda.report()
    assert report1 is report2  # same object — cache works


# ---------------------------------------------------------------------------
# Plugin-level unit tests
# ---------------------------------------------------------------------------

def test_missing_plugin():
    from noweda.plugins.missing import MissingDataPlugin
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, None]})
    result = MissingDataPlugin().run(df)
    assert abs(result["a"] - 1 / 3) < 0.01
    assert result["b"] == 1.0


def test_duplicates_plugin():
    from noweda.plugins.duplicates import DuplicatesPlugin
    df = pd.DataFrame({"x": [1, 1, 2], "y": [10, 10, 20]})
    result = DuplicatesPlugin().run(df)
    assert result["duplicate_rows"] == 1
    assert result["constant_columns"] == []


def test_schema_plugin_id_detection():
    from noweda.plugins.schema import SchemaPlugin
    df = pd.DataFrame({"id": range(100), "category": ["A"] * 50 + ["B"] * 50})
    result = SchemaPlugin().run(df)
    assert result["id"]["role"] == "id_candidate"
    assert "categorical" in result["category"]["role"]


def test_pii_plugin_detects_email():
    from noweda.plugins.pii import PIIDetectorPlugin
    df = pd.DataFrame({"email": ["user@example.com", "other@test.org", "nomail"]})
    result = PIIDetectorPlugin().run(df)
    assert "email" in result
    assert result["email"]["emails_detected"] == 2


def test_html_report_creates_file(tmp_path):
    from noweda.report.html import generate_html_report
    report = {
        "scores": {"data_quality": 85, "risk": 10, "model_readiness": 80},
        "insights": ["Test insight one.", "Test insight two."],
        "results": {"schema": {}, "missing": {}, "duplicates": {}, "outliers": {}, "pii": {}, "encoding": {}},
    }
    out = str(tmp_path / "report.html")
    generate_html_report(report, out)
    assert os.path.exists(out)
    content = open(out).read()
    assert "NowEDA" in content
    assert "Test insight one." in content
