"""Behavioral regressions for 0.1.4 reliability fixes."""
import base64
import builtins
import json

import numpy as np
import pandas as pd
import pytest

import noweda
from noweda.ml_utils import calculate_vif
from noweda.plugins.encoding import EncodingDetectionPlugin
from noweda.plugins.pii import PIIDetectorPlugin
from noweda.report.html import generate_html_report
from noweda.report.json import generate_json_report
from noweda.scoring.scorer import Scorer


@pytest.mark.parametrize("card", ["4111111111111111", "4111 1111 1111 1111",
                                  "4111-1111-1111-1111", "3782 822463 10005"])
def test_card_formats_and_duplicate_indices(card):
    df = pd.DataFrame({"contact": [card, "202-555-0101", "202-555-0102", None]}, index=[0] * 4)
    assert PIIDetectorPlugin().run(df) == {"contact": {"credit_card": 1, "phone": 2}}


def test_pii_counts_cells_and_preserves_separate_phone_in_card_cell():
    df = pd.DataFrame({"contact": ["4111 1111 1111 1111; 4111-1111-1111-1111; call 202-555-0101"]})
    assert PIIDetectorPlugin().run(df) == {"contact": {"credit_card": 1, "phone": 1}}


@pytest.mark.parametrize("value", ["4111-1111-1111-1112", "4111111111111112", "12345678901"])
def test_invalid_cards_and_long_numbers_are_not_phones(value):
    assert PIIDetectorPlugin().run(pd.DataFrame({"value": [value]})) == {}


def _orthogonal_features():
    return pd.DataFrame({"x": [-1.] * 4 + [1.] * 4,
                         "y": [-1., -1., 1., 1.] * 2,
                         "noise": [-1., 1.] * 4})


def test_vif_multivariate_without_statsmodels(monkeypatch):
    original = builtins.__import__
    def without_statsmodels(name, *args, **kwargs):
        if name.startswith("statsmodels"):
            raise ImportError("Not installed")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_statsmodels)
    df = _orthogonal_features()
    df["z"] = df.x + df.y + df.noise
    features = df[["x", "y", "z"]].copy()
    assert calculate_vif(features) == pytest.approx({"x": 2., "y": 2., "z": 3.})
    assert calculate_vif(features * [10, .1, -3] + [100, -30, 500]) == pytest.approx(
        {"x": 2., "y": 2., "z": 3.})
    features["z"] = features.x + features.y
    assert all(np.isinf(v) for v in calculate_vif(features).values())


def test_large_finite_vif_is_not_reported_as_infinite():
    df = _orthogonal_features()
    df["z"] = df.x + 1e-4 * df.noise
    assert calculate_vif(df[["x", "z"]])["x"] == pytest.approx(1e8 + 1, rel=1e-6)


def test_vif_constants_missing_infinite_and_insufficient_rows():
    df = _orthogonal_features()[["x", "y"]].astype("Float64")
    df.loc[8] = [None, 5]
    df.loc[9] = [np.inf, 6]
    # pandas 1.3 may cast extension columns to object when appending rows.
    df = df.astype("Float64")
    df["constant"] = 7.
    vif = calculate_vif(df)
    assert vif["x"] == pytest.approx(1)
    assert vif["y"] == pytest.approx(1)
    assert np.isnan(vif["constant"])
    for frame in [df.iloc[:0], df.iloc[:1], pd.DataFrame({"a": [1., 2.], "b": [2., 3.]})]:
        assert all(np.isnan(v) for v in calculate_vif(frame).values())


@pytest.mark.parametrize("columns", [[0, 1, 2], [0, "category", ("group", "value")]])
def test_non_string_labels_in_reports(columns, capsys, tmp_path):
    df = pd.DataFrame({columns[0]: range(30), columns[1]: ["same"] * 30,
                       columns[2]: np.arange(30) * 100.})
    df.eda.statsall()
    output = capsys.readouterr().out
    assert "Full Statistical Report" in output
    assert "Multicollinearity Detected" in output
    assert all(str(col) in output for col in columns)
    assert list(df.eda.summary()["stats"]) == columns
    df.eda.duplicates_df()
    generate_html_report(df.eda.report(), tmp_path / "report.html")
    generate_json_report(df.eda.report(), tmp_path / "report.json")
    df.eda.mlall()
    df.eda.compare(df.rename(columns={columns[0]: "renamed"}))


@pytest.mark.parametrize("frame", [pd.DataFrame(), pd.DataFrame({"x": pd.Series(dtype="float64")}),
                                    pd.DataFrame(index=[0, 1])])
def test_empty_reports_are_informative(frame, capsys, tmp_path):
    frame.eda.statsall()
    assert "Empty DataFrame" in capsys.readouterr().out
    report = frame.eda.report()
    assert "Scores are not informative" in report["insights"][0]
    generate_json_report(report, tmp_path / "empty.json")


def test_duplicate_column_labels_fail_clearly():
    with pytest.raises(ValueError, match="unique column labels"):
        pd.DataFrame([[1, 2]], columns=["a", "a"]).eda.report()


@pytest.mark.parametrize("value", ["John", "Mary", "Jonathan", "test", "aGVsbG8gd29ybGQ=\n"])
def test_ordinary_names_and_invalid_base64_do_not_raise_risk(value):
    report = pd.DataFrame({"name": [value] * 20}).eda.report()
    assert report["results"]["encoding"] == {}
    assert report["scores"]["risk"] == 0


def test_base64_evidence_and_thresholds():
    encoded = base64.b64encode(b"hello world").decode()
    plugin = EncodingDetectionPlugin()
    df = pd.DataFrame({"payload": [encoded] * 16 + ["John"] * 4})
    assert plugin.run(df) == {"payload": "possible_base64"}
    expected = {"sample_size": 20, "matches": 16, "confidence": .8}
    assert plugin.details["payload"] == expected
    assert df.eda.report()["encoding_details"]["payload"] == expected
    assert df.eda.encoding_df(include_confidence=True).iloc[0]["Confidence"] == .8
    assert list(df.eda.encoding_df().columns) == ["Column", "Encoding_Type"]
    for values in [[encoded] * 5, [encoded] * 15 + ["John"] * 5]:
        assert plugin.run(pd.DataFrame({"payload": values})) == {}
        assert plugin.details == {}
    assert plugin.is_base64(base64.b64encode(b"hello world!").decode())  # no padding
    assert plugin.is_base64(base64.b64encode(b"\xff" * 8).decode())  # binary


def _strict_load(path):
    def reject(value):
        pytest.fail("Invalid JSON constant: " + value)
    return json.loads(path.read_text(), parse_constant=reject)


def test_json_export_normalizes_nonfinite_without_mutating_report(tmp_path):
    report = pd.DataFrame({"value": [1.]}).eda.report()
    report["custom"] = {"array": np.array([np.inf, -np.inf, np.nan]), "null": pd.NA,
                        "count": np.int64(2), "unicode": "é"}
    path = tmp_path / "report.json"
    generate_json_report(report, path)
    loaded = _strict_load(path)
    assert loaded["results"]["stats"]["value"]["std"] is None
    assert loaded["custom"] == {"array": [None] * 3, "null": None, "count": 2, "unicode": "é"}
    assert np.isnan(report["results"]["stats"]["value"]["std"])


@pytest.mark.parametrize("invalid", [{"columns": {1: "one", "1": "other"}}, {"custom": object()}])
def test_invalid_json_export_preserves_existing_file(tmp_path, invalid):
    path = tmp_path / "report.json"
    path.write_text("existing report")
    with pytest.raises((ValueError, TypeError)):
        generate_json_report(invalid, path)
    assert path.read_text() == "existing report"


def test_cli_json_export_is_strict(tmp_path, monkeypatch, capsys):
    from noweda.cli import main
    source, output = tmp_path / "one.csv", tmp_path / "one.json"
    source.write_text("value\n1\n")
    monkeypatch.setattr("sys.argv", ["noweda", str(source), "--json", str(output)])
    main()
    assert _strict_load(output)["results"]["stats"]["value"]["std"] is None


def test_html_shows_escaped_evidence_and_score_contributions(tmp_path):
    df = pd.DataFrame({"<payload>": ["aGVsbG8gd29ybGQ="] * 10, "value": [0.] * 9 + [100.]})
    path = tmp_path / "report.html"
    generate_html_report(df.eda.report(), path)
    from lxml import html
    page = html.fromstring(path.read_text())
    assert not page.xpath("//payload")
    assert "<payload>" in page.text_content()
    assert "Score Contributions" in page.text_content()
    assert "observed rate: 10.00%" in page.text_content()
    assert "10 / 10" in page.text_content()


def _outlier_rule(results):
    _, details = Scorer().compute(results, explain=True)
    return next(rule for rule in details if rule["rule"] == "outliers")


@pytest.mark.parametrize("outliers,penalty", [(0, 0), (10, 0), (11, -5), (50, -5), (51, -10)])
def test_outlier_rate_thresholds_and_size_invariance(outliers, penalty):
    for scale in [1, 100]:
        rule = _outlier_rule({"outliers": {"x": outliers * scale},
                              "stats": {"x": {"count": 1000 * scale}}})
        assert rule["contributions"]["data_quality"] == penalty
        assert rule["contributions"]["model_readiness"] == penalty
        assert rule["evidence"]["rate"] == outliers / 1000


def test_outlier_denominator_excludes_missing_and_non_numeric_cells():
    rule = _outlier_rule({"outliers": {"x": 2, "y": 0},
                          "stats": {"x": {"count": 10, "missing": 100},
                                    "y": {"count": 10}, "text": {"count": 1000}}})
    assert rule["evidence"]["rate"] == .1
    assert rule["contributions"]["data_quality"] == -10
    assert _outlier_rule({"outliers": {"x": 100}})["evidence"]["rate"] is None


def test_score_breakdown_reconciles_including_clamp():
    df = pd.DataFrame({str(i): [None] * 9 + ["a@example.com"] for i in range(20)})
    report = df.eda.report()
    for name, start in [("data_quality", 100), ("model_readiness", 100), ("risk", 0)]:
        assert start + sum(rule["contributions"][name] for rule in report["score_breakdown"]) == report["scores"][name]
    assert report["scores"]["data_quality"] == 0
    assert report["scores"] == Scorer().compute(report["results"])
