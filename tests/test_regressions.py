"""Regression coverage for the 0.1.3 release."""
import numpy as np
import pandas as pd
import pytest

import noweda
from noweda.ml_recommendations import _profile
from noweda.ml_utils import cramers_v
from noweda.plugins import PIIDetectorPlugin, SchemaPlugin, EncodingDetectionPlugin


@pytest.mark.parametrize("dtype", ["object", "string", "category"])
def test_text_dtypes_detect_pii_and_profile_categories(dtype):
    df = pd.DataFrame({"email": pd.Series(["a@example.com", "b@example.org", None], dtype=dtype)})
    assert PIIDetectorPlugin().run(df)["email"]["email"] == 2
    assert SchemaPlugin().run(df)["email"]["role"] != "unknown"
    assert df.eda.summary()["stats"]["email"]["top_freq"] == 1


def test_inferred_strings_detected():
    df = pd.DataFrame({"email": ["a@example.com", "b@example.org"]})
    assert df.eda.pii_df()["Count"].tolist() == [2]


@pytest.mark.parametrize("dtype", ["string", "category"])
def test_encoded_extension_strings(dtype):
    df = pd.DataFrame({"encoded": pd.Series(["aGVsbG8gd29ybGQ="] * 10, dtype=dtype)})
    assert EncodingDetectionPlugin().run(df) == {"encoded": "possible_base64"}


def test_empty_text_is_not_datetime():
    df = pd.DataFrame({"empty": pd.Series([None] * 3, dtype=object)})
    assert SchemaPlugin().run(df)["empty"]["role"] != "datetime"


def test_cached_report_refreshes_after_value_schema_and_dtype_changes():
    df = pd.DataFrame({"a": [1., 2., 3.]})
    report = df.eda.report()
    assert df.eda.report() is report
    df.loc[0, "a"] = np.nan
    assert df.eda.summary()["missing"]["a"] == pytest.approx(1 / 3)
    df.rename(columns={"a": "renamed"}, inplace=True)
    assert "renamed" in df.eda.summary()["schema"]
    df["renamed"] = df["renamed"].astype("Float64")
    assert df.eda.summary()["schema"]["renamed"]["dtype"] == "Float64"
    report = df.eda.report()
    assert df.eda.refresh() is not report


@pytest.mark.parametrize("dtype", ["Int64", "Float64"])
@pytest.mark.parametrize("values", [[None, None], [1, None]])
def test_nullable_statistics_are_unavailable_without_crashing(dtype, values):
    df = pd.DataFrame({"a": pd.Series(values, dtype=dtype)})
    report = df.eda.report()
    assert np.isnan(report["results"]["stats"]["a"]["std"])
    assert 0 <= report["scores"]["data_quality"] <= 100


@pytest.mark.parametrize("x,y,expected", [
    (["a", "a", "b", "b"], ["c", "d", "c", "d"], 0),
    (["a", "a", "b", "b"], ["c", "c", "d", "d"], 1),
    (["a", None, "b", "b"], ["c", "c", "d", None], 1),
])
def test_cramers_v_known_associations(x, y, expected):
    # Repeated index labels must not multiply observations.
    assert cramers_v(pd.Series(x, index=[0] * 4), pd.Series(y, index=[0] * 4)) == pytest.approx(expected)


def test_cramers_v_matches_chi_squared_reference():
    from scipy.stats import chi2_contingency
    x = pd.Series(list("aaabbbcccc"))
    y = pd.Series(list("xyyxyyxyxy"))
    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table, correction=False)[0]
    assert cramers_v(x, y) == pytest.approx(np.sqrt(chi2 / (len(x) * (min(table.shape) - 1))))


def test_cramers_v_degenerate_is_unavailable():
    assert np.isnan(cramers_v(["a", "a"], ["x", "y"]))
    assert np.isnan(cramers_v([], []))
    with pytest.raises(ValueError, match="paired"):
        cramers_v([1], [1, 2])


def test_balanced_multiclass_and_untargeted_features_not_imbalanced():
    df = pd.DataFrame({"label": list("abcd") * 25, "feature": ["a"] * 99 + ["b"]})
    assert not _profile(df, {}, {}, {}, {}, target="label")["has_imbalance"]
    assert not _profile(df, {}, {}, {}, {})["has_imbalance"]


def test_numeric_class_target_detected_and_excluded_from_features():
    df = pd.DataFrame({"label": [0] * 90 + [1] * 10, "feature": range(100)})
    profile = _profile(df, {}, {}, {}, {}, target="label")
    assert profile["imbalanced_cols"] == {"label": .9}
    assert profile["numeric_cols"] == ["feature"]
    with pytest.raises(ValueError, match="Target"):
        df.eda.mlall(target="absent")


def test_jsonl_defaults_and_chunk_validation(tmp_path):
    path = tmp_path / "rows.JSONL"
    path.write_text('{"a":1}\n{"a":2}\n')
    assert noweda.read(path)["a"].tolist() == [1, 2]
    assert [len(c) for c in noweda.read_chunked(path, chunksize=1, concat=False)] == [1, 1]
    for size in [0, -1, True, 1.5]:
        with pytest.raises(ValueError, match="positive integer"):
            noweda.read_chunked(path, chunksize=size)


def test_large_csv_options_and_parquet_options_bypass_spark(tmp_path, monkeypatch):
    import noweda.io as io
    def unexpected_spark(*args, **kwargs):
        pytest.fail("Pandas reader options must never be sent to Spark")
    monkeypatch.setattr(io, "_load_with_spark", unexpected_spark)
    monkeypatch.setattr(io.os.path, "getsize", lambda _: 512 * 1024 * 1024)
    path = tmp_path / "rows.csv"
    path.write_text("id,other\n001,x\n002,y\n")
    assert noweda.read(path, dtype={"id": str}, usecols=["id"])["id"].tolist() == ["001", "002"]
    pytest.importorskip("pyarrow")
    parquet = tmp_path / "rows.parquet"
    pd.DataFrame({"a": [1], "b": [2]}).to_parquet(parquet)
    assert list(noweda.read(parquet, columns=["a"]).columns) == ["a"]


def test_categorical_heatmap_preserves_associations(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matrices = []
    def capture():
        for ax in plt.gcf().axes:
            if ax.get_title() == "Categorical Association (Cramér's V)":
                matrices.append(np.asarray(ax.images[0].get_array()))
    monkeypatch.setattr(plt, "show", capture)
    df = pd.DataFrame({"first": list("aabb") * 20, "second": list("xxyy") * 20})
    try:
        df.eda.vizall()
        assert len(matrices) == 1
        assert matrices[0][0, 1] == pytest.approx(1)
    finally:
        plt.close("all")


def test_aliases_share_cache_but_copies_do_not():
    df = pd.DataFrame({"a": [1., 2., 3.]})
    report = df.eda.report()
    assert df.noweda.report() is report
    copied = df.copy()
    assert copied.eda.report() is not report
    copied.loc[0, "a"] = np.nan
    assert copied.eda.summary()["missing"]["a"] > 0
    assert df.eda.summary()["missing"]["a"] == 0


def test_wide_schema_rename_invalidates_cache():
    df = pd.DataFrame(np.ones((2, 100)), columns=[f"col{i}" for i in range(100)])
    df.eda.report()
    df.rename(columns={"col50": "renamed"}, inplace=True)
    assert "renamed" in df.eda.summary()["schema"]


def test_printed_nullable_report(capsys):
    df = pd.DataFrame({"a": pd.Series([1, None], dtype="Int64")})
    df.eda.statsall()
    assert "Full Statistical Report" in capsys.readouterr().out


def test_unused_target_categories_do_not_create_imbalance():
    labels = pd.Categorical(list("abab"), categories=list("abc"))
    df = pd.DataFrame({"target": labels})
    assert not _profile(df, {}, {}, {}, {}, target="target")["has_imbalance"]


def test_heatmap_calculation_failure_is_visible_as_unavailable(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import noweda.ml_utils as utils
    def unavailable(*args):
        raise ValueError("Cannot calculate association")
    monkeypatch.setattr(utils, "cramers_v", unavailable)
    labels = []
    def capture():
        for ax in plt.gcf().axes:
            if ax.get_title() == "Categorical Association (Cramér's V)":
                labels.extend((text.get_text(), text.get_color()) for text in ax.texts)
    monkeypatch.setattr(plt, "show", capture)
    try:
        pd.DataFrame({"a": list("aabb") * 20, "b": list("xxyy") * 20}).eda.vizall()
        assert labels.count(("N/A", "black")) == 2
    finally:
        plt.close("all")
