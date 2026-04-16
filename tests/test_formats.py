"""
Comprehensive format + edge case tests for NowEDA.

Every test creates its own file in a tmp_path fixture so there are no
leftover artefacts and tests are fully isolated.
"""
import io
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

import noweda
from noweda.io import read

# ---------------------------------------------------------------------------
# Shared sample DataFrame used across format tests
# ---------------------------------------------------------------------------

def _sample_df():
    return pd.DataFrame({
        "id":         [1, 2, 3, 4, 5],
        "name":       ["Alice", "Bob", "Carol", "David", "Eve"],
        "score":      [91.5, 82.0, None, 77.3, 95.1],
        "category":   ["A", "B", "A", "C", "B"],
        "active":     [True, False, True, True, False],
    })


# ===========================================================================
# FORMAT TESTS
# ===========================================================================

class TestCSV:
    def test_read(self, tmp_path):
        p = tmp_path / "data.csv"
        _sample_df().to_csv(p, index=False)
        df = read(str(p))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_accessor_works(self, tmp_path):
        p = tmp_path / "data.csv"
        _sample_df().to_csv(p, index=False)
        df = read(str(p))
        assert df.noweda.score()["data_quality"] > 0

    def test_uppercase_extension(self, tmp_path):
        p = tmp_path / "DATA.CSV"
        _sample_df().to_csv(p, index=False)
        df = read(str(p))
        assert len(df) == 5


class TestTSV:
    def test_read(self, tmp_path):
        p = tmp_path / "data.tsv"
        _sample_df().to_csv(p, sep="\t", index=False)
        df = read(str(p))
        assert len(df) == 5
        assert list(df.columns) == list(_sample_df().columns)

    def test_tab_extension(self, tmp_path):
        p = tmp_path / "data.tab"
        _sample_df().to_csv(p, sep="\t", index=False)
        df = read(str(p))
        assert len(df) == 5

    def test_pipe_delimited_via_kwargs(self, tmp_path):
        p = tmp_path / "data.csv"
        _sample_df().to_csv(p, sep="|", index=False)
        df = read(str(p), sep="|")
        assert len(df) == 5


class TestExcel:
    def test_xlsx(self, tmp_path):
        p = tmp_path / "data.xlsx"
        _sample_df().to_excel(p, index=False)
        df = read(str(p))
        assert len(df) == 5

    def test_xlsx_sheet_name_kwarg(self, tmp_path):
        p = tmp_path / "data.xlsx"
        with pd.ExcelWriter(p) as w:
            _sample_df().to_excel(w, sheet_name="Results", index=False)
        df = read(str(p), sheet_name="Results")
        assert len(df) == 5

    def test_ods(self, tmp_path):
        p = tmp_path / "data.ods"
        _sample_df().to_excel(p, index=False, engine="odf")
        df = read(str(p))
        assert len(df) == 5


class TestJSON:
    def test_records_orient(self, tmp_path):
        p = tmp_path / "data.json"
        _sample_df().to_json(p, orient="records")
        df = read(str(p))
        assert len(df) == 5

    def test_index_orient(self, tmp_path):
        p = tmp_path / "data.json"
        _sample_df().to_json(p, orient="index")
        df = read(str(p))
        assert len(df) == 5


class TestXML:
    def test_read(self, tmp_path):
        p = tmp_path / "data.xml"
        _sample_df()[["id", "name", "score"]].to_xml(p, index=False)
        df = read(str(p))
        assert len(df) == 5


class TestHTML:
    def test_read(self, tmp_path):
        p = tmp_path / "data.html"
        _sample_df().to_html(p, index=False)
        df = read(str(p))
        assert len(df) == 5

    def test_htm_extension(self, tmp_path):
        p = tmp_path / "data.htm"
        _sample_df().to_html(p, index=False)
        df = read(str(p))
        assert len(df) == 5


class TestParquet:
    pytest.importorskip("pyarrow", reason="pyarrow not installed")

    def test_read(self, tmp_path):
        p = tmp_path / "data.parquet"
        _sample_df().to_parquet(p, index=False)
        df = read(str(p))
        assert len(df) == 5

    def test_accessor_works(self, tmp_path):
        p = tmp_path / "data.parquet"
        _sample_df().to_parquet(p, index=False)
        df = read(str(p))
        insights = df.noweda.insights()
        assert isinstance(insights, list)


class TestFeather:
    pytest.importorskip("pyarrow", reason="pyarrow not installed")

    def test_read(self, tmp_path):
        p = tmp_path / "data.feather"
        # feather doesn't support boolean in older pyarrow — use int
        df_out = _sample_df().copy()
        df_out["active"] = df_out["active"].astype(int)
        df_out.to_feather(p)
        df = read(str(p))
        assert len(df) == 5


class TestPickle:
    def test_read(self, tmp_path):
        p = tmp_path / "data.pkl"
        _sample_df().to_pickle(p)
        df = read(str(p))
        assert len(df) == 5

    def test_pickle_extension(self, tmp_path):
        p = tmp_path / "data.pickle"
        _sample_df().to_pickle(p)
        df = read(str(p))
        assert len(df) == 5


class TestStata:
    def test_read(self, tmp_path):
        p = tmp_path / "data.dta"
        # Stata doesn't support boolean or object with NaN — simplify
        df_out = pd.DataFrame({
            "id":    pd.array([1, 2, 3, 4, 5], dtype="int32"),
            "score": [91.5, 82.0, 0.0, 77.3, 95.1],
        })
        df_out.to_stata(p, write_index=False)
        df = read(str(p))
        assert len(df) == 5


# ===========================================================================
# ERROR HANDLING TESTS
# ===========================================================================

class TestErrorHandling:
    def test_unsupported_extension_raises(self):
        # Extension check happens before file existence check
        with pytest.raises(ValueError, match="Unsupported"):
            read("data.avro")

    def test_file_not_found_raises(self):
        # Known extension but file doesn't exist → FileNotFoundError
        with pytest.raises(FileNotFoundError):
            read("nonexistent_file_xyz_abc.csv")

    def test_missing_pyarrow_raises_helpful_error(self, tmp_path, monkeypatch):
        """If pyarrow is not installed, the error message must mention the fix."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyarrow":
                raise ImportError("No module named 'pyarrow'")
            return real_import(name, *args, **kwargs)

        p = tmp_path / "data.parquet"
        _sample_df().to_parquet(p, index=False)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="pip install noweda"):
            read(str(p))

    def test_supported_extensions_listed_in_error(self):
        # Unsupported extension → error message lists all valid ones
        with pytest.raises(ValueError) as exc_info:
            read("data.xyz123")
        msg = str(exc_info.value)
        assert ".csv" in msg
        assert ".parquet" in msg


# ===========================================================================
# EDGE CASE TESTS
# ===========================================================================

class TestEdgeCases:
    def test_all_numeric_df(self, tmp_path):
        p = tmp_path / "numeric.csv"
        pd.DataFrame(np.random.rand(20, 5), columns=list("ABCDE")).to_csv(p, index=False)
        df = read(str(p))
        report = df.noweda.report()
        assert "correlation" in report["results"]

    def test_all_string_df(self, tmp_path):
        p = tmp_path / "strings.csv"
        pd.DataFrame({
            "a": ["foo", "bar", "baz"] * 5,
            "b": ["x", "y", "z"] * 5,
        }).to_csv(p, index=False)
        df = read(str(p))
        report = df.noweda.report()
        assert report["results"]["correlation"] == {}  # no numeric cols

    def test_single_column_df(self, tmp_path):
        p = tmp_path / "single.csv"
        pd.DataFrame({"value": [1, 2, 3, 4, 5]}).to_csv(p, index=False)
        df = read(str(p))
        assert df.noweda.score()["data_quality"] == 100

    def test_all_null_column(self, tmp_path):
        p = tmp_path / "nulls.csv"
        pd.DataFrame({
            "good": [1, 2, 3],
            "empty": [None, None, None],
        }).to_csv(p, index=False)
        df = read(str(p))
        missing = df.noweda.summary()["missing"]
        assert missing["empty"] == 1.0

    def test_duplicate_rows(self, tmp_path):
        p = tmp_path / "dups.csv"
        base = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        pd.concat([base, base]).to_csv(p, index=False)
        df = read(str(p))
        dups = df.noweda.summary()["duplicates"]
        assert dups["duplicate_rows"] == 3

    def test_constant_column(self, tmp_path):
        p = tmp_path / "const.csv"
        pd.DataFrame({
            "id":  [1, 2, 3],
            "flag": [0, 0, 0],
        }).to_csv(p, index=False)
        df = read(str(p))
        consts = df.noweda.summary()["duplicates"]["constant_columns"]
        assert "flag" in consts

    def test_large_ish_df(self, tmp_path):
        """10k rows should complete without error."""
        p = tmp_path / "large.csv"
        pd.DataFrame({
            "x": np.random.rand(10_000),
            "y": np.random.randint(0, 100, 10_000),
            "label": np.random.choice(["A", "B", "C"], 10_000),
        }).to_csv(p, index=False)
        df = read(str(p))
        assert len(df.noweda.insights()) > 0

    def test_high_skew_detected_in_insights(self, tmp_path):
        p = tmp_path / "skew.csv"
        # Construct data that is deterministically, heavily right-skewed:
        # 190 values near 1, 10 extreme outliers → skewness always >> 2
        data = [1.0] * 190 + [100.0, 200.0, 300.0, 400.0, 500.0,
                               600.0, 700.0, 800.0, 900.0, 1000.0]
        pd.DataFrame({"amount": data}).to_csv(p, index=False)
        df = read(str(p))
        insights = df.noweda.insights()
        skew_insights = [i for i in insights if "skew" in i.lower()]
        assert len(skew_insights) > 0

    def test_pii_risk_score_elevated(self, tmp_path):
        p = tmp_path / "pii.csv"
        pd.DataFrame({
            "email": [f"user{i}@example.com" for i in range(20)],
            "value": range(20),
        }).to_csv(p, index=False)
        df = read(str(p))
        assert df.noweda.score()["risk"] >= 15

    def test_no_warnings_emitted(self, tmp_path):
        """No UserWarnings should escape from noweda internals."""
        p = tmp_path / "data.csv"
        _sample_df().to_csv(p, index=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = read(str(p))
            _ = df.noweda.report()
        noweda_warnings = [
            w for w in caught
            if "noweda" in str(w.filename).lower()
        ]
        assert noweda_warnings == [], f"Unexpected warnings: {noweda_warnings}"

    def test_kwargs_forwarded_to_reader(self, tmp_path):
        """Extra kwargs (like nrows) are passed through to the pandas reader."""
        p = tmp_path / "data.csv"
        pd.DataFrame({"x": range(100)}).to_csv(p, index=False)
        df = read(str(p), nrows=10)
        assert len(df) == 10
