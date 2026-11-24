import pandas as pd
import pytest

import noweda as eda


def _write_csv(path, rows=12):
    pd.DataFrame(
        {
            "customer_id": range(rows),
            "amount": [float(i) for i in range(rows)],
            "churned": [i % 2 for i in range(rows)],
        }
    ).to_csv(path, index=False)


def test_large_read_returns_disk_backed_dataset(tmp_path):
    path = tmp_path / "data.csv"
    _write_csv(path)

    data = eda.read(path, mode="large", chunksize=5)

    assert isinstance(data, eda.LargeDataset)
    assert data.shape == (12, 3)
    assert data.columns == ["customer_id", "amount", "churned"]
    preview = data.head(2)
    assert list(preview["customer_id"]) == [0, 1]
    assert len(preview) == 2


def test_large_report_marks_sampled_results(tmp_path, capsys):
    path = tmp_path / "data.csv"
    _write_csv(path)

    report = eda.read(path, mode="large", chunksize=5).eda.report()

    assert "sample-based" in capsys.readouterr().out
    assert report["large_mode"] == {
        "source": str(path),
        "rows": 12,
        "columns": 3,
        "sample_rows": 5,
        "result_scope": "sample-based except dataset dimensions",
    }


def test_large_mode_rejects_unsupported_format_and_bad_chunksize(tmp_path):
    path = tmp_path / "data.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(path, index=False)

    with pytest.raises(ValueError, match="Large mode supports"):
        eda.read(path, mode="large")
    _write_csv(path.with_suffix(".csv"))
    with pytest.raises(ValueError, match="positive integer"):
        eda.read(path.with_suffix(".csv"), mode="large", chunksize="10000")


def test_large_mode_supported_methods_are_clear(tmp_path):
    path = tmp_path / "data.csv"
    _write_csv(path)
    data = eda.read(path, mode="large")

    with pytest.raises(AttributeError, match="not available in large mode"):
        data.eda.insights()


def test_large_mode_uses_ten_thousand_rows_by_default(tmp_path, capsys):
    path = tmp_path / "data.csv"
    _write_csv(path, rows=10_001)

    eda.read(path, mode="large").eda.mlall(target="churned")

    assert "10,000 of 10,001 rows" in capsys.readouterr().out


def test_small_mode_uses_all_rows_unless_sample_is_requested(capsys):
    frame = pd.DataFrame({"amount": range(20), "target": [0, 1] * 10})

    frame.eda.mlall(target="target")
    assert "sample-based results" not in capsys.readouterr().out

    frame.eda.mlall(target="target", sample=5)
    assert "5 of 20 rows" in capsys.readouterr().out


def test_ordered_file_sample_includes_later_rows_and_html_scope(tmp_path):
    from noweda.report.html import generate_html_report

    path = tmp_path / "ordered.csv"
    frame = pd.DataFrame({"_noweda_row": range(150), "label": [0] * 120 + [1] * 30})
    frame.to_csv(path, index=False)
    data = eda.read(path, mode="large", chunksize=100)
    assert list(data.head(5)["_noweda_row"]) == list(range(5))
    report = data.eda.report()
    assert 0 < report["results"]["stats"]["label"]["mean"] < 1
    fresh = eda.read(path, mode="large", chunksize=100)
    pd.testing.assert_frame_equal(data._sample(), fresh._sample())
    assert data._sample()["_noweda_row"].is_monotonic_increasing
    page = tmp_path / "report.html"
    generate_html_report(report, page)
    assert "sample of 100 of 150 rows" in page.read_text()


@pytest.mark.parametrize("method", ["statsall", "vizall", "mlall"])
def test_large_sample_is_validated_before_analysis(tmp_path, method):
    path = tmp_path / "data.csv"
    _write_csv(path)
    data = eda.read(path, mode="large")
    for invalid in (False, -1, "10", 1.5):
        with pytest.raises(ValueError, match="sample must be a positive integer"):
            getattr(data.eda, method)(sample=invalid)


def test_large_explicit_none_uses_method_default(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    path = tmp_path / "data.csv"
    _write_csv(path, rows=10_001)
    data = eda.read(path, mode="large", chunksize=5)
    result = data.eda.vizall(sample=None, target="churned", max_plots=1)
    plt.close(result["figures"][0])
    assert result["scope"] == {"rows": 10_001, "sample_rows": 10_000, "sample_based": True}
