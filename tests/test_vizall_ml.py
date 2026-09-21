import numpy as np
import pandas as pd
import pytest


pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _close_figures(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    yield
    plt.close("all")


def _classification_frame(rows=240):
    rng = np.random.RandomState(7)
    signal = rng.normal(size=rows)
    return pd.DataFrame({
        "customer_id": np.arange(rows),
        "email": ["person{}@example.org".format(index) for index in range(rows)],
        "linear_signal": signal,
        "curved_signal": signal ** 2,
        "segment": np.where(signal > 0.5, "high", np.where(signal < -0.5, "low", "mid")),
        "fraud_flag": (signal + rng.normal(scale=0.35, size=rows) > 0).astype(int),
    })


def test_vizall_returns_ranked_classification_diagnostics():
    frame = _classification_frame()

    result = frame.eda.vizall(target="fraud_flag", max_plots=7)

    assert result["problem_type"] == "classification"
    assert result["target"] == "fraud_flag"
    assert result["plots_generated"] <= 7
    assert "Target class distribution" in result["plot_titles"]
    assert "customer_id" in result["excluded_features"]
    assert "email" in result["excluded_features"]
    assert "fraud_flag" not in result["selected_features"]
    assert result["model_signals"]
    assert len(result["figures"]) <= result["plots_generated"]


def test_vizall_returns_regression_relationships_and_signals():
    rng = np.random.RandomState(11)
    feature = np.linspace(-3, 3, 240)
    frame = pd.DataFrame({
        "feature": feature,
        "category": np.where(feature > 0, "positive", "negative"),
        "target": 4 * feature + rng.normal(scale=0.4, size=len(feature)),
    })

    result = frame.eda.vizall(target="target", max_plots=5)

    assert result["problem_type"] == "regression"
    assert "Regression target distribution" in result["plot_titles"]
    assert result["associations"][0]["column"] == "feature"
    assert any("linear regression" in signal for signal in result["model_signals"])
    assert result["plots_generated"] <= 5


def test_vizall_plot_budget_counts_panels_not_only_figures():
    frame = _classification_frame()

    result = frame.eda.vizall(target="fraud_flag", max_plots=3)

    assert result["plots_generated"] == 3
    assert sum(len(figure.axes) for figure in result["figures"]) >= 3


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "10"])
def test_vizall_rejects_invalid_plot_budget(value):
    frame = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})

    with pytest.raises(ValueError, match="max_plots must be a positive integer"):
        frame.eda.vizall(target="target", max_plots=value)


def test_vizall_rejects_missing_or_empty_target():
    frame = pd.DataFrame({"feature": [1, 2, 3], "target": [None, None, None]})

    with pytest.raises(ValueError, match="Target column not found"):
        frame.eda.vizall(target="missing")
    with pytest.raises(ValueError, match="has no observed values"):
        frame.eda.vizall(target="target")


def test_vizall_sampling_scope_is_returned(capsys):
    frame = _classification_frame(rows=100)

    result = frame.eda.vizall(sample=40, target="fraud_flag", max_plots=2)

    assert result["scope"] == {"rows": 100, "sample_rows": 40, "sample_based": True}
    assert "40 of 100 rows" in capsys.readouterr().out


def test_large_vizall_forwards_target_and_plot_budget(tmp_path):
    import noweda as eda

    path = tmp_path / "classification.csv"
    _classification_frame(rows=80).to_csv(path, index=False)

    result = eda.read(path, mode="large", chunksize=30).eda.vizall(
        target="fraud_flag", max_plots=4, sample=30
    )

    assert result["problem_type"] == "classification"
    assert result["plots_generated"] <= 4
    assert result["scope"] == {"rows": 80, "sample_rows": 30, "sample_based": True}


def test_vizall_supports_mixed_hashable_column_labels():
    frame = pd.DataFrame({
        ("feature", "one"): np.arange(60, dtype=float),
        "feature_two": np.arange(60, dtype=float) ** 2,
        ("target", "flag"): [0, 1] * 30,
    })

    result = frame.eda.vizall(target=("target", "flag"), max_plots=5)

    assert result["problem_type"] == "classification"
    assert result["target"] == ("target", "flag")


def test_mlall_reuses_visual_diagnostic_evidence():
    frame = _classification_frame()

    plan = frame.eda.mlall(target="fraud_flag", plan=True)

    assert plan["visual_diagnostics"]["problem_type"] == "classification"
    assert plan["visual_diagnostics"]["associations"]
    assert plan["visual_diagnostics"]["model_signals"]


def test_vizall_flags_near_deterministic_features_as_possible_leakage():
    target = np.array([0, 1] * 80)
    frame = pd.DataFrame({
        "leaked_score": target.astype(float),
        "ordinary_feature": np.linspace(0, 1, len(target)),
        "target": target,
    })

    result = frame.eda.vizall(target="target", max_plots=3)

    assert any("Possible target leakage" in signal for signal in result["model_signals"])


def test_vizall_ignores_unused_target_categories_for_imbalance():
    labels = pd.Categorical(["a", "b"] * 40, categories=["a", "b", "unused"])
    frame = pd.DataFrame({"feature": np.arange(80), "target": labels})

    result = frame.eda.vizall(target="target", max_plots=2)

    assert result["target_summary"]["classes"] == 2
    assert not any("Class imbalance" in item for item in result["observations"])
