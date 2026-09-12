"""Tests for the single-method task-aware ML guidance API."""

import numpy as np
import pandas as pd
import pytest

import noweda  # noqa: F401 - registers the DataFrame accessor
from noweda.ml_tasks import PROBLEM_TYPES, build_ml_guidance


def _names(result):
    return [item["name"] for item in result["recommendations"]]


def _plan(df, *args, **kwargs):
    """Use the public structured-result option without duplicating test noise."""
    kwargs["plan"] = True
    return df.eda.mlall(*args, **kwargs)


def test_mlall_is_the_only_public_ml_guidance_method():
    df = pd.DataFrame({"amount": [10, 20, 30]})
    assert hasattr(df.eda, "mlall")
    assert not hasattr(df.eda, "ml_plan")


def test_mlall_without_objective_assesses_dataset_and_returns_plan_on_request(capsys):
    df = pd.DataFrame({
        "customer_id": range(1000, 1040),
        "amount": np.arange(40, dtype=float),
        "frequency": np.arange(40) % 5,
        "country": ["us", "ca"] * 20,
    })

    result = _plan(df)
    output = capsys.readouterr().out

    assert result["problem_type"] is None
    assert result["assessment"]["unsupervised_readiness"] == "high"
    assert result["assessment"]["supervised_readiness"] == "not_assessed"
    assert "customer_id" in result["assessment"]["likely_identifiers"]
    assert "amount" in result["assessment"]["usable_features"]
    assert all(direction["problem_type"] in PROBLEM_TYPES for direction in result["assessment"]["directions"])
    assert "Dataset ML assessment" in output
    assert "Recommended analytical directions" in output
    assert "Estimated fit:" in output
    assert "★" in output


def test_mlall_default_prints_without_returning_a_plan(capsys):
    df = pd.DataFrame({"x": range(40), "y": np.arange(40) % 5})

    result = df.eda.mlall()
    output = capsys.readouterr().out

    assert result is None
    assert "Dataset ML assessment" in output


def test_automatic_assessment_lists_only_possible_targets_not_selected_ones():
    df = pd.DataFrame({
        "amount": np.arange(40, dtype=float),
        "account_age": np.arange(40) % 10,
        "fraud_flag": [0] * 30 + [1] * 10,
    })

    result = _plan(df)
    candidate = result["assessment"]["target_candidates"][0]

    assert result["target"] is None
    assert candidate["column"] == "fraud_flag"
    assert candidate["problem_type"] == "classification"
    assert candidate["readiness"] == "ready"
    assert "fraud_flag" in result["assessment"]["excluded_candidate_targets"]


def test_binary_classification_is_inferred_and_target_is_excluded():
    df = pd.DataFrame({
        "amount": np.arange(40, dtype=float),
        "channel": ["web", "store"] * 20,
        "fraud_flag": [0] * 30 + [1] * 10,
    })

    result = _plan(df, target="fraud_flag")

    assert result["problem_type"] == "classification"
    assert result["problem_subtype"] == "binary"
    assert result["inferred"] is True
    assert "two distinct numeric" in result["inference_reason"]
    assert result["features"] == ["amount", "channel"]
    assert result["target_summary"]["class_counts"] == {"0": 30, "1": 10}
    assert result["target_summary"]["readiness"] == "ready"
    assert any("2:1" in warning for warning in result["warnings"])
    assert any("precision-recall AUC" in item for item in result["evaluation"])
    assert "Logistic Regression" in _names(result)
    assert not any("Regressor" in name for name in _names(result))
    scores = [candidate["score"] for candidate in result["recommendations"]]
    assert scores == sorted(scores, reverse=True)
    assert all(1 <= score <= 5 for score in scores)


def test_mlall_prints_heuristic_stars_and_five_point_score(capsys):
    df = pd.DataFrame({"x": range(40), "label": [0, 1] * 20})

    df.eda.mlall(target="label")
    output = capsys.readouterr().out

    assert "ranked by estimated fit" in output
    assert "Estimated fit:" in output
    assert "★" in output
    assert "/5)" in output
    assert "no models were trained or measured" in output


def test_low_cardinality_integer_target_is_inferred_as_multiclass():
    df = pd.DataFrame({"x": np.arange(60), "label": [0, 1, 2] * 20})

    result = _plan(df, target="label")

    assert result["problem_type"] == "classification"
    assert result["problem_subtype"] == "multiclass"
    assert "low-cardinality integer" in result["inference_reason"]
    assert any("macro F1" in item for item in result["evaluation"])


def test_continuous_numeric_target_is_inferred_as_regression():
    df = pd.DataFrame({
        "x": np.linspace(0, 1, 50),
        "segment": ["a", "b"] * 25,
        "price": np.linspace(12.5, 95.0, 50),
    })

    result = _plan(df, target="price")

    assert result["problem_type"] == "regression"
    assert result["problem_subtype"] == "continuous"
    assert "high-cardinality numeric" in result["inference_reason"]
    assert "Ridge / Elastic Net Regression" in _names(result)
    assert not any("Classifier" in name for name in _names(result))
    assert any("RMSE" in item for item in result["evaluation"])


@pytest.mark.parametrize(
    "problem_type, expected_method, expected_evaluation",
    [
        ("clustering", "K-Means / MiniBatchKMeans", "silhouette"),
        ("anomaly_detection", "Isolation Forest", "alert volume"),
        ("dimensionality_reduction", "PCA", "explained variance"),
    ],
)
def test_unsupervised_tasks_return_task_specific_guidance(
    problem_type, expected_method, expected_evaluation
):
    df = pd.DataFrame({"x": np.arange(40), "y": np.arange(40) % 5})

    result = _plan(df, problem_type=problem_type)

    assert result["target"] is None
    assert expected_method in _names(result)
    assert any(expected_evaluation in item for item in result["evaluation"])
    scores = [candidate["score"] for candidate in result["recommendations"]]
    assert scores == sorted(scores, reverse=True)
    assert all(1 <= score <= 5 for score in scores)


def test_categorical_clustering_prioritizes_a_mixed_type_method(capsys):
    df = pd.DataFrame({
        "region": ["north", "south"] * 20,
        "tier": ["basic", "plus", "premium", "basic"] * 10,
    })
    result = _plan(df, problem_type="clustering")
    output = capsys.readouterr().out
    assert result["recommendations"][0]["name"] == "K-Modes / K-Prototypes"
    assert result["recommendations"][0]["score"] == 4.5
    assert "K-Modes or K-Prototypes" in output
    assert "K-Modes / K-Prototypes" not in output


@pytest.mark.parametrize(
    "alias, expected",
    [
        ("anomaly", "anomaly_detection"),
        ("outlier-detection", "anomaly_detection"),
        ("reduction", "dimensionality_reduction"),
    ],
)
def test_problem_type_aliases(alias, expected):
    df = pd.DataFrame({"x": range(10), "y": range(10, 20)})
    assert _plan(df, problem_type=alias)["problem_type"] == expected


def test_classification_alias_validates_requested_subtype():
    df = pd.DataFrame({"x": range(30), "label": [0, 1, 2] * 10})

    assert _plan(df, target="label", problem_type="multiclass")["problem_subtype"] == "multiclass"
    with pytest.raises(ValueError, match="Requested binary classification"):
        _plan(df, target="label", problem_type="binary")


def test_features_can_be_selected_and_must_not_include_target():
    df = pd.DataFrame({"keep": range(40), "drop": range(40), "label": [0, 1] * 20})

    result = _plan(df, target="label", features=["keep"])
    assert result["features"] == ["keep"]

    with pytest.raises(ValueError, match="target must not"):
        _plan(df, target="label", features=["keep", "label"])
    with pytest.raises(ValueError, match="not found"):
        _plan(df, target="label", features=["missing"])
    with pytest.raises(ValueError, match="at least one"):
        _plan(df, target="label", features=[])
    with pytest.raises(TypeError, match="not a string"):
        _plan(df, target="label", features="keep")


def test_non_string_column_labels_are_supported():
    df = pd.DataFrame({0: range(40), ("outcome", 1): [False, True] * 20})
    result = _plan(df, target=("outcome", 1), features=[0])
    assert result["target"] == ("outcome", 1)
    assert result["features"] == [0]


def test_partial_labels_are_reported_without_claiming_a_model_can_use_unlabeled_rows():
    df = pd.DataFrame({"x": range(100), "label": [0, 1] * 20 + [None] * 60})
    result = _plan(df, target="label", problem_type="classification")
    assert result["target_summary"]["missing"] == 60
    assert result["target_summary"]["label_coverage"] == 0.40
    assert result["target_summary"]["readiness"] == "partial"
    assert any("Supervised training uses those rows only" in warning for warning in result["warnings"])


def test_unlabeled_target_falls_back_to_unsupervised_assessment(capsys):
    df = pd.DataFrame({
        "amount": np.arange(40, dtype=float),
        "frequency": np.arange(40) % 5,
        "fraud_flag": [None] * 40,
    })
    result = _plan(df, target="fraud_flag", problem_type="classification")
    output = capsys.readouterr().out

    assert result["supervised_unavailable"] is True
    assert result["target_summary"]["readiness"] == "unlabeled"
    assert result["assessment"]["directions"]
    assert "Requested supervised task is not ready" in output
    assert "No labeled observations" in output


def test_unlabeled_target_without_features_reports_low_unsupervised_readiness():
    df = pd.DataFrame({"fraud_flag": [None] * 4})
    result = _plan(df, target="fraud_flag", problem_type="classification")

    assert result["supervised_unavailable"] is True
    assert result["assessment"]["usable_features"] == []
    assert result["assessment"]["unsupervised_readiness"] == "low"


def test_tiny_class_and_possible_target_leakage_are_reported():
    df = pd.DataFrame({
        "source_value": np.arange(30, dtype=float),
        "derived_value": np.arange(30, dtype=float) * 2,
        "event_timestamp": pd.date_range("2025-01-01", periods=30, freq="D"),
        "label": ["rare"] + ["common"] * 29,
    })
    classification = _plan(df, target="label", problem_type="classification")
    assert classification["target_summary"]["smallest_class_count"] == 1
    assert any("stratified splitting is not possible" in warning for warning in classification["warnings"])
    assert any("time-aware validation" in warning for warning in classification["warnings"])

    regression = _plan(
        df,
        target="derived_value",
        problem_type="regression",
        features=["source_value"],
    )
    assert any("near-perfect correlation" in warning for warning in regression["warnings"])


def test_temporal_guidance_uses_schema_roles_not_name_substrings():
    df = pd.DataFrame({
        "monthly_income": np.arange(30, dtype=float),
        "lifetime_value": np.arange(30, dtype=float) * 5,
        "event_timestamp": pd.date_range("2025-01-01", periods=30, freq="D"),
        "label": ["yes", "no"] * 15,
    })

    result = _plan(df, target="label", problem_type="classification")
    warning = next(item for item in result["warnings"] if "Temporal feature" in item)

    assert "event_timestamp" in warning
    assert "monthly_income" not in warning
    assert "lifetime_value" not in warning


def test_likely_identifier_target_is_flagged_for_review():
    df = pd.DataFrame({"x": range(40), "case_id": range(1000, 1040)})
    result = _plan(df, target="case_id")
    assert any("target is marked as a likely identifier" in warning for warning in result["warnings"])


def test_constant_targets_are_rejected():
    df = pd.DataFrame({"x": range(3), "target": [1, 1, 1]})
    with pytest.raises(ValueError, match="constant"):
        _plan(df, target="target")


def test_regression_requires_finite_numeric_target():
    report = {
        "results": {"missing": {}, "outliers": {}, "correlation": {}, "stats": {}, "schema": {}},
        "scores": {},
    }
    text_target = pd.DataFrame({"x": range(3), "target": ["low", "medium", "high"]})
    with pytest.raises(ValueError, match="numeric"):
        build_ml_guidance(text_target, report, target="target", problem_type="regression")

    infinite_target = pd.DataFrame({"x": range(3), "target": [1.0, 2.0, np.inf]})
    with pytest.raises(ValueError, match="infinite"):
        build_ml_guidance(infinite_target, report, target="target", problem_type="regression")


def test_problem_type_and_target_contract_errors_are_clear():
    df = pd.DataFrame({"x": range(10), "label": [0, 1] * 5})

    with pytest.raises(ValueError, match="requires target"):
        _plan(df, problem_type="classification")
    with pytest.raises(ValueError, match="only accepted"):
        _plan(df, problem_type="clustering", target="label")
    with pytest.raises(ValueError, match="Forecasting is planned"):
        _plan(df, problem_type="forecasting")
    with pytest.raises(ValueError, match="Unsupported problem_type"):
        _plan(df, problem_type="ranking")
    with pytest.raises(ValueError, match="Target column not found"):
        _plan(df, target="absent")


def test_large_supervised_guidance_omits_quadratic_kernel_candidates():
    size = 100_000
    df = pd.DataFrame({"x": np.arange(size), "label": np.arange(size) % 2})
    report = {
        "results": {"missing": {}, "outliers": {}, "correlation": {}, "stats": {}, "schema": {}},
        "scores": {},
    }

    result = build_ml_guidance(df, report, target="label", problem_type="classification")
    assert "Support Vector Classifier" not in _names(result)


def test_mlall_does_not_mutate_the_dataframe():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "label": [0, 1, 0]})
    before = df.copy(deep=True)
    _plan(df, target="label")
    pd.testing.assert_frame_equal(df, before)
