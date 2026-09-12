"""Task-aware ML guidance introduced for the 0.1.5 release."""

import numpy as np
import pandas as pd
import pytest

import noweda  # noqa: F401 - registers the DataFrame accessor
from noweda.ml_tasks import PROBLEM_TYPES, build_ml_plan


def _names(plan):
    return [item["name"] for item in plan["recommendations"]]


def test_mlall_without_objective_prompts_for_a_supported_task(capsys):
    df = pd.DataFrame({"amount": [10, 20, 30], "group": ["a", "b", "a"]})

    result = df.eda.mlall()
    output = capsys.readouterr().out

    assert result is None
    assert "No ML objective selected" in output
    assert all(problem_type in output for problem_type in PROBLEM_TYPES)
    assert "Candidate methods" not in output
    assert "Rating" not in output
    assert "★" not in output


def test_binary_classification_is_inferred_and_target_is_excluded():
    df = pd.DataFrame({
        "amount": np.arange(40, dtype=float),
        "channel": ["web", "store"] * 20,
        "fraud_flag": [0] * 30 + [1] * 10,
    })

    plan = df.eda.ml_plan(target="fraud_flag")

    assert plan["problem_type"] == "classification"
    assert plan["problem_subtype"] == "binary"
    assert plan["inferred"] is True
    assert "two distinct numeric" in plan["inference_reason"]
    assert plan["features"] == ["amount", "channel"]
    assert plan["target_summary"]["class_counts"] == {"0": 30, "1": 10}
    assert any("2:1" in warning for warning in plan["warnings"])
    assert any("precision-recall AUC" in item for item in plan["evaluation"])
    assert "Logistic Regression" in _names(plan)
    assert not any("Regressor" in name for name in _names(plan))
    scores = [candidate["score"] for candidate in plan["recommendations"]]
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
    assert "no models were trained or measured" in output


def test_low_cardinality_integer_target_is_inferred_as_multiclass():
    df = pd.DataFrame({"x": np.arange(60), "label": [0, 1, 2] * 20})

    plan = df.eda.ml_plan(target="label")

    assert plan["problem_type"] == "classification"
    assert plan["problem_subtype"] == "multiclass"
    assert "low-cardinality integer" in plan["inference_reason"]
    assert any("macro F1" in item for item in plan["evaluation"])


def test_continuous_numeric_target_is_inferred_as_regression():
    df = pd.DataFrame({
        "x": np.linspace(0, 1, 50),
        "segment": ["a", "b"] * 25,
        "price": np.linspace(12.5, 95.0, 50),
    })

    plan = df.eda.ml_plan(target="price")

    assert plan["problem_type"] == "regression"
    assert plan["problem_subtype"] == "continuous"
    assert "high-cardinality numeric" in plan["inference_reason"]
    assert "Ridge / Elastic Net Regression" in _names(plan)
    assert not any("Classifier" in name for name in _names(plan))
    assert any("RMSE" in item for item in plan["evaluation"])


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

    plan = df.eda.ml_plan(problem_type=problem_type)

    assert plan["target"] is None
    assert expected_method in _names(plan)
    assert any(expected_evaluation in item for item in plan["evaluation"])
    scores = [candidate["score"] for candidate in plan["recommendations"]]
    assert scores == sorted(scores, reverse=True)
    assert all(1 <= score <= 5 for score in scores)


def test_categorical_clustering_prioritizes_a_mixed_type_method():
    df = pd.DataFrame({
        "region": ["north", "south"] * 20,
        "tier": ["basic", "plus", "premium", "basic"] * 10,
    })
    plan = df.eda.ml_plan(problem_type="clustering")
    assert plan["recommendations"][0]["name"] == "K-Modes / K-Prototypes"
    assert plan["recommendations"][0]["score"] == 4.5


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
    assert df.eda.ml_plan(problem_type=alias)["problem_type"] == expected


def test_classification_alias_validates_requested_subtype():
    df = pd.DataFrame({"x": range(30), "label": [0, 1, 2] * 10})

    assert df.eda.ml_plan(target="label", problem_type="multiclass")["problem_subtype"] == "multiclass"
    with pytest.raises(ValueError, match="Requested binary classification"):
        df.eda.ml_plan(target="label", problem_type="binary")


def test_features_can_be_selected_and_must_not_include_target():
    df = pd.DataFrame({"keep": range(40), "drop": range(40), "label": [0, 1] * 20})

    plan = df.eda.ml_plan(target="label", features=["keep"])
    assert plan["features"] == ["keep"]

    with pytest.raises(ValueError, match="target must not"):
        df.eda.ml_plan(target="label", features=["keep", "label"])
    with pytest.raises(ValueError, match="not found"):
        df.eda.ml_plan(target="label", features=["missing"])
    with pytest.raises(ValueError, match="at least one"):
        df.eda.ml_plan(target="label", features=[])
    with pytest.raises(TypeError, match="not a string"):
        df.eda.ml_plan(target="label", features="keep")


def test_non_string_column_labels_are_supported():
    df = pd.DataFrame({0: range(40), ("outcome", 1): [False, True] * 20})
    plan = df.eda.ml_plan(target=("outcome", 1), features=[0])
    assert plan["target"] == ("outcome", 1)
    assert plan["features"] == [0]


def test_missing_labels_are_reported():
    df = pd.DataFrame({"x": range(40), "label": [0, 1] * 19 + [None, None]})
    plan = df.eda.ml_plan(target="label", problem_type="classification")
    assert plan["target_summary"]["missing"] == 2
    assert any("missing target values" in warning for warning in plan["warnings"])


def test_tiny_class_and_possible_target_leakage_are_reported():
    df = pd.DataFrame({
        "source_value": np.arange(30, dtype=float),
        "derived_value": np.arange(30, dtype=float) * 2,
        "label": ["rare"] + ["common"] * 29,
    })
    classification = df.eda.ml_plan(target="label", problem_type="classification")
    assert classification["target_summary"]["smallest_class_count"] == 1
    assert any("stratified splitting is not possible" in warning for warning in classification["warnings"])

    regression = df.eda.ml_plan(
        target="derived_value",
        problem_type="regression",
        features=["source_value"],
    )
    assert any("near-perfect correlation" in warning for warning in regression["warnings"])


def test_likely_identifier_target_is_flagged_for_review():
    df = pd.DataFrame({"x": range(40), "case_id": range(1000, 1040)})
    plan = df.eda.ml_plan(target="case_id")
    assert any("target is marked as a likely identifier" in warning for warning in plan["warnings"])


@pytest.mark.parametrize(
    "values, message",
    [
        ([None, None, None], "no nonmissing observations"),
        ([1, 1, 1], "constant"),
    ],
)
def test_invalid_targets_are_rejected(values, message):
    df = pd.DataFrame({"x": range(len(values)), "target": values})
    with pytest.raises(ValueError, match=message):
        df.eda.ml_plan(target="target")


def test_regression_requires_finite_numeric_target():
    report = {
        "results": {"missing": {}, "outliers": {}, "correlation": {}, "stats": {}, "schema": {}},
        "scores": {},
    }
    text_target = pd.DataFrame({"x": range(3), "target": ["low", "medium", "high"]})
    with pytest.raises(ValueError, match="numeric"):
        build_ml_plan(text_target, report, target="target", problem_type="regression")

    infinite_target = pd.DataFrame({"x": range(3), "target": [1.0, 2.0, np.inf]})
    with pytest.raises(ValueError, match="infinite"):
        build_ml_plan(infinite_target, report, target="target", problem_type="regression")


def test_problem_type_and_target_contract_errors_are_clear():
    df = pd.DataFrame({"x": range(10), "label": [0, 1] * 5})

    with pytest.raises(ValueError, match="requires target"):
        df.eda.ml_plan(problem_type="classification")
    with pytest.raises(ValueError, match="only accepted"):
        df.eda.ml_plan(problem_type="clustering", target="label")
    with pytest.raises(ValueError, match="Forecasting is planned"):
        df.eda.ml_plan(problem_type="forecasting")
    with pytest.raises(ValueError, match="Unsupported problem_type"):
        df.eda.ml_plan(problem_type="ranking")
    with pytest.raises(ValueError, match="Target column not found"):
        df.eda.ml_plan(target="absent")


def test_large_supervised_plans_omit_quadratic_kernel_candidates():
    size = 100_000
    df = pd.DataFrame({"x": np.arange(size), "label": np.arange(size) % 2})
    report = {
        "results": {"missing": {}, "outliers": {}, "correlation": {}, "stats": {}, "schema": {}},
        "scores": {},
    }

    plan = build_ml_plan(df, report, target="label", problem_type="classification")
    assert "Support Vector Classifier" not in _names(plan)


def test_ml_plan_does_not_mutate_the_dataframe():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "label": [0, 1, 0]})
    before = df.copy(deep=True)
    df.eda.ml_plan(target="label")
    pd.testing.assert_frame_equal(df, before)
