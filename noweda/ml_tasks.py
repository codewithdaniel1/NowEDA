"""Task-aware ML guidance built from a NowEDA profile, without training models."""

import math

import numpy as np
import pandas as pd

from noweda.dtypes import is_textual
from noweda.ml_recommendations import _profile


PROBLEM_TYPES = (
    "classification",
    "regression",
    "clustering",
    "anomaly_detection",
    "dimensionality_reduction",
)

_SUPERVISED_TYPES = ("classification", "regression")
_UNSUPERVISED_TYPES = (
    "clustering",
    "anomaly_detection",
    "dimensionality_reduction",
)

# A column name is only a hint.  NowEDA never treats these as an automatically
# selected prediction target; it presents them for the user to review.
_TARGET_NAME_HINTS = (
    "target", "label", "outcome", "response", "class", "fraud", "churn",
    "default", "flag", "status", "converted", "conversion",
)
_TEMPORAL_NAME_HINTS = ("date", "time", "timestamp", "period", "month", "year")
_IDENTIFIER_NAME_HINTS = ("_id", "id_", "uuid", "guid", "identifier", "record_key")

_ALIASES = {
    "binary": "classification",
    "binary_classification": "classification",
    "multiclass": "classification",
    "multiclass_classification": "classification",
    "anomaly": "anomaly_detection",
    "outlier_detection": "anomaly_detection",
    "dimensionality": "dimensionality_reduction",
    "reduction": "dimensionality_reduction",
}


def _normalise_problem_type(problem_type):
    if problem_type is None:
        return None, None
    value = str(problem_type).strip().lower().replace("-", "_").replace(" ", "_")
    requested_subtype = None
    if value in ("binary", "binary_classification"):
        requested_subtype = "binary"
    elif value in ("multiclass", "multiclass_classification"):
        requested_subtype = "multiclass"
    value = _ALIASES.get(value, value)
    if value == "forecasting":
        raise ValueError(
            "Forecasting is planned but not supported yet; it requires a time column, "
            "forecast horizon, and time-aware validation."
        )
    if value == "auto":
        return None, None
    if value not in PROBLEM_TYPES:
        raise ValueError(
            "Unsupported problem_type {!r}. Choose one of: {}.".format(
                problem_type, ", ".join(PROBLEM_TYPES)
            )
        )
    return value, requested_subtype


def _column_position(df, column, description):
    if column not in df.columns:
        raise ValueError("{} column not found: {!r}".format(description, column))
    location = df.columns.get_loc(column)
    if not isinstance(location, (int, np.integer)):
        raise ValueError("NowEDA requires unique column labels for ML guidance.")
    return int(location)


def _feature_frame(df, target=None, features=None):
    if features is None:
        selected = [col for col in df.columns if col != target]
    else:
        if isinstance(features, (str, bytes)):
            raise TypeError("features must be a sequence of column labels, not a string")
        selected = list(features)
        if not selected:
            raise ValueError("features must contain at least one column label")
        missing = [col for col in selected if col not in df.columns]
        if missing:
            raise ValueError("Feature column(s) not found: {}".format(missing))
        if len(set(selected)) != len(selected):
            raise ValueError("features contains duplicate column labels")
        if target is not None and target in selected:
            raise ValueError("The target must not also appear in features")
    if not selected:
        raise ValueError("At least one feature column is required")
    positions = [_column_position(df, col, "Feature") for col in selected]
    return df.iloc[:, positions].copy(), selected


def _column_name_text(column):
    """Return a conservative text representation for name-based hints."""
    return str(column).strip().lower()


def _has_name_hint(column, hints):
    name = _column_name_text(column)
    return any(hint in name for hint in hints)


def _looks_like_identifier(column):
    name = _column_name_text(column)
    return name == "id" or _has_name_hint(column, _IDENTIFIER_NAME_HINTS)


def _label_readiness(observed_count, total_count, problem_type=None, class_counts=None):
    """Describe whether a named outcome has enough observed labels to proceed."""
    coverage = observed_count / total_count if total_count else 0.0
    if observed_count == 0:
        return "unlabeled", coverage
    if observed_count < 30:
        return "limited", coverage
    if problem_type == "classification" and class_counts and min(class_counts.values()) < 5:
        return "limited", coverage
    if coverage < 0.80:
        return "partial", coverage
    return "ready", coverage


def _candidate_target(df, column, schema):
    """Return a possible target only when both name and values support the idea."""
    name_hint = _has_name_hint(column, _TARGET_NAME_HINTS)
    role = schema.get(column, {}).get("role")
    series = df.iloc[:, _column_position(df, column, "Candidate")]
    observed = series.dropna()
    total = int(len(series))
    observed_count = int(len(observed))

    # An all-missing column can still be a useful candidate when its name makes
    # its intended use clear (for example, fraud_flag before labeling begins).
    if not name_hint or role == "id_candidate":
        return None

    unique = int(observed.nunique())
    if observed_count and unique < 2:
        return None

    problem_type = None
    subtype = None
    reason = "name suggests a possible outcome"
    class_counts = None
    if observed_count:
        try:
            problem_type, type_reason = _infer_supervised_type(observed)
        except ValueError:
            return None
        reason = "{}; {}".format(reason, type_reason)
        if problem_type == "classification":
            subtype = "binary" if unique == 2 else "multiclass"
            class_counts = {str(label): int(count) for label, count in observed.value_counts().items()}
        else:
            subtype = "continuous"

    readiness, coverage = _label_readiness(
        observed_count, total, problem_type, class_counts=class_counts
    )
    score = 3.0
    if observed_count:
        score += 0.5
    if readiness == "ready":
        score += 0.5
    elif readiness == "unlabeled":
        score -= 0.5
    return {
        "column": column,
        "score": _fit_score(score),
        "problem_type": problem_type,
        "problem_subtype": subtype,
        "observations": total,
        "usable_observations": observed_count,
        "label_coverage": coverage,
        "readiness": readiness,
        "reason": reason,
    }


def _target_candidates(df, results):
    schema = results.get("schema", {})
    candidates = [
        candidate for column in df.columns
        for candidate in [_candidate_target(df, column, schema)]
        if candidate is not None
    ]
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def _assessment_features(df, selected_features, results, candidates):
    """Exclude likely identifiers and possible outcomes from automatic unsupervised guidance."""
    schema = results.get("schema", {})
    candidate_columns = {candidate["column"] for candidate in candidates}
    id_features = [
        column for column in selected_features
        if (
            schema.get(column, {}).get("role") == "id_candidate"
            and _looks_like_identifier(column)
        )
    ]
    excluded_candidates = [
        column for column in selected_features if column in candidate_columns
    ]
    usable = [
        column for column in selected_features
        if column not in id_features and column not in excluded_candidates
    ]
    return df.loc[:, usable].copy(), usable, id_features, excluded_candidates


def _assessment_direction(problem_type, recommendations, profile):
    """Summarize one unsupervised direction without claiming measured performance."""
    top = recommendations[0]
    if problem_type == "anomaly_detection":
        reason = (
            "Multiple usable numeric features can be screened for unusual combinations."
            if profile["n_numeric"] >= 2
            else "Feature encoding is needed before unusual observations can be compared."
        )
    elif problem_type == "clustering":
        reason = (
            "Multiple usable features can support exploratory segmentation after preprocessing."
            if profile["n_cols"] >= 2
            else "More than one usable feature is needed for meaningful segmentation."
        )
    else:
        reason = (
            "Several features may benefit from compact exploratory representations."
            if profile["n_cols"] >= 2
            else "More than one usable feature is needed for dimensionality reduction."
        )
    return {
        "problem_type": problem_type,
        "score": top["score"],
        "reason": reason,
        "recommendations": recommendations,
    }


def _infer_supervised_type(target):
    observed = target.dropna()
    unique = int(observed.nunique())
    n = len(observed)
    dtype = observed.dtype
    if pd.api.types.is_bool_dtype(dtype) or is_textual(observed):
        return "classification", "categorical or boolean target values"
    if not pd.api.types.is_numeric_dtype(dtype):
        raise ValueError(
            "Cannot infer a supervised problem from target dtype {!s}; specify "
            "problem_type explicitly or convert the target.".format(dtype)
        )
    if unique == 2:
        return "classification", "two distinct numeric target values"
    low_cardinality = unique <= min(20, max(3, int(math.sqrt(max(n, 1)))))
    if pd.api.types.is_integer_dtype(dtype) and low_cardinality:
        return "classification", "low-cardinality integer target values"
    return "regression", "continuous or high-cardinality numeric target values"


def _target_summary(df, target, problem_type, requested_subtype=None):
    position = _column_position(df, target, "Target")
    series = df.iloc[:, position]
    observed = series.dropna()
    if observed.empty:
        return {
            "column": target,
            "dtype": str(series.dtype),
            "observations": int(len(series)),
            "usable_observations": 0,
            "missing": int(series.isna().sum()),
            "unique": 0,
            "label_coverage": 0.0,
            "readiness": "unlabeled",
            "subtype": None,
        }, [
            "Target {!r} has no labeled observations, so supervised training cannot begin.".format(target),
            "Review label acquisition, or consider an explicitly selected unsupervised objective."
        ]
    unique = int(observed.nunique())
    if unique < 2:
        raise ValueError("Target {!r} is constant; at least two outcomes are required".format(target))

    summary = {
        "column": target,
        "dtype": str(series.dtype),
        "observations": int(len(series)),
        "usable_observations": int(len(observed)),
        "missing": int(series.isna().sum()),
        "unique": unique,
    }
    warnings = []
    if summary["missing"]:
        warnings.append(
            "{} row(s) have missing target values and cannot be used for supervised training or evaluation."
            .format(summary["missing"])
        )
    if len(observed) < 30:
        warnings.append("Only {} labeled rows are available; validation estimates will be unstable.".format(len(observed)))

    if problem_type == "classification":
        counts = observed.value_counts()
        subtype = "binary" if unique == 2 else "multiclass"
        if requested_subtype and requested_subtype != subtype:
            raise ValueError(
                "Requested {} classification, but target {!r} has {} observed classes."
                .format(requested_subtype, target, unique)
            )
        summary["subtype"] = subtype
        summary["class_counts"] = {str(label): int(count) for label, count in counts.items()}
        summary["smallest_class_count"] = int(counts.min())
        summary["imbalanced"] = bool(counts.max() / counts.min() > 2)
        if unique > max(20, int(math.sqrt(len(observed)))):
            warnings.append(
                "The target has {} classes; verify that this is classification rather than an identifier."
                .format(unique)
            )
        if summary["imbalanced"]:
            warnings.append("Class counts differ by more than 2:1; use stratified splits and class-aware metrics.")
        if summary["smallest_class_count"] < 2:
            warnings.append(
                "At least one class has fewer than two rows, so ordinary stratified splitting is not possible."
            )
        elif summary["smallest_class_count"] < 5:
            warnings.append(
                "At least one class has fewer than five rows; validation for that class will be unstable."
            )
    else:
        if pd.api.types.is_bool_dtype(observed.dtype) or not pd.api.types.is_numeric_dtype(observed.dtype):
            raise ValueError("Regression requires a numeric, non-boolean target")
        numeric = observed.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(numeric).all():
            raise ValueError("Regression target contains infinite values")
        summary["subtype"] = "continuous"
        if unique <= 10:
            warnings.append(
                "The numeric target has only {} distinct values; classification or ordinal modeling may also be plausible."
                .format(unique)
            )
    readiness, coverage = _label_readiness(
        len(observed), len(series), problem_type, summary.get("class_counts")
    )
    summary["label_coverage"] = coverage
    summary["readiness"] = readiness
    if readiness == "partial":
        warnings.append(
            "Only {:.1%} of rows have labels. Supervised training uses those rows only; review label acquisition or a semi-supervised strategy before modeling."
            .format(coverage)
        )
    elif readiness == "limited":
        warnings.append(
            "Only {} labeled rows are available; collect more labels before relying on supervised validation."
            .format(len(observed))
        )
    return summary, warnings


def _fit_score(base, *adjustments):
    """Clamp a transparent heuristic score to 1-5 in half-point steps."""
    value = base + sum(adjustments)
    return min(5.0, max(1.0, round(value * 2) / 2))


def _recommendation(name, why, caution, score):
    return {"name": name, "why": why, "caution": caution, "score": score}


def _rank(recommendations):
    """Order candidates by dataset-fit heuristic, preserving ties."""
    return sorted(recommendations, key=lambda item: item["score"], reverse=True)


def _classification_recommendations(profile):
    recs = [
        _recommendation(
            "Logistic Regression",
            "Interpretable probability baseline for binary or multiclass outcomes.",
            "Encode categoricals, scale numeric features, and use regularization for correlated features.",
            _fit_score(
                4.0,
                0.5 if profile["small"] else 0,
                -0.5 if profile["mostly_categorical"] else 0,
                -0.5 if profile["high_missing"] else 0,
                -0.5 if profile["has_high_corr"] else 0,
            ),
        ),
        _recommendation(
            "Random Forest Classifier",
            "Captures nonlinear interactions and provides a robust tree-based baseline.",
            "Impute missing values and validate depth to limit overfitting, especially on small data.",
            _fit_score(
                4.0,
                -0.5 if profile["small"] else 0,
                -0.5 if profile["high_missing"] else 0,
                -0.5 if profile["large"] else 0,
            ),
        ),
        _recommendation(
            "Gradient-Boosted Trees",
            "Strong candidate for nonlinear tabular relationships and mixed feature types.",
            "Tune with validation data and early stopping; performance is not established by this recommendation.",
            _fit_score(
                4.5,
                -0.5 if profile["small"] else 0,
                -0.5 if profile["high_missing"] else 0,
                -0.5 if profile["mostly_categorical"] else 0,
            ),
        ),
    ]
    if not profile["large"]:
        recs.append(_recommendation(
            "Support Vector Classifier",
            "Can model complex boundaries on small or medium numeric datasets.",
            "Requires scaling and can become expensive as row count grows.",
            _fit_score(
                3.5,
                0.5 if profile["small"] else 0,
                -0.5 if profile["mostly_categorical"] else 0,
                -0.5 if profile["high_missing"] else 0,
                -0.5 if profile["wide"] else 0,
            ),
        ))
    return _rank(recs)


def _regression_recommendations(profile):
    recs = [
        _recommendation(
            "Ridge / Elastic Net Regression",
            "Interpretable regularized baseline for a continuous outcome.",
            "Encode categoricals, scale features, and inspect nonlinear residual patterns.",
            _fit_score(
                4.0,
                0.5 if profile["small"] else 0,
                -0.5 if profile["mostly_categorical"] else 0,
                -0.5 if profile["high_missing"] else 0,
            ),
        ),
        _recommendation(
            "Random Forest Regressor",
            "Captures nonlinear effects and feature interactions with limited distribution assumptions.",
            "Impute missing values and tune tree depth to control variance.",
            _fit_score(
                4.0,
                -0.5 if profile["small"] else 0,
                -0.5 if profile["high_missing"] else 0,
                -0.5 if profile["large"] else 0,
            ),
        ),
        _recommendation(
            "Gradient-Boosted Tree Regressor",
            "Flexible candidate for nonlinear tabular regression.",
            "Use early stopping and compare against the regularized linear baseline.",
            _fit_score(
                4.5,
                -0.5 if profile["small"] else 0,
                -0.5 if profile["high_missing"] else 0,
                -0.5 if profile["mostly_categorical"] else 0,
            ),
        ),
    ]
    if not profile["large"]:
        recs.append(_recommendation(
            "Support Vector Regression",
            "Useful for smooth nonlinear relationships in small or medium datasets.",
            "Scale inputs and the target; runtime grows quickly with row count.",
            _fit_score(
                3.5,
                0.5 if profile["small"] else 0,
                -0.5 if profile["mostly_categorical"] else 0,
                -0.5 if profile["high_missing"] else 0,
                -0.5 if profile["wide"] else 0,
            ),
        ))
    return _rank(recs)


def _clustering_recommendations(profile):
    recs = [
        _recommendation(
            "K-Means / MiniBatchKMeans",
            "Fast centroid baseline for scaled numeric features.",
            "Assumes roughly compact clusters and requires choosing the number of clusters.",
            _fit_score(
                4.0 if profile["mostly_numeric"] else 2.5,
                -0.5 if profile["high_missing"] else 0,
                -0.5 if profile["wide"] else 0,
            ),
        ),
        _recommendation(
            "DBSCAN",
            "Finds nonspherical dense groups and labels sparse observations as noise.",
            "Sensitive to scaling and neighborhood parameters; distance degrades in many dimensions.",
            _fit_score(
                3.5 if profile["mostly_numeric"] else 2.5,
                -0.5 if profile["large"] else 0,
                -0.5 if profile["wide"] else 0,
                -0.5 if profile["high_missing"] else 0,
            ),
        ),
        _recommendation(
            "Agglomerative Clustering",
            "Exposes hierarchical group structure without centroid assumptions.",
            "Memory and runtime make it best suited to smaller datasets.",
            _fit_score(
                3.5,
                0.5 if profile["small"] else 0,
                -1.5 if profile["large"] else 0,
                -0.5 if profile["mostly_categorical"] else 0,
            ),
        ),
    ]
    if profile["mostly_categorical"]:
        recs.insert(0, _recommendation(
            "K-Modes / K-Prototypes",
            "Uses categorical or mixed-type dissimilarities instead of raw Euclidean distance.",
            "Requires an additional implementation and careful feature weighting.",
            _fit_score(4.5 if profile["mostly_categorical"] else 3.5),
        ))
    return _rank(recs)


def _anomaly_recommendations(profile):
    recommendations = [
        _recommendation(
            "Isolation Forest",
            "Scales well and detects unusual multivariate combinations without labels.",
            "The contamination assumption changes the alert threshold and needs domain review.",
            _fit_score(
                4.0,
                -0.5 if profile["mostly_categorical"] else 0,
                -0.5 if profile["high_missing"] else 0,
            ),
        ),
        _recommendation(
            "Local Outlier Factor",
            "Finds observations that are unusual relative to their local neighborhood.",
            "Scale features; scoring new unseen rows requires novelty mode.",
            _fit_score(
                3.5,
                0.5 if profile["small"] else 0,
                -0.5 if profile["mostly_categorical"] else 0,
                -0.5 if profile["wide"] else 0,
            ),
        ),
    ]
    if not profile["large"]:
        recommendations.append(_recommendation(
            "One-Class SVM",
            "Can learn nonlinear boundaries around typical observations on smaller datasets.",
            "Sensitive to scaling and hyperparameters and expensive as row count grows.",
            _fit_score(
                3.0,
                0.5 if profile["small"] else 0,
                -0.5 if profile["mostly_categorical"] else 0,
                -0.5 if profile["wide"] else 0,
            ),
        ))
    return _rank(recommendations)


def _reduction_recommendations(profile):
    recs = [
        _recommendation(
            "PCA",
            "Reproducible linear compression for scaled dense numeric features.",
            "Components may be hard to interpret and cannot directly consume raw categoricals.",
            _fit_score(
                4.0 if profile["mostly_numeric"] else 2.5,
                -0.5 if profile["high_missing"] else 0,
            ),
        ),
        _recommendation(
            "Truncated SVD",
            "Works with sparse encoded matrices and avoids centering them.",
            "Like PCA, it captures linear structure and component meaning may be unclear.",
            _fit_score(
                4.0 if profile["wide"] or profile["n_categorical"] else 3.5,
                -0.5 if profile["high_missing"] else 0,
            ),
        ),
        _recommendation(
            "UMAP / t-SNE for exploration",
            "Useful for visualizing possible nonlinear neighborhoods in two or three dimensions.",
            "Plots can change with parameters and random seeds; do not treat visual groups as validated clusters.",
            _fit_score(
                3.5,
                0.5 if profile["small"] else 0,
                -0.5 if profile["large"] else 0,
                -0.5 if profile["high_missing"] else 0,
            ),
        ),
    ]
    if profile["mostly_categorical"]:
        recs.insert(0, _recommendation(
            "Multiple Correspondence Analysis",
            "Purpose-built linear exploration for categorical indicators.",
            "Requires categorical encoding choices and an additional implementation.",
            _fit_score(4.5),
        ))
    return _rank(recs)


def _preprocessing(problem_type, profile, target_summary):
    steps = []
    if target_summary and target_summary["missing"]:
        steps.append("Remove rows with missing target values before splitting X and y.")
    if profile["high_missing"] or profile["max_missing_pct"] > 0:
        steps.append("Choose imputers on training data only; preserve missingness indicators when meaningful.")
    if profile["n_categorical"]:
        steps.append("Encode categorical features inside the fitted pipeline; handle unseen categories.")
    if profile["n_numeric"]:
        steps.append("Scale numeric features for distance-, margin-, and component-based methods.")
    if profile["has_high_corr"]:
        steps.append("Review correlated features; regularization or dimensionality reduction may improve stability.")
    if problem_type == "classification" and target_summary["imbalanced"]:
        steps.append("Use class weights or resampling within training folds; never resample validation rows.")
    if problem_type in ("clustering", "anomaly_detection", "dimensionality_reduction"):
        steps.append("Exclude identifiers and features unavailable at the point where the method will be used.")
    return steps


def _evaluation(problem_type, target_summary):
    if problem_type == "classification":
        if target_summary["subtype"] == "binary":
            return [
                "Use stratified holdout or stratified cross-validation when every class has enough rows.",
                "Compare ROC-AUC, precision-recall AUC, F1, confusion matrix, and probability calibration.",
            ]
        return [
            "Use stratified holdout or cross-validation when every class has enough rows.",
            "Compare macro F1, weighted F1, per-class recall, confusion matrix, and log loss.",
        ]
    if problem_type == "regression":
        return [
            "Use holdout or K-fold validation; use grouped or time-aware splits when rows are dependent.",
            "Compare MAE, RMSE, R², residual plots, and errors across important subgroups.",
        ]
    if problem_type == "clustering":
        return [
            "Compare silhouette and stability across seeds or resamples, then inspect cluster usefulness with domain experts.",
            "Internal cluster scores do not prove that groups are meaningful for the intended decision.",
        ]
    if problem_type == "anomaly_detection":
        return [
            "Review top-ranked anomalies and alert volume with domain experts; test stability across windows or samples.",
            "If labels exist, use precision-recall metrics and cost at an operational threshold.",
        ]
    return [
        "For PCA/SVD, inspect explained variance, reconstruction error, and downstream validation performance.",
        "For visualization embeddings, test stability across parameters and random seeds.",
    ]


def _selected_results(results, selected_features):
    return {
        key: ({col: value for col, value in value.items() if col in selected_features}
              if key in ("missing", "outliers", "stats", "schema", "correlation") else value)
        for key, value in results.items()
    }


def _automatic_assessment(df, report, features=None, target=None, target_summary=None):
    """Assess plausible ML directions without selecting a target on the user's behalf."""
    results = report.get("results", {})
    if features is None:
        selected_features = [column for column in df.columns if column != target]
    else:
        _, selected_features = _feature_frame(df, target=target, features=features)
    candidates = _target_candidates(df, results)
    feature_df, usable_features, id_features, candidate_features = _assessment_features(
        df, selected_features, results, candidates
    )
    selected_results = _selected_results(results, usable_features)
    profile = _profile(
        feature_df,
        selected_results.get("stats", {}),
        selected_results.get("schema", {}),
        report.get("scores", {}),
        selected_results,
    )

    warnings = []
    if id_features:
        warnings.append(
            "Likely identifier feature(s) were excluded from automatic ML assessment: {}."
            .format(id_features)
        )
    if candidate_features:
        warnings.append(
            "Possible target column(s) were excluded from automatic unsupervised guidance: {}."
            .format(candidate_features)
        )
    if len(usable_features) < 2:
        unsupervised_readiness = "low"
        warnings.append(
            "At least two usable non-identifier features are needed for meaningful unsupervised guidance."
        )
    elif profile["high_missing"]:
        unsupervised_readiness = "moderate"
        warnings.append(
            "High missingness lowers unsupervised readiness until an imputation strategy is chosen."
        )
    else:
        unsupervised_readiness = "high"

    directions = []
    recommendations = []
    if unsupervised_readiness != "low":
        factories = {
            "clustering": _clustering_recommendations,
            "anomaly_detection": _anomaly_recommendations,
            "dimensionality_reduction": _reduction_recommendations,
        }
        for problem_type in _UNSUPERVISED_TYPES:
            task_recommendations = factories[problem_type](profile)
            directions.append(_assessment_direction(problem_type, task_recommendations, profile))
            top = dict(task_recommendations[0])
            top["problem_type"] = problem_type
            recommendations.append(top)
        directions.sort(key=lambda item: item["score"], reverse=True)
        recommendations.sort(key=lambda item: item["score"], reverse=True)

    if candidates:
        supervised_readiness = candidates[0]["readiness"]
    else:
        supervised_readiness = "not_assessed"
        warnings.append(
            "No likely target candidates were found. Select target= explicitly for supervised guidance."
        )

    assessment = {
        "supervised_readiness": supervised_readiness,
        "unsupervised_readiness": unsupervised_readiness,
        "target_candidates": candidates,
        "likely_identifiers": id_features,
        "excluded_candidate_targets": candidate_features,
        "usable_features": usable_features,
        "directions": directions,
    }
    return {
        "problem_type": None,
        "problem_subtype": None,
        "target": target,
        "target_summary": target_summary,
        "features": usable_features,
        "inferred": False,
        "inference_reason": None,
        "recommendations": recommendations,
        "preprocessing": _preprocessing("clustering", profile, None) if usable_features else [],
        "evaluation": [],
        "warnings": warnings,
        "assessment": assessment,
        "supervised_unavailable": False,
        "supported_problem_types": list(PROBLEM_TYPES),
    }


def build_ml_guidance(df, report, target=None, problem_type=None, features=None):
    """Build explainable ML guidance without fitting or evaluating models."""
    resolved, requested_subtype = _normalise_problem_type(problem_type)
    inferred = False
    inference_reason = None

    if resolved is None and target is None:
        return _automatic_assessment(df, report, features=features)

    supervised = resolved in _SUPERVISED_TYPES or resolved is None
    if supervised and target is None:
        raise ValueError("problem_type {!r} requires target=".format(resolved or "auto"))
    if not supervised and target is not None:
        raise ValueError("target is only accepted for classification or regression")

    results = report.get("results", {})
    if resolved is None:
        position = _column_position(df, target, "Target")
        observed = df.iloc[:, position].dropna()
        if observed.empty:
            summary, target_warnings = _target_summary(df, target, None)
            plan = _automatic_assessment(
                df, report, features=features, target=target, target_summary=summary
            )
            plan["warnings"] = target_warnings + plan["warnings"]
            return plan
        resolved, inference_reason = _infer_supervised_type(observed)
        inferred = True

    target_summary = None
    warnings = []
    if resolved in _SUPERVISED_TYPES:
        target_summary, target_warnings = _target_summary(
            df, target, resolved, requested_subtype=requested_subtype
        )
        warnings.extend(target_warnings)
        if target_summary["readiness"] == "unlabeled":
            plan = _automatic_assessment(
                df, report, features=features, target=target, target_summary=target_summary
            )
            plan.update({
                "problem_type": resolved,
                "target": target,
                "target_summary": target_summary,
                "inferred": inferred,
                "inference_reason": inference_reason,
                "warnings": warnings + plan["warnings"],
                "supervised_unavailable": True,
            })
            return plan

    if (
        target is not None
        and results.get("schema", {}).get(target, {}).get("role") == "id_candidate"
    ):
        warnings.append(
            "The target is marked as a likely identifier; confirm that it represents the intended outcome."
        )

    feature_df, selected_features = _feature_frame(df, target=target, features=features)
    if target is not None:
        correlation = results.get("correlation", {})
        near_perfect = []
        for feature in selected_features:
            value = correlation.get(target, {}).get(feature)
            if value is None:
                value = correlation.get(feature, {}).get(target)
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and abs(value) >= 0.98:
                near_perfect.append(feature)
        if near_perfect:
            warnings.append(
                "Feature(s) with near-perfect correlation to the target need leakage review: {}."
                .format(near_perfect)
            )
        temporal_features = [
            feature for feature in selected_features
            if _has_name_hint(feature, _TEMPORAL_NAME_HINTS)
        ]
        if temporal_features:
            warnings.append(
                "Temporal feature(s) detected: {}. Use time-aware validation when row order can reveal future information."
                .format(temporal_features)
            )

    selected_results = _selected_results(results, selected_features)
    profile = _profile(
        feature_df,
        selected_results.get("stats", {}),
        selected_results.get("schema", {}),
        report.get("scores", {}),
        selected_results,
    )
    profile["target"] = target
    id_features = [
        col for col in selected_features
        if results.get("schema", {}).get(col, {}).get("role") == "id_candidate"
    ]
    if id_features:
        warnings.append("Likely identifier feature(s) should be reviewed or excluded: {}.".format(id_features))

    factories = {
        "classification": _classification_recommendations,
        "regression": _regression_recommendations,
        "clustering": _clustering_recommendations,
        "anomaly_detection": _anomaly_recommendations,
        "dimensionality_reduction": _reduction_recommendations,
    }
    recommendations = factories[resolved](profile)
    return {
        "problem_type": resolved,
        "problem_subtype": target_summary.get("subtype") if target_summary else None,
        "target": target,
        "target_summary": target_summary,
        "features": selected_features,
        "inferred": inferred,
        "inference_reason": inference_reason,
        "recommendations": recommendations,
        "preprocessing": _preprocessing(resolved, profile, target_summary),
        "evaluation": _evaluation(resolved, target_summary),
        "warnings": warnings,
        "assessment": None,
        "supervised_unavailable": False,
        "supported_problem_types": list(PROBLEM_TYPES),
    }


def _stars(score):
    full = int(math.floor(score + 0.5))
    return "★" * full + "☆" * (5 - full)


def _readiness_label(value):
    return {
        "ready": "Ready",
        "partial": "Partial labels",
        "limited": "Limited labels",
        "unlabeled": "No labels",
        "high": "High",
        "moderate": "Moderate",
        "low": "Low",
        "not_assessed": "No likely target",
    }.get(value, str(value).replace("_", " ").title())


def format_ml_plan(plan):
    """Print a task plan in terminals and notebook text output."""
    bold, cyan, green, yellow, reset = "\033[1m", "\033[36m", "\033[32m", "\033[33m", "\033[0m"
    bar = "=" * 70
    print("\n{}{}{}{}".format(bold, cyan, bar, reset))
    print("{}{}  NowEDA · Task-Aware ML Guidance{}".format(bold, cyan, reset))
    print("{}{}{}{}".format(bold, cyan, bar, reset))

    assessment = plan.get("assessment")
    if assessment is not None:
        print("\n  {}Dataset ML assessment{}".format(bold, reset))
        print("  Usable features: {}".format(len(assessment["usable_features"])))
        print("  Supervised readiness: {}".format(
            _readiness_label(assessment["supervised_readiness"])
        ))
        print("  Unsupervised readiness: {}".format(
            _readiness_label(assessment["unsupervised_readiness"])
        ))
        if plan["target"] is not None:
            summary = plan["target_summary"]
            print("  Selected target: {!r}".format(plan["target"]))
            print("  Usable labels: {} / {} ({:.1%})".format(
                summary["usable_observations"], summary["observations"], summary["label_coverage"]
            ))

        candidates = assessment["target_candidates"]
        if candidates:
            print("\n  {}Potential targets — review before selecting one{}".format(bold, reset))
            for candidate in candidates[:3]:
                detail = candidate["problem_type"] or "labels unavailable"
                print("    • {!r}: {} labels ({:.1%}), {}, {}".format(
                    candidate["column"],
                    candidate["usable_observations"],
                    candidate["label_coverage"],
                    detail,
                    _readiness_label(candidate["readiness"]),
                ))

        if plan.get("supervised_unavailable"):
            print("\n  {}Requested supervised task is not ready{}".format(yellow, reset))
            print("  No labeled observations are available for the selected target.")

        if assessment["directions"]:
            print("\n  {}Recommended analytical directions (estimated fit){}".format(bold, reset))
            print("  " + "-" * 62)
            for direction in assessment["directions"]:
                print("\n  {}{}{}".format(bold, direction["problem_type"], reset))
                print("    {}Estimated fit:{} {} ({:.1f}/5)".format(
                    cyan, reset, _stars(direction["score"]), direction["score"]
                ))
                print("    {}Why:{} {}".format(green, reset, direction["reason"]))
                top = direction["recommendations"][0]
                print("    {}Start with:{} {}".format(green, reset, top["name"]))

        if plan["warnings"]:
            print("\n  {}Review before modeling:{}".format(yellow, reset))
            for warning in plan["warnings"]:
                print("    • {}".format(warning))

        if plan["preprocessing"]:
            print("\n  {}Preparation{}".format(bold, reset))
            print("  " + "-" * 62)
            for index, step in enumerate(plan["preprocessing"], 1):
                print("    {}. {}".format(index, step))

        print("\n  Select target= for supervised guidance, or problem_type= for a focused unsupervised review.")
        print("  Guidance is based on data characteristics; no models were trained or measured.")
        print("{}{}{}\n".format(cyan, bar, reset))
        return

    print("\n  {}Problem type:{} {}".format(bold, reset, plan["problem_type"]))
    if plan["problem_subtype"]:
        print("  {}Subtype:{} {}".format(bold, reset, plan["problem_subtype"]))
    if plan["target"] is not None:
        print("  {}Target:{} {!r}".format(bold, reset, plan["target"]))
        summary = plan["target_summary"]
        print("  {}Usable labels:{} {} / {}".format(
            bold, reset, summary["usable_observations"], summary["observations"]
        ))
    if plan["inferred"]:
        print("  {}Inference:{} {}. Override problem_type if this does not match your objective."
              .format(yellow, reset, plan["inference_reason"]))

    if plan["warnings"]:
        print("\n  {}Review before modeling:{}".format(yellow, reset))
        for warning in plan["warnings"]:
            print("    • {}".format(warning))

    print("\n  {}Candidate methods (ranked by estimated fit){}".format(bold, reset))
    print("  " + "-" * 62)
    for rec in plan["recommendations"]:
        print("\n  {}{}{}".format(bold, rec["name"], reset))
        print("    {}Estimated fit:{} {} ({:.1f}/5)".format(
            cyan, reset, _stars(rec["score"]), rec["score"]
        ))
        print("    {}Why:{} {}".format(green, reset, rec["why"]))
        print("    {}Caution:{} {}".format(yellow, reset, rec["caution"]))

    print("\n  {}Preprocessing{}".format(bold, reset))
    print("  " + "-" * 62)
    for index, step in enumerate(plan["preprocessing"], 1):
        print("    {}. {}".format(index, step))

    print("\n  {}Validation and metrics{}".format(bold, reset))
    print("  " + "-" * 62)
    for item in plan["evaluation"]:
        print("    • {}".format(item))
    print("\n  Guidance is based on data characteristics; no models were trained or measured.")
    print("{}{}{}\n".format(cyan, bar, reset))
