"""Shared profiling helper for task-aware ML guidance.

The recommendation API lives in :mod:`noweda.ml_tasks`. This module retains the
profile helper used by older code and regression tests.
"""

from noweda.dtypes import is_textual


def _profile(df, stats, schema, scores, results, target=None):
    """Build the feature profile used by the task-specific recommenders."""
    del schema  # Kept in the signature for compatibility with earlier releases.
    missing = results.get("missing", {})
    outliers = results.get("outliers", {})
    correlation = results.get("correlation", {})

    if target is not None and target not in df.columns:
        raise ValueError("Target column not found: {!r}".format(target))
    target_values = df[target] if target is not None else None
    feature_df = df.drop(columns=[target]) if target is not None else df

    missing = {column: value for column, value in missing.items() if column in feature_df.columns}
    outliers = {column: value for column, value in outliers.items() if column in feature_df.columns}
    correlation = {
        column: {
            other: value for other, value in values.items()
            if other in feature_df.columns
        }
        for column, values in correlation.items()
        if column in feature_df.columns
    }
    numeric_columns = [
        column for column in feature_df.columns
        if feature_df[column].dtype.kind in ("i", "u", "f")
    ]
    categorical_columns = [
        column for column in feature_df.columns if is_textual(feature_df[column])
    ]
    row_count, column_count = feature_df.shape

    skewed_count = sum(
        1 for column in numeric_columns
        if column in stats and abs(stats[column].get("skewness", 0)) > 1
    )
    has_high_correlation = any(
        first != second and abs(value) > 0.85
        for first, values in correlation.items()
        for second, value in values.items()
    )
    max_missing = max(missing.values()) if missing else 0
    high_cardinality = [
        column for column in categorical_columns
        if feature_df[column].nunique() > 20
    ]

    observed_numeric_cells = sum(
        int(feature_df[column].notna().sum()) for column in numeric_columns
    )
    total_outliers = sum(outliers.values()) if outliers else 0
    outlier_fraction = (
        total_outliers / observed_numeric_cells if observed_numeric_cells else 0
    )

    imbalanced_columns = {}
    if target_values is not None:
        counts = target_values.value_counts()
        counts = counts[counts > 0]
        if len(counts) > 1 and counts.max() / counts.min() > 2:
            imbalanced_columns[target] = float(counts.max() / counts.sum())

    return {
        "target": target,
        "n_rows": row_count,
        "n_cols": column_count,
        "n_numeric": len(numeric_columns),
        "n_categorical": len(categorical_columns),
        "numeric_cols": numeric_columns,
        "cat_cols": categorical_columns,
        "small": row_count < 1_000,
        "medium": 1_000 <= row_count < 100_000,
        "large": row_count >= 100_000,
        "wide": column_count > 50,
        "mostly_numeric": len(numeric_columns) >= len(categorical_columns),
        "mostly_categorical": len(categorical_columns) > len(numeric_columns),
        "mixed": len(numeric_columns) >= 2 and len(categorical_columns) >= 1,
        "has_high_corr": has_high_correlation,
        "n_skewed": skewed_count,
        "max_missing_pct": max_missing,
        "high_missing": max_missing > 0.20,
        "high_card_cats": high_cardinality,
        "outlier_heavy": outlier_fraction > 0.05,
        "has_imbalance": bool(imbalanced_columns),
        "imbalanced_cols": imbalanced_columns,
        "dq": scores.get("data_quality", 0),
        "model_readiness": scores.get("model_readiness", 0),
        "risk": scores.get("risk", 0),
    }
