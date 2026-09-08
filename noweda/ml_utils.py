"""ML-specific utilities for feature analysis and preprocessing recommendations."""

import numpy as np
import pandas as pd


def calculate_vif(df, numeric_cols=None):
    """
    Calculate Variance Inflation Factor (VIF) for numeric columns.
    VIF > 5-10 indicates problematic multicollinearity.

    Uses complete, finite observations and an intercept in each regression.
    Constant responses or no residual degrees of freedom return NaN.
    Returns dict: {column: vif_value}; fewer than two columns returns {}.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if len(numeric_cols) < 2:
        return {}

    # Regress each feature on ALL other features. Centering includes an
    # intercept, so results do not depend on offsets or optional dependencies.
    values = df[numeric_cols].to_numpy(dtype=float, na_value=np.nan)
    values = values[np.isfinite(values).all(axis=1)]
    vif_data = dict.fromkeys(numeric_cols, float("nan"))
    if len(values) < 2:
        return vif_data

    # Scale before centering to avoid overflow with large finite values.
    scales = np.max(np.abs(values), axis=0)
    values = values / np.where(scales == 0, 1, scales)
    values -= values.mean(axis=0)
    norms = np.linalg.norm(values, axis=0)
    variable = norms > 0
    values /= np.where(variable, norms, 1)

    for i, col in enumerate(numeric_cols):
        if not variable[i]:
            continue  # VIF is undefined for a constant response.
        predictors = values[:, variable & (np.arange(len(numeric_cols)) != i)]
        try:
            coef, _, rank, _ = np.linalg.lstsq(predictors, values[:, i], rcond=None)
            if len(values) <= rank + 1:
                continue  # No residual degrees of freedom after the intercept.
            residual = values[:, i] - predictors @ coef
            residual_ss = float(residual @ residual)
            tolerance = (np.finfo(float).eps * max(values.shape)) ** 2
            vif_data[col] = float("inf") if residual_ss <= tolerance else max(1.0, 1 / residual_ss)
        except np.linalg.LinAlgError:
            pass
    return vif_data


def cramers_v(x, y):
    """Cramér's V from paired observations; undefined associations return NaN.

    Pair by position, including when indices contain duplicate labels.
    """
    if len(x) != len(y):
        raise ValueError("Cramér's V requires equally sized paired inputs")
    pairs = pd.DataFrame({"x": pd.Series(x).reset_index(drop=True),
                          "y": pd.Series(y).reset_index(drop=True)}).dropna()
    observed = pd.crosstab(pairs["x"], pairs["y"]).to_numpy(dtype=float)
    if min(observed.shape, default=0) < 2:
        return float("nan")
    n = observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    chi2 = np.sum((observed - expected) ** 2 / expected)
    return float(np.clip(np.sqrt(chi2 / (n * (min(observed.shape) - 1))), 0, 1))


def mutual_information(x, y, bins=10):
    """Calculate mutual information between two variables (numeric or categorical)."""
    try:
        from sklearn.metrics import mutual_info_score
        from sklearn.preprocessing import KBinsDiscretizer

        # Handle NaN
        mask = ~(pd.isna(x) | pd.isna(y))
        x_clean, y_clean = x[mask], y[mask]

        if len(x_clean) < 2:
            return 0

        # Discretize if numeric
        if np.issubdtype(x_clean.dtype, np.number):
            x_clean = pd.qcut(x_clean, q=bins, labels=False, duplicates='drop')
        if np.issubdtype(y_clean.dtype, np.number):
            y_clean = pd.qcut(y_clean, q=bins, labels=False, duplicates='drop')

        x_clean = pd.Categorical(x_clean).codes
        y_clean = pd.Categorical(y_clean).codes

        return mutual_info_score(x_clean, y_clean)
    except ImportError:
        return np.nan


def get_scaling_recommendation(col):
    """Return scaling recommendation based on data characteristics."""
    if col.dtype.kind not in ('i', 'u', 'f'):
        return "categorical"

    col_clean = col.dropna()
    if len(col_clean) == 0:
        return "unknown"

    min_val, max_val = col_clean.min(), col_clean.max()
    range_val = max_val - min_val

    # If range is very large or mean >> median, likely needs scaling
    skew = col_clean.skew()
    if range_val > 100 or (pd.notna(skew) and abs(skew) > 1):
        return "scale (StandardScaler or MinMaxScaler)"

    return "optional"


def get_transformation_suggestion(col):
    """Suggest transformations for skewed numeric columns."""
    if col.dtype.kind not in ('i', 'u', 'f'):
        return None

    col_clean = col.dropna()
    if len(col_clean) < 3:
        return None

    skewness = col_clean.skew()

    # Only suggest if right-skewed (positive) and all positive values
    if skewness > 1 and col_clean.min() > 0:
        if abs(skewness) > 2:
            return "log transform (highly skewed)"
        else:
            return "sqrt or log transform (moderately skewed)"

    return None


def cardinality_warning(col):
    """Return warning if categorical column has problematic cardinality."""
    if col.dtype.kind not in ('O',):
        return None

    n_unique = col.nunique()
    n_total = len(col)

    if n_unique > 1000:
        return "very high cardinality (>1000) — consider target encoding or dropping"
    elif n_unique > 100:
        return "high cardinality (>100) — one-hot encoding may create too many features"

    return None


def rare_category_detection(col, threshold=0.01):
    """Detect rare categories (< threshold % of data)."""
    if col.dtype.kind not in ('O',):
        return {}

    col_clean = col.dropna()
    value_counts = col_clean.value_counts(normalize=True)
    rare = value_counts[value_counts < threshold]

    return rare.to_dict() if len(rare) > 0 else {}


def assess_column_quality(col, role="unknown"):
    """
    Assess the quality of a single column and return a summary.

    Returns: (quality_status, issues_list)
    where quality_status is '✓ Good', '⚠ Check', or '✗ Issue'
    """
    issues = []
    col_clean = col.dropna()
    n_total = len(col)
    n_missing = col.isna().sum()
    missing_pct = (n_missing / n_total * 100) if n_total > 0 else 0

    # Check for high missingness
    if missing_pct > 50:
        issues.append(f"High missingness ({missing_pct:.0f}%)")
    elif missing_pct > 20:
        issues.append(f"Moderate missingness ({missing_pct:.0f}%)")
    elif missing_pct > 0:
        issues.append(f"Minor missingness ({missing_pct:.1f}%)")

    # Check for constant columns
    if col.nunique() <= 1:
        issues.append("Constant or single value")

    # Check for high cardinality categoricals
    if role == "categorical" and col.nunique() > 50:
        issues.append(f"High cardinality ({col.nunique()} unique)")

    # Check for skewness in numeric
    if col.dtype.kind in ('i', 'u', 'f') and len(col_clean) > 2:
        skewness = col_clean.skew()
        if abs(skewness) > 2:
            issues.append(f"Highly skewed ({skewness:.2f})")
        elif abs(skewness) > 1:
            issues.append(f"Moderately skewed ({skewness:.2f})")

    # Check for outliers
    if col.dtype.kind in ('i', 'u', 'f') and len(col_clean) > 3:
        Q1 = col_clean.quantile(0.25)
        Q3 = col_clean.quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            outliers = col_clean[(col_clean < Q1 - 1.5*IQR) | (col_clean > Q3 + 1.5*IQR)]
            if len(outliers) > 0:
                outlier_pct = (len(outliers) / len(col_clean) * 100)
                if outlier_pct > 10:
                    issues.append(f"Outliers detected ({outlier_pct:.1f}%)")

    # Determine status
    if len(issues) == 0:
        return ("✓ Good", [])
    elif len(issues) == 1 and missing_pct > 0 and missing_pct <= 5:
        return ("✓ Good", issues)  # Minor issue only
    elif len(issues) <= 2:
        return ("⚠ Check", issues)
    else:
        return ("✗ Issue", issues)
