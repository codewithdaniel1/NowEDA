"""ML-specific utilities for feature analysis and preprocessing recommendations."""

import numpy as np
import pandas as pd


def calculate_vif(df, numeric_cols=None):
    """
    Calculate Variance Inflation Factor (VIF) for numeric columns.
    VIF > 5-10 indicates problematic multicollinearity.

    Returns dict: {column: vif_value}
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if len(numeric_cols) < 2:
        return {}

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif_data = {}
        # Drop NaN before calculation
        df_clean = df[numeric_cols].dropna()
        if len(df_clean) == 0:
            return {}
        for i, col in enumerate(numeric_cols):
            try:
                vif = variance_inflation_factor(df_clean.values, i)
                vif_data[col] = vif if vif < 1e6 else float('inf')
            except Exception:
                vif_data[col] = np.nan
        return vif_data
    except ImportError:
        # Fallback: calculate VIF manually using correlation
        vif_data = {}
        corr_matrix = df[numeric_cols].corr().abs()
        for col in numeric_cols:
            # Simple approximation: 1 / (1 - max_correlation^2)
            other_corr = corr_matrix[col].drop(col)
            max_corr = other_corr.max() if len(other_corr) > 0 else 0
            if max_corr < 1.0:
                vif = 1 / (1 - max_corr**2) if max_corr < 0.99 else np.inf
            else:
                vif = np.inf
            vif_data[col] = vif
        return vif_data


def cramers_v(x, y):
    """Calculate Cramér's V statistic for categorical-categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ((confusion_matrix / confusion_matrix.sum().sum()) ** 2 /
            (confusion_matrix.sum(axis=0) / confusion_matrix.sum().sum() *
             confusion_matrix.sum(axis=1).values[:, None] / confusion_matrix.sum().sum())).sum().sum() * len(x)
    chi2 = np.minimum(chi2, len(x) * (min(confusion_matrix.shape) - 1))
    return np.sqrt(chi2 / (len(x) * (min(confusion_matrix.shape) - 1))) if min(confusion_matrix.shape) > 1 else 0


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
    if range_val > 100 or abs(col_clean.skew()) > 1:
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
