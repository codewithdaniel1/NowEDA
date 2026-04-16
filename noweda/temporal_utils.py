"""Temporal data analysis utilities."""

import pandas as pd
import numpy as np


def detect_temporal_columns(df):
    """
    Detect datetime columns and infer their frequencies.

    Returns: dict of {column_name: (dtype, frequency, confidence)}
    """
    import warnings
    temporal = {}

    for col in df.columns:
        # Direct datetime dtype
        if df[col].dtype.kind == "M":
            freq = infer_frequency(df[col])
            temporal[col] = ("datetime64", freq, 0.99)
            continue

        # Try parsing object columns as datetime
        if df[col].dtype == "object":
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                # Check if >80% of values parsed successfully
                valid_pct = parsed.notna().sum() / len(df)
                if valid_pct > 0.8:
                    freq = infer_frequency(parsed)
                    temporal[col] = ("datetime", freq, valid_pct)
            except Exception:
                pass

    return temporal


def infer_frequency(dt_series):
    """
    Infer the frequency of a datetime series.

    Returns: str like 'daily', 'hourly', 'monthly', 'unknown'
    """
    dt_series = pd.to_datetime(dt_series, errors="coerce").dropna()

    if len(dt_series) < 2:
        return "unknown"

    # Sort and compute differences
    dt_sorted = dt_series.sort_values()
    diffs = dt_sorted.diff().dropna()

    if len(diffs) == 0:
        return "unknown"

    # Most common difference
    most_common_diff = diffs.mode()[0] if len(diffs) > 0 else None

    if most_common_diff is None:
        return "unknown"

    # Map to frequency
    if most_common_diff == pd.Timedelta(days=1):
        return "daily"
    elif most_common_diff == pd.Timedelta(hours=1):
        return "hourly"
    elif most_common_diff == pd.Timedelta(minutes=1):
        return "minutely"
    elif most_common_diff.days >= 28 and most_common_diff.days <= 31:
        return "monthly"
    elif most_common_diff.days >= 7 and most_common_diff.days <= 7:
        return "weekly"
    elif most_common_diff.days == 0:
        return "sub-hourly"
    else:
        return f"~{most_common_diff.days}d" if most_common_diff.days > 0 else "unknown"


def stationarity_test(series):
    """
    Test if a time series is stationary (Augmented Dickey-Fuller test).

    Returns: (is_stationary, p_value)
    """
    try:
        from statsmodels.tsa.stattools import adfuller

        # Remove NaN
        clean_series = series.dropna()
        if len(clean_series) < 10:
            return None, None

        result = adfuller(clean_series, autolag="AIC")
        p_value = result[1]
        is_stationary = p_value < 0.05

        return is_stationary, p_value
    except ImportError:
        return None, None
    except Exception:
        return None, None


def detect_seasonality(series, period=12):
    """
    Detect seasonality in a time series.

    Returns: (has_seasonality, strength)
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose

        clean_series = series.dropna()
        if len(clean_series) < 2 * period:
            return False, 0

        decomposition = seasonal_decompose(clean_series, model="additive", period=period)
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        # Strength of seasonality
        strength = 1 - (residual.var() / (seasonal + residual).var())
        strength = max(0, min(1, strength))  # Clamp to [0, 1]

        has_seasonality = strength > 0.1

        return has_seasonality, strength
    except Exception:
        return False, 0
