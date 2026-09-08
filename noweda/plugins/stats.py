from noweda.dtypes import is_textual
import pandas as pd
from .base import BasePlugin


def _stat_float(value):
    return float("nan") if pd.isna(value) else float(value)


class StatsPlugin(BasePlugin):
    """Compute descriptive statistics for numeric and categorical columns."""

    name = "stats"

    def run(self, df):
        result = {}

        for col in df.columns:
            series = df[col]
            col_stats = {
                "dtype": str(series.dtype),
                "count": int(series.count()),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique()),
            }

            if series.dtype.kind in ("i", "u", "f"):
                col_stats.update({
                    "mean": _stat_float(series.mean()),
                    "median": _stat_float(series.median()) if col_stats["count"] else float("nan"),
                    "std": _stat_float(series.std()),
                    "min": _stat_float(series.min()),
                    "max": _stat_float(series.max()),
                    "q25": _stat_float(series.quantile(0.25)),
                    "q75": _stat_float(series.quantile(0.75)),
                    "skewness": _stat_float(series.skew()),
                    "kurtosis": _stat_float(series.kurt()),
                })
            elif is_textual(series):
                top = series.value_counts()
                if not top.empty:
                    col_stats["top_value"] = str(top.index[0])
                    col_stats["top_freq"] = int(top.iloc[0])

            result[col] = col_stats

        return result
