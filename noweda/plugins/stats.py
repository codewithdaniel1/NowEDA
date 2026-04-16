from .base import BasePlugin


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
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "q25": float(series.quantile(0.25)),
                    "q75": float(series.quantile(0.75)),
                    "skewness": float(series.skew()),
                })
            elif series.dtype == "object" or str(series.dtype) == "category":
                top = series.value_counts()
                if not top.empty:
                    col_stats["top_value"] = str(top.index[0])
                    col_stats["top_freq"] = int(top.iloc[0])

            result[col] = col_stats

        return result
