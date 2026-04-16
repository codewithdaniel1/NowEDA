import warnings

import pandas as pd

from .base import BasePlugin


# Cardinality thresholds
_ID_UNIQUENESS = 0.95   # column is likely an ID/key
_CAT_CARDINALITY = 0.05  # column is likely categorical


class SchemaPlugin(BasePlugin):
    """Infer semantic column roles: id, categorical, numeric, datetime, text."""

    name = "schema"

    def _infer_role(self, series, n_rows):
        dtype = series.dtype
        n_unique = series.nunique()
        uniqueness = n_unique / n_rows if n_rows > 0 else 0

        # Datetime detection
        if dtype.kind == "M":
            return "datetime"

        # Try to parse object columns as datetime (suppress noisy pandas warnings)
        if dtype == "object":
            sample = series.dropna().astype(str).head(20)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pd.to_datetime(sample)
                return "datetime"
            except Exception:
                pass

        # Numeric
        if dtype.kind in ("i", "u", "f"):
            # Only flag integer-typed columns with perfect uniqueness as IDs.
            # Float columns (salary, score) are almost never true keys.
            # Require uniqueness == 1.0 so numeric columns with any duplicates
            # (like age appearing twice) are NOT incorrectly flagged.
            if dtype.kind in ("i", "u") and uniqueness == 1.0 and n_unique > 5:
                return "id_candidate"
            if uniqueness <= _CAT_CARDINALITY and n_unique <= 20:
                return "categorical_numeric"
            return "numeric"

        # Object / string
        if dtype == "object":
            # Require near-perfect uniqueness for string ID detection
            if uniqueness >= 0.98 and n_unique > 5:
                return "id_candidate"
            if uniqueness <= _CAT_CARDINALITY and n_unique <= 50:
                return "categorical"
            return "text"

        return "unknown"

    def run(self, df):
        n_rows = len(df)
        result = {}

        for col in df.columns:
            result[col] = {
                "dtype": str(df[col].dtype),
                "role": self._infer_role(df[col], n_rows),
                "unique": int(df[col].nunique()),
                "uniqueness_ratio": round(df[col].nunique() / n_rows, 4) if n_rows > 0 else 0,
            }

        return result
