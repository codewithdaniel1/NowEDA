from .base import BasePlugin


class DuplicatesPlugin(BasePlugin):
    """Detect duplicate rows and constant (zero-variance) columns."""

    name = "duplicates"

    def run(self, df):
        n_rows = len(df)

        duplicate_rows = int(df.duplicated().sum())
        duplicate_pct = round(duplicate_rows / n_rows, 4) if n_rows > 0 else 0.0

        constant_columns = [
            col for col in df.columns if df[col].nunique(dropna=False) <= 1
        ]

        return {
            "duplicate_rows": duplicate_rows,
            "duplicate_rows_pct": duplicate_pct,
            "constant_columns": constant_columns,
        }
