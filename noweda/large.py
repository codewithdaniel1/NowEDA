"""Out-of-core dataset support for NowEDA.

LargeDataset deliberately exposes the familiar ``data.eda`` workflow while
keeping the source data on disk.  DuckDB supplies exact file-level metadata;
the existing pandas EDA engine runs against a bounded, deterministic sample.
Every method that uses that sample announces it before returning a result.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd


_LARGE_FORMATS = {".csv", ".tsv", ".tab", ".txt", ".parquet"}


class LargeDataset:
    """A disk-backed NowEDA dataset returned by ``eda.read(mode='large')``.

    The source is never loaded in full.  Exact metadata such as the row count
    and columns comes from DuckDB; exploratory results use a bounded sample.
    """

    def __init__(self, file_path: str, *, chunksize: int = 100_000, **read_kwargs: Any):
        if isinstance(chunksize, bool) or not isinstance(chunksize, int) or chunksize <= 0:
            raise ValueError("chunksize must be a positive integer")
        self.path = str(Path(file_path))
        self.chunksize = chunksize
        self.read_kwargs = read_kwargs
        self._sample_dfs = {}
        self._row_count: Optional[int] = None
        self._columns: Optional[list[str]] = None
        self._accessor: Optional[LargeEDAAccessor] = None

        self._duckdb = self._require_duckdb()
        self._connection = self._duckdb.connect(":memory:")
        self._create_source_view()

    @staticmethod
    def _require_duckdb():
        try:
            import duckdb
        except ImportError:  # pragma: no cover - package dependency guard
            raise ImportError(
                "Large mode requires DuckDB. Reinstall NowEDA to include its standard dependencies."
            ) from None
        return duckdb

    def _create_source_view(self) -> None:
        ext = Path(self.path).suffix.lower()
        if ext not in _LARGE_FORMATS:
            supported = ", ".join(sorted(_LARGE_FORMATS))
            raise ValueError(
                f"Large mode supports: {supported}. Got: '{ext}'."
            )
        if self.read_kwargs:
            unsupported = ", ".join(sorted(self.read_kwargs))
            raise ValueError(
                "Reader options are not yet supported in large mode "
                f"({unsupported}). Configure the file delimiter or use mode='small'."
            )
        quoted_path = self.path.replace("'", "''")
        if ext == ".parquet":
            source = f"read_parquet('{quoted_path}')"
        else:
            options = "delim='\\t'" if ext in {".tsv", ".tab"} else ""
            source = f"read_csv_auto('{quoted_path}'{', ' + options if options else ''})"
        self._connection.execute(f"CREATE VIEW noweda_source AS SELECT * FROM {source}")

    @property
    def columns(self) -> list[str]:
        if self._columns is None:
            self._columns = list(self._connection.execute("DESCRIBE noweda_source").fetchdf()["column_name"])
        return self._columns

    @property
    def row_count(self) -> int:
        if self._row_count is None:
            self._row_count = int(self._connection.execute("SELECT count(*) FROM noweda_source").fetchone()[0])
        return self._row_count

    @property
    def shape(self) -> tuple[int, int]:
        return (self.row_count, len(self.columns))

    def __len__(self) -> int:
        return self.row_count

    def head(self, n: int = 5) -> pd.DataFrame:
        """Return the first *n* rows as a pandas DataFrame without loading the source.

        This mirrors ``pandas.DataFrame.head()`` and is exact: it reads only
        the requested leading rows from the disk-backed source.
        """
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError("n must be an integer")
        if n < 0:
            n = max(self.row_count + n, 0)
        return self._connection.execute(
            "SELECT * FROM noweda_source LIMIT ?", [n]
        ).fetchdf()

    def _sample(self, rows: Optional[int] = None) -> pd.DataFrame:
        rows = min(rows if rows is not None else self.chunksize, self.row_count)
        if rows not in self._sample_dfs:
            # A stable bounded sample keeps all exploratory operations safe for
            # files with millions of rows.  The notice makes this explicit.
            self._sample_dfs[rows] = self._connection.execute(
                "SELECT * FROM noweda_source LIMIT ?", [rows]
            ).fetchdf()
        return self._sample_dfs[rows]

    @property
    def eda(self) -> "LargeEDAAccessor":
        if self._accessor is None:
            self._accessor = LargeEDAAccessor(self)
        return self._accessor

    noweda = eda


class LargeEDAAccessor:
    """Large-mode facade for the supported NowEDA methods."""

    _SUPPORTED = {"statsall", "report", "vizall", "mlall", "profile_column", "compare"}

    def __init__(self, dataset: LargeDataset):
        self._dataset = dataset

    def _announce_sample(self, method: str, *, rows: Optional[int] = None) -> None:
        total = self._dataset.row_count
        sample_rows = min(rows if rows is not None else self._dataset.chunksize, total)
        if sample_rows == total:
            print(
                f"\nNowEDA large mode · {method}: running on all {total:,} rows; "
                "results are not sample-based.\n"
            )
            return
        print(
            f"\nNowEDA large mode · {method}: sample-based results use "
            f"{sample_rows:,} of {total:,} rows. Estimates may differ from the full dataset.\n"
        )

    def _pandas_accessor(self, sample=None):
        # Importing here avoids an accessor import cycle during package setup.
        return self._dataset._sample(sample).eda

    def statsall(self, sample=10_000):
        """Print exploratory analysis from a bounded sample; row count is exact."""
        self._announce_sample("statsall()", rows=sample)
        print(f"Exact dataset size: {self._dataset.row_count:,} rows × {len(self._dataset.columns)} columns")
        return self._pandas_accessor(sample).statsall(sample=sample)

    def report(self):
        """Return a sampled report with large-mode metadata describing its scope."""
        self._announce_sample("report()")
        report = self._pandas_accessor().report()
        report = dict(report)
        sampled = self._dataset.row_count > self._dataset.chunksize
        report["large_mode"] = {
            "source": self._dataset.path,
            "rows": self._dataset.row_count,
            "columns": len(self._dataset.columns),
            "sample_rows": min(self._dataset.chunksize, self._dataset.row_count),
            "result_scope": (
                "sample-based except dataset dimensions" if sampled else "full dataset"
            ),
        }
        return report

    def vizall(self, sample=10_000):
        """Render charts from the large-mode sample."""
        self._announce_sample("vizall()", rows=sample)
        # Do not let the pandas accessor take a second, hidden sample of the
        # already bounded large-mode sample. An explicit ``sample=`` remains
        # available for callers who want smaller charts.
        return self._pandas_accessor(sample).vizall(sample=sample)

    def mlall(self, target=None, problem_type=None, features=None, plan=False, sample=10_000):
        """Create ML guidance from the sample and identify it as estimated."""
        self._announce_sample("mlall()", rows=sample)
        return self._pandas_accessor(sample).mlall(
            target=target, problem_type=problem_type, features=features, plan=plan,
            sample=sample,
        )

    def profile_column(self, column_name):
        """Profile a column from the sample."""
        self._announce_sample("profile_column()")
        return self._pandas_accessor().profile_column(column_name)

    def compare(self, other):
        """Compare two sampled large datasets, or a large dataset and pandas frame."""
        self._announce_sample("compare()")
        if isinstance(other, LargeDataset):
            other = other._sample()
            print(
                "Comparison note: the second large dataset is also sampled; "
                "schema and displayed dimensions use its sample."
            )
        return self._pandas_accessor().compare(other)

    def __getattr__(self, name: str):
        if name not in self._SUPPORTED:
            raise AttributeError(
                f"'{name}' is not available in large mode yet. Supported methods: "
                f"{', '.join(sorted(self._SUPPORTED))}."
            )
        raise AttributeError(name)
