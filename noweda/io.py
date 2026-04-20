"""
NowEDA file ingestion layer.

Supports all major tabular data formats. Optional formats (Parquet, Feather,
ORC, HDF5, SPSS) require additional dependencies — a clear error is raised if
the dependency is missing. For large Spark-friendly inputs, Spark is used
automatically while still returning a pandas DataFrame.
"""

import os
import pandas as pd
from noweda.ui import loading


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read(file_path, **kwargs):
    """
    Load any supported file into a pandas DataFrame.
    Large Spark-friendly files are routed through Spark automatically when it
    helps performance, but the returned object is always a pandas DataFrame.

    Supported formats
    -----------------
    No extra deps required:
        .csv  .tsv  .tab  .txt       — delimited text
        .xlsx .xls  .xlsm .xlsb
        .ods  .odf  .odt             — spreadsheets
        .json                        — JSON
        .xml                         — XML
        .html .htm                   — HTML table
        .dta                         — Stata
        .sas7bdat .xpt               — SAS
        .pkl .pickle                 — Python pickle

    Requires `pip install noweda[parquet]`  (pyarrow):
        .parquet  .feather  .orc

    Requires `pip install noweda[hdf]`  (tables):
        .h5  .hdf  .hdf5

    Requires `pip install noweda[spss]`  (pyreadstat):
        .sav  .zsav

    Parameters
    ----------
    file_path : str
        Path to the data file.
    **kwargs
        Forwarded directly to the underlying pandas reader
        (e.g. sheet_name='Sheet1' for Excel, sep='|' for custom CSV).

    Returns
    -------
    pandas.DataFrame
    """
    ext = _extension(file_path)
    loader = _LOADERS.get(ext)

    if loader is None:
        supported = ", ".join(sorted(_LOADERS.keys()))
        raise ValueError(
            f"Unsupported file extension: '{ext}'\n"
            f"Supported extensions: {supported}"
        )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with loading(f"NowEDA · Reading {os.path.basename(file_path)}"):
        if _should_use_spark(file_path, ext):
            try:
                return _load_with_spark(file_path, ext, **dict(kwargs))
            except Exception:
                # Fall back to the pandas reader if Spark is unavailable or
                # the Spark path cannot handle the provided options.
                pass

        return loader(file_path, **kwargs)


def read_chunked(file_path, chunksize=10_000, concat=True, **kwargs):
    """
    Read a large CSV or JSON file in chunks to avoid loading it all into memory.

    This is useful for files larger than available RAM. Pandas natively supports
    chunked reading via the `chunksize` parameter for CSV and JSON files.

    Supported formats for chunked reading
    ----------
    .csv  .tsv  .tab  .txt  — delimited text (any delimiter)
    .json  .jsonl           — JSON and newline-delimited JSON

    Other formats (Excel, Parquet, XML, etc.) do not support chunked iteration
    and must be read with the standard `read()` function.

    Parameters
    ----------
    file_path : str
        Path to a CSV, TSV, or JSON file.
    chunksize : int, default 10_000
        Number of rows to read per chunk.
    concat : bool, default True
        If True, concatenate all chunks into a single DataFrame and return it.
        If False, return a generator yielding chunk DataFrames one at a time
        (memory-efficient for large files).
    **kwargs
        Forwarded directly to the underlying pandas reader
        (e.g. sep='|' for custom CSV, na_values=['NA'], etc.).

    Returns
    -------
    pandas.DataFrame (if concat=True) or generator of DataFrames (if concat=False)
        If concat=True: a single DataFrame with all rows concatenated.
        If concat=False: a generator that yields one chunk at a time.

    Raises
    ------
    ValueError
        If the file format does not support chunked reading.
    FileNotFoundError
        If the file does not exist.

    Example
    -------
    >>> import noweda as eda
    >>> # Read a large CSV in chunks and concatenate into one DataFrame
    >>> df = eda.read_chunked('huge_file.csv', chunksize=50_000)
    >>> len(df)  # Total rows
    5000000

    >>> # Process chunks one at a time without loading all into memory
    >>> for chunk in eda.read_chunked('huge_file.csv', chunksize=50_000, concat=False):
    ...     print(chunk.shape)  # Process each chunk
    ...     # (50000, 10)
    ...     # (50000, 10)
    ...     # ...
    """
    ext = _extension(file_path)

    if ext not in _CHUNKED_FORMATS:
        supported = ", ".join(sorted(_CHUNKED_FORMATS))
        raise ValueError(
            f"Chunked reading is only supported for: {supported}\n"
            f"Got: '{ext}'. Use eda.read() for other formats (Excel, XML, etc.)."
        )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    use_spark = _should_use_spark(file_path, ext)

    if use_spark:
        try:
            return _load_chunked_with_spark(
                file_path,
                ext,
                chunksize=chunksize,
                concat=concat,
                **dict(kwargs),
            )
        except Exception:
            # Fall back to pandas chunking if Spark is unavailable or the
            # Spark path cannot handle the provided input/options.
            pass

    def _pandas_iterator():
        if ext in (".csv", ".tsv", ".tab", ".txt"):
            if ext in (".tsv", ".tab"):
                kwargs.setdefault("sep", "\t")
            return pd.read_csv(file_path, chunksize=chunksize, **kwargs)
        kwargs.setdefault("lines", True)
        return pd.read_json(file_path, chunksize=chunksize, **kwargs)

    iterator = _pandas_iterator()

    if concat:
        with loading(f"NowEDA · Reading {os.path.basename(file_path)} in chunks"):
            return pd.concat(list(iterator), ignore_index=True)

    def _chunk_generator():
        with loading(f"NowEDA · Reading {os.path.basename(file_path)} in chunks"):
            for chunk in iterator:
                yield chunk

    return _chunk_generator()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extension(file_path):
    """Return the lower-cased file extension including the dot."""
    _, ext = os.path.splitext(file_path)
    return ext.lower()


def _should_use_spark(file_path, ext):
    """Return True when Spark should be tried first for this file."""
    if ext not in _SPARK_FORMATS:
        return False

    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    except OSError:
        return False

    return file_size_mb >= _SPARK_AUTO_THRESHOLD_MB


def _load_with_spark(path, ext, **kw):
    """Load Spark-friendly formats via PySpark and convert to pandas."""
    spark_df = _spark_dataframe(path, ext, **kw)
    return spark_df.toPandas()


def _load_chunked_with_spark(path, ext, chunksize, concat, **kw):
    """Load a chunked Spark-friendly file via Spark and batch into pandas chunks."""
    spark_df = _spark_dataframe(path, ext, **kw)
    iterator = _spark_chunk_iterator(spark_df, chunksize)
    message = f"NowEDA · Reading {os.path.basename(path)} in chunks"

    if concat:
        with loading(message):
            return pd.concat(list(iterator), ignore_index=True)

    def _chunk_generator():
        with loading(message):
            for chunk in iterator:
                yield chunk

    return _chunk_generator()


def _spark_dataframe(path, ext, **kw):
    """Return a Spark DataFrame for Spark-friendly file formats."""
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        raise ImportError(
            "Spark support requires 'pyspark'. Reinstall NowEDA with its "
            "standard dependencies or install pyspark in this environment."
        ) from None

    spark = SparkSession.builder.appName("NowEDA").getOrCreate()

    if ext in (".csv", ".tsv", ".tab", ".txt"):
        if ext in (".tsv", ".tab"):
            kw.setdefault("sep", "\t")
        kw.setdefault("header", True)
        kw.setdefault("inferSchema", True)
        return spark.read.options(**kw).csv(path)

    if ext in (".json", ".jsonl"):
        kw.pop("lines", None)
        kw.setdefault("multiLine", False)
        return spark.read.options(**kw).json(path)

    if ext == ".parquet":
        return spark.read.options(**kw).parquet(path)

    if ext == ".orc":
        return spark.read.options(**kw).orc(path)

    raise ValueError(
        f"Spark reader does not support '{ext}'. "
        "Use the pandas reader for this format."
    )


def _spark_chunk_iterator(spark_df, chunksize):
    """Yield pandas DataFrames from a Spark DataFrame in fixed-size batches."""
    columns = list(spark_df.columns)
    batch = []

    for row in spark_df.toLocalIterator():
        batch.append(row.asDict(recursive=True))
        if len(batch) >= chunksize:
            yield pd.DataFrame.from_records(batch, columns=columns)
            batch = []

    if batch:
        yield pd.DataFrame.from_records(batch, columns=columns)


# -- loaders -----------------------------------------------------------------

def _load_csv(path, **kw):
    return pd.read_csv(path, **kw)


def _load_tsv(path, **kw):
    kw.setdefault("sep", "\t")
    return pd.read_csv(path, **kw)


def _load_excel(path, **kw):
    return pd.read_excel(path, **kw)


def _load_json(path, **kw):
    return pd.read_json(path, **kw)


def _load_xml(path, **kw):
    return pd.read_xml(path, **kw)


def _load_html(path, **kw):
    tables = pd.read_html(path, **kw)
    if not tables:
        raise ValueError(f"No tables found in HTML file: {path}")
    return tables[0]


def _load_parquet(path, **kw):
    _require("pyarrow", "parquet", ".parquet")
    return pd.read_parquet(path, **kw)


def _load_feather(path, **kw):
    _require("pyarrow", "parquet", ".feather")
    return pd.read_feather(path, **kw)


def _load_orc(path, **kw):
    _require("pyarrow", "parquet", ".orc")
    return pd.read_orc(path, **kw)


def _load_hdf(path, **kw):
    _require("tables", "hdf", ".h5/.hdf5")
    return pd.read_hdf(path, **kw)


def _load_stata(path, **kw):
    return pd.read_stata(path, **kw)


def _load_sas(path, **kw):
    return pd.read_sas(path, **kw)


def _load_spss(path, **kw):
    _require("pyreadstat", "spss", ".sav")
    return pd.read_spss(path, **kw)


def _load_pickle(path, **kw):
    return pd.read_pickle(path, **kw)


def _require(package, extra, fmt):
    try:
        __import__(package)
    except ImportError:
        raise ImportError(
            f"Reading {fmt} files requires '{package}'.\n"
            f"Install it with:  pip install noweda[{extra}]"
        ) from None


# ---------------------------------------------------------------------------
# Formats that support chunked reading via pandas
# ---------------------------------------------------------------------------

_CHUNKED_FORMATS = {".csv", ".tsv", ".tab", ".txt", ".json", ".jsonl"}

# ---------------------------------------------------------------------------
# Formats that can be accelerated by Spark
# ---------------------------------------------------------------------------

_SPARK_FORMATS = {".csv", ".tsv", ".tab", ".txt", ".parquet", ".orc"}

# Default file-size threshold for Spark auto-routing.
_SPARK_AUTO_THRESHOLD_MB = 128

# ---------------------------------------------------------------------------
# Extension → loader dispatch table
# ---------------------------------------------------------------------------

_LOADERS = {
    # Delimited text
    ".csv":      _load_csv,
    ".tsv":      _load_tsv,
    ".tab":      _load_tsv,
    ".txt":      _load_csv,   # assume CSV; caller can pass sep='\t' if needed
    # Spreadsheets
    ".xlsx":     _load_excel,
    ".xls":      _load_excel,
    ".xlsm":     _load_excel,
    ".xlsb":     _load_excel,
    ".ods":      _load_excel,
    ".odf":      _load_excel,
    ".odt":      _load_excel,
    # JSON
    ".json":     _load_json,
    ".jsonl":    _load_json,  # JSON Lines — pandas read_json handles it
    # XML
    ".xml":      _load_xml,
    # HTML
    ".html":     _load_html,
    ".htm":      _load_html,
    # Binary columnar (need pyarrow)
    ".parquet":  _load_parquet,
    ".feather":  _load_feather,
    ".orc":      _load_orc,
    # HDF5 (need tables)
    ".h5":       _load_hdf,
    ".hdf":      _load_hdf,
    ".hdf5":     _load_hdf,
    # Statistical software
    ".dta":      _load_stata,      # Stata
    ".sas7bdat": _load_sas,        # SAS
    ".xpt":      _load_sas,        # SAS transport
    ".sav":      _load_spss,       # SPSS
    ".zsav":     _load_spss,       # SPSS compressed
    # Python pickle
    ".pkl":      _load_pickle,
    ".pickle":   _load_pickle,
}
