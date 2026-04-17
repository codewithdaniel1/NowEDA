"""
NowEDA file ingestion layer.

Supports all major tabular data formats. Optional formats (Parquet, Feather,
ORC, HDF5, SPSS) require additional dependencies — a clear error is raised if
the dependency is missing.
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read(file_path, **kwargs):
    """
    Load any supported file into a pandas DataFrame.

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

    # Get iterator from appropriate loader
    if ext in (".csv", ".tsv", ".tab", ".txt"):
        # Set default separator for TSV/TAB
        if ext in (".tsv", ".tab"):
            kwargs.setdefault("sep", "\t")
        iterator = pd.read_csv(file_path, chunksize=chunksize, **kwargs)
    else:  # .json or .jsonl
        # Default to newline-delimited JSON
        kwargs.setdefault("lines", True)
        iterator = pd.read_json(file_path, chunksize=chunksize, **kwargs)

    # Return either concatenated or as generator
    if concat:
        return pd.concat(list(iterator), ignore_index=True)
    else:
        return iterator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extension(file_path):
    """Return the lower-cased file extension including the dot."""
    _, ext = os.path.splitext(file_path)
    return ext.lower()


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
