# Reading Data

`noweda.read()` is your single entry point for loading any tabular dataset. It auto-detects the file format from the extension, routes to the correct pandas reader, and returns a standard DataFrame with the `df.noweda` accessor ready to use.

---

## Basic Usage

```python
import noweda

df = noweda.read("path/to/file.csv")
```

All `**kwargs` are forwarded directly to the underlying pandas reader, so every option you know from `pd.read_csv`, `pd.read_excel`, etc. works exactly the same way.

---

## Supported Formats

### Delimited Text

=== "CSV (.csv)"

    ```python
    # Basic
    df = noweda.read("data.csv")

    # With options
    df = noweda.read("data.csv", sep=",", encoding="utf-8", nrows=1000)
    df = noweda.read("data.csv", skiprows=2, header=0)
    df = noweda.read("data.csv", usecols=["id", "name", "amount"])
    df = noweda.read("data.csv", dtype={"id": str, "amount": float})
    df = noweda.read("data.csv", parse_dates=["created_at"])
    df = noweda.read("data.csv", na_values=["N/A", "NULL", "-"])
    ```

    File extensions recognized: `.csv`, `.CSV` (case-insensitive)

=== "TSV (.tsv / .tab)"

    ```python
    # Auto-detected as tab-separated
    df = noweda.read("data.tsv")
    df = noweda.read("data.tab")

    # Override separator for non-standard delimiters
    df = noweda.read("data.csv", sep="|")    # pipe-delimited
    df = noweda.read("data.csv", sep=";")    # semicolon-delimited
    ```

    File extensions recognized: `.tsv`, `.tab`

=== "Plain Text (.txt)"

    ```python
    # Treated as CSV by default; pass sep= for other delimiters
    df = noweda.read("data.txt")
    df = noweda.read("data.txt", sep="\t")
    ```

---

### Spreadsheets

=== "Excel (.xlsx)"

    ```python
    # Basic — reads the first sheet
    df = noweda.read("data.xlsx")

    # Specific sheet
    df = noweda.read("data.xlsx", sheet_name="Sales Q1")
    df = noweda.read("data.xlsx", sheet_name=0)   # by index

    # Skip header rows
    df = noweda.read("data.xlsx", skiprows=3, header=0)

    # Select specific columns
    df = noweda.read("data.xlsx", usecols="A:E")
    df = noweda.read("data.xlsx", usecols=[0, 1, 4])
    ```

    File extensions recognized: `.xlsx`, `.xlsm`, `.xlsb`

=== "Legacy Excel (.xls)"

    ```python
    df = noweda.read("data.xls")
    df = noweda.read("data.xls", sheet_name="Sheet1")
    ```

    File extensions recognized: `.xls`

=== "OpenDocument (.ods)"

    ```python
    df = noweda.read("data.ods")
    df = noweda.read("data.odf")
    df = noweda.read("data.odt")
    ```

    Requires `odfpy`: `pip install odfpy`

---

### JSON

=== ".json"

    ```python
    # Records orient (most common)
    # [{"col1": val1, "col2": val2}, ...]
    df = noweda.read("data.json")

    # Other orients
    df = noweda.read("data.json", orient="columns")
    df = noweda.read("data.json", orient="index")
    df = noweda.read("data.json", orient="split")

    # Nested JSON with normalization
    df = noweda.read("data.json", orient="records")
    ```

=== ".jsonl (JSON Lines)"

    Each line is one JSON object — common for streaming / log data:

    ```python
    df = noweda.read("events.jsonl", lines=True)
    ```

---

### XML

```python
df = noweda.read("data.xml")

# Specify XPath to the row elements
df = noweda.read("data.xml", xpath=".//record")

# Specify namespace
df = noweda.read("data.xml", namespaces={"ns": "http://example.com/ns"})
```

!!! note
    `lxml` is required and installed automatically with NowEDA.

---

### HTML

```python
# Reads the first `<table>` found in the file
df = noweda.read("data.html")
df = noweda.read("data.htm")

# Select a specific table by index (0-based)
df = noweda.read("data.html", match="Revenue")   # match by text
```

!!! tip
    HTML tables are very common on web pages. Export any web table to `.html` and NowEDA will parse it.

---

### Parquet / Feather / ORC

!!! warning "Requires pyarrow"
    ```bash
    pip install "noweda[parquet]"
    ```

=== "Parquet"

    ```python
    df = noweda.read("data.parquet")

    # Specific columns (efficient — only reads those columns from disk)
    df = noweda.read("data.parquet", columns=["id", "amount", "date"])

    # Specific row groups
    df = noweda.read("data.parquet", filters=[("amount", ">", 1000)])
    ```

    Parquet is the industry standard for large analytical datasets.
    It stores data column-by-column with compression — much faster than CSV for big files.

=== "Feather"

    ```python
    df = noweda.read("data.feather")
    df = noweda.read("data.feather", columns=["col1", "col2"])
    ```

    Feather is optimized for fast read/write within Python/R workflows.

=== "ORC"

    ```python
    df = noweda.read("data.orc")
    df = noweda.read("data.orc", columns=["id", "value"])
    ```

    ORC is common in Hadoop and Spark pipelines.

---

### HDF5

!!! warning "Requires tables (PyTables)"
    ```bash
    pip install "noweda[hdf]"
    ```

```python
df = noweda.read("data.h5")
df = noweda.read("data.hdf5")
df = noweda.read("data.hdf")

# Specify the key (dataset name inside the HDF5 file)
df = noweda.read("data.h5", key="/transactions")
df = noweda.read("data.h5", key="df")
```

HDF5 stores multiple datasets in a single file under named keys. Use `h5py` or `pandas.HDFStore` to inspect what keys a file contains.

---

### Statistical Software

=== "Stata (.dta)"

    ```python
    df = noweda.read("data.dta")

    # Convert categoricals (Stata uses numeric codes with labels)
    df = noweda.read("data.dta", convert_categoricals=True)
    ```

    No extra dependencies needed.

=== "SAS (.sas7bdat / .xpt)"

    ```python
    df = noweda.read("data.sas7bdat")
    df = noweda.read("data.xpt")      # SAS transport format

    # Chunk large SAS files
    df = noweda.read("data.sas7bdat", chunksize=10000)
    ```

    No extra dependencies needed.

=== "SPSS (.sav)"

    !!! warning "Requires pyreadstat"
        ```bash
        pip install "noweda[spss]"
        ```

    ```python
    df = noweda.read("data.sav")
    df = noweda.read("data.zsav")    # compressed SPSS

    # Apply value labels
    df = noweda.read("data.sav", apply_value_labels=True)
    ```

---

### Python Pickle

```python
df = noweda.read("data.pkl")
df = noweda.read("data.pickle")
```

!!! warning "Security note"
    Only load pickle files from sources you trust. Pickle files can execute arbitrary code on load.

---

## Error Handling

### Unsupported format

```python
noweda.read("data.avro")
# ValueError: Unsupported file extension: '.avro'
# Supported extensions: .csv, .feather, .h5, .hdf, ...
```

### File not found

```python
noweda.read("missing.csv")
# FileNotFoundError: File not found: missing.csv
```

### Missing optional dependency

```python
noweda.read("data.parquet")
# ImportError: Reading .parquet files requires 'pyarrow'.
# Install it with:  pip install noweda[parquet]
```

---

## Complete Format Reference

| Extension | Format | Extra Dep |
|---|---|---|
| `.csv` | Comma-separated values | — |
| `.tsv` `.tab` | Tab-separated values | — |
| `.txt` | Plain text (assumed CSV) | — |
| `.xlsx` `.xlsm` `.xlsb` | Excel (modern) | — |
| `.xls` | Excel (legacy) | — |
| `.ods` `.odf` `.odt` | OpenDocument spreadsheet | `odfpy` |
| `.json` `.jsonl` | JSON / JSON Lines | — |
| `.xml` | XML | `lxml` (auto-installed) |
| `.html` `.htm` | HTML table | `lxml` (auto-installed) |
| `.parquet` | Apache Parquet | `pyarrow` |
| `.feather` | Apache Feather | `pyarrow` |
| `.orc` | Apache ORC | `pyarrow` |
| `.h5` `.hdf` `.hdf5` | HDF5 | `tables` |
| `.dta` | Stata | — |
| `.sas7bdat` `.xpt` | SAS | — |
| `.sav` `.zsav` | SPSS | `pyreadstat` |
| `.pkl` `.pickle` | Python pickle | — |
