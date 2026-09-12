# Installation

## Requirements

- Python 3.8 or higher
- pip

NowEDA's core dependencies are installed automatically:

| Package | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 1.3 | DataFrame engine |
| `numpy` | ≥ 1.21 | Numeric computation |
| `pyspark` | ≥ 3.4 | Spark-backed ingestion for large files |
| `duckdb` | ≥ 1.1 | Disk-backed large-mode analytics |
| `pyarrow` | ≥ 14 | Columnar file support and interchange |
| `openpyxl` | ≥ 3.1 | Excel (.xlsx) reading |
| `lxml` | ≥ 4.6 | XML and HTML parsing |

---

## Standard Install

```bash
pip install noweda
```

This gives you full support for:
CSV, TSV, XLSX/XLSM, JSON/JSONL, XML, HTML, Stata, SAS, Pickle, Parquet,
Feather, and ORC. Explicit `mode="large"` keeps CSV, TSV, TXT, and Parquet
sources on disk for the supported `.eda` workflow.

---

## Install with Optional Format Support

Some formats and analysis features require additional libraries:

```bash
pip install "noweda[excel]"  # XLS, XLSB, ODS/ODF/ODT
pip install "noweda[viz]"    # Matplotlib charts and SciPy KDE overlays
pip install "noweda[ml]"     # Statsmodels and scikit-learn diagnostics
```

Install other formats as extras:

=== "HDF5"

    ```bash
    pip install "noweda[hdf]"
    ```

    Installs `tables` (PyTables). Enables `.h5`, `.hdf`, `.hdf5`.

=== "SPSS"

    ```bash
    pip install "noweda[spss]"
    ```

    Installs `pyreadstat`. Enables `.sav`, `.zsav`.

=== "Everything"

    ```bash
    pip install "noweda[full]"
    ```

    Installs all optional dependencies at once.

---

## Install from Source (Development)

If you want to contribute or run from the latest code:

```bash
git clone https://github.com/codewithdaniel1/NowEDA.git
cd NowEDA

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

# Install in editable mode
pip install -e .

# Install dev tools
pip install pytest mkdocs mkdocs-material
```

---

## Verify Installation

```python
import noweda as eda
print(eda.__version__)   # 0.2.0

import pandas as pd
df = pd.DataFrame({"x": [1, 2, 3]})
print(df.noweda.score())    # {'data_quality': 100, 'risk': 0, 'model_readiness': 100}
```

---

## What If an Optional Dependency Is Missing?

NowEDA gives you a clear, actionable error instead of a confusing traceback:

```python
eda.read("data.parquet")
# ImportError: Reading .parquet files requires 'pyarrow'.
# Reinstall or upgrade NowEDA to include its standard pyarrow dependency.
```

---

## Upgrading

```bash
pip install --upgrade noweda
```

---

## Uninstalling

```bash
pip uninstall noweda
```
