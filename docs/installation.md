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
| `openpyxl` | ≥ 3.0 | Excel (.xlsx) reading |
| `lxml` | ≥ 4.6 | XML and HTML parsing |

---

## Standard Install

```bash
pip install noweda
```

This gives you full support for:
CSV, TSV, Excel, JSON, XML, HTML, Stata, SAS, Pickle, and Spark-backed large-file loading.

---

## Install with Optional Format Support

Some formats require additional libraries. Install them as extras:

=== "Parquet / Feather / ORC"

    ```bash
    pip install "noweda[parquet]"
    ```

    Installs `pyarrow`. Enables `.parquet`, `.feather`, `.orc`.

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
print(eda.__version__)   # 0.1.0

import pandas as pd
df = pd.DataFrame({"x": [1, 2, 3]})
print(df.noweda.score())    # {'data_quality': 100, 'risk': 0, 'model_readiness': 97}
```

---

## What If an Optional Dependency Is Missing?

NowEDA gives you a clear, actionable error instead of a confusing traceback:

```python
eda.read("data.parquet")
# ImportError: Reading .parquet files requires 'pyarrow'.
# Install it with:  pip install noweda[parquet]
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
