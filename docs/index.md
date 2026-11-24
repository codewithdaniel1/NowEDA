# NowEDA

**Automated Exploratory Data Analysis — built as a native pandas extension.**

[![PyPI version](https://img.shields.io/pypi/v/noweda?label=PyPI)](https://pypi.org/project/noweda/)

Use NowEDA to explore a table and choose a first model to try.

## Your first prediction question

Suppose you want to predict whether a customer will leave. The **target** is
the column that records what happened, such as `churned`. The **features** are
the details you would know beforehand, such as how often the customer logged in.

```python
import noweda as eda

df = eda.read("customers.csv")          # Use your file and target column.
df.eda.statsall()                       # Check missing values and duplicates.
charts = df.eda.vizall(target="churned")
df.eda.mlall(target="churned")           # Pick models to try first.
```

Install `noweda[viz]` for the charts. For a yes/no outcome, look at where a
straight or curved boundary makes mistakes. For a numeric outcome, look at
where a fitted line consistently predicts too high or too low. Each view gives
you a question to test with your own data. [How to read the charts](visual-diagnostics.md)

---

## Supported Formats

NowEDA supports **28 file extensions** across major tabular data formats. Some
formats require an optional dependency extra; see [Installation](installation.md).

| Category | Extensions |
|---|---|
| Delimited text | `.csv` `.tsv` `.tab` `.txt` |
| Spreadsheets | `.xlsx` `.xls` `.xlsm` `.xlsb` `.ods` `.odf` `.odt` |
| JSON | `.json` `.jsonl` |
| XML | `.xml` |
| HTML | `.html` `.htm` |
| Columnar (pyarrow) | `.parquet` `.feather` `.orc` |
| HDF5 (tables) | `.h5` `.hdf` `.hdf5` |
| Statistical software | `.dta` `.sas7bdat` `.xpt` `.sav` `.zsav` |
| Python pickle | `.pkl` `.pickle` |

---

## Built-in Plugins

| Plugin | What it analyzes |
|---|---|
| **Schema** | Column roles — id, categorical, numeric, datetime, text |
| **Stats** | Mean, median, std, skewness, min/max, top values |
| **Missing** | Per-column missing rate |
| **Duplicates** | Duplicate rows, constant (zero-variance) columns |
| **Correlation** | Pearson correlation matrix for numeric columns |
| **Outliers** | IQR-based outlier count per numeric column |
| **PII** | Email, phone, SSN and payment-card signals |
| **Encoding** | Base64 and obfuscation signal detection |

---

## Installation

```bash
pip install noweda
```

PySpark is included in the standard install. Charts require `pip install "noweda[viz]"`; additional spreadsheet formats require `pip install "noweda[excel]"`.

For additional format support:

```bash
# Parquet, Feather, and ORC are included with pip install noweda
pip install "noweda[hdf]"       # HDF5
pip install "noweda[spss]"      # SPSS
pip install "noweda[full]"      # Everything
```

→ [Full installation guide](installation.md)

---

## CLI

```bash
noweda data.csv
noweda data.csv --html report.html --json report.json
```

→ [CLI reference](cli.md)
