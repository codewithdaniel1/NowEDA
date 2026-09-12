# NowEDA

**Automated Exploratory Data Analysis — built as a native pandas extension.**

[![PyPI version](https://img.shields.io/pypi/v/noweda?label=PyPI)](https://pypi.org/project/noweda/)

NowEDA profiles pandas DataFrames with heuristic quality scores, pattern-based PII detection, outlier analysis, correlations and readable insights. Use `df.noweda` or the equivalent `df.eda` accessor. For data that should stay on disk, explicit `eda.read(..., mode="large")` supports the same core `.eda` workflow and clearly marks sample-based findings.

---

## Why NowEDA?

Most EDA tools give you charts and tables. NowEDA gives you **answers**.

| Other EDA tools | NowEDA |
|---|---|
| "Here are histograms" | "Column 'salary' is heavily right-skewed — consider a log transform" |
| Show missing value counts | "Column 'email' is 32% missing — imputation recommended" |
| Show correlation matrix | "Very strong correlation (0.99) between 'age' and 'salary' — one may be redundant" |
| No security awareness | "Column 'email' contains 17 PII email addresses — mask before sharing" |
| Mix unrelated ML algorithms | Task-aware candidates for your selected target and problem type |
| Requires specific file format | Supports 28 file extensions, with optional readers where needed |

---

## Core Design Principles

**1. Pandas-native** — `df.noweda.*` feels like part of pandas itself. No wrappers, no new object types to learn.

**2. Zero fragile dependencies** — NowEDA does not wrap or depend on other EDA libraries (ydata-profiling, Sweetviz, D-Tale). Every analysis is implemented from scratch, so no upstream changes can break your workflow.

**3. Plugin-based** — Every analysis step is an independent, swappable plugin. Run only what you need, or add your own.

**4. Security-aware** — Built-in PII and encoding detection that no other EDA tool provides out of the box.

---

## Quick Example

```python
import noweda as eda

# Works exactly like pandas — just swap pd.read_csv for eda.read
df = eda.read("transactions.csv")

# All normal pandas operations still work
print(df.head())
print(df.describe())

# Start with one complete assessment.
df.noweda.statsall()

# Add charts when a visual answer will help.
# df.noweda.vizall()

# State the target or unsupervised objective before requesting ML guidance
df.noweda.mlall(target="fraud_flag", problem_type="classification")
```

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
| **PII** | Email address detection (extensible) |
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
