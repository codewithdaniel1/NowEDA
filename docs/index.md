# NowEDA

**Automated Exploratory Data Analysis — built as a native pandas extension with Spark acceleration built in.**

NowEDA is a lightweight, modular Python framework that turns any dataset into instant insight. Load any file format, call `df.noweda.*`, and get a complete EDA report — data quality scoring, PII detection, outlier analysis, correlation mapping, and human-readable insights — with zero boilerplate. Large supported files are automatically accelerated with Spark, and notebook/CLI runs show a loading indicator while work is in progress.

---

## Why NowEDA?

Most EDA tools give you charts and tables. NowEDA gives you **answers**.

| Other EDA tools | NowEDA |
|---|---|
| "Here are histograms" | "Column 'salary' is heavily right-skewed — consider a log transform" |
| Show missing value counts | "Column 'email' is 32% missing — imputation recommended" |
| Show correlation matrix | "Very strong correlation (0.99) between 'age' and 'salary' — one may be redundant" |
| No security awareness | "Column 'email' contains 17 PII email addresses — mask before sharing" |
| Requires specific file format | Works with 28 file extensions out of the box |

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

# Now layer on automated intelligence
print(df.noweda.insights())   # human-readable insight list
print(df.noweda.score())      # data_quality, risk, model_readiness
print(df.noweda.summary())    # raw results from every plugin
report = df.noweda.report()   # everything in one structured dict
```

**Example output:**

```
Insights:
 - Likely identifier column(s) detected: id. Consider excluding from modelling.
 - Column 'email' has high missing rate (32%) — imputation recommended.
 - Very strong correlation (0.99) between 'age' and 'salary'. One may be redundant.
 - PII detected in column 'email': 17 email address(es) found. Mask before sharing.
 - Data quality score is acceptable (77). Minor issues present.
 - Moderate risk level (25). Review PII and encoded columns before sharing.

Scores:
{'data_quality': 77, 'risk': 25, 'model_readiness': 53}
```

---

## Supported Formats

NowEDA reads **28 file extensions** across all major tabular data formats:

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

Spark acceleration is included in the standard install.

For additional format support:

```bash
pip install "noweda[parquet]"   # Parquet, Feather, ORC
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
