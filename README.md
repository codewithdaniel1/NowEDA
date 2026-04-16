# NowEDA

**Automated Exploratory Data Analysis — built as a native pandas extension.**

NowEDA is a lightweight, modular Python framework that turns any dataset into instant insight. Load any file, call `df.noweda.*`, and get a full EDA report — including data quality scoring, PII detection, outlier analysis, and human-readable insights — with zero boilerplate.

---

## 5 Powerful Methods — Everything You Need for ML-Ready EDA

NowEDA provides 5 powerful methods that do **everything** an ML engineer needs before modeling:

`import noweda as eda`

### 1. `df.eda.statsall()` — Complete Statistical & ML Analysis Report

Prints a rich, colour-coded report with comprehensive data profiling:

**Data Profile:**
- **Scores** — data quality (0–100), risk level, model readiness (0–100)
- **Column analysis** — dtypes, inferred roles with confidence scores, diversity metrics
- **Column quality summary** — ✓ Good / ⚠ Check / ✗ Issue flags for each column
- **Numeric stats** — mean, std, quartiles, skewness, outlier counts
- **Categorical stats** — cardinality, balance (entropy-based diversity), top values with %
- **Temporal analysis** — auto-detects datetime columns, infers frequency (daily/weekly/monthly), tests stationarity, detects seasonality
- **Insights** — human-readable findings about data quality, patterns, issues

**Preprocessing Guidance:**
- ⚠️ **Multicollinearity detection** (VIF > 5) with code snippets
- 🔧 **Scaling recommendations** with `StandardScaler` and `MinMaxScaler` code
- 📊 **Transformation suggestions** (log, sqrt for skewed data) with examples
- 🚨 **Cardinality warnings** (high-cardinality categoricals)
- 🏷️ **Rare category detection** (<1% frequency)
- ❌ **Missing data strategy** (impute vs drop) with code examples

### 2. `df.eda.mlall()` — ML Algorithm Recommendations & Preprocessing Pipeline

Provides expert guidance on which algorithms to try first, with data-specific reasoning:

**🤖 Supervised Learning (Classification & Regression)**
- **Supervised algorithms** — Linear/Logistic Regression, Random Forest, Gradient Boosting (XGBoost), SVM, KNN, Neural Networks, Naive Bayes
- **Rating system** — ⭐ (1–5 stars) based on your data characteristics
- **Reasoning** — ✓ Why it fits + ✗ When to avoid + ⚠ Before fitting warnings
- **Class imbalance detection** — ⚠️ Alerts if target variable is imbalanced with actionable solutions (SMOTE, stratified split, class weights)

**📊 Unsupervised Learning (Clustering & Dimensionality Reduction)**
- **Clustering** — K-Means, DBSCAN, Hierarchical Clustering with data-specific ratings
- **Reduction** — PCA, t-SNE/UMAP with dimensionality guidance
- **Anomaly Detection** — Isolation Forest for outlier scoring

**🔧 Multicollinearity Guidance**
- Detects high correlations (|r| > 0.85) and explains the problem
- Provides three solutions: (1) Drop correlated features, (2) Use PCA, (3) Use Ridge/Lasso

**📋 Data Preprocessing Pipeline**
- Step-by-step instructions tailored to your dataset
- Code snippets for imputation, encoding, scaling, and handling outliers
- Handles special cases: high missingness, skewness, outliers, multicollinearity

### 3. `df.eda.vizall()` — Advanced Auto-Visualizations

Auto-renders 7+ types of publication-quality charts (no matplotlib code needed):

- **Histograms + KDE** — Numeric distributions with density overlays
- **Bar charts** — Top categorical values with percentage labels
- **Box plots** — Numeric distributions by categorical features (spot relationships)
- **Feature variance ranking** — Horizontal bar chart showing information content
- **Pair plots** — Top correlated feature pairs with regression lines
- **Categorical association heatmap** — Cramér's V strength between categorical columns
- **Missing data heatmap** — Correlation patterns in missingness (reveals if missingness is random or correlated)
- **Correlation heatmap** — Pearson correlations between all numeric columns
- **Time-series plots** — Line charts for datetime columns + numeric trends

### 4. `df.eda.profile_column(column_name)` — Deep Dive Into a Single Column

Detailed analysis of one column for understanding distributions and transformations:

- **Type information** — Inferred data type and role with confidence
- **Distribution analysis** — Detects if data is symmetric, moderately skewed, or highly skewed
- **Transformation recommendations** — Suggests log/sqrt transforms with statistical justification
- **Outlier explanation** — Quantifies outliers as % of data
- **Cardinality insights** — For categorical columns, shows all/top values with frequencies

Example:
```python
df.eda.profile_column('age')      # Deep dive into 'age' column
df.eda.profile_column('category') # Deep dive into 'category' column
```

### 5. `df.eda.compare(other_df)` — Compare Two Datasets

Detect schema drift and data distribution changes between datasets:

- **Dimension changes** — Rows and columns added/removed
- **Score regression** — Data quality, model readiness, and risk changes
- **Schema drift** — Columns added/removed, column roles changed
- **Risk changes** — New PII detected, risks added/removed

Example:
```python
df_train.eda.compare(df_test)     # Check for train/test drift
df_v1.eda.compare(df_v2)          # Detect schema changes between versions
```

---

## Optional: Quick-Access One-Liner Methods

For quick access to individual analysis tables during interactive exploration. All methods return **pandas DataFrames** for easy inspection and manipulation:

### Data Quality & Scores

#### `df.eda.scores_df()`
Returns a DataFrame with data quality, model readiness, and risk scores.

| Score | Range | Meaning |
|---|---|---|
| `data_quality` | 0–100 | Penalised for missing values, duplicates, constants, outliers |
| `model_readiness` | 0–100 | Penalised for skew, untyped columns, high missingness |
| `risk` | 0+ | Added per PII column (+15) and encoded column (+10) |

**Returns:** DataFrame with shape (3, 1), containing scores.

```python
scores = df.eda.scores_df()
print(scores)
#               Value
# data_quality    87.5
# model_readiness  82.1
# risk            15.0
```

---

#### `df.eda.insights_df(full_line=True)`
Returns a DataFrame with human-readable insights about data quality, patterns, and issues.

**Parameters:**
- `full_line` (bool, default=True) — If True, shows complete insight text. If False, truncates at ~50 characters for compact display.

**Returns:** DataFrame with columns: `[Insight]`

```python
# Full insights (default)
insights = df.eda.insights_df()

# Compact insights (truncated for brief review)
insights = df.eda.insights_df(full_line=False)
```

---

### Schema & Column Information

#### `df.eda.schema_df()`
Returns a DataFrame with inferred column roles, types, and confidence scores.

**Returns:** DataFrame with columns: `[Column, Role, Type, Confidence]`

```python
schema = df.eda.schema_df()
print(schema)
#      Column       Role      Type  Confidence
# 0      user_id        id     int64        0.99
# 1      age      numeric   int64        0.95
# 2      name         text    object       0.88
# 3      signup_date  datetime  datetime64   0.98
```

---

#### `df.eda.stats_df()`
Returns a DataFrame with descriptive statistics (mean, std, quartiles, skewness, etc.) for numeric columns.

**Returns:** DataFrame with columns: `[Column, Mean, Std, Min, Q1, Median, Q3, Max, Skewness, Kurtosis]`

```python
stats = df.eda.stats_df()
```

---

### Missing Data & Duplicates

#### `df.eda.missing_df(format='percentage')`
Returns a DataFrame with missing data rates per column. Choose between percentages or counts.

**Parameters:**
- `format` (str, default='percentage') — Display format: `'percentage'` shows missing data as %, `'number'` shows absolute missing counts.

**Returns:** DataFrame with columns: `[Column, Missing_Data]`

```python
# Default: show missing data as percentage
missing = df.eda.missing_df()
print(missing)
#        Column Missing_Data
# 0         age          2.5%
# 1       salary         15.3%

# Alternative: show absolute missing counts
missing = df.eda.missing_df(format='number')
print(missing)
#        Column Missing_Data
# 0         age            50
# 1       salary          306
```

---

#### `df.eda.duplicates_df()`
Returns a DataFrame summarising duplicate rows and constant columns.

**Returns:** DataFrame with metrics: `[Total Rows, Duplicate Rows, Constant Columns]`

```python
duplicates = df.eda.duplicates_df()
print(duplicates)
#                   Metric  Value
# 0             Total Rows  2000
# 1        Duplicate Rows    15
# 2      Constant Columns     3
```

---

### Correlations & Outliers

#### `df.eda.correlation_df()`
Returns the Pearson correlation matrix between all numeric columns.

**Returns:** DataFrame — correlation matrix (symmetric, numeric columns × numeric columns)

```python
corr = df.eda.correlation_df()
print(corr)
#            age    salary    score
# age       1.00     0.65     0.42
# salary    0.65     1.00     0.78
# score     0.42     0.78     1.00
```

---

#### `df.eda.outliers_df(format='number')`
Returns outlier counts per numeric column. Choose between absolute counts or percentages.

**Parameters:**
- `format` (str, default='number') — Display format: `'number'` shows outlier counts, `'percentage'` shows as % of total rows.

**Returns:** DataFrame with columns: `[Column, Outliers]`

```python
# Default: show outlier counts
outliers = df.eda.outliers_df()
print(outliers)
#      Column  Outliers
# 0       age         12
# 1    salary         25

# Alternative: show outliers as percentage
outliers = df.eda.outliers_df(format='percentage')
print(outliers)
#      Column  Outliers
# 0       age      0.6%
# 1    salary      1.2%
```

---

### PII & Encoding Detection

#### `df.eda.pii_df()`
Returns a DataFrame listing all detected PII (Personally Identifiable Information) fields.

**Returns:** DataFrame with columns: `[Column, PII_Type, Sample]`

```python
pii = df.eda.pii_df()
print(pii)
#        Column PII_Type                Sample
# 0  customer_email    email  john@example.com
# 1  user_address    email   sarah@test.org
```

---

#### `df.eda.encoding_df()`
Returns a DataFrame listing columns with detected encoding signals (Base64, obfuscation patterns).

**Returns:** DataFrame with columns: `[Column, Encoding_Type, Sample]`

```python
encoding = df.eda.encoding_df()
print(encoding)
#        Column       Encoding_Type               Sample
# 0  customer_id    base64_signal   aGVsbG8gd29ybGQ=
# 1  secret_key    obfuscation     xxxxxxxx1234xxxx
```

---

## Usage Summary

| Method | Returns | Key Parameters |
|---|---|---|
| `statsall()` | Prints comprehensive report | — |
| `mlall()` | Prints ML recommendations | — |
| `vizall()` | Displays visualizations | — |
| `profile_column(col)` | Prints single-column analysis | `col` — column name |
| `compare(other_df)` | Prints schema drift report | `other_df` — comparison dataset |
| `scores_df()` | DataFrame (3 rows) | — |
| `insights_df()` | DataFrame | `full_line` (default=True) |
| `schema_df()` | DataFrame | — |
| `stats_df()` | DataFrame | — |
| `missing_df()` | DataFrame | `format` (default='percentage') |
| `duplicates_df()` | DataFrame | — |
| `correlation_df()` | DataFrame | — |
| `outliers_df()` | DataFrame | `format` (default='number') |
| `pii_df()` | DataFrame | — |
| `encoding_df()` | DataFrame | — |

---

## Understanding the API

**The 5 Powerful Methods** are designed for comprehensive analysis in one call:
- `statsall()` — Complete statistical & ML analysis
- `mlall()` — Algorithm recommendations
- `vizall()` — Auto-visualizations
- `profile_column()` — Deep dive into one column
- `compare()` — Schema drift detection

**The 10 One-Liner Methods** are for interactive exploration and specific queries:
- Return DataFrames directly for easy inspection
- Support flexible formatting options (percentages, truncation, etc.)
- Designed for quick analysis without boilerplate
- Recommended for Jupyter notebooks and interactive workflows

---

## Features

| Feature | Description |
|---|---|
| **5 powerful methods** | `statsall()`, `mlall()`, `vizall()`, `profile_column()`, `compare()` |
| **10 convenience methods** | Quick access to individual tables for interactive exploration |
| Universal ingestion | CSV, Excel, JSON, XML, HTML, Parquet, Feather — 28 formats |
| Native pandas accessor | `df.eda.*` — feels like pandas |
| **ML-Ready Analysis** | Algorithm ratings, class imbalance detection, preprocessing pipeline with code |
| **Advanced Visualizations** | 9+ auto-chart types, missing data heatmap, time-series plots |
| **Confidence scores** | Data type inference confidence per column |
| **Column quality flags** | ✓ Good / ⚠ Check / ✗ Issue for every column |
| **Temporal analysis** | Datetime detection, frequency inference, stationarity & seasonality tests |
| Plugin architecture | Every analysis is a swappable plugin |
| Schema inference | Auto-detects IDs, categoricals, datetimes, text with confidence scores |
| Data quality scoring | 0–100 quality + model-readiness score + risk level |
| Multicollinearity detection | VIF calculation with actionable recommendations + code |
| Class imbalance detection | Identifies imbalanced classes, suggests SMOTE/class_weight |
| Correlation explanations | Explains why high correlations are problematic + solutions |
| PII detection | Email addresses + extensible patterns |
| Encoding detection | Base64 and obfuscation signals |
| Outlier detection | IQR-based, per numeric column with % quantification |
| Duplicate detection | Exact row duplicates + constant columns |
| Categorical diversity | Entropy-based balance metric per column |
| Actionable insights | Human-readable text, not just numbers |
| HTML report | Dark-themed, stakeholder-ready export |
| CLI | One-liner from the terminal |

---

## Installation

```bash
cd NowEDA
pip install -e .
```

> Requires Python 3.8+ and pandas 1.3+.

---

## Quick Start

```python
import noweda as eda

df = eda.read("data.csv")

# The 5 Powerful Methods — use these for comprehensive analysis
df.eda.statsall()                    # Complete statistical analysis report
df.eda.mlall()                       # ML algorithm recommendations & preprocessing
df.eda.vizall()                      # Auto-render 9+ visualizations
df.eda.profile_column('age')         # Deep dive into a single column
df.eda.compare(df_test)              # Detect schema drift between datasets

# Optional: Quick-access one-liner methods for interactive exploration
scores = df.eda.scores_df()          # Returns DataFrame with quality/readiness/risk scores
insights = df.eda.insights_df()      # Returns DataFrame with insights
schema = df.eda.schema_df()          # Returns DataFrame with column roles & types
stats = df.eda.stats_df()            # Returns DataFrame with statistics
missing = df.eda.missing_df()        # Returns DataFrame with missing data (default: %)
missing = df.eda.missing_df(format='number')  # Show missing counts instead
duplicates = df.eda.duplicates_df()  # Returns DataFrame with duplicate info
corr = df.eda.correlation_df()       # Returns correlation matrix
outliers = df.eda.outliers_df()      # Returns DataFrame with outlier counts (default: #)
outliers = df.eda.outliers_df(format='percentage')  # Show outliers as %
pii = df.eda.pii_df()                # Returns DataFrame with PII findings
encoding = df.eda.encoding_df()      # Returns DataFrame with encoding signals
```

### All Supported Formats

NowEDA supports 28 file formats through pandas:

```python
import noweda as eda

df = eda.read("data.csv")               # CSV
df = eda.read("data.xlsx")              # Excel (.xlsx, .xls, .ods)
df = eda.read("data.json")              # JSON
df = eda.read("data.xml")               # XML
df = eda.read("data.html")              # HTML tables
df = eda.read("data.parquet")           # Parquet
df = eda.read("data.feather")           # Feather
df = eda.read("data.h5")                # HDF5
df = eda.read("data.sav")               # SPSS (.sav, .zsav)
```

Any `**kwargs` are forwarded to the underlying pandas reader:

```python
df = eda.read("data.xlsx", sheet_name="Sheet1")
df = eda.read("data.json", orient="records")
```

---

## CLI

```bash
# Print insights and scores to the terminal
noweda data.csv

# Export a dark-themed HTML report
noweda data.csv --html report.html

# Export a JSON report
noweda data.csv --json report.json

# Both at once
noweda data.csv --html report.html --json report.json
```

---

## Plugin System

Every analysis step is an independent plugin. You can swap, extend, or disable plugins.

```python
import noweda as eda
from noweda.core.engine import AutoEDAEngine
from noweda.plugins.missing import MissingDataPlugin
from noweda.plugins.pii import PIIDetectorPlugin

# Run only the plugins you want
engine = AutoEDAEngine([MissingDataPlugin(), PIIDetectorPlugin()])
report = engine.run_df(df)
```

### Built-in plugins

| Plugin | Name key | What it detects |
|---|---|---|
| `SchemaPlugin` | `schema` | Column roles: id, categorical, numeric, datetime, text |
| `StatsPlugin` | `stats` | Descriptive stats: mean, median, std, skewness, etc. |
| `MissingDataPlugin` | `missing` | Per-column missing rate |
| `DuplicatesPlugin` | `duplicates` | Duplicate rows, constant columns |
| `CorrelationPlugin` | `correlation` | Pearson correlation matrix (numeric columns) |
| `OutlierPlugin` | `outliers` | IQR-based outlier counts per column |
| `PIIDetectorPlugin` | `pii` | Email addresses (extensible to SSN, phone, etc.) |
| `EncodingDetectionPlugin` | `encoding` | Base64-encoded strings |

### Writing a custom plugin

```python
import noweda as eda
from noweda.plugins.base import BasePlugin

class MyPlugin(BasePlugin):
    name = "my_check"

    def run(self, df):
        # return any JSON-serialisable dict
        return {"total_rows": len(df)}
```

---

## HTML Report

```bash
noweda examples/sample.csv --html report.html
```

The report includes:

- Score cards (quality, risk, model readiness)
- Actionable insights list
- Column schema table with inferred roles
- Missing value bars
- Duplicate and constant column summary
- Outlier counts
- PII findings (highlighted)
- Encoding signals (highlighted)

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Project Structure

```
NowEDA/
├── noweda/
│   ├── __init__.py               # exposes noweda.read()
│   ├── io.py                     # file ingestion (all formats)
│   ├── accessor.py               # df.eda.* pandas accessor (5 methods)
│   ├── ml_utils.py               # ML utility functions (VIF, cardinality, etc.)
│   ├── ml_recommendations.py     # Algorithm recommendations + preprocessing pipeline
│   ├── temporal_utils.py         # Temporal analysis (datetime, stationarity, seasonality)
│   ├── core/
│   │   └── engine.py             # orchestrates plugins → scorer → insights
│   ├── plugins/
│   │   ├── base.py
│   │   ├── schema.py             # with confidence scores
│   │   ├── stats.py
│   │   ├── missing.py
│   │   ├── duplicates.py
│   │   ├── correlation.py
│   │   ├── outliers.py
│   │   ├── pii.py
│   │   └── encoding.py
│   ├── scoring/
│   │   └── scorer.py
│   ├── insights/
│   │   └── generator.py
│   ├── report/
│   │   └── html.py
│   └── cli.py
├── examples/
│   └── sample.csv
├── tests/
│   ├── test_basic.py
│   └── test_formats.py
└── pyproject.toml
```

---

## Roadmap

**Completed:**
- [x] Visualization layer (9+ chart types including missing data heatmap)
- [x] ML algorithm recommendations with data-specific ratings
- [x] Class imbalance detection and guidance
- [x] Temporal data analysis (stationarity, seasonality)
- [x] Dataset comparison (drift detection)
- [x] Column-level quality summaries
- [x] Code snippets in preprocessing pipeline

**Coming Soon:**
- [ ] Additional PII patterns (phone, SSN, credit card)
- [ ] Streaming / chunked ingestion for large files
- [ ] PyPI publish
- [ ] Web dashboard UI
- [ ] Feature interaction detection
- [ ] Correlation explanation for numeric features

---

## Author

**Daniel Peng** — [danielpeng95@gmail.com]
