# NowEDA

**Automated Exploratory Data Analysis — built as a native pandas extension.**

NowEDA is a lightweight, modular Python framework that turns any dataset into instant insight. Load any file, call `df.noweda.*`, and get a full EDA report — including data quality scoring, PII detection, outlier analysis, and human-readable insights — with zero boilerplate.

---

## ML-Ready EDA — One Call Does It All

NowEDA now does **everything** an ML engineer needs before modeling:

### `df.noweda.statsall()` — Complete Statistical Report + Algorithm Recommendations
Prints a rich, colour-coded report with:

**Data Profile:**
- **Scores** — data quality, risk, model readiness
- **Column analysis** — dtypes, roles, diversity metrics
- **Numeric stats** — mean, std, quartiles, skewness
- **Categorical stats** — cardinality, diversity, frequency distribution
- **Insights** — human-readable findings

**Preprocessing Guidance:**
- ⚠️ **Multicollinearity detection** (VIF > 5) with actionable recommendations
- 🔧 **Scaling recommendations** (StandardScaler, MinMaxScaler)
- 📊 **Transformation suggestions** (log, sqrt for skewed data)
- 🚨 **Cardinality warnings** (high-cardinality categoricals)
- 🏷️ **Rare category detection** (<1% frequency)
- ❌ **Missing data strategy** (impute vs drop)

**🤖 ML Algorithm Recommendations:**
- **Supervised Learning** — Recommends (Linear Regression, Random Forest, XGBoost, SVM, KNN, Neural Networks, Naive Bayes) with:
  - ⭐ Star ratings based on data fit
  - ✓ Why each algorithm suits your data
  - ✗ When to avoid it and why
- **Unsupervised Learning** — Recommends (K-Means, DBSCAN, Hierarchical Clustering, PCA/t-SNE/UMAP) with reasoning
- **Data Preprocessing Pipeline** — Step-by-step instructions

### `df.noweda.vizall()` — Advanced ML Visualizations
Auto-renders 7+ types of charts:
- Histograms + KDE for numeric columns
- Bar charts with % labels for categoricals
- **Box plots** — numeric distributions by categorical (spot relationships)
- **Feature variance ranking** — importance at a glance
- **Pair plots** — top correlated features with regression lines
- **Categorical heatmap** — association strength (Cramér's V)
- Correlation heatmap, missing value bars, time-series

## Features

| Feature | Description |
|---|---|
| Universal ingestion | CSV, Excel, JSON, XML, HTML — 28 formats |
| Native pandas accessor | `df.noweda.*` — feels like pandas |
| **ML Preprocessing Guide** | Multicollinearity (VIF), scaling recommendations, transformations, cardinality warnings |
| **Advanced Visualizations** | Box plots, pair plots, feature variance ranking, categorical heatmaps |
| Plugin architecture | Every analysis is a swappable plugin |
| Schema inference | Auto-detects IDs, categoricals, datetimes, text |
| Data quality scoring | 0–100 quality + model-readiness score |
| Risk scoring | PII and encoding risk level |
| PII detection | Email addresses + extensible patterns |
| Encoding detection | Base64 and obfuscation signals |
| Outlier detection | IQR-based, per numeric column |
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

# Still a regular pandas DataFrame — nothing changes
print(df.head())
print(df.describe())

# NowEDA layer
print(df.eda.insights())   # human-readable insight list
print(df.eda.score())      # quality, risk, model_readiness
print(df.eda.summary())    # raw plugin results
report = df.eda.report()   # full structured dict

# Power methods
df.eda.statsall()          # full rich report: dtypes + stats + scores + insights
df.eda.vizall()            # auto-render best charts for every column
df.eda.mlall()             # ML algorithm recommendations + preprocessing pipeline
```

### All supported formats

```python
import noweda as eda

df = eda.read("data.csv")
df = eda.read("data.xlsx", sheet_name="Sheet1")
df = eda.read("data.json")
df = eda.read("data.xml")
df = eda.read("data.html")
```

Any `**kwargs` are forwarded to the underlying pandas reader.

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

## Score Breakdown

| Score | Range | Meaning |
|---|---|---|
| `data_quality` | 0–100 | Penalised for missing values, duplicates, constants, outliers |
| `model_readiness` | 0–100 | Penalised for skew, untyped columns, high missingness |
| `risk` | 0+ | Added per PII column (+15) and encoded column (+10) |

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
│   ├── __init__.py          # exposes noweda.read()
│   ├── io.py                # file ingestion (all formats)
│   ├── accessor.py          # df.noweda.* pandas accessor
│   ├── core/
│   │   └── engine.py        # orchestrates plugins → scorer → insights
│   ├── plugins/
│   │   ├── base.py
│   │   ├── schema.py
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
│   └── test_basic.py
└── pyproject.toml
```

---

## Roadmap

- [ ] Visualization layer (histograms, correlation heatmap)
- [ ] Dataset fingerprinting / hash-based change detection
- [ ] Additional PII patterns (phone, SSN, credit card)
- [ ] Streaming / chunked ingestion for large files
- [ ] PyPI publish
- [ ] Web dashboard UI

---

## Author

**Daniel Peng** — [danielpeng95@gmail.com]
