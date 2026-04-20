# Quick Start

Get up and running in under 5 minutes.

---

## Step 1 — Install

```bash
pip install noweda
```

---

## Step 2 — Load your data

`eda.read()` replaces every `pd.read_*` call. The file format is detected automatically from the extension.

```python
import noweda as eda

df = eda.read("data.csv")
df = eda.read("data.xlsx")
df = eda.read("data.json")
df = eda.read("data.parquet")  # requires: pip install "noweda[parquet]"
```

`**kwargs` are forwarded directly to the underlying pandas reader:

```python
df = eda.read("data.xlsx", sheet_name="Sales Q1")
df = eda.read("data.csv", nrows=1000, encoding="latin-1")
```

For very large Spark-friendly files, `eda.read()` can switch to PySpark automatically. Spark is bundled with NowEDA, so there is no extra install flag or backend setting.

The returned object is a **standard pandas DataFrame** — every pandas method still works:

```python
df.head()
df.describe()
df["column"].value_counts()
df.groupby("category").mean()
```

---

## Step 3 — Run EDA

The `df.noweda` accessor is available on every DataFrame after importing noweda.

### Get human-readable insights

```python
for insight in df.noweda.insights():
    print(insight)
```

```
Likely identifier column(s) detected: id. Consider excluding from modelling.
Column 'email' has high missing rate (32%) — imputation recommended.
Very strong correlation (0.99) between 'age' and 'salary'. One may be redundant.
PII detected in column 'email': 17 email address(es) found. Mask before sharing.
Data quality score is acceptable (77). Minor issues present.
Moderate risk level (25). Review PII and encoded columns before sharing.
```

### Get scores

```python
print(df.noweda.score())
```

```python
{
    'data_quality': 77,    # 0–100, higher = cleaner
    'risk': 25,            # 0+, higher = more sensitive data present
    'model_readiness': 53  # 0–100, higher = more ready for ML
}
```

### Get the full plugin results

```python
summary = df.noweda.summary()

summary["schema"]       # column roles (id, numeric, categorical, datetime, text)
summary["stats"]        # descriptive statistics per column
summary["missing"]      # per-column missing rates
summary["duplicates"]   # duplicate rows, constant columns
summary["correlation"]  # Pearson correlation matrix
summary["outliers"]     # IQR-based outlier counts
summary["pii"]          # PII findings
summary["encoding"]     # encoding detection results
```

### Get everything at once

```python
report = df.noweda.report()
# report["results"]  → all plugin outputs
# report["scores"]   → quality, risk, readiness
# report["insights"] → human-readable list
```

---

## Step 4 — Export a report

### HTML report (shareable, dark-themed)

```python
from noweda.report.html import generate_html_report

report = df.noweda.report()
generate_html_report(report, "my_report.html")
```

Or via CLI:

```bash
noweda data.csv --html my_report.html
```

### JSON export

```bash
noweda data.csv --json my_report.json
```

---

## Step 5 — Use selectively (run specific plugins only)

You don't have to run all plugins. Compose exactly what you need:

```python
from noweda.core.engine import AutoEDAEngine
from noweda.plugins.missing import MissingDataPlugin
from noweda.plugins.pii import PIIDetectorPlugin

engine = AutoEDAEngine([MissingDataPlugin(), PIIDetectorPlugin()])
report = engine.run_df(df)

print(report["results"]["missing"])
print(report["results"]["pii"])
```

---

## What's Next?

- [Reading Data](user-guide/reading-data.md) — all 28 supported formats
- [The NowEDA Accessor](user-guide/accessor.md) — full `df.noweda.*` API
- [Plugin Reference](plugins/overview.md) — what each plugin detects and how
- [Scoring System](scoring.md) — how scores are calculated
- [Writing Custom Plugins](plugins/custom.md) — extend NowEDA for your use case
