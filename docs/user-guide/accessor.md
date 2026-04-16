# The NowEDA Accessor

The `df.noweda` accessor is a [pandas extension accessor](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors) registered on every DataFrame. It is the primary interface for all NowEDA functionality.

---

## How It Works

When you `import noweda`, the accessor is registered globally on the `pandas.DataFrame` class. This means it's available on **any** DataFrame — not just ones loaded with `noweda.read()`.

```python
import noweda
import pandas as pd

# Works on DataFrames from any source
df = pd.read_csv("data.csv")     # regular pandas load
df.noweda.insights()              # NowEDA is available

df2 = noweda.read("data.xlsx")   # NowEDA load
df2.noweda.insights()             # also works
```

---

## Lazy Evaluation

NowEDA does **not** run analysis when you load data. Analysis is deferred until you first call a `df.noweda.*` method. The result is then **cached** on the accessor instance, so subsequent calls are instant.

```python
df = noweda.read("data.csv")      # no analysis runs here

df.noweda.score()                  # analysis runs here (first call)
df.noweda.insights()               # instant — uses the cache
df.noweda.summary()                # instant — uses the cache
df.noweda.report()                 # instant — uses the cache
```

!!! note "Cache lifetime"
    The cache lives on the accessor instance, which is tied to the DataFrame object. If you create a new DataFrame (e.g. by filtering or copying), the new object has no cache and will re-run analysis on first access.

    ```python
    df_filtered = df[df["amount"] > 1000]
    df_filtered.noweda.score()   # re-runs analysis on the filtered data
    ```

---

## Methods

### `df.noweda.insights()`

Returns a list of human-readable, actionable insight strings derived from all plugin results and scores.

```python
insights = df.noweda.insights()
# Returns: List[str]

for insight in insights:
    print(insight)
```

**Example output:**

```
Likely identifier column(s) detected: id. Consider excluding from modelling.
Datetime column(s) detected: join_date. Temporal features may be valuable.
Column 'encoded_field' is 72% missing — consider dropping it.
Column 'email' has high missing rate (32%) — imputation recommended.
Minor missing values in: 'name', 'salary', 'join_date', 'score'.
Very strong correlation (0.99) between 'age' and 'salary'. One may be redundant.
PII detected in column 'email': 17 email address(es) found. Mask before sharing.
Column 'encoded_field' may contain encoded data (possible_base64).
Data quality score is acceptable (77). Minor issues present.
Moderate risk level (25). Review PII and encoded columns before sharing.
```

Insights cover: schema roles, missing values, duplicate rows, high skewness, outliers, correlations, PII findings, encoding signals, and overall quality/risk assessment.

→ See [Insights Engine](../insights.md) for the full rule list.

---

### `df.noweda.score()`

Returns a dictionary of three scores.

```python
scores = df.noweda.score()
# Returns: Dict[str, int | float]
```

| Key | Range | Description |
|---|---|---|
| `data_quality` | 0 – 100 | Penalised for missing values, duplicates, constant columns, outliers |
| `risk` | 0+ | Increased per PII column (+15) and encoded column (+10) |
| `model_readiness` | 0 – 100 | Penalised for high skew, untyped columns, high missingness |

**Example output:**

```python
{'data_quality': 77, 'risk': 25, 'model_readiness': 53}
```

→ See [Scoring System](../scoring.md) for full scoring logic.

---

### `df.noweda.summary()`

Returns the raw output from every plugin as a nested dictionary. Use this when you need the numbers, not just the text.

```python
summary = df.noweda.summary()
# Returns: Dict[str, Any]
```

**Keys:**

| Key | Type | Content |
|---|---|---|
| `schema` | `dict` | Column roles and uniqueness info |
| `stats` | `dict` | Descriptive statistics per column |
| `missing` | `dict` | Per-column missing rate (0.0 – 1.0) |
| `duplicates` | `dict` | Duplicate rows count, constant columns list |
| `correlation` | `dict` | Pearson correlation matrix (numeric cols only) |
| `outliers` | `dict` | IQR outlier count per numeric column |
| `pii` | `dict` | PII findings per column |
| `encoding` | `dict` | Encoding detection results per column |

**Example:**

```python
summary = df.noweda.summary()

# Missing rates
print(summary["missing"])
# {'id': 0.0, 'name': 0.08, 'email': 0.32, 'salary': 0.04}

# Schema roles
print(summary["schema"])
# {'id': {'dtype': 'int64', 'role': 'id_candidate', 'unique': 25, ...},
#  'department': {'dtype': 'object', 'role': 'categorical', ...}}

# Outlier counts
print(summary["outliers"])
# {'age': 1, 'salary': 3, 'score': 0}

# Correlation (numeric columns only)
print(summary["correlation"]["salary"]["score"])
# 0.712
```

---

### `df.noweda.report()`

Returns everything in one structured dictionary. This is the canonical full output of NowEDA.

```python
report = df.noweda.report()
# Returns: Dict[str, Any]
```

**Structure:**

```python
{
    "results": {          # raw plugin outputs (same as summary())
        "schema":     {...},
        "stats":      {...},
        "missing":    {...},
        "duplicates": {...},
        "correlation":{...},
        "outliers":   {...},
        "pii":        {...},
        "encoding":   {...},
    },
    "scores": {           # quality, risk, model_readiness
        "data_quality":    77,
        "risk":            25,
        "model_readiness": 53,
    },
    "insights": [         # human-readable list
        "Likely identifier...",
        "Column 'email' has high...",
        ...
    ]
}
```

Use `report()` when passing data to the HTML generator, serializing to JSON, or integrating NowEDA into a pipeline.

---

## Accessing Individual Plugin Results

You can drill into any plugin's output directly via `summary()`:

```python
summary = df.noweda.summary()

# Schema
for col, info in summary["schema"].items():
    print(f"{col}: {info['role']}")

# Stats — skewness for numeric columns
for col, stats in summary["stats"].items():
    if "skewness" in stats:
        print(f"{col} skewness: {stats['skewness']:.2f}")

# PII
for col, findings in summary["pii"].items():
    print(f"PII in '{col}': {findings}")

# Correlations above 0.7
corr = summary["correlation"]
for col1 in corr:
    for col2, val in corr[col1].items():
        if col1 < col2 and abs(val) > 0.7:
            print(f"{col1} ↔ {col2}: {val:.3f}")
```

---

## Using NowEDA Without `noweda.read()`

The accessor works on any DataFrame. Just make sure noweda is imported so the accessor gets registered:

```python
import noweda          # registers df.noweda on all DataFrames
import pandas as pd

df = pd.read_sql("SELECT * FROM transactions", conn)
df.noweda.insights()   # works

df2 = pd.read_feather("cached_data.feather")
df2.noweda.score()     # works
```

---

## Running a Subset of Plugins

By default `df.noweda.*` runs all 8 plugins. To run only specific ones:

```python
from noweda.core.engine import AutoEDAEngine
from noweda.plugins.missing import MissingDataPlugin
from noweda.plugins.pii import PIIDetectorPlugin
from noweda.scoring.scorer import Scorer
from noweda.insights.generator import InsightGenerator

engine = AutoEDAEngine([MissingDataPlugin(), PIIDetectorPlugin()])
report = engine.run_df(df)

print(report["results"]["missing"])
print(report["results"]["pii"])
```

This is useful when you only care about specific checks (e.g. a quick PII scan before sharing data externally).
