# API Reference

Complete reference for all public functions and classes in NowEDA.

---

## `eda.read()` / `noweda.read()`

```python
import noweda as eda

eda.read(file_path, mode="small", chunksize=100_000, **kwargs) → pandas.DataFrame | LargeDataset
```

Load a supported file into a pandas DataFrame, or explicitly create a
disk-backed `LargeDataset`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `file_path` | `str` | Path to the data file |
| `mode` | `"small"` or `"large"` | `small` loads pandas; `large` keeps CSV, TSV, TXT, and Parquet data on disk |
| `chunksize` | `int` | Bounded sample size used by large-mode exploratory methods |
| `**kwargs` | any | Forwarded to the underlying pandas reader |

**Returns:** `pandas.DataFrame` in small mode, or `LargeDataset` in large mode.

**Raises:**
- `ValueError` — Unsupported file extension
- `FileNotFoundError` — File does not exist
- `ImportError` — Optional dependency not installed (Parquet, HDF5, SPSS)

**Examples:**

```python
df = eda.read("data.csv")
df = eda.read("data.xlsx", sheet_name="Q1")
df = eda.read("data.csv", nrows=500, encoding="latin-1")
```

Large mode supports `statsall()`, `report()`, `vizall()`, `mlall()`,
`profile_column()`, and `compare()` through `data.eda`. It calculates source
dimensions exactly, and prints a sample-size notice before every exploratory,
sample-based result.

```python
data = eda.read("data.csv", mode="large", chunksize=100_000)
data.eda.statsall()
```

---

## `NowEDAAccessor`

Registered as `df.noweda` on every `pandas.DataFrame` after `import noweda as eda`.

```python
import noweda as eda
import pandas as pd

df = pd.DataFrame({"x": [1, 2, 3]})
df.noweda    # → NowEDAAccessor instance
```

---

### `df.noweda.insights()`

```python
df.noweda.insights() → List[str]
```

Returns a list of human-readable, actionable insight strings.

Analysis runs on first call and is cached. Subsequent calls scan a fingerprint of the data and schema, reusing the report only if they have not changed.

---

### `df.noweda.score()`

```python
df.noweda.score() → Dict[str, int | float]
```

Returns:

```python
{
    "data_quality":    int,   # 0–100
    "risk":            int,   # 0+
    "model_readiness": int,   # 0–100
}
```

---

### `df.noweda.summary()`

```python
df.noweda.summary() → Dict[str, Any]
```

Returns raw plugin results:

```python
{
    "schema":     Dict,
    "stats":      Dict,
    "missing":    Dict,
    "duplicates": Dict,
    "correlation":Dict,
    "outliers":   Dict,
    "pii":        Dict,
    "encoding":   Dict,
}
```

---

### `df.noweda.report()`

```python
df.noweda.report() → Dict[str, Any]
```

Returns the complete report:

```python
{
    "results":  Dict,   # same as summary()
    "scores":   Dict,   # same as score()
    "insights": List,   # same as insights()
}
```

---

### `df.noweda.statsall()`

```python
df.noweda.statsall() → None
```

Prints a rich, fully-formatted analysis report to the terminal or notebook. Combines everything in one call:

- **Scores** — data_quality, model_readiness, risk (colour-coded)
- **Column overview** — dtype, inferred role, unique count, missing count per column
- **Numeric statistics** — count, mean, std, min, 25%, median, 75%, max, skewness (highlighted when |skew| > 1)
- **Categorical statistics** — count, unique values, top value, top frequency
- **Insights** — full human-readable list
- **Plugin summary** — raw output from outliers, duplicates, PII, encoding plugins

Returns `None`; output is printed directly (suitable for notebooks and terminals).

---

### `df.noweda.vizall()`

```python
df.noweda.vizall() → None
```

Auto-renders the best visualizations for your dataset based on column types:

| Chart | Condition |
|---|---|
| Histogram + KDE overlay | Every numeric column |
| Bar chart (top 15) | Every categorical column with ≤ 30 unique values |
| Correlation heatmap | When ≥ 2 numeric columns exist |
| Missing value bar chart | When any column has missing values |
| Time-series line plot | When datetime + numeric columns both exist |

Install `noweda[viz]` for Matplotlib charts and SciPy KDE overlays.

Returns `None`; charts are rendered inline (Jupyter) or displayed in a window (terminal).

---

### `df.noweda.refresh()`

Forces fresh analysis and returns the complete report dictionary.

### `df.noweda.mlall(target=None, problem_type=None, features=None, plan=False)`

Prints task-aware ML guidance with heuristic stars and `/5` estimated dataset-fit
scores, cautions, preprocessing, validation, and metrics. No models are trained or
evaluated. Set `plan=True` to return the structured result after printing it.

With no objective, `mlall()` assesses likely supervised and unsupervised directions:
it reports possible targets for review, label readiness, likely identifiers, usable
features, and ranked unsupervised directions. It never silently selects a target.

Classification and regression require `target`. When a target is supplied without
`problem_type`, NowEDA infers classification or regression from target dtype and
cardinality and includes the reason in the result. Unsupervised tasks reject a
target. `features` optionally limits which input columns are profiled; it cannot
contain the target.

```python
df.noweda.mlall()

plan = df.noweda.mlall(target="segment", plan=True)
df.noweda.mlall(
    problem_type="clustering",
    features=["age", "income"],
)
```

When `plan=True`, the result includes `problem_type`, `problem_subtype`, `target`,
`target_summary`, `features`, `inferred`, `inference_reason`, `recommendations`,
`assessment`, `preprocessing`, `evaluation`, and `warnings`. Each recommendation
contains `name`, `score`, `why`, and `caution`; `score` is a 1–5 estimated
dataset-fit value rather than measured model performance.

See [Task-Aware ML Guidance](ml-guidance.md) for inference rules, validation, and
complete examples.

### Table schemas

`schema_df()` returns `Column`, `dtype`, `role`, `confidence`, `unique`, and
`uniqueness_ratio`. `stats_df()` includes all columns, with numeric fields
`mean`, `median`, `std`, `min`, `max`, `q25`, `q75`, `skewness`, and `kurtosis`,
and categorical fields `top_value` and `top_freq` where applicable.
`encoding_df()` returns `Column` and `Encoding_Type` with possible Base64 signals.
Use `encoding_df(include_confidence=True)` to add `Sample_Size`, `Matches` and
`Confidence` (the sample match fraction, not a probability).

---

## `AutoEDAEngine`

```python
from noweda.core.engine import AutoEDAEngine

engine = AutoEDAEngine(plugins: List[BasePlugin])
```

Orchestrates plugins → scorer → insight generator.

### `engine.run_df(df)`

```python
engine.run_df(df: pandas.DataFrame) → Dict[str, Any]
```

Runs all plugins on `df` and returns a full report dict.

---

## `BasePlugin`

```python
from noweda.plugins.base import BasePlugin
```

Base class for all plugins.

| Attribute/Method | Description |
|---|---|
| `name: str` | Key used in `results` dict — must be unique |
| `run(df) → dict` | Run analysis on DataFrame, return JSON-serialisable dict |

---

## Built-in Plugin Classes

| Class | Import | Result key |
|---|---|---|
| `SchemaPlugin` | `noweda.plugins.schema` | `schema` |
| `StatsPlugin` | `noweda.plugins.stats` | `stats` |
| `MissingDataPlugin` | `noweda.plugins.missing` | `missing` |
| `DuplicatesPlugin` | `noweda.plugins.duplicates` | `duplicates` |
| `CorrelationPlugin` | `noweda.plugins.correlation` | `correlation` |
| `OutlierPlugin` | `noweda.plugins.outliers` | `outliers` |
| `PIIDetectorPlugin` | `noweda.plugins.pii` | `pii` |
| `EncodingDetectionPlugin` | `noweda.plugins.encoding` | `encoding` |

### `default_plugins()`

```python
from noweda.plugins import default_plugins

plugins = default_plugins()   # → List[BasePlugin] (all 8 plugins)
```

---

## `Scorer`

```python
from noweda.scoring.scorer import Scorer

scorer = Scorer()
scores = scorer.compute(results: dict) → dict
```

Computes `data_quality`, `risk`, and `model_readiness` from plugin results.

---

## `InsightGenerator`

```python
from noweda.insights.generator import InsightGenerator

generator = InsightGenerator()
insights = generator.generate(results: dict, scores: dict) → List[str]
```

Generates human-readable insights from plugin results and scores.

---

## `generate_html_report()`

```python
from noweda.report.html import generate_html_report

generate_html_report(report: dict, output_path: str) → None
```

Writes a self-contained HTML report to `output_path`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `report` | `dict` | Full report dict from `df.noweda.report()` |
| `output_path` | `str` | Path to write the `.html` file |

---

## CLI Entry Point

```python
from noweda.cli import main

main()   # reads sys.argv
```

Or from the command line:

```bash
noweda data.csv [--html output.html] [--json output.json]
```


## Release 0.1.4 report additions

`report()` adds `score_breakdown` and `encoding_details` while retaining the
existing `results`, `scores` and `insights`. See [scoring](scoring.md) and
[encoding detection](plugins/encoding.md) for field definitions.

```python
from noweda.report.json import generate_json_report
generate_json_report(df.eda.report(), "report.json")
```

This UTF-8 JSON exporter converts nonfinite numbers to `null` and column labels
to strings. Colliding labels and unsupported custom values raise before writing.

`calculate_vif(df, numeric_cols=None)` in `noweda.ml_utils` uses NumPy least
squares with an intercept and all other numeric features as predictors. Rows
with missing or infinite values in any selected feature are excluded. Constant
responses and fits without residual degrees of freedom return NaN; exact
collinearity with sufficient observations returns infinity. Fewer than two
selected columns returns an empty dictionary. No optional ML install is needed.
