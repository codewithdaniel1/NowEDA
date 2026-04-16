# API Reference

Complete reference for all public functions and classes in NowEDA.

---

## `noweda.read()`

```python
noweda.read(file_path, **kwargs) → pandas.DataFrame
```

Load any supported file into a pandas DataFrame.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `file_path` | `str` | Path to the data file |
| `**kwargs` | any | Forwarded to the underlying pandas reader |

**Returns:** `pandas.DataFrame`

**Raises:**
- `ValueError` — Unsupported file extension
- `FileNotFoundError` — File does not exist
- `ImportError` — Optional dependency not installed (Parquet, HDF5, SPSS)

**Examples:**

```python
df = noweda.read("data.csv")
df = noweda.read("data.xlsx", sheet_name="Q1")
df = noweda.read("data.csv", nrows=500, encoding="latin-1")
```

---

## `NowEDAAccessor`

Registered as `df.noweda` on every `pandas.DataFrame` after `import noweda`.

```python
import noweda
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

Analysis runs on first call and is cached. Subsequent calls return immediately.

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
