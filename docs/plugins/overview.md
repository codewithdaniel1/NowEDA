# Plugin System Overview

NowEDA's analysis is entirely plugin-based. Every check — missing values, correlations, PII detection — is an independent class that runs on the DataFrame and returns a JSON-serialisable result. Plugins are composable, swappable, and easy to extend.

---

## Default Plugin Stack

When you call `df.noweda.*`, these 8 plugins run in order:

| # | Plugin | Key in results | What it checks |
|---|---|---|---|
| 1 | `SchemaPlugin` | `schema` | Column roles, data types, uniqueness |
| 2 | `StatsPlugin` | `stats` | Descriptive statistics, skewness |
| 3 | `MissingDataPlugin` | `missing` | Per-column missing rates |
| 4 | `DuplicatesPlugin` | `duplicates` | Duplicate rows, constant columns |
| 5 | `CorrelationPlugin` | `correlation` | Pearson correlation (numeric cols) |
| 6 | `OutlierPlugin` | `outliers` | IQR-based outlier counts |
| 7 | `PIIDetectorPlugin` | `pii` | Email addresses and PII patterns |
| 8 | `EncodingDetectionPlugin` | `encoding` | Base64 and encoded string signals |

---

## How Plugins Work

Every plugin inherits from `BasePlugin` and implements one method: `run(df)`.

```python
class BasePlugin:
    name = "base"           # key used in results dict

    def run(self, df):
        raise NotImplementedError
```

The engine calls each plugin, collects the results under `plugin.name`, then passes the full results dict to the Scorer and InsightGenerator.

```
DataFrame
    │
    ▼
[SchemaPlugin] → results["schema"]
[StatsPlugin]  → results["stats"]
[MissingPlugin]→ results["missing"]
      ...
    │
    ▼
Scorer → scores
    │
    ▼
InsightGenerator → insights
```

---

## Running a Custom Set of Plugins

You don't have to run all 8. Use `AutoEDAEngine` directly to compose exactly what you need:

```python
from noweda.core.engine import AutoEDAEngine
from noweda.plugins.missing import MissingDataPlugin
from noweda.plugins.pii import PIIDetectorPlugin
from noweda.plugins.outliers import OutlierPlugin

engine = AutoEDAEngine([
    MissingDataPlugin(),
    PIIDetectorPlugin(),
    OutlierPlugin(),
])

report = engine.run_df(df)
print(report["results"])
```

---

## Plugin Reference Pages

- [Schema Inference](schema.md)
- [Descriptive Stats](stats.md)
- [Missing Values](missing.md)
- [Duplicates & Constants](duplicates.md)
- [Correlations](correlation.md)
- [Outlier Detection](outliers.md)
- [PII Detection](pii.md)
- [Encoding Detection](encoding.md)
- [Writing Custom Plugins](custom.md)
