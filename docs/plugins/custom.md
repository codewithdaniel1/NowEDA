# Writing Custom Plugins

NowEDA is built on a plugin system. Any analysis logic you want — custom business rules, domain-specific checks, regulatory compliance detection — can be added as a plugin in minutes.

---

## The BasePlugin Contract

Every plugin must inherit from `BasePlugin` and implement `run()`:

```python
from noweda.plugins.base import BasePlugin

class MyPlugin(BasePlugin):
    name = "my_check"          # key in results["my_check"]

    def run(self, df):
        # df: pandas DataFrame
        # return: any JSON-serialisable dict
        return {}
```

**Rules:**
- `name` must be a unique string — it becomes the key in `report["results"]`
- `run()` must accept a pandas DataFrame and return a dict
- The returned dict must be JSON-serialisable (no numpy types, no DataFrames)

---

## Example 1 — Row Count Plugin

The simplest possible plugin:

```python
from noweda.plugins.base import BasePlugin

class RowCountPlugin(BasePlugin):
    name = "row_count"

    def run(self, df):
        return {
            "rows":    len(df),
            "columns": len(df.columns),
        }
```

---

## Example 2 — Negative Value Detector

Flag numeric columns that contain negative values (e.g., quantities or prices should never be negative):

```python
from noweda.plugins.base import BasePlugin

class NegativeValuePlugin(BasePlugin):
    name = "negatives"

    def run(self, df):
        result = {}
        for col in df.select_dtypes(include="number").columns:
            neg_count = int((df[col] < 0).sum())
            if neg_count > 0:
                result[col] = {"negative_count": neg_count}
        return result
```

---

## Example 3 — Future Date Detector

Flag datetime columns that contain dates in the future (useful for transaction data where future dates indicate bad data entry):

```python
import pandas as pd
from datetime import datetime
from noweda.plugins.base import BasePlugin

class FutureDatePlugin(BasePlugin):
    name = "future_dates"

    def run(self, df):
        result = {}
        now = pd.Timestamp(datetime.utcnow())

        for col in df.columns:
            if df[col].dtype.kind == "M":    # datetime dtype
                future = (df[col] > now).sum()
                if future > 0:
                    result[col] = {"future_dates": int(future)}

        return result
```

---

## Example 4 — Column Name Validator

Check that column names follow a naming convention (snake_case):

```python
import re
from noweda.plugins.base import BasePlugin

class ColumnNamingPlugin(BasePlugin):
    name = "naming"

    SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")

    def run(self, df):
        violations = [
            col for col in df.columns
            if not self.SNAKE_CASE.match(col)
        ]
        return {
            "non_snake_case_columns": violations,
            "compliant": len(violations) == 0,
        }
```

---

## Using Your Plugin

### Option A — Replace the default stack

```python
from noweda.core.engine import AutoEDAEngine
from noweda.plugins import default_plugins
from my_plugins import NegativeValuePlugin, FutureDatePlugin

engine = AutoEDAEngine(
    default_plugins() + [NegativeValuePlugin(), FutureDatePlugin()]
)

report = engine.run_df(df)
print(report["results"]["negatives"])
print(report["results"]["future_dates"])
```

### Option B — Run only your plugins

```python
from noweda.core.engine import AutoEDAEngine
from my_plugins import NegativeValuePlugin

engine = AutoEDAEngine([NegativeValuePlugin()])
report = engine.run_df(df)
```

### Option C — Use with `df.noweda.*`

Monkey-patch the default plugin list before analysis runs:

```python
import noweda.plugins as _plugins
from my_plugins import NegativeValuePlugin

original_defaults = _plugins.default_plugins

def patched_defaults():
    return original_defaults() + [NegativeValuePlugin()]

_plugins.default_plugins = patched_defaults

# Now df.noweda.* uses your extended plugin list
df = noweda.read("data.csv")
print(df.noweda.summary()["negatives"])
```

---

## Integrating with Scoring and Insights

Custom plugins can influence scores and insights by adding logic to the Scorer and InsightGenerator — or you can simply read from `results` yourself after the engine runs.

### Option A — Read results directly (simplest)

```python
report = engine.run_df(df)
negatives = report["results"]["negatives"]

if negatives:
    print("WARNING: Negative values found:")
    for col, info in negatives.items():
        print(f"  {col}: {info['negative_count']} negative values")
```

### Option B — Subclass InsightGenerator

```python
from noweda.insights.generator import InsightGenerator

class ExtendedInsightGenerator(InsightGenerator):
    def generate(self, results, scores):
        insights = super().generate(results, scores)

        # Add insight for negative values
        negatives = results.get("negatives", {})
        for col, info in negatives.items():
            insights.append(
                f"Column '{col}' has {info['negative_count']} negative "
                f"value(s) — check if this is expected."
            )

        return insights
```

Then wire it into the engine:

```python
from noweda.core.engine import AutoEDAEngine
from noweda.scoring.scorer import Scorer

class CustomEngine(AutoEDAEngine):
    def run_df(self, df):
        results = {plugin.name: plugin.run(df) for plugin in self.plugins}
        scores = Scorer().compute(results)
        insights = ExtendedInsightGenerator().generate(results, scores)
        return {"results": results, "scores": scores, "insights": insights}
```

---

## Plugin Checklist

Before shipping your plugin, verify:

- [ ] `name` is unique and descriptive
- [ ] `run()` handles empty DataFrames (`len(df) == 0`)
- [ ] `run()` handles DataFrames with no columns of the expected type
- [ ] Return value contains only Python built-in types (`int`, `float`, `str`, `list`, `dict`, `bool`, `None`)
- [ ] No mutations to the input DataFrame (`df` should be read-only)
- [ ] Any exceptions are handled or allowed to propagate clearly

```python
class RobustPlugin(BasePlugin):
    name = "robust_example"

    def run(self, df):
        if df.empty:
            return {}

        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            return {}

        # safe to proceed
        return {col: float(df[col].mean()) for col in numeric_cols}
```
