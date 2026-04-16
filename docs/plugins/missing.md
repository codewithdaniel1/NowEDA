# Missing Values Plugin

**Class:** `MissingDataPlugin`  
**Result key:** `missing`  
**Import:** `from noweda.plugins.missing import MissingDataPlugin`

---

## What It Does

The Missing Values plugin computes the fraction of null/NaN values in each column. It counts `None`, `NaN`, `pd.NA`, and `pd.NaT` as missing.

---

## Output Format

```python
{
    "id":             0.0,      # 0% missing — no nulls
    "name":           0.04,     # 4% missing
    "email":          0.32,     # 32% missing — high
    "salary":         0.08,     # 8% missing
    "encoded_field":  0.72      # 72% missing — critical
}
```

Values are floats in the range `[0.0, 1.0]`. Every column in the DataFrame appears in the output.

---

## Severity Thresholds

NowEDA uses these thresholds to generate insights and penalise scores:

| Missing rate | Severity | Insight generated | Quality penalty | Readiness penalty |
|---|---|---|---|---|
| `0.0` | None | (no insight) | 0 | 0 |
| `0 < x ≤ 0.3` | Low | Minor missing values noted | −2 | −3 |
| `0.3 < x ≤ 0.5` | High | Imputation recommended | −5 | −8 |
| `> 0.5` | Critical | Consider dropping column | −10 | −15 |

---

## How to Use the Results

```python
summary = df.noweda.summary()
missing = summary["missing"]

# Columns with any missing values
has_missing = {col: pct for col, pct in missing.items() if pct > 0}
print(has_missing)

# Critical columns (>50% missing)
critical = {col: pct for col, pct in missing.items() if pct > 0.5}
print("Consider dropping:", list(critical.keys()))

# Sort by missing rate descending
sorted_missing = sorted(missing.items(), key=lambda x: -x[1])
for col, pct in sorted_missing:
    bar = "█" * int(pct * 20)
    print(f"{col:20s} {pct:5.1%}  {bar}")
```

**Example output:**
```
encoded_field        72.0%  ██████████████
email                32.0%  ██████
salary                8.0%  █
name                  4.0%
id                    0.0%
```

---

## Using This Plugin Standalone

```python
import pandas as pd
import numpy as np
from noweda.plugins.missing import MissingDataPlugin

df = pd.DataFrame({
    "a": [1, 2, None, 4, 5],
    "b": [None, None, None, 4, 5],
    "c": [1, 2, 3, 4, 5],
})

plugin = MissingDataPlugin()
result = plugin.run(df)

print(result)
# {'a': 0.2, 'b': 0.6, 'c': 0.0}
```

---

## Common Strategies for Missing Data

Once NowEDA identifies which columns have missing data and at what rate, you can decide how to handle them:

| Situation | Recommended action |
|---|---|
| <5% missing, numeric | Mean or median imputation |
| <5% missing, categorical | Mode imputation or "Unknown" label |
| 5–30% missing | Advanced imputation (KNN, iterative) |
| 30–50% missing | Impute carefully, or use as a feature itself (missingness indicator) |
| >50% missing | Drop the column, or keep only if it carries critical domain info |

```python
# Example: act on NowEDA's findings
missing = df.noweda.summary()["missing"]

# Drop columns with >50% missing
cols_to_drop = [col for col, pct in missing.items() if pct > 0.5]
df_clean = df.drop(columns=cols_to_drop)

# Median-impute columns with <30% missing (numeric only)
for col, pct in missing.items():
    if 0 < pct <= 0.3 and df[col].dtype.kind in ("i", "u", "f"):
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
```
