# Descriptive Stats Plugin

**Class:** `StatsPlugin`  
**Result key:** `stats`  
**Import:** `from noweda.plugins.stats import StatsPlugin`

---

## What It Does

The Stats plugin computes descriptive statistics for every column. Numeric columns get full distribution metrics including skewness. Categorical/string columns get frequency information.

---

## Output Format

### Numeric columns

```python
{
    "salary": {
        "dtype":   "float64",
        "count":   23,         # number of non-null values
        "missing": 2,          # number of nulls
        "unique":  23,         # distinct value count
        "mean":    97826.09,
        "median":  92000.0,
        "std":     40521.32,   # standard deviation
        "min":     32000.0,
        "max":     200000.0,
        "q25":     62000.0,    # 25th percentile
        "q75":     130000.0,   # 75th percentile
        "skewness":0.52        # Fisher skewness
    }
}
```

### String / categorical columns

```python
{
    "department": {
        "dtype":     "object",
        "count":     25,
        "missing":   0,
        "unique":    4,
        "top_value": "Marketing",   # most frequent value
        "top_freq":  8              # count of most frequent value
    }
}
```

---

## Understanding Skewness

Skewness measures how asymmetric a distribution is.

| Skewness value | Meaning |
|---|---|
| Close to `0` | Roughly symmetric (normal-like) |
| `> 1` | Moderately right-skewed (long tail on the right) |
| `> 2` | Heavily right-skewed — NowEDA flags this in insights |
| `< -1` | Moderately left-skewed (long tail on the left) |
| `< -2` | Heavily left-skewed — NowEDA flags this in insights |

**Right-skewed example (salary data):**
```
Most employees earn $50k–$90k
A few executives earn $500k+
→ Skewness ≈ 2.3 (right-skewed)
→ Insight: "Column 'salary' is heavily right-skewed — log transform may help"
```

---

## How to Use the Results

```python
summary = df.noweda.summary()
stats = summary["stats"]

# Print mean and std for all numeric columns
for col, s in stats.items():
    if "mean" in s:
        print(f"{col}: mean={s['mean']:.2f}, std={s['std']:.2f}")

# Find heavily skewed columns
skewed = [(col, s["skewness"]) for col, s in stats.items()
          if "skewness" in s and abs(s["skewness"]) > 2]
print("Heavily skewed:", skewed)

# Check top values for categorical columns
for col, s in stats.items():
    if "top_value" in s:
        print(f"{col}: most common = '{s['top_value']}' ({s['top_freq']} times)")
```

---

## Using This Plugin Standalone

```python
import pandas as pd
import numpy as np
from noweda.plugins.stats import StatsPlugin

df = pd.DataFrame({
    "amount":   np.random.exponential(scale=100, size=500),
    "category": np.random.choice(["A", "B", "C"], size=500),
})

plugin = StatsPlugin()
result = plugin.run(df)

print(result["amount"]["skewness"])   # ~2.0+ (exponential is right-skewed)
print(result["category"]["top_value"])
```

---

## Relationship to Other Plugins

- **StatsPlugin → InsightGenerator**: Columns with `abs(skewness) > 2` trigger a "heavily skewed" insight
- **StatsPlugin → Scorer**: High-skew columns reduce `model_readiness` score (-5 per column)
