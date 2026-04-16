# Outlier Detection Plugin

**Class:** `OutlierPlugin`  
**Result key:** `outliers`  
**Import:** `from noweda.plugins.outliers import OutlierPlugin`

---

## What It Does

The Outlier plugin detects statistical outliers in every numeric column using the **IQR (Interquartile Range)** method. It reports how many rows fall outside the normal range for each column.

---

## The IQR Method

The IQR method is robust to extreme values and works well for most distributions without assuming normality.

**How it works:**

1. Compute `Q1` (25th percentile) and `Q3` (75th percentile)
2. Compute `IQR = Q3 - Q1`
3. Lower fence = `Q1 - 1.5 × IQR`
4. Upper fence = `Q3 + 1.5 × IQR`
5. Any value below the lower fence or above the upper fence is an outlier

```
  [outlier]   |  Q1 -------- median -------- Q3  |   [outlier]
──────────────────────────────────────────────────────────────▶
         lower fence                        upper fence
         (Q1 - 1.5×IQR)                   (Q3 + 1.5×IQR)
```

---

## Output Format

```python
{
    "age":    0,    # no outliers
    "salary": 3,    # 3 rows fall outside the IQR fence
    "score":  1
}
```

Only numeric columns appear. String, datetime, and boolean columns are skipped.

---

## How to Use the Results

```python
summary = df.noweda.summary()
outliers = summary["outliers"]

# Columns with outliers
has_outliers = {col: n for col, n in outliers.items() if n > 0}
print(has_outliers)

# View the actual outlier rows for a specific column
col = "salary"
q1 = df[col].quantile(0.25)
q3 = df[col].quantile(0.75)
iqr = q3 - q1
outlier_rows = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
print(outlier_rows)
```

---

## Outlier Thresholds and Scoring

| Total outlier count | Quality penalty | Readiness penalty |
|---|---|---|
| `> 50` across all columns | −10 | −10 |
| `> 10` across all columns | −5 | −5 |
| `≤ 10` | 0 | 0 |

---

## Dealing with Outliers

Once NowEDA identifies outlier counts, you have several options depending on context:

| Strategy | When to use |
|---|---|
| **Investigate** | Always first — are they data errors or legitimate extreme values? |
| **Remove** | Data entry errors, measurement mistakes |
| **Cap (Winsorize)** | Keep values but limit to fence boundary (preserves row count) |
| **Transform** | Log, sqrt, or Box-Cox transform to reduce skew |
| **Keep** | When outliers are genuinely important (fraud amounts, rare events) |

```python
# Example: Cap salary at the IQR fence (winsorize)
col = "salary"
q1 = df[col].quantile(0.25)
q3 = df[col].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

df[col] = df[col].clip(lower=lower, upper=upper)
```

---

## Using This Plugin Standalone

```python
import pandas as pd
from noweda.plugins.outliers import OutlierPlugin

df = pd.DataFrame({
    "normal":  [10, 12, 11, 13, 12, 11, 10, 12, 13, 11],
    "extreme": [10, 12, 11, 13, 12, 11, 10, 12, 999, 11],   # 999 is an outlier
})

plugin = OutlierPlugin()
result = plugin.run(df)

print(result)
# {'normal': 0, 'extreme': 1}
```

---

## Limitations

- **IQR works best for roughly symmetric distributions.** For highly skewed data, even non-problematic extreme values may be flagged. Check the skewness from the Stats plugin alongside outlier results.
- **IQR does not flag outliers relative to other columns.** A `salary` of $500k is not flagged as an outlier if many other salaries are also $500k, even if it's far from the median.
- **Small datasets are less reliable.** With fewer than ~30 rows, the percentile estimates are noisy and outlier detection becomes unreliable.
