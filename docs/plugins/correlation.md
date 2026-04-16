# Correlation Plugin

**Class:** `CorrelationPlugin`  
**Result key:** `correlation`  
**Import:** `from noweda.plugins.correlation import CorrelationPlugin`

---

## What It Does

The Correlation plugin computes the **Pearson correlation coefficient** between every pair of numeric columns. It only operates on columns with a numeric dtype — string, datetime, and boolean columns are excluded.

---

## Output Format

The result is a nested dictionary representing the full correlation matrix.

```python
{
    "age": {
        "age":    1.0,
        "salary": 0.99,
        "score":  0.63
    },
    "salary": {
        "age":    0.99,
        "salary": 1.0,
        "score":  0.71
    },
    "score": {
        "age":    0.63,
        "salary": 0.71,
        "score":  1.0
    }
}
```

If there are no numeric columns: returns `{}`.

---

## Interpreting Correlation Values

| Value | Interpretation |
|---|---|
| `1.0` | Perfect positive correlation (same column) |
| `0.7 – 0.99` | Strong positive — NowEDA flags this in insights |
| `0.9 – 0.99` | Very strong positive — likely redundant, flagged with higher urgency |
| `0.3 – 0.7` | Moderate positive |
| `0.0 – 0.3` | Weak or no linear relationship |
| `-0.3 – 0.0` | Weak negative |
| `-0.7 – -0.3` | Moderate negative |
| `< -0.7` | Strong negative — also flagged in insights |

---

## Insight Thresholds

| Absolute correlation | Insight |
|---|---|
| `> 0.9` | "Very strong correlation (X.XX) between 'A' and 'B'. One may be redundant." |
| `> 0.7` | "Strong correlation (X.XX) between 'A' and 'B'." |

Each pair is reported only once (A↔B, not both A↔B and B↔A).

---

## How to Use the Results

```python
summary = df.noweda.summary()
corr = summary["correlation"]

# Find all strong correlations (above 0.7, excluding self-correlation)
seen = set()
for col1, row in corr.items():
    for col2, val in row.items():
        pair = tuple(sorted([col1, col2]))
        if col1 != col2 and pair not in seen and abs(val) > 0.7:
            seen.add(pair)
            print(f"{col1} ↔ {col2}: {val:.3f}")
```

```
age ↔ salary: 0.990
salary ↔ score: 0.712
```

### Convert to a pandas DataFrame for further analysis

```python
import pandas as pd

corr_df = pd.DataFrame(summary["correlation"])
print(corr_df.round(2))
```

```
        age  salary  score
age    1.00    0.99   0.63
salary 0.99    1.00   0.71
score  0.63    0.71   1.00
```

---

## Important Caveats

**Pearson correlation measures linear relationships only.** Two variables can be strongly related in a non-linear way and still show a low Pearson coefficient.

**Correlation does not imply causation.** A high correlation between `age` and `salary` could simply reflect that both increase with career progression, not that age causes salary.

**Outliers strongly affect Pearson correlation.** If the outlier plugin reports outliers in a column, treat the corresponding correlations with caution.

---

## Using This Plugin Standalone

```python
import pandas as pd
from noweda.plugins.correlation import CorrelationPlugin

df = pd.DataFrame({
    "x": [1, 2, 3, 4, 5],
    "y": [2, 4, 5, 4, 5],   # moderate positive correlation with x
    "z": [5, 4, 3, 2, 1],   # perfect negative correlation with x
    "label": ["a", "b", "c", "d", "e"]   # ignored (not numeric)
})

plugin = CorrelationPlugin()
result = plugin.run(df)

print(result["x"]["z"])   # -1.0
print(result["x"]["y"])   # ~0.87
```
