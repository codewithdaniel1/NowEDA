# Duplicates & Constants Plugin

**Class:** `DuplicatesPlugin`  
**Result key:** `duplicates`  
**Import:** `from noweda.plugins.duplicates import DuplicatesPlugin`

---

## What It Does

The Duplicates plugin detects two types of data quality issues:

1. **Duplicate rows** — rows that are identical across all columns
2. **Constant columns** — columns where every value is the same (zero variance)

Both indicate data quality problems that should be resolved before analysis or modelling.

---

## Output Format

```python
{
    "duplicate_rows":      3,       # number of exact duplicate rows
    "duplicate_rows_pct":  0.12,    # as a fraction (12% of total rows)
    "constant_columns":    ["flag", "source"]   # list of zero-variance columns
}
```

If there are no duplicates: `duplicate_rows = 0`, `duplicate_rows_pct = 0.0`  
If there are no constant columns: `constant_columns = []`

---

## Duplicate Row Detection

Duplicate detection uses `pandas.DataFrame.duplicated()` with default settings — a row is a duplicate if **all** column values are identical to another row.

```python
summary = df.noweda.summary()
dups = summary["duplicates"]

print(f"Duplicate rows: {dups['duplicate_rows']} ({dups['duplicate_rows_pct']:.1%})")
```

### How to remove duplicates

```python
# Based on NowEDA findings
dups = df.noweda.summary()["duplicates"]

if dups["duplicate_rows"] > 0:
    df_clean = df.drop_duplicates()
    print(f"Removed {dups['duplicate_rows']} duplicate rows")
```

### Partial duplicates (subset of columns)

The plugin checks full-row duplicates. To check duplicates on a subset of columns:

```python
# Check duplicates on specific columns (outside NowEDA)
partial_dups = df.duplicated(subset=["user_id", "date"]).sum()
print(f"Partial duplicates: {partial_dups}")
```

---

## Constant Column Detection

A constant column has only one unique value across the entire dataset (including NaN — a column of all NaN values is also constant). These columns carry zero information and are useless for analysis or modelling.

```python
summary = df.noweda.summary()
const_cols = summary["duplicates"]["constant_columns"]

print(f"Constant columns: {const_cols}")

# Drop them
df_clean = df.drop(columns=const_cols)
```

**Common causes of constant columns:**

- Filtering that makes a column uniform (e.g., all rows have `status = "active"`)
- Export artifacts (a column that was always a default value)
- Template columns that were never filled in

---

## Score and Insight Impact

| Condition | Quality penalty | Readiness penalty | Insight |
|---|---|---|---|
| Duplicate rows >10% | −10 | — | "X duplicate row(s) detected (Y%). De-duplication recommended." |
| Duplicate rows >0 | −3 | — | "X duplicate row(s) detected." |
| Each constant column | −3 | −5 | "Constant columns detected: [list]." |

---

## Using This Plugin Standalone

```python
import pandas as pd
from noweda.plugins.duplicates import DuplicatesPlugin

data = pd.DataFrame({
    "id":     [1, 2, 2, 3, 4],         # row 1 and 2 are duplicates
    "value":  [10, 20, 20, 30, 40],
    "source": ["web", "web", "web", "web", "web"],   # constant column
})

plugin = DuplicatesPlugin()
result = plugin.run(data)

print(result)
# {
#     'duplicate_rows': 1,
#     'duplicate_rows_pct': 0.2,
#     'constant_columns': ['source']
# }
```
