# Schema Inference Plugin

**Class:** `SchemaPlugin`  
**Result key:** `schema`  
**Import:** `from noweda.plugins.schema import SchemaPlugin`

---

## What It Does

The Schema plugin inspects every column and assigns it a **semantic role** — what kind of data does this column represent? It goes beyond `dtype` to tell you whether a column is an ID, a category, free text, or a datetime, regardless of how it's stored.

---

## Output Format

```python
{
    "column_name": {
        "dtype":             "object",       # pandas dtype string
        "role":              "categorical",  # inferred semantic role
        "unique":            4,              # number of distinct values
        "uniqueness_ratio":  0.16            # unique / total rows
    },
    ...
}
```

---

## Column Roles

| Role | Criteria | Example columns |
|---|---|---|
| `id_candidate` | Integer column with 100% unique values and >5 rows | `id`, `user_id`, `order_id` |
| `id_candidate` | String column with ≥98% unique values and >5 rows | `uuid`, `transaction_ref` |
| `numeric` | Float or int column, not ID-like, not low-cardinality | `salary`, `score`, `age` |
| `categorical_numeric` | Numeric column with ≤5% uniqueness and ≤20 distinct values | `flag` (0/1), `rating` (1-5) |
| `categorical` | String column with ≤5% uniqueness and ≤50 distinct values | `department`, `status`, `country` |
| `datetime` | Native datetime dtype, or string column parseable as dates | `join_date`, `created_at` |
| `text` | High-cardinality string column | `name`, `email`, `description` |
| `unknown` | Cannot be classified | Mixed-type or complex columns |

---

## Datetime Detection

The plugin detects datetimes in two ways:

1. **Native datetime dtype** — if pandas already parsed the column as `datetime64`, it's classified immediately.

2. **String parsing** — for `object` dtype columns, the plugin takes a sample of up to 20 non-null values and attempts `pd.to_datetime()`. If it succeeds, the column is classified as `datetime`. This catches columns like `"2024-01-15"` or `"Jan 15, 2024"` stored as strings.

---

## How to Use the Results

```python
summary = df.noweda.summary()
schema = summary["schema"]

# Find all ID-like columns
id_cols = [col for col, info in schema.items() if info["role"] == "id_candidate"]
print("Identifier columns:", id_cols)

# Find all categorical columns
cat_cols = [col for col, info in schema.items() if "categorical" in info["role"]]
print("Categorical columns:", cat_cols)

# Find datetime columns
dt_cols = [col for col, info in schema.items() if info["role"] == "datetime"]
print("Datetime columns:", dt_cols)

# Get uniqueness ratio for all columns
for col, info in schema.items():
    print(f"{col}: {info['uniqueness_ratio']:.1%} unique — {info['role']}")
```

---

## Using This Plugin Standalone

```python
import pandas as pd
from noweda.plugins.schema import SchemaPlugin

df = pd.DataFrame({
    "user_id":    [1, 2, 3, 4, 5],
    "status":     ["active", "inactive", "active", "active", "inactive"],
    "revenue":    [100.5, 250.0, 75.0, 310.0, 88.5],
    "signup_date":["2024-01-01", "2024-02-15", "2024-03-10", "2024-04-01", "2024-05-20"],
})

plugin = SchemaPlugin()
result = plugin.run(df)

for col, info in result.items():
    print(f"{col:15s} → {info['role']}")
```

```
user_id         → id_candidate
status          → categorical
revenue         → numeric
signup_date     → datetime
```

---

## Why This Matters

Knowing the role of each column upfront prevents common data prep mistakes:

- **ID columns** should never be used as features in a model
- **Categorical** columns need encoding (one-hot, label encoding) before ML
- **Datetime** columns can be expanded into useful features (day of week, month, etc.)
- **Text** columns may need NLP preprocessing
- **High-cardinality strings** that look numeric should be investigated
