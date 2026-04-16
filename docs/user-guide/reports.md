# Understanding Reports

A NowEDA report is a structured dictionary with three top-level keys: `results`, `scores`, and `insights`. This page explains how to read and work with every part of it.

---

## Report Structure

```python
report = df.noweda.report()
```

```
report
├── results           ← raw plugin outputs
│   ├── schema        ← column roles
│   ├── stats         ← descriptive statistics
│   ├── missing       ← per-column missing rates
│   ├── duplicates    ← row duplicates & constant columns
│   ├── correlation   ← Pearson correlation matrix
│   ├── outliers      ← IQR-based outlier counts
│   ├── pii           ← PII findings
│   └── encoding      ← encoding signals
├── scores            ← quality, risk, model_readiness
└── insights          ← human-readable text list
```

---

## results

### `results["schema"]`

Column role inference. Each column gets a `role` and uniqueness metrics.

```python
{
    "id": {
        "dtype": "int64",
        "role": "id_candidate",      # likely a primary key
        "unique": 25,
        "uniqueness_ratio": 1.0
    },
    "department": {
        "dtype": "object",
        "role": "categorical",       # low-cardinality string
        "unique": 4,
        "uniqueness_ratio": 0.16
    },
    "salary": {
        "dtype": "float64",
        "role": "numeric",
        "unique": 23,
        "uniqueness_ratio": 0.92
    },
    "join_date": {
        "dtype": "object",
        "role": "datetime",          # detected via parsing
        "unique": 24,
        "uniqueness_ratio": 0.96
    }
}
```

**Possible roles:**

| Role | Meaning |
|---|---|
| `id_candidate` | Near-perfect uniqueness — likely a key/identifier |
| `numeric` | Continuous numeric column |
| `categorical` | Low-cardinality string column |
| `categorical_numeric` | Numeric column with very few distinct values |
| `datetime` | Detected as date/time (by dtype or string parsing) |
| `text` | High-cardinality string column (free text) |
| `unknown` | Could not be classified |

---

### `results["stats"]`

Descriptive statistics per column. Numeric columns get full stats including skewness.

```python
{
    "salary": {
        "dtype": "float64",
        "count": 23,           # non-null count
        "missing": 2,          # null count
        "unique": 23,          # distinct values
        "mean": 97826.09,
        "median": 92000.0,
        "std": 40521.32,
        "min": 32000.0,
        "max": 200000.0,
        "q25": 62000.0,
        "q75": 130000.0,
        "skewness": 0.52       # positive = right-skewed
    },
    "department": {
        "dtype": "object",
        "count": 25,
        "missing": 0,
        "unique": 4,
        "top_value": "Marketing",  # most frequent value
        "top_freq": 8              # how many times it appears
    }
}
```

---

### `results["missing"]`

Per-column missing rate as a float between `0.0` and `1.0`.

```python
{
    "id":             0.0,     # no missing values
    "name":           0.04,    # 4% missing
    "email":          0.32,    # 32% missing — flagged as high
    "salary":         0.08,
    "encoded_field":  0.72     # 72% missing — critical
}
```

---

### `results["duplicates"]`

```python
{
    "duplicate_rows":      2,      # exact duplicate row count
    "duplicate_rows_pct":  0.08,   # as a fraction of total rows
    "constant_columns":    ["flag"] # columns with only one unique value
}
```

---

### `results["correlation"]`

Pearson correlation matrix for numeric columns only. Stored as a nested dict.

```python
{
    "age": {
        "age":    1.0,
        "salary": 0.99,     # very strong — flagged in insights
        "score":  0.63
    },
    "salary": {
        "age":    0.99,
        "salary": 1.0,
        "score":  0.71      # strong
    },
    ...
}
```

Values range from `-1.0` (perfect negative) to `1.0` (perfect positive). `0.0` means no linear relationship.

---

### `results["outliers"]`

IQR-based outlier count per numeric column.

```python
{
    "age":    0,
    "salary": 3,    # 3 values fall outside 1.5 × IQR
    "score":  1
}
```

The IQR method flags a value as an outlier if it is below `Q1 - 1.5 × IQR` or above `Q3 + 1.5 × IQR`.

---

### `results["pii"]`

PII findings per column.

```python
{
    "email": {
        "emails_detected": 17    # number of email pattern matches
    }
}
```

Only columns with at least one match appear here. Empty dict = no PII detected.

---

### `results["encoding"]`

Columns where encoded data was detected.

```python
{
    "encoded_field": "possible_base64"
}
```

---

## scores

```python
{
    "data_quality":    77,    # 0–100
    "risk":            25,    # 0+
    "model_readiness": 53     # 0–100
}
```

→ See [Scoring System](../scoring.md) for detailed penalty and scoring logic.

---

## insights

A flat list of human-readable strings, ordered by category:

```python
[
    # Schema
    "Likely identifier column(s) detected: id. Consider excluding from modelling.",
    "Datetime column(s) detected: join_date. Temporal features may be valuable.",

    # Missing
    "Column 'encoded_field' is 72% missing — consider dropping it.",
    "Column 'email' has high missing rate (32%) — imputation recommended.",
    "Minor missing values in: 'name', 'salary', 'join_date', 'score'.",

    # Correlation
    "Very strong correlation (0.99) between 'age' and 'salary'. One may be redundant.",
    "Strong correlation (0.71) between 'salary' and 'score'.",

    # PII
    "PII detected in column 'email': 17 email address(es) found. Mask before sharing.",

    # Encoding
    "Column 'encoded_field' may contain encoded data (possible_base64). Inspect for obfuscation.",

    # Overall
    "Data quality score is acceptable (77). Minor issues present.",
    "Moderate risk level (25). Review PII and encoded columns before sharing."
]
```

---

## Exporting Reports

### To JSON

```python
import json

report = df.noweda.report()
with open("report.json", "w") as f:
    json.dump(report, f, indent=2)
```

### To HTML

```python
from noweda.report.html import generate_html_report

report = df.noweda.report()
generate_html_report(report, "report.html")
```

### Via CLI

```bash
noweda data.csv --json report.json --html report.html
```

→ See [HTML Reports](../html-reports.md) for what the HTML report contains.
