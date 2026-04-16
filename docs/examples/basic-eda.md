# Example: Basic EDA Workflow

A complete end-to-end walkthrough using NowEDA on a realistic employee dataset.

---

## Setup

```bash
pip install noweda
```

```python
import noweda as eda
```

---

## Step 1 — Load the data

```python
df = eda.read("employees.csv")
print(df.shape)    # (25, 10)
print(df.dtypes)
```

```
id                int64
name             object
email            object
age               int64
salary          float64
department       object
join_date        object
score           float64
flag              int64
encoded_field    object
dtype: object
```

---

## Step 2 — Standard pandas exploration

Since `df` is a regular pandas DataFrame, nothing changes:

```python
df.head(3)
df.describe()
df["department"].value_counts()
df.groupby("department")["salary"].mean()
```

---

## Step 3 — Run NowEDA

```python
# Get scores first — fast overview
scores = df.noweda.score()
print(scores)
```

```python
{'data_quality': 77, 'risk': 25, 'model_readiness': 53}
```

The data quality is acceptable (77), but there's a moderate risk signal (25) and model readiness is low (53) — meaning preprocessing is needed before ML.

---

## Step 4 — Read the insights

```python
for insight in df.noweda.insights():
    print(insight)
```

```
Likely identifier column(s) detected: id. Consider excluding from modelling.
Datetime column(s) detected: join_date. Temporal features may be valuable.
Column 'encoded_field' is 72% missing — consider dropping it.
Column 'email' has high missing rate (32%) — imputation recommended.
Minor missing values in: 'name', 'salary', 'join_date', 'score'.
Very strong correlation (0.99) between 'age' and 'salary'. One may be redundant.
Strong correlation (0.71) between 'salary' and 'score'.
PII detected in column 'email': 17 email address(es) found. Mask before sharing.
Column 'encoded_field' may contain encoded data (possible_base64). Inspect for obfuscation.
Data quality score is acceptable (77). Minor issues present.
Moderate risk level (25). Review PII and encoded columns before sharing.
```

---

## Step 5 — Drill into specific results

```python
summary = df.noweda.summary()

# Which columns have missing data?
missing = {k: v for k, v in summary["missing"].items() if v > 0}
print("Columns with missing data:", missing)
```

```python
{'name': 0.04, 'email': 0.32, 'salary': 0.08, 'join_date': 0.08,
 'score': 0.12, 'encoded_field': 0.72}
```

```python
# What roles did schema inference assign?
for col, info in summary["schema"].items():
    print(f"{col:20s}: {info['role']}")
```

```
id                  : id_candidate
name                : text
email               : text
age                 : numeric
salary              : numeric
department          : categorical
join_date           : datetime
score               : numeric
flag                : numeric
encoded_field       : text
```

```python
# Outliers
outliers = {k: v for k, v in summary["outliers"].items() if v > 0}
print("Outliers:", outliers)
```

```python
{'salary': 2, 'score': 0}
```

---

## Step 6 — Act on findings

```python
import pandas as pd

# 1. Drop the nearly-empty encoded column
df_clean = df.drop(columns=["encoded_field"])

# 2. Drop the ID column (not useful for modelling)
df_clean = df_clean.drop(columns=["id"])

# 3. Redact PII
df_clean["email"] = "[REDACTED]"

# 4. Impute minor missing values
df_clean["salary"] = df_clean["salary"].fillna(df_clean["salary"].median())
df_clean["score"]  = df_clean["score"].fillna(df_clean["score"].median())

# 5. Drop rows with missing name (only 1 row)
df_clean = df_clean.dropna(subset=["name"])

# 6. Parse join_date as datetime
df_clean["join_date"] = pd.to_datetime(df_clean["join_date"])

print(df_clean.shape)    # fewer columns, no critical missing values
```

---

## Step 7 — Re-evaluate

```python
df_clean.noweda.score()
```

```python
{'data_quality': 95, 'risk': 0, 'model_readiness': 88}
```

After cleaning: quality jumped from 77 → 95, risk dropped to 0, model readiness jumped from 53 → 88.

---

## Step 8 — Export the report

```bash
noweda employees.csv --html employees_report.html
```

Share `employees_report.html` with stakeholders — no Python needed to read it.
