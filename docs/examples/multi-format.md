# Example: Working with Multiple Formats

NowEDA reads 28 file extensions. This example shows how to load and analyse data from several different formats in one workflow.

---

## Loading Different Formats

```python
import noweda

# Delimited text
csv_df   = noweda.read("sales.csv")
tsv_df   = noweda.read("events.tsv")

# Spreadsheets
xlsx_df  = noweda.read("report.xlsx", sheet_name="Summary")
ods_df   = noweda.read("budget.ods")

# Structured data
json_df  = noweda.read("api_response.json")
xml_df   = noweda.read("records.xml")
html_df  = noweda.read("web_table.html")

# Big data formats (requires: pip install "noweda[parquet]")
parquet_df = noweda.read("transactions.parquet")
feather_df = noweda.read("cache.feather")

# Statistical software
stata_df  = noweda.read("survey.dta")
sas_df    = noweda.read("results.sas7bdat")

# Python
pickle_df = noweda.read("model_data.pkl")
```

---

## Scanning Multiple Files at Once

```python
import os
import pandas as pd
import noweda

def scan_directory(folder):
    """Run NowEDA on every supported file in a folder."""
    supported = {
        ".csv", ".tsv", ".xlsx", ".xls", ".json",
        ".xml", ".html", ".parquet", ".pkl", ".dta"
    }

    results = {}

    for fname in os.listdir(folder):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in supported:
            continue

        path = os.path.join(folder, fname)
        try:
            df = noweda.read(path)
            results[fname] = {
                "rows":         len(df),
                "columns":      len(df.columns),
                "scores":       df.noweda.score(),
                "top_insights": df.noweda.insights()[:3],
            }
        except Exception as e:
            results[fname] = {"error": str(e)}

    return results


findings = scan_directory("data/")

for filename, info in findings.items():
    if "error" in info:
        print(f"  {filename}: ERROR — {info['error']}")
    else:
        s = info["scores"]
        print(f"  {filename} ({info['rows']} rows): "
              f"quality={s['data_quality']}, risk={s['risk']}")
```

---

## Comparing Two Versions of the Same Dataset

A common use case: compare a new data export against last month's version.

```python
import noweda

old = noweda.read("customers_march.csv")
new = noweda.read("customers_april.csv")

old_scores = old.noweda.score()
new_scores = new.noweda.score()

print("Score comparison:")
for metric in ["data_quality", "risk", "model_readiness"]:
    delta = new_scores[metric] - old_scores[metric]
    direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
    print(f"  {metric:20s}: {old_scores[metric]} → {new_scores[metric]}  {direction} {abs(delta)}")
```

```
Score comparison:
  data_quality        : 77 → 85  ↑ 8
  risk                : 25 → 10  ↓ 15
  model_readiness     : 53 → 71  ↑ 18
```

---

## Converting Formats

Load one format, transform, export another:

```python
import noweda

# Load Excel, inspect, save as Parquet (much faster for big data)
df = noweda.read("large_report.xlsx")

print(df.noweda.score())

# Check it's clean before converting
if df.noweda.score()["data_quality"] >= 70:
    df.to_parquet("large_report.parquet", index=False)
    print("Saved as Parquet.")
else:
    print("Data quality too low — clean before converting.")
```

---

## Format-Specific Tips

### Excel: multiple sheets

```python
import pandas as pd

# Read all sheets and scan each one
xl = pd.ExcelFile("workbook.xlsx")

for sheet in xl.sheet_names:
    df = noweda.read("workbook.xlsx", sheet_name=sheet)
    score = df.noweda.score()
    print(f"Sheet '{sheet}': quality={score['data_quality']}, rows={len(df)}")
```

### Parquet: column selection for large files

```python
# Only load the columns you need — much faster for wide Parquet files
df = noweda.read("big_data.parquet", columns=["user_id", "amount", "date"])
df.noweda.score()
```

### JSON: nested data

```python
# Normalize nested JSON before passing to NowEDA
import pandas as pd
import json

with open("nested.json") as f:
    data = json.load(f)

df = pd.json_normalize(data["records"])   # flatten nested structure
df.noweda.insights()
```

### HTML: multiple tables on a page

```python
import pandas as pd

# pd.read_html returns ALL tables on the page
tables = pd.read_html("webpage.html")
print(f"Found {len(tables)} tables")

# Scan each one
for i, table in enumerate(tables):
    score = table.noweda.score()
    print(f"Table {i}: {len(table)} rows, quality={score['data_quality']}")
```
