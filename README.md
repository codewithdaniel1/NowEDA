<div align="left">
  <img src="https://raw.githubusercontent.com/codewithdaniel1/NowEDA/main/assets/noweda-wordmark-logo.svg" alt="NowEDA" width="700" />
</div>

[![PyPI version](https://img.shields.io/pypi/v/noweda?cacheSeconds=300)](https://pypi.org/project/noweda/)

# NowEDA

Exploratory data analysis through a native pandas accessor. Profile columns,
find missing data and duplicates, flag potential PII, explore relationships,
and generate reports with `df.eda` or its equivalent alias `df.noweda`.

[Documentation](https://codewithdaniel1.github.io/NowEDA/) ·
[API reference](https://codewithdaniel1.github.io/NowEDA/api-reference/) ·
[Changelog](https://codewithdaniel1.github.io/NowEDA/changelog/) ·
[Report an issue](https://github.com/codewithdaniel1/NowEDA/issues)

## Install

Requires Python 3.8 or later. pip selects dependency versions compatible with
your Python version.

```bash
pip install noweda
pip install "noweda[viz]"  # Add charts and density overlays
```

## Quick start

This example runs without downloading a dataset:

```python
import pandas as pd
import noweda as eda

# Or load your own file: df = eda.read("data.csv")
df = pd.DataFrame({
    "age": [25, 31, None, 42],
    "income": [42000, 55000, 47000, 68000],
    "segment": ["A", "B", "A", "B"],
})

# Start with one complete assessment.
df.eda.statsall()

# Add charts when they help answer the next question.
# df.eda.vizall()

# Name an outcome before asking for supervised ML guidance.
# df.eda.mlall(target="segment")
```

### Large files

Use explicit large mode when a CSV, TSV, TXT, or Parquet file should stay on
disk. The same core `.eda` workflow supports `statsall()`, `report()`,
`vizall()`, `mlall()`, `profile_column()`, and `compare()`.

```python
data = eda.read("data.csv", mode="large", chunksize=100_000)
data.head()  # Reads only the requested leading rows from disk.
data.eda.statsall()
data.eda.mlall(target="fraud_flag")
```

Large mode calculates the dataset dimensions exactly. Exploratory findings,
charts, profiles, comparisons, and ML guidance use a bounded sample and print
its size before running, so estimates are never presented as full-data results.
`statsall()`, `vizall()`, and `mlall()` use `sample=10_000` by default in large
mode; pass another positive integer to adjust that scope. In small mode those
methods use every loaded row by default, and accept the same `sample=` override.

## Analysis methods

### 1. `df.eda.statsall()` — Statistical profile

Prints scores, column roles, numeric and categorical statistics, missingness,
outliers, and preprocessing suggestions. VIF uses multivariate regression
in the standard install. Optional `noweda[ml]` dependencies add time-series
diagnostics. Constant columns and insufficient observations yield unavailable VIF.

### 2. `df.eda.mlall()` — Task-aware ML guidance

With no arguments, NowEDA assesses whether supervised or unsupervised analysis is
plausible. It presents possible target columns for review, excludes likely IDs,
and ranks appropriate unsupervised directions. It never silently selects a target.

```python
# Assess a dataset when you do not yet know the ML objective.
df.eda.mlall()

# Infer binary/multiclass classification or regression from the named target.
df.eda.mlall(target="segment")

# State the objective explicitly when you already know it.
df.eda.mlall(target="segment", problem_type="classification")
df.eda.mlall(problem_type="clustering", features=["age", "income"])

# Get the printed guidance and the structured result for use in code.
plan = df.eda.mlall(target="segment", plan=True)
```

**Supported problem types**

`classification`, `regression`, `clustering`, `anomaly_detection`, and
`dimensionality_reduction`. Binary and multiclass are classification subtypes.
Forecasting is planned separately because it needs a time column, horizon, and
time-aware validation.

**Supervised tasks**

Classification and regression require `target=`. If `problem_type` is omitted,
NowEDA infers one from the target dtype and cardinality, shows the reason, and lets
you override it. The target is excluded from features. It reports label coverage,
partial labels, small samples, imbalance, likely IDs, and potential leakage before
recommending a supervised workflow.

**Unsupervised tasks**

Clustering, anomaly detection, and dimensionality reduction do not accept a target.
Use `features=` to limit the analysis to columns available for that objective.
When no usable labels are available for a selected target, NowEDA explains that
supervised training cannot begin and presents unsupervised directions instead.

**Honest guidance**

Recommendations are task-specific candidates with preprocessing, validation,
metrics, and cautions. Stars and `/5` values are estimated dataset-fit ratings
within the selected task, not measured accuracy or expected performance. NowEDA
does not train models in this step. See the [ML guidance documentation](https://codewithdaniel1.github.io/NowEDA/ml-guidance/).

### 3. `df.eda.vizall()` — Automatic charts

Install `noweda[viz]` to draw distributions, correlations, categorical associations,
missingness, and applicable temporal plots. Unsupported or undefined categorical
associations appear as `N/A`, not zero association.

```python
df.eda.vizall()               # Small mode: use every loaded row
df.eda.vizall(sample=5_000)   # Use a bounded sample
```

Statistical reports use the complete DataFrame; visualization sampling affects
only charts.

### 4. `df.eda.profile_column("age")` — One-column profile

Inspect a column's distribution, missingness, outliers, and suggested transforms.

### 5. `df.eda.compare(other_df)` — Compare reports

Compare dimensions, inferred column roles, scores, and detected risks between
two DataFrames. This is not a formal statistical test for distribution drift.

## Advanced: tables and programmatic reports

Most notebook and pipeline code should start with `report()`. The individual
tables below are focused extracts for dashboards, tests, and custom workflows.

| Method | Output |
|---|---|
| `scores_df()` | `Value` column indexed by score name |
| `insights_df()` | `Insight` column |
| `schema_df()` | `Column`, `dtype`, `role`, `confidence`, `unique`, `uniqueness_ratio` |
| `stats_df()` | Per-column statistics; numeric fields include `mean`, `std`, `q25`, `median`, `q75`, `skewness`, `kurtosis` |
| `missing_df()` | Missing percentages; use `format="number"` for counts |
| `duplicates_df()` | Duplicate rows and constant columns |
| `correlation_df()` | Numeric Pearson correlation matrix |
| `outliers_df()` | IQR outlier counts; use `format="percentage"` for percentages |
| `pii_df()` | `Column`, `PII_Type`, `Count` |
| `encoding_df()` | `Column`, `Encoding_Type`; `include_confidence=True` adds sample evidence |
| `summary()` | Dictionary of raw plugin outputs |
| `report()` | `results`, `scores`, `insights`, `score_breakdown`, and `encoding_details` |

In 0.1.4, outlier penalties use the fraction of observed numeric values flagged
by the IQR rule: above 1% deducts 5 points; above 5% deducts 10 from quality and
readiness. `report()["score_breakdown"]` explains the score contributions.
Base64 detection requires at least six matches and an 80% sample match rate,
with extra evidence beyond simply being decodable. Its confidence field is the
sample match fraction, not a probability that data is encoded or malicious.

Column labels may be integers, strings, or tuples but must be unique. Empty
DataFrames produce an explanatory message; their scores are not informative.

Reports are cached. Each request fingerprints the DataFrame's values and schema,
recomputing analysis when they change. This check scans the data; cached access is
not constant-time. Use `df.eda.refresh()` to explicitly force a new report.

PII detection uses patterns and can miss sensitive values or flag false positives.
A risk score of zero means no configured pattern was detected, not that data is
safe to share. Quality and readiness scores are heuristics.

## File formats and optional dependencies

```python
df = eda.read("data.csv", dtype={"customer_id": str})
# eda.read("data.jsonl") defaults to lines=True
```

| Formats / feature | Installation |
|---|---|
| CSV, TSV, TXT, JSON/JSONL, XLSX/XLSM, XML, HTML, Stata, SAS, Pickle | `pip install noweda` |
| XLS, XLSB, ODS/ODF/ODT | `pip install "noweda[excel]"` |
| Parquet, Feather, ORC | Included with `pip install noweda` |
| HDF5 | `pip install "noweda[hdf]"` |
| SPSS | `pip install "noweda[spss]"` |
| Charts and KDE overlays | `pip install "noweda[viz]"` |
| Stationarity and seasonality dependencies | `pip install "noweda[ml]"` |
| All optional analysis and format dependencies | `pip install "noweda[full]"` |

Only read Pickle files from trusted sources, since loading them can execute code.

## Large files

Use explicit large mode for CSV, TSV, TXT, and Parquet files that should stay
on disk:

```python
data = eda.read("large.csv", mode="large", chunksize=100_000)
data.eda.report()
```

Large mode reports exact dimensions. Its exploratory methods print a clear
sample-size notice before returning sample-based estimates. `read_chunked()`
remains available for manual streaming of CSV and line-delimited JSON.

For data that does not fit in memory, iterate over CSV or line-delimited JSON:

```python
for chunk in eda.read_chunked("large.csv", chunksize=10000, concat=False):
    print(chunk.eda.missing_df())
```

The default `concat=True` retains all chunks and allocates a combined DataFrame;
it is appropriate only when the data and concatenation overhead fit in RAM.
Per-chunk scores describe each chunk, not the whole dataset.

If processing may stop early, wrap the generator in `contextlib.closing()`:

```python
from contextlib import closing

with closing(eda.read_chunked("large.csv", concat=False)) as chunks:
    for chunk in chunks:
        print(chunk.eda.missing_df())
        break  # Optional: the reader closes and the indicator reports stopped.
```

Running indicators show elapsed time; 100% appears only after completion.
Closing a stream early displays "Stopped before completion".

## Export and CLI

```python
from noweda.report.html import generate_html_report
from noweda.report.json import generate_json_report

generate_html_report(df.eda.report(), "report.html")
generate_json_report(df.eda.report(), "report.json")
```

```bash
noweda data.csv --html report.html --json report.json
```

JSON exports replace undefined/nonfinite statistics with `null`. Column labels
become strings; colliding labels raise an error before overwriting a file.
The in-memory report retains its original labels and numeric values.

## Development

```bash
git clone https://github.com/codewithdaniel1/NowEDA.git
cd NowEDA
pip install -e ".[test]"
python -m pytest tests/
```

NowEDA is alpha software, licensed under MIT. See the
[documentation](https://codewithdaniel1.github.io/NowEDA/) for plugins and examples.
