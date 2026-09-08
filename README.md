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

print(df.eda.scores_df())
print(df.eda.missing_df())
df.eda.statsall()
```

## Analysis methods

### 1. `df.eda.statsall()` — Statistical profile

Prints scores, column roles, numeric and categorical statistics, missingness,
outliers, and preprocessing suggestions. VIF uses multivariate regression
in the standard install. Optional `noweda[ml]` dependencies add time-series
diagnostics. Constant columns and insufficient observations yield unavailable VIF.

### 2. `df.eda.mlall()` — ML recommendations and preprocessing guidance

Heuristic recommendations help choose starting points for experimentation.
Ratings are not measured model performance or guarantees of model readiness.

**Supervised Learning (Classification & Regression)**

- Baseline and alternative algorithms with reasons and preprocessing guidance.
- For classification, use `df.eda.mlall(target="segment")` to check class balance.
  The target is excluded from feature recommendations. Class imbalance is flagged
  when the most frequent observed class has more than twice the count of the least
  frequent observed class. Without a target, class balance is not assessed.
- Numeric targets are treated as class labels when supplied; omit `target` for
  regression guidance without a class-balance check.

**Unsupervised Learning (Clustering & Dimensionality Reduction)**

- Suggestions for clustering, dimensionality reduction, and anomaly detection.

**Multicollinearity Guidance**

- High-correlation warnings and suggestions for feature selection or regularization.

**Data Preprocessing Pipeline**

- Suggested steps for missing values, encoding, scaling, and outliers.
- Example snippets are guidance; adapt them to your train/test split and task.

### 3. `df.eda.vizall()` — Automatic charts

Install `noweda[viz]` to draw distributions, correlations, categorical associations,
missingness, and applicable temporal plots. Unsupported or undefined categorical
associations appear as `N/A`, not zero association.

```python
df.eda.vizall()              # Samples 10,000 rows when the input exceeds 50,000
# df.eda.vizall(sample=5000)  # Choose a sample size
# df.eda.vizall(sample=False) # Plot the complete dataset
```

Statistical reports use the complete DataFrame; visualization sampling affects
only charts.

### 4. `df.eda.profile_column("age")` — One-column profile

Inspect a column's distribution, missingness, outliers, and suggested transforms.

### 5. `df.eda.compare(other_df)` — Compare reports

Compare dimensions, inferred column roles, scores, and detected risks between
two DataFrames. This is not a formal statistical test for distribution drift.

## Tables and programmatic reports

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
| Parquet, Feather, ORC | `pip install "noweda[parquet]"` |
| HDF5 | `pip install "noweda[hdf]"` |
| SPSS | `pip install "noweda[spss]"` |
| Charts and KDE overlays | `pip install "noweda[viz]"` |
| Stationarity and seasonality dependencies | `pip install "noweda[ml]"` |
| All optional analysis and format dependencies | `pip install "noweda[full]"` |

Only read Pickle files from trusted sources, since loading them can execute code.

## Large files

CSV, JSON, and all chunked reads use pandas consistently regardless of file size.
PySpark remains a standard dependency in 0.1.4. Large Parquet/ORC files (at least
128 MB) without reader options may use Spark, with a pandas fallback on failure.
Spark requires a compatible Java installation. Reads with options use pandas.
Spark loading still collects the final pandas DataFrame into local memory and
is not guaranteed to be faster.

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
