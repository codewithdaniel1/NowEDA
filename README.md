# NowEDA

**Automated Exploratory Data Analysis — built as a native pandas extension with Spark acceleration built in.**

NowEDA is a lightweight, modular Python framework that turns any dataset into instant insight. Load any file, call `df.noweda.*`, and get a full EDA report — including data quality scoring, PII detection, outlier analysis, and human-readable insights — with zero boilerplate.

---

## 5 Powerful Methods — Everything You Need for ML-Ready EDA

NowEDA provides 5 powerful methods that do **everything** an ML engineer needs before modeling:

`import noweda as eda`

### 1. `df.eda.statsall()` — Complete Statistical & ML Analysis Report

Prints a rich, colour-coded report with comprehensive data profiling:

**Data Profile:**
- **Scores** — data quality (0–100), risk level, model readiness (0–100)
- **Column analysis** — dtypes, inferred roles with confidence scores, diversity metrics
- **Column quality summary** — ✓ Good / ⚠ Check / ✗ Issue flags for each column
- **Numeric stats** — mean, std, quartiles, skewness, outlier counts
- **Categorical stats** — cardinality, balance (entropy-based diversity), top values with %
- **Temporal analysis** — auto-detects datetime columns, infers frequency (daily/weekly/monthly), tests stationarity, detects seasonality
- **Insights** — human-readable findings about data quality, patterns, issues

**Preprocessing Guidance:**
- **Multicollinearity detection** (VIF > 5) with code snippets
- **Scaling recommendations** with `StandardScaler` and `MinMaxScaler` code
- **Transformation suggestions** (log, sqrt for skewed data) with examples
- **Cardinality warnings** (high-cardinality categoricals)
- **Rare category detection** (<1% frequency)
- **Missing data strategy** (impute vs drop) with code examples

### 2. `df.eda.mlall()` — ML Algorithm Recommendations & Preprocessing Pipeline

Provides expert guidance on which algorithms to try first, with data-specific reasoning:

** Supervised Learning (Classification & Regression)**
- **Supervised algorithms** — Linear/Logistic Regression, Random Forest, Gradient Boosting (XGBoost), SVM, KNN, Neural Networks, Naive Bayes
- **Rating system** — ***** (1–5 stars) based on your data characteristics
- **Reasoning** — ✓ Why it fits + ✗ When to avoid + ⚠ Before fitting warnings
- **Class imbalance detection** — Alerts if target variable is imbalanced with actionable solutions (SMOTE, stratified split, class weights)

** Unsupervised Learning (Clustering & Dimensionality Reduction)**
- **Clustering** — K-Means, DBSCAN, Hierarchical Clustering with data-specific ratings
- **Reduction** — PCA, t-SNE/UMAP with dimensionality guidance
- **Anomaly Detection** — Isolation Forest for outlier scoring

** Multicollinearity Guidance**
- Detects high correlations (|r| > 0.85) and explains the problem
- Provides three solutions: (1) Drop correlated features, (2) Use PCA, (3) Use Ridge/Lasso

** Data Preprocessing Pipeline**
- Step-by-step instructions tailored to your dataset
- Code snippets for imputation, encoding, scaling, and handling outliers
- Handles special cases: high missingness, skewness, outliers, multicollinearity

### 3. `df.eda.vizall()` — Advanced Auto-Visualizations

Auto-renders 7+ types of publication-quality charts (no matplotlib code needed):

- **Histograms + KDE** — Numeric distributions with density overlays
- **Bar charts** — Top categorical values with percentage labels
- **Box plots** — Numeric distributions by categorical features (spot relationships)
- **Feature variance ranking** — Horizontal bar chart showing information content
- **Pair plots** — Top correlated feature pairs with regression lines
- **Categorical association heatmap** — Cramér's V strength between categorical columns
- **Missing data heatmap** — Correlation patterns in missingness (reveals if missingness is random or correlated)
- **Correlation heatmap** — Pearson correlations between all numeric columns
- **Time-series plots** — Line charts for datetime columns + numeric trends

**Smart Column Filtering:**

`vizall()` automatically excludes unhelpful columns from visualization:

- **ID columns** — customer_id, user_id, record_id, etc. (not useful for pattern analysis)
- **High-cardinality PII (string)** — email addresses, etc. (>95% unique strings)
- **Numeric PII** — credit_card, ssn, account_number, zip codes, etc. (no meaning as features)
- **Continuous numeric features** — account_balance, transaction_count, etc. (kept for analysis!)

This ensures visualizations focus on meaningful patterns, not noise from identifiers.

**For Large Datasets (>50K rows):**

By default, `vizall()` automatically samples 10,000 rows for visualization while keeping full-dataset analysis:

```python
# Automatic sampling for large files (no code change needed)
df.eda.vizall()  # 100K rows → renders charts from 10K sample

# Custom sample size
df.eda.vizall(sample=5_000)  # Use 5,000 rows for visualization

# No sampling (full dataset, may be slow)
df.eda.vizall(sample=False)  # Use all 100K rows (warning: slow on huge files)
```

**Note:** Sampling is only for visualization speed. All statistical analysis (`statsall()`, `scores_df()`, etc.) always uses the complete dataset.

### 4. `df.eda.profile_column(column_name)` — Deep Dive Into a Single Column

Detailed analysis of one column for understanding distributions and transformations:

- **Type information** — Inferred data type and role with confidence
- **Distribution analysis** — Detects if data is symmetric, moderately skewed, or highly skewed
- **Transformation recommendations** — Suggests log/sqrt transforms with statistical justification
- **Outlier explanation** — Quantifies outliers as % of data
- **Cardinality insights** — For categorical columns, shows all/top values with frequencies

Example:
```python
df.eda.profile_column('age')      # Deep dive into 'age' column
df.eda.profile_column('category') # Deep dive into 'category' column
```

### 5. `df.eda.compare(other_df)` — Compare Two Datasets

Detect schema drift and data distribution changes between datasets:

- **Dimension changes** — Rows and columns added/removed
- **Score regression** — Data quality, model readiness, and risk changes
- **Schema drift** — Columns added/removed, column roles changed
- **Risk changes** — New PII detected, risks added/removed

Example:
```python
df_train.eda.compare(df_test)     # Check for train/test drift
df_v1.eda.compare(df_v2)          # Detect schema changes between versions
```

---

## Optional: Quick-Access One-Liner Methods

For quick access to individual analysis tables during interactive exploration. All methods return **pandas DataFrames** for easy inspection and manipulation:

### Data Quality & Scores

#### `df.eda.scores_df()`
Returns a DataFrame with data quality, model readiness, and risk scores.

| Score | Range | Meaning |
|---|---|---|
| `data_quality` | 0–100 | Penalised for missing values, duplicates, constants, outliers |
| `model_readiness` | 0–100 | Penalised for skew, untyped columns, high missingness |
| `risk` | 0+ | Added per PII column (+15) and encoded column (+10) |

**Returns:** DataFrame with shape (3, 1), containing scores.

```python
scores = df.eda.scores_df()
print(scores)
#               Value
# data_quality    87.5
# model_readiness  82.1
# risk            15.0
```

---

#### `df.eda.insights_df(full_line=True)`
Returns a DataFrame with human-readable insights about data quality, patterns, and issues.

**Insights include:**
- **Identifier columns** — Detected customer_id, user_id, etc. columns
- **Low-cardinality categoricals** — first_name, last_name, region, etc.
- **Temporal columns** — Detected datetime columns with frequency/seasonality analysis
- **Missing data patterns** — Columns with missingness
- **PII warnings** — Email addresses, phone numbers, SSNs, credit cards found (actionable alerts)
- **Encoded data detection** — Base64 or other obfuscation
- **Data quality assessment** — Overall scoring and risk levels

**Parameters:**
- `full_line` (bool, default=True) — If True, shows complete insight text. If False, truncates at ~50 characters for compact display.

**Returns:** DataFrame with columns: `[Insight]`

**Example:**
```python
# Full insights (default)
insights = df.eda.insights_df()

# Output example:
# Likely identifier column(s) detected: customer_id. Consider dropping...
# Columns with low cardinality (likely categorical): first_name, last_name
# Datetime column(s) detected: last_transaction, signup_date. Temporal...
# PII detected in column 'email': 95 email address(es), 0 phone number(s). Mask...
# PII detected in column 'phone': 87 phone number(s). Mask or remove...
# High risk level (70). Sensitive data likely present.
# Data quality score is excellent (>=95).

# Compact insights (truncated for brief review)
insights = df.eda.insights_df(full_line=False)
```

---

### Schema & Column Information

#### `df.eda.schema_df()`
Returns a DataFrame with inferred column roles, types, and confidence scores.

**Returns:** DataFrame with columns: `[Column, Role, Type, Confidence]`

```python
schema = df.eda.schema_df()
print(schema)
#      Column       Role      Type  Confidence
# 0      user_id        id     int64        0.99
# 1      age      numeric   int64        0.95
# 2      name         text    object       0.88
# 3      signup_date  datetime  datetime64   0.98
```

---

#### `df.eda.stats_df()`
Returns a DataFrame with descriptive statistics (mean, std, quartiles, skewness, etc.) for numeric columns.

**Returns:** DataFrame with columns: `[Column, Mean, Std, Min, Q1, Median, Q3, Max, Skewness, Kurtosis]`

```python
stats = df.eda.stats_df()
```

---

### Missing Data & Duplicates

#### `df.eda.missing_df(format='percentage')`
Returns a DataFrame with missing data rates per column. Choose between percentages or counts.

**Parameters:**
- `format` (str, default='percentage') — Display format: `'percentage'` shows missing data as %, `'number'` shows absolute missing counts.

**Returns:** DataFrame with columns: `[Column, Missing_Data]`

```python
# Default: show missing data as percentage
missing = df.eda.missing_df()
print(missing)
#        Column Missing_Data
# 0         age          2.5%
# 1       salary         15.3%

# Alternative: show absolute missing counts
missing = df.eda.missing_df(format='number')
print(missing)
#        Column Missing_Data
# 0         age            50
# 1       salary          306
```

---

#### `df.eda.duplicates_df()`
Returns a DataFrame summarising duplicate rows and constant columns.

**Returns:** DataFrame with metrics: `[Total Rows, Duplicate Rows, Constant Columns]`

```python
duplicates = df.eda.duplicates_df()
print(duplicates)
#                   Metric  Value
# 0             Total Rows  2000
# 1        Duplicate Rows    15
# 2      Constant Columns     3
```

---

### Correlations & Outliers

#### `df.eda.correlation_df()`
Returns the Pearson correlation matrix between all numeric columns.

**Returns:** DataFrame — correlation matrix (symmetric, numeric columns × numeric columns)

```python
corr = df.eda.correlation_df()
print(corr)
#            age    salary    score
# age       1.00     0.65     0.42
# salary    0.65     1.00     0.78
# score     0.42     0.78     1.00
```

---

#### `df.eda.outliers_df(format='number')`
Returns outlier counts per numeric column. Choose between absolute counts or percentages.

**Parameters:**
- `format` (str, default='number') — Display format: `'number'` shows outlier counts, `'percentage'` shows as % of total rows.

**Returns:** DataFrame with columns: `[Column, Outliers]`

```python
# Default: show outlier counts
outliers = df.eda.outliers_df()
print(outliers)
#      Column  Outliers
# 0       age         12
# 1    salary         25

# Alternative: show outliers as percentage
outliers = df.eda.outliers_df(format='percentage')
print(outliers)
#      Column  Outliers
# 0       age      0.6%
# 1    salary      1.2%
```

---

### PII & Encoding Detection

#### `df.eda.pii_df()`
Returns a DataFrame listing all detected PII (Personally Identifiable Information) fields.

**Returns:** DataFrame with columns: `[Column, PII_Type, Count]`
- **Column**: Column name containing PII
- **PII_Type**: Type of PII detected ('email', 'phone', 'ssn', 'credit_card')
- **Count**: Number of values matching this PII pattern

```python
pii = df.eda.pii_df()
print(pii)
#        Column     PII_Type  Count
# 0  customer_email      email  95017
# 1  customer_phone      phone  89875
# 2  ssn_column           ssn  84941
```

---

#### `df.eda.encoding_df()`
Returns a DataFrame listing columns with detected encoding signals (Base64, obfuscation patterns).

**Returns:** DataFrame with columns: `[Column, Encoding_Type, Sample]`

```python
encoding = df.eda.encoding_df()
print(encoding)
#        Column       Encoding_Type               Sample
# 0  customer_id    base64_signal   aGVsbG8gd29ybGQ=
# 1  secret_key    obfuscation     xxxxxxxx1234xxxx
```

---

## Usage Summary

| Method | Returns | Key Parameters |
|---|---|---|
| `statsall()` | Prints comprehensive report | — |
| `mlall()` | Prints ML recommendations | — |
| `vizall()` | Displays visualizations | — |
| `profile_column(col)` | Prints single-column analysis | `col` — column name |
| `compare(other_df)` | Prints schema drift report | `other_df` — comparison dataset |
| `scores_df()` | DataFrame (3 rows) | — |
| `insights_df()` | DataFrame | `full_line` (default=True) |
| `schema_df()` | DataFrame | — |
| `stats_df()` | DataFrame | — |
| `missing_df()` | DataFrame | `format` (default='percentage') |
| `duplicates_df()` | DataFrame | — |
| `correlation_df()` | DataFrame | — |
| `outliers_df()` | DataFrame | `format` (default='number') |
| `pii_df()` | DataFrame | — |
| `encoding_df()` | DataFrame | — |

---

## Understanding the API

**The 5 Powerful Methods** are designed for comprehensive analysis in one call:
- `statsall()` — Complete statistical & ML analysis
- `mlall()` — Algorithm recommendations
- `vizall()` — Auto-visualizations
- `profile_column()` — Deep dive into one column
- `compare()` — Schema drift detection

**The 10 One-Liner Methods** are for interactive exploration and specific queries:
- Return DataFrames directly for easy inspection
- Support flexible formatting options (percentages, truncation, etc.)
- Designed for quick analysis without boilerplate
- Recommended for Jupyter notebooks and interactive workflows

---

## Features

| Feature | Description |
|---|---|
| **5 powerful methods** | `statsall()`, `mlall()`, `vizall()`, `profile_column()`, `compare()` |
| **10 convenience methods** | Quick access to individual tables for interactive exploration |
| Universal ingestion | CSV, Excel, JSON, XML, HTML, Parquet, Feather — 28 formats |
| Native pandas accessor | `df.eda.*` — feels like pandas |
| **ML-Ready Analysis** | Algorithm ratings, class imbalance detection, preprocessing pipeline with code |
| **Advanced Visualizations** | 9+ auto-chart types, missing data heatmap, time-series plots |
| **Confidence scores** | Data type inference confidence per column |
| **Column quality flags** | ✓ Good / ⚠ Check / ✗ Issue for every column |
| **Temporal analysis** | Datetime detection, frequency inference, stationarity & seasonality tests |
| Plugin architecture | Every analysis is a swappable plugin |
| Schema inference | Auto-detects IDs, categoricals, datetimes, text with confidence scores |
| Data quality scoring | 0–100 quality + model-readiness score + risk level |
| Multicollinearity detection | VIF calculation with actionable recommendations + code |
| Class imbalance detection | Identifies imbalanced classes, suggests SMOTE/class_weight |
| Correlation explanations | Explains why high correlations are problematic + solutions |
| PII detection | Email addresses + extensible patterns |
| Encoding detection | Base64 and obfuscation signals |
| Outlier detection | IQR-based, per numeric column with % quantification |
| Duplicate detection | Exact row duplicates + constant columns |
| Categorical diversity | Entropy-based balance metric per column |
| Actionable insights | Human-readable text, not just numbers |
| HTML report | Dark-themed, stakeholder-ready export |
| CLI | One-liner from the terminal |

---

## Installation

```bash
cd NowEDA
pip install -e .
```

> Requires Python 3.8+ and pandas 1.3+.

---

## Quick Start

```python
import noweda as eda

df = eda.read("data.csv")

# The 5 Powerful Methods — use these for comprehensive analysis
df.eda.statsall()                    # Complete statistical analysis report
df.eda.mlall()                       # ML algorithm recommendations & preprocessing
df.eda.vizall()                      # Auto-render 9+ visualizations
df.eda.profile_column('age')         # Deep dive into a single column
df.eda.compare(df_test)              # Detect schema drift between datasets

# Optional: Quick-access one-liner methods for interactive exploration
scores = df.eda.scores_df()          # Returns DataFrame with quality/readiness/risk scores
insights = df.eda.insights_df()      # Returns DataFrame with insights
schema = df.eda.schema_df()          # Returns DataFrame with column roles & types
stats = df.eda.stats_df()            # Returns DataFrame with statistics
missing = df.eda.missing_df()        # Returns DataFrame with missing data (default: %)
missing = df.eda.missing_df(format='number')  # Show missing counts instead
duplicates = df.eda.duplicates_df()  # Returns DataFrame with duplicate info
corr = df.eda.correlation_df()       # Returns correlation matrix
outliers = df.eda.outliers_df()      # Returns DataFrame with outlier counts (default: #)
outliers = df.eda.outliers_df(format='percentage')  # Show outliers as %
pii = df.eda.pii_df()                # Returns DataFrame with PII findings
encoding = df.eda.encoding_df()      # Returns DataFrame with encoding signals
```

### All Supported Formats

NowEDA supports 28 file formats through pandas, with Spark acceleration built in for large CSV, TSV, TXT, Parquet, and ORC files:

```python
import noweda as eda

df = eda.read("data.csv")               # CSV
df = eda.read("data.xlsx")              # Excel (.xlsx, .xls, .ods)
df = eda.read("data.json")              # JSON
df = eda.read("data.xml")               # XML
df = eda.read("data.html")              # HTML tables
df = eda.read("data.parquet")           # Parquet
df = eda.read("data.feather")           # Feather
df = eda.read("data.h5")                # HDF5
df = eda.read("data.sav")               # SPSS (.sav, .zsav)
```

Any `**kwargs` are forwarded to the underlying pandas reader:

```python
df = eda.read("data.xlsx", sheet_name="Sheet1")
df = eda.read("data.json", orient="records")
```

For very large Spark-friendly files, Spark-backed loading happens automatically:

```python
df = eda.read("big.csv")      # Uses Spark automatically for large supported files
df = eda.read("big.parquet")  # Also handled automatically
```

Spark ships with NowEDA, so no extra install flag is needed.

When a method is running, NowEDA shows a traditional percentage progress line in notebooks and the CLI so the user can see what is happening.

### Chunked Ingestion for Large Files

For files larger than available RAM, use `read_chunked()` to process data in chunks. This is essential for datasets that would otherwise exhaust system memory.
When a chunked file is Spark-friendly and large enough, NowEDA will use Spark automatically before yielding pandas chunks.

#### Mode 1: Concatenate All Chunks (Simpler)

Read the entire file in chunks, but automatically concatenate into a single DataFrame:

```python
# Read a 10 GB CSV file in 100k-row chunks, return single concatenated DataFrame
df = eda.read_chunked("huge_file.csv", chunksize=100_000)
print(df.shape)  # Full dataset shape

# Run NowEDA analysis on the complete concatenated result
scores = df.eda.scores_df()
pii = df.eda.pii_df()
```

**Use this when:** Your data fits in RAM after being read, but you want to avoid holding the entire file on disk in memory during the read operation.

#### Mode 2: Process Chunks One-at-a-Time (Memory-Efficient)

Iterate through chunks without concatenating—only one chunk in memory at a time:

```python
# Process a 1 TB JSON file in 50k-row chunks—never load more than 50k rows
for chunk in eda.read_chunked("massive_file.json", chunksize=50_000, concat=False):
    print(f"Processing {chunk.shape[0]} rows...")
    
    # Analyze this chunk
    chunk_pii = chunk.eda.pii_df()
    chunk_scores = chunk.eda.scores_df()
    
    # Save results, upload, or aggregate
    process_and_save(chunk_pii)
```

**Use this when:** Your final dataset is too large to fit in RAM, or you need to process data in a streaming fashion (e.g., aggregate results across chunks).

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `file_path` | — | Path to `.csv`, `.tsv`, `.json`, `.jsonl`, or `.txt` file |
| `chunksize` | `10_000` | Number of rows per chunk. Larger chunks = fewer iterations but more memory |
| `concat` | `True` | If `True`, concatenate all chunks into one DataFrame. If `False`, return a generator |
| `**kwargs` | — | Forward to pandas reader (`sep='\|'` for custom CSV delimiters, `lines=True` for JSONL, etc.) |

#### Supported Formats

- ✓ `.csv`, `.tsv`, `.tab`, `.txt` — Delimited text
- ✓ `.json`, `.jsonl` — JSON and newline-delimited JSON

Other formats (Excel, Parquet, XML, HDF5, etc.) must be read with the standard `read()` function.

#### Examples

```python
import noweda as eda
import pandas as pd

# Example 1: Read a large CSV with custom delimiter
df = eda.read_chunked("data.csv", chunksize=50_000, sep=";")

# Example 2: Aggregate stats across chunks (memory-efficient)
all_stats = []
for chunk in eda.read_chunked("huge_file.csv", chunksize=20_000, concat=False):
    stats = chunk.eda.stats_df()
    all_stats.append(stats)
combined_stats = pd.concat(all_stats)

# Example 3: Detect PII across large JSON dataset
for i, chunk in enumerate(eda.read_chunked("large.jsonl", chunksize=100_000, concat=False)):
    pii = chunk.eda.pii_df()
    if len(pii) > 0:
        print(f"Chunk {i}: Found {len(pii)} PII instances")
        pii.to_csv(f"pii_chunk_{i}.csv", index=False)

# Example 4: Process TSV with missing value handling
df = eda.read_chunked("data.tsv", chunksize=50_000, na_values=['NA', 'N/A', ''])

# Example 5: Stream large JSON Lines file for real-time processing
total_records = 0
for chunk in eda.read_chunked("events.jsonl", chunksize=10_000, concat=False):
    total_records += len(chunk)
    # Process chunk: save to database, filter, aggregate, etc.
    scores = chunk.eda.scores_df()
    print(f"Processed {total_records:,} records...")
```

#### Choosing the Right Mode

| Scenario | Method | Rationale |
|----------|--------|-----------|
| File fits in RAM after loading | `read_chunked(..., concat=True)` | Simpler API, faster analysis, data fits memory |
| File is larger than RAM | `read_chunked(..., concat=False)` | Essential for truly large files; prevents memory overflow |
| Need to save results per-chunk | `concat=False` + loop | Export, upload, or persist chunks before next one |
| Analyzing for quality metrics | `concat=True` | Get full dataset stats in one call |
| Real-time stream processing | `concat=False` + loop | Process and discard chunks incrementally |
| Detecting PII across dataset | Either | Both work; `concat=False` is better for huge files |

#### Memory Usage Tips

- **Chunk size matters**: Larger chunks = fewer loops but more memory per iteration
  - Start with 10,000-50,000 rows depending on column count and data types
  - For very wide tables (100+ columns), use smaller chunks
  - For simple numeric data, use larger chunks (100,000+)

- **Monitor memory**: On a machine with 16 GB RAM, you can typically hold 5-10 million rows

- **Custom parameters matter**: `dtype={'col': 'int32'}` or `na_values` can reduce memory usage
  ```python
  df = eda.read_chunked("data.csv", chunksize=50_000, dtype={'id': 'int32', 'category': 'category'})
  ```

#### When to Use `read_chunked()` vs `read()`

Use `read_chunked()` when:
- File is > 500 MB
- Available RAM is < 3× file size
- You need to process data in batches
- You're doing streaming/real-time analysis

Use standard `read()` when:
- File is < 100 MB
- RAM available is > 10× file size
- You need fast analysis of the complete dataset
- You're using formats like Excel, Parquet, XML, HDF5 (which don't support chunking)

---

## How-To Guide: Chunked Ingestion

### Real-World Scenario: Processing Customer Data (100,000+ rows)

**Problem:** You have a `customer_data.csv` (19.5 MB, 100,000 rows) that takes 15 seconds to load all at once. You want to detect PII, check data quality, and save findings without waiting.

**Solution: Stream with Chunked Ingestion**

```python
import noweda as eda
import pandas as pd

FILE = 'customer_data.csv'

# Method 1: Quick Quality Check (load all, analyze once)
print("Loading customer data...")
df = eda.read_chunked(FILE, chunksize=25_000)  # Reads in 4 chunks, returns 1 DataFrame

scores = df.eda.scores_df()
pii = df.eda.pii_df()
print(f"Data Quality Score: {scores.loc['data_quality', 'Value']:.1f}")
print(f"PII Detected: {len(pii)} instances across {pii['Column'].nunique()} columns")
```

```python
# Method 2: Chunk-by-chunk Analysis (for very large files or real-time processing)
print("Streaming customer data...")

all_pii = []
chunk_count = 0

for chunk in eda.read_chunked(FILE, chunksize=25_000, concat=False):
    chunk_count += 1
    
    # Analyze this chunk
    pii = chunk.eda.pii_df()
    if len(pii) > 0:
        pii['chunk_num'] = chunk_count
        all_pii.append(pii)
    
    # Example: Save chunk analysis
    chunk.eda.scores_df().to_csv(f'scores_chunk_{chunk_count}.csv')
    
    print(f"✓ Processed chunk {chunk_count} ({len(chunk):,} rows)")

# Combine all PII findings
if all_pii:
    combined_pii = pd.concat(all_pii, ignore_index=True)
    print(f"\n✓ Found {combined_pii['Count'].sum():,} PII instances across {chunk_count} chunks")
    combined_pii.to_csv('all_pii_findings.csv', index=False)
```

### Common Patterns

**Pattern 1: Batch Processing**
```python
# Export one CSV per chunk
for i, chunk in enumerate(eda.read_chunked('data.csv', chunksize=50_000, concat=False), 1):
    chunk.to_csv(f'batch_{i:03d}.csv', index=False)
    print(f"Saved batch {i}")
```

**Pattern 2: Aggregating Metrics**
```python
# Collect statistics across all chunks
stats_list = []
for chunk in eda.read_chunked('data.csv', chunksize=100_000, concat=False):
    stats_list.append(chunk.eda.stats_df())

combined_stats = pd.concat(stats_list).groupby(level=0).mean()
print(combined_stats)
```

**Pattern 3: Filtering & Saving**
```python
# Find and export rows matching certain criteria
for chunk in eda.read_chunked('data.csv', chunksize=50_000, concat=False):
    # Filter: keep only high-risk records
    pii = chunk.eda.pii_df()
    high_risk = chunk[chunk['risk_score'] > 75]
    
    if len(high_risk) > 0:
        high_risk.to_csv('high_risk_records.csv', mode='a', index=False)
```

**Pattern 4: Real-Time Monitoring**
```python
# Process as-it-arrives with early stopping
for i, chunk in enumerate(eda.read_chunked('data.csv', chunksize=10_000, concat=False)):
    pii = chunk.eda.pii_df()
    
    # Alert if PII found
    if len(pii) > 0:
        print(f"⚠️  ALERT: PII detected in chunk {i+1}")
        print(pii)
    
    # Stop early if we find enough issues
    if i > 100:  # Only scan first 100 chunks for speed
        print("Stopping after 100 chunks for quick review")
        break
```

### Quick Reference: Chunked Ingestion

**30-second summary:**

```python
# Load large file, return single DataFrame
df = eda.read_chunked("huge_file.csv", chunksize=50_000)

# Or stream chunks one at a time (memory efficient)
for chunk in eda.read_chunked("huge_file.csv", chunksize=50_000, concat=False):
    pii = chunk.eda.pii_df()  # Analyze each chunk
```

| Question | Answer |
|----------|--------|
| File fits in RAM? | Use `concat=True` (default) — get single DataFrame |
| File > RAM? | Use `concat=False` — iterate chunks, minimal memory |
| Supported formats? | CSV, TSV, JSON, JSONL, TXT. Not: Excel, Parquet, XML, HDF5 |
| Memory concerns? | See "Memory Usage Tips" section above |
| Detect PII large file? | See "Common Patterns" → "Pattern 3" above |
| Need help? | See "How-To Guide: Chunked Ingestion" section above |

---

## CLI

```bash
# Print insights and scores to the terminal
noweda data.csv

# Export a dark-themed HTML report
noweda data.csv --html report.html

# Export a JSON report
noweda data.csv --json report.json

# Both at once
noweda data.csv --html report.html --json report.json
```

---

## Plugin System

Every analysis step is an independent plugin. You can run all plugins together (default), or cherry-pick specific plugins for custom analysis workflows. This is useful when you want lightweight analysis without running the full suite.

### Built-in Plugins

| Plugin | Returns | Key Use Case |
|---|---|---|
| `SchemaPlugin` | Column roles & types with confidence scores | Understand data structure and inferred column purposes |
| `StatsPlugin` | Descriptive statistics per column | Analyze distributions, skewness, quartiles |
| `MissingDataPlugin` | Missing value rates | Assess data completeness |
| `DuplicatesPlugin` | Duplicate rows & constant columns | Identify redundancy and zero-variance features |
| `CorrelationPlugin` | Pearson correlation matrix | Detect multicollinearity and feature relationships |
| `OutlierPlugin` | IQR-based outlier counts | Identify anomalous values per column |
| `PIIDetectorPlugin` | PII column findings | Flag columns containing sensitive data (emails, phone, SSN, credit cards) |
| `EncodingDetectionPlugin` | Encoding signals | Detect Base64 or obfuscated data |

### Running All Plugins (Default)

When you call the 5 powerful methods (`statsall()`, `mlall()`, etc.), all 8 plugins run automatically:

```python
import noweda as eda

df = eda.read("data.csv")

# This runs ALL plugins internally
df.eda.statsall()  # Reports include schema, stats, missing, duplicates, correlation, outliers, pii, encoding
```

### Running Specific Plugins Only

Run a subset of plugins for lighter-weight analysis:

```python
from noweda.core.engine import AutoEDAEngine
from noweda.plugins.missing import MissingDataPlugin
from noweda.plugins.pii import PIIDetectorPlugin
from noweda.plugins.outliers import OutlierPlugin

# Example 1: Check only for data quality issues (missing + duplicates + outliers)
engine = AutoEDAEngine([
    MissingDataPlugin(),
    OutlierPlugin(),
])
report = engine.run_df(df)

# Access results
missing_data = report['results']['missing']    # Dict: {col_name: missing_rate}
outlier_counts = report['results']['outliers']  # Dict: {col_name: count}

# Example 2: Check only for security/privacy issues (PII + encoding)
from noweda.plugins.encoding import EncodingDetectionPlugin

engine = AutoEDAEngine([
    PIIDetectorPlugin(),
    EncodingDetectionPlugin(),
])
report = engine.run_df(df)

pii_findings = report['results']['pii']           # Dict: {col_name: pii_type}
encoding_signals = report['results']['encoding']  # Dict: {col_name: encoding_type}

# Example 3: Schema + Stats only (lightweight column analysis)
from noweda.plugins.schema import SchemaPlugin
from noweda.plugins.stats import StatsPlugin

engine = AutoEDAEngine([
    SchemaPlugin(),
    StatsPlugin(),
])
report = engine.run_df(df)

schema_info = report['results']['schema']  # Dict: {col_name: {dtype, role, confidence}}
stats_info = report['results']['stats']    # Dict: {col_name: {mean, std, min, max, ...}}

print("Column Roles:")
for col, info in schema_info.items():
    print(f"  {col}: {info['role']} (confidence: {info['confidence']:.2f})")

print("\nMissing Rates:")
for col, rate in missing_data.items():
    if rate > 0:
        print(f"  {col}: {rate*100:.1f}%")
```

### Understanding Plugin Output

Each plugin returns a dictionary under `report['results'][plugin_name]`:

#### SchemaPlugin Output
```python
{
    'user_id': {'dtype': 'int64', 'role': 'id', 'confidence': 0.99, 'unique': 1000, 'uniqueness_ratio': 1.0},
    'email': {'dtype': 'object', 'role': 'text', 'confidence': 0.85, 'unique': 998, 'uniqueness_ratio': 0.998},
    'age': {'dtype': 'int64', 'role': 'numeric', 'confidence': 0.98, 'unique': 60, 'uniqueness_ratio': 0.06}
}
```

**Key Fields:**
- `role`: Detected column type (id, numeric, categorical, datetime, text)
- `confidence`: How confident (0-1) the schema detection is
- `unique`: Count of unique values
- `uniqueness_ratio`: Unique / total rows

#### StatsPlugin Output
```python
{
    'age': {
        'dtype': 'int64',
        'count': 1000,
        'missing': 5,
        'unique': 60,
        'mean': 35.2,
        'median': 34.0,
        'std': 12.5,
        'min': 18.0,
        'max': 85.0,
        'q25': 28.0,
        'q75': 42.0,
        'skewness': 0.15
    }
}
```

#### MissingDataPlugin Output
```python
{
    'user_id': 0.0,
    'age': 0.005,  # 0.5% missing
    'salary': 0.02  # 2% missing
}
```

#### OutlierPlugin Output
```python
{
    'age': 3,      # 3 outliers detected
    'salary': 15,  # 15 outliers detected
    'score': 0     # No outliers
}
```

#### PIIDetectorPlugin Output
```python
{
    'customer_email': {'email': 42, 'phone': 3},  # Multiple PII types per column
    'ssn_field': {'ssn': 50},
    'card_number': {'credit_card': 28},  # Luhn-validated
    'notes': {'email': 5, 'phone': 2}
}
```

NowEDA detects four types of PII:
- **Email**: `user@example.com`
- **Phone**: `(555) 123-4567`, `555-123-4567`, `+1-800-555-1234`, etc. (US/intl formats)
- **SSN**: `123-45-6789` or `123 45 6789` format
- **Credit Card**: Visa, Mastercard, Amex, Discover with Luhn algorithm validation to reduce false positives

### Writing a Custom Plugin

Build your own plugin to add custom analysis:

```python
from noweda.plugins.base import BasePlugin
from noweda.core.engine import AutoEDAEngine

class CustomQualityPlugin(BasePlugin):
    """Example: Flag columns with low cardinality."""
    name = "custom_quality"

    def run(self, df):
        results = {}
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.05:  # Less than 5% unique values
                results[col] = f"Low cardinality ({unique_ratio:.1%})"
        return results

# Use your custom plugin
custom_plugin = CustomQualityPlugin()
engine = AutoEDAEngine([custom_plugin])
report = engine.run_df(df)

print(report['results']['custom_quality'])
```

### Combining Plugins

Mix and match built-in and custom plugins:

```python
from noweda.plugins.schema import SchemaPlugin
from noweda.plugins.missing import MissingDataPlugin

engine = AutoEDAEngine([
    SchemaPlugin(),
    MissingDataPlugin(),
    CustomQualityPlugin(),  # Your custom plugin
])
report = engine.run_df(df)
```

---

## HTML Report

```bash
noweda examples/sample.csv --html report.html
```

The report includes:

- Score cards (quality, risk, model readiness)
- Actionable insights list
- Column schema table with inferred roles
- Missing value bars
- Duplicate and constant column summary
- Outlier counts
- PII findings (highlighted)
- Encoding signals (highlighted)

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Project Structure

```
NowEDA/
├── noweda/
│   ├── __init__.py               # exposes noweda.read() and noweda.read_chunked()
│   ├── io.py                     # file ingestion (all formats)
│   ├── accessor.py               # df.eda.* pandas accessor (5 methods)
│   ├── ml_utils.py               # ML utility functions (VIF, cardinality, etc.)
│   ├── ml_recommendations.py     # Algorithm recommendations + preprocessing pipeline
│   ├── temporal_utils.py         # Temporal analysis (datetime, stationarity, seasonality)
│   ├── core/
│   │   └── engine.py             # orchestrates plugins → scorer → insights
│   ├── plugins/
│   │   ├── base.py
│   │   ├── schema.py             # with confidence scores
│   │   ├── stats.py
│   │   ├── missing.py
│   │   ├── duplicates.py
│   │   ├── correlation.py
│   │   ├── outliers.py
│   │   ├── pii.py
│   │   └── encoding.py
│   ├── scoring/
│   │   └── scorer.py
│   ├── insights/
│   │   └── generator.py
│   ├── report/
│   │   └── html.py
│   └── cli.py
├── examples/
│   └── sample.csv
├── tests/
│   ├── test_basic.py
│   └── test_formats.py
└── pyproject.toml
```

---

## Roadmap

**Completed:**
- [x] Visualization layer (9+ chart types including missing data heatmap)
- [x] ML algorithm recommendations with data-specific ratings
- [x] Class imbalance detection and guidance
- [x] Temporal data analysis (stationarity, seasonality)
- [x] Dataset comparison (drift detection)
- [x] Column-level quality summaries
- [x] Code snippets in preprocessing pipeline
- [x] Additional PII patterns (phone, SSN, credit card with Luhn validation)
- [x] Streaming / chunked ingestion for large files (`read_chunked()`)

**Coming Soon:**
- [ ] PyPI publish
- [ ] Feature interaction detection
- [ ] Correlation explanation for numeric features

---

## Author

**Daniel Peng** — [danielpeng95@gmail.com]
