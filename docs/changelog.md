# Changelog

All notable changes to NowEDA are documented here.

---

## [0.2.1] — Unreleased

### Fixed

- `statsall()` now shows identifier, PII, and temporal-field guidance even on
  small datasets where high cardinality alone would not trigger a warning.
- ML guidance uses plain-language separators in terminal and notebook output,
  such as `K-Modes or K-Prototypes` and `5,000 of 5,000` labels.
- ML guidance now uses inferred datetime roles for time-aware validation advice,
  avoiding false positives from names such as `monthly_income`.

### Changed

- `eda.read(..., mode="large")` now provides a disk-backed CSV, TSV, TXT, and
  Parquet workflow with `head()`, `statsall()`, `vizall()`, `mlall()`,
  `profile_column()`, `compare()`, and `report()`. Large-mode analysis defaults
  to `sample=10_000` and clearly labels sample-based findings; small mode keeps
  full-input analysis by default and accepts an optional `sample=` override.
- Documentation now leads with the core `read()`, `statsall()`, `vizall()`,
  `mlall()`, `profile_column()`, `compare()`, and `report()` workflow; table
  extracts, plugins, scoring internals, and extension points are grouped as
  advanced material.
- Generic `statsall()` preprocessing guidance now excludes binary indicators
  and likely identifiers, avoiding recommendations to scale probable targets
  such as `fraud_flag`.
- Documentation distinguishes standard readers from formats that need an
  optional dependency extra.
- Score summaries now use plain-language labels such as `79 out of 100`.
- `statsall()` now keeps its temporal, plugin, and ML-preparation sections
  visible in a fixed order, using concise summaries and explicit empty states.
- `statsall()` now lists every scaling recommendation instead of shortening the
  list, and the playground renders `insights_df()` as a left-aligned table.
- Replaced the playground dataset with a deterministic 100,000-row, 32-column
  synthetic dataset that demonstrates the current profiling, quality, privacy,
  encoding, and ML-guidance capabilities without using real personal data.
- Updated package license metadata to the current PyPA format.

---

## [0.2.0] — 2026-09-12

### Added

- `mlall()` with no arguments now assesses possible target candidates, supervised
  and unsupervised readiness, likely identifiers, usable features, and ranked
  unsupervised directions without selecting a target automatically.
- Named targets now report label coverage and readiness. Targets with no observed
  labels explain why supervised training cannot begin and show unsupervised
  directions instead.
- Added temporal-feature warnings for supervised guidance and explicit cautions
  that anomalies are not automatically business or security outcomes.

### Changed

- `mlall(..., plan=True)` now returns the structured guidance after printing it.
- Removed the public `ml_plan()` accessor; `mlall()` is the single public ML
  guidance method.
- `statsall()` now gives high-uniqueness guidance based on each field's inferred
  role, so continuous measures, dates, PII, and free text are not presented as
  ordinary categorical features.

### Fixed

- Automatic `mlall()` assessment supports mixed column labels, including tuple
  labels, on pandas 1.3.

---

## [0.1.5] — 2026-09-12

### Added

- Added `ml_plan(target=None, problem_type=None, features=None)` for structured,
  task-aware ML guidance and expanded `mlall()` to accept the same arguments.
- Added classification, regression, clustering, anomaly-detection, and
  dimensionality-reduction plans with task-specific preprocessing, validation,
  metrics, and cautions.
- Added target validation, binary/multiclass subtype detection, transparent
  classification/regression inference, explicit feature selection, and warnings
  for missing labels, small samples or classes, class imbalance, likely
  identifiers, possible target leakage, and ambiguous targets.

### Changed

- `mlall()` without an objective now lists the supported problem types instead of
  presenting a mixed ranking of supervised and unsupervised algorithms.
- ML stars and `/5` scores now represent estimated dataset fit within the selected
  task and are clearly separated from measured model performance. NowEDA does not
  fit models in this workflow.
- Documented the ML objective contract throughout the README, website, API reference,
  FAQ, examples, and playground. Forecasting remains separate until NowEDA can accept
  time columns, horizons, series identifiers, and time-aware validation.

---

## [0.1.4] — 2026-09-08

### Fixed

- Chunk readers close on exhaustion, explicit closure and errors. Loading indicators stop on generator closure or keyboard interruption; running tasks show elapsed time instead of an artificial percentage, and only completed operations show 100%.
- Updated the playground for pandas chunking, full-stream memory comparisons, measured DataFrame sizes, 0.1.4 evidence fields and strict JSON export. Removed obsolete saved outputs.
- PII detection recognizes supported card numbers with spaces or hyphens and checks their Luhn checksum. Counts represent matching cells, even with duplicate row indices; separate phone numbers in a card-containing cell remain detectable.
- VIF uses NumPy multivariate least squares with an intercept in every installation. Constant columns and insufficient complete observations return NaN; exact estimable collinearity returns infinity. Large finite VIF values are no longer rounded to infinity.
- Statistical reports support integer and tuple column labels and explain empty inputs. Duplicate column labels fail with a clear error instead of silently losing results.
- Base64 detection rejects short ordinary words such as `John`, validates encoding strictly, and requires additional evidence plus at least six matches and an 80% match rate in the first 20 nonmissing values.
- CLI JSON exports map undefined/nonfinite numbers to `null` and enforce strict JSON. The new `generate_json_report()` helper provides the same behavior in Python without changing the in-memory report.

### Changed

- Outlier penalties use the fraction of observed numeric cells flagged as IQR outliers: more than 1% deducts 5 points; more than 5% deducts 10 points from quality and readiness. Scores may differ from 0.1.3 for the same data.
- Reports add `score_breakdown` (including clamping adjustments) and `encoding_details` (sample size, matches and empirical confidence). Existing plugin output shapes remain unchanged.
- `encoding_df(include_confidence=True)` exposes sample evidence. HTML reports display score contributions and Base64 sample counts.
- JSON object keys use string column labels; ambiguous label conversions are rejected before writing.

---

## [0.1.3] — 2026-09-08

### Fixed

- PII, encoding, schema and categorical analysis support pandas string and categorical dtypes, including pandas 3 defaults.
- Cached reports refresh after DataFrame values or schema change; `refresh()` forces a new report.
- Cramér's V uses paired observations and the Pearson chi-square formula; undefined heatmap entries display N/A.
- Nullable numeric statistics no longer crash on undefined reductions; kurtosis is included.
- JSONL files default to line-delimited parsing.
- Restored `summary()` and exposed `__version__`.
- Corrected README bold labels, installation instructions, table schemas and image URLs.

### Changed

- `mlall(target="label")` checks the named classification target only. No target means no class-balance assessment. Imbalance means a largest/smallest observed class count ratio greater than 2.
- CSV/JSON and all chunked reads use pandas for consistent parsing. Automatic Spark routing is limited to large Parquet/ORC files without reader options.
- Added `excel`, `viz`, `ml`, and `test` extras; `full` includes every optional runtime feature.
- Release builds require tests and package checks before publication.

---

## [0.1.2] — 2026-04-20

### Changed

- Spark-backed loading is now part of the standard install, with automatic routing for large supported files
- `read_chunked()` now uses Spark automatically for large CSV/JSON files when appropriate
- Notebook, CLI, and accessor methods show a loading indicator while analysis is running
- GitHub Actions now publishes to PyPI automatically when the package version in `pyproject.toml` is new

---

## [0.1.0] — 2026-04-16

### Added

**Core**
- `noweda.read()` — unified file loading API supporting 28 file extensions
- `df.noweda` — pandas accessor for zero-boilerplate EDA
- `AutoEDAEngine` — modular engine orchestrating plugins → scorer → insights
- Lazy evaluation with per-accessor result caching

**Format support**
- `.csv`, `.tsv`, `.tab`, `.txt` — delimited text (auto-detects tab separator)
- `.xlsx`, `.xls`, `.xlsm`, `.xlsb`, `.ods`, `.odf`, `.odt` — spreadsheets
- `.json`, `.jsonl` — JSON and JSON Lines
- `.xml` — XML
- `.html`, `.htm` — HTML tables
- `.parquet`, `.feather`, `.orc` — columnar formats (optional: `pip install "noweda[parquet]"`)
- `.h5`, `.hdf`, `.hdf5` — HDF5 (optional: `pip install "noweda[hdf]"`)
- `.dta` — Stata
- `.sas7bdat`, `.xpt` — SAS
- `.sav`, `.zsav` — SPSS (optional: `pip install "noweda[spss]"`)
- `.pkl`, `.pickle` — Python pickle
- File extension detection is case-insensitive (`.CSV` works the same as `.csv`)

**Plugins (8 built-in)**
- `SchemaPlugin` — column role inference (id, categorical, numeric, datetime, text)
- `StatsPlugin` — descriptive statistics including skewness
- `MissingDataPlugin` — per-column missing rates
- `DuplicatesPlugin` — duplicate rows and constant columns
- `CorrelationPlugin` — Pearson correlation matrix
- `OutlierPlugin` — IQR-based outlier detection
- `PIIDetectorPlugin` — email address detection
- `EncodingDetectionPlugin` — Base64 encoding signal detection

**Scoring**
- `data_quality` score (0–100)
- `risk` score (0+)
- `model_readiness` score (0–100)

**Insights**
- 9 insight categories covering all plugin outputs and scores
- Human-readable, actionable text for every finding

**Reporting**
- Dark-themed, self-contained HTML report with score cards, tables, and visual bars
- JSON export
- CLI: `noweda <file> [--html OUTPUT] [--json OUTPUT]`

**Error handling**
- Extension check before file existence check — unsupported formats get a clear error listing valid options
- Missing optional dependency errors include the exact `pip install` command

---

## Upcoming

Features planned for future releases:

- conda-forge package
- Web dashboard UI

### 2027 maintenance roadmap

- Raise the supported Python baseline to 3.11 after Python 3.10 reaches end
  of life, and test each maintained Python release in CI.
- Maintain pandas 3 compatibility, especially string, categorical, datetime,
  and missing-value inference used by NowEDA's schema rules.
- Make large-mode samples more representative for ordered source files while
  continuing to label estimated findings clearly.
- Review DuckDB, PyArrow, and PySpark dependency support; keep local large-mode
  installation straightforward and leave distributed execution opt-in.
- Automate dependency updates and build checks, including wheel and source
  distribution validation before releases.
