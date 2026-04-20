# Changelog

All notable changes to NowEDA are documented here.

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

- Visualisation layer (histograms, correlation heatmap, distribution plots)
- Additional PII patterns (phone numbers, SSNs, credit card numbers)
- Dataset fingerprinting (hash-based change detection)
- conda-forge package
- Web dashboard UI
