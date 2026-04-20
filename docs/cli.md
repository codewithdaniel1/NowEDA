# CLI Reference

NowEDA ships with a command-line tool installed as `noweda`. After `pip install noweda`, you can run analysis directly from your terminal without writing any Python.

---

## Basic Usage

```bash
noweda <file>
```

Reads the file, runs all plugins, and prints a rich report to the terminal — including row/column counts, dtypes, descriptive statistics, and actionable insights.

While the file is loading and the report is being built, the CLI shows a small live status indicator so it is clear what NowEDA is doing.

---

## Commands and Options

```
noweda <file> [--html OUTPUT] [--json OUTPUT]

Arguments:
  file              Path to the dataset file (any supported format)

Options:
  --html OUTPUT     Export an HTML report to OUTPUT path
  --json OUTPUT     Export a JSON report to OUTPUT path
  --help            Show this message and exit
```

---

## Examples

### Print full report to terminal

```bash
noweda data.csv
```

```
========================================================================
  NowEDA · data.csv
========================================================================
  Rows    : 25
  Columns : 10

Scores
------------------------------------------------------------
  Data Quality    : 77 / 100
  Model Readiness : 53 / 100
  Risk            : 25  (0 = no risk)

Columns
------------------------------------------------------------
  Column          Dtype          Role                    Unique  Missing
  --------------- -------------- ----------------------  ------  -------
  id              int64          id_candidate                25        0
  name            object         text                        24        1
  email           object         text                        17        8
  age             int64          numeric                     15        0
  salary          float64        numeric                     23        2
  department      object         categorical                  4        0
  join_date       object         datetime                    24        2
  score           float64        numeric                     22        3
  flag            int64          categorical_numeric          2        0
  encoded_field   object         text                         7       18

Numeric Statistics
------------------------------------------------------------
  Column     Count         Mean          Std        Min        25%        50%        75%        Max     Skew
  ---------- --------  ------------ ------------  ---------- ---------- ---------- ---------- ----------  --------
  age           25         32.6          9.5          21         25         31         40         55     0.52
  salary        23       62500        18300        30000      49000      60000      72000     110000     0.89
  ...

Insights
------------------------------------------------------------
  • Likely identifier column(s) detected: id. Consider excluding from modelling.
  • Column 'encoded_field' is 72% missing — consider dropping it.
  • PII detected in column 'email': 17 email address(es) found. Mask before sharing.
  • Data quality score is acceptable (77). Minor issues present.
  • Moderate risk level (25). Review PII and encoded columns before sharing.
========================================================================
```

---

### Export an HTML report

```bash
noweda data.csv --html report.html
```

Opens `report.html` in any browser. Includes score cards, insights, schema table, missing value bars, PII and encoding warnings.

---

### Export a JSON report

```bash
noweda data.csv --json report.json
```

The JSON file contains the full report: plugin results, scores, and insights.

---

### Export both at once

```bash
noweda data.csv --html report.html --json report.json
```

---

### Use with any supported format

```bash
noweda transactions.xlsx
noweda events.parquet
noweda customers.json
noweda records.dta
```

---

## Redirecting Output

The terminal output is plain text and can be piped or redirected:

```bash
# Save terminal output to a text file
noweda data.csv > analysis.txt

# Pipe insights through grep
noweda data.csv | grep "PII"
noweda data.csv | grep "missing"
```

---

## Using the CLI in Scripts

```bash
#!/bin/bash

# Example: automated daily data quality check
FILE="daily_export.csv"
REPORT="reports/$(date +%Y-%m-%d).html"

echo "Running NowEDA on $FILE..."
noweda "$FILE" --html "$REPORT" --json "${REPORT%.html}.json"
echo "Report saved to $REPORT"
```

---

## Installing the CLI

The CLI is installed automatically with the package:

```bash
pip install noweda
noweda --help
```

If `noweda` is not found after installing, your Python Scripts directory may not be on your PATH. Try:

```bash
python -m noweda.cli data.csv
```

Or activate your virtual environment first:

```bash
source .venv/bin/activate
noweda data.csv
```
