# CLI Reference

NowEDA ships with a command-line tool installed as `noweda`. After `pip install noweda`, you can run analysis directly from your terminal without writing any Python.

---

## Basic Usage

```bash
noweda <file>
```

Reads the file, runs all plugins, and prints insights and scores to the terminal.

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

### Print insights to terminal

```bash
noweda data.csv
```

```
=== NOWEDA INSIGHTS ===
- Likely identifier column(s) detected: id. Consider excluding from modelling.
- Datetime column(s) detected: join_date. Temporal features may be valuable.
- Column 'encoded_field' is 72% missing — consider dropping it.
- Column 'email' has high missing rate (32%) — imputation recommended.
- PII detected in column 'email': 17 email address(es) found. Mask before sharing.
- Data quality score is acceptable (77). Minor issues present.
- Moderate risk level (25). Review PII and encoded columns before sharing.

=== SCORES ===
{'data_quality': 77, 'risk': 25, 'model_readiness': 53}
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
