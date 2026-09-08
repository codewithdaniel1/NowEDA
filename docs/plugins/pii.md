# PII Detection Plugin

**Class:** `PIIDetectorPlugin`  
**Result key:** `pii`  
**Import:** `from noweda.plugins.pii import PIIDetectorPlugin`

## What It Does

Scans object, string and categorical columns for email addresses, US-style phone
numbers, formatted US Social Security numbers and supported credit card patterns.
Numeric and datetime columns are skipped; load identifiers as strings to preserve
leading zeros and formatting.

## Output Format

```python
{"contact": {"credit_card": 1, "phone": 2}}
```

Counts represent **cells containing at least one match of each type**. Two card
numbers in one cell count once. Duplicate row index labels still count as separate
observations. Columns without matches are omitted.

## How Detection Works

Card candidates can contain spaces or hyphens. The digits must match supported
Visa (13/16 digits), Mastercard (51–55 prefixes), American Express or Discover
(6011/65 prefixes) patterns and pass the Luhn checksum. A checksum match does not
verify that an account exists. Other issuers, newer ranges and 19-digit cards are
not currently detected.

Card-shaped spans are excluded from phone matching, including candidates that
fail the checksum. A separate phone elsewhere in the same cell is still checked.
Phone patterns cover ten-digit US-style numbers, with an optional `+1`, separators
and parentheses. SSN patterns require `NNN-NN-NNNN` or space-separated groups.
These patterns do not validate whether a phone or SSN was issued.

## Using This Plugin Standalone

```python
import pandas as pd
from noweda.plugins.pii import PIIDetectorPlugin

df = pd.DataFrame({"contact": [
    "4111 1111 1111 1111",
    "202-555-0101",
    "202-555-0102",
]}, index=[0, 0, 0])
print(PIIDetectorPlugin().run(df))
# {'contact': {'credit_card': 1, 'phone': 2}}
```

## Scoring Impact

Each column with any PII signal adds 15 risk points, regardless of how many types
or cells match. `df.eda.pii_df()` returns `Column`, `PII_Type` and `Count`.

## Limitations

These are pattern-based signals: false positives and missed sensitive values are
possible. A zero risk score means no configured pattern matched, not that data is
safe to share. Adjacent numeric identifiers without clear separators can be
ambiguous. Review findings in context before using them for data governance.
