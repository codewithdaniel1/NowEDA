# PII Detection Plugin

**Class:** `PIIDetectorPlugin`  
**Result key:** `pii`  
**Import:** `from noweda.plugins.pii import PIIDetectorPlugin`

---

## What It Does

The PII (Personally Identifiable Information) Detection plugin scans string columns for patterns that indicate sensitive personal data. Out of the box, it detects **email addresses**. The plugin is designed to be extended with additional patterns (phone numbers, SSNs, credit card numbers, etc.).

---

## Output Format

```python
{
    "email": {
        "emails_detected": 17    # count of matching values in this column
    }
}
```

Only columns with at least one match appear. Empty dict = no PII detected.

---

## How Detection Works

The plugin scans each `object`-dtype column using a regular expression:

```
[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+
```

This matches standard email address formats. The count represents the number of **rows** (not characters) in the column that contain at least one email address.

---

## Scoring Impact

| PII columns found | Risk score increase |
|---|---|
| Per column | +15 |

A dataset with 2 columns containing emails gets `risk += 30`.

---

## Insight Generated

```
PII detected in column 'email': 17 email address(es) found. Mask or remove before sharing.
```

---

## How to Use the Results

```python
summary = df.noweda.summary()
pii = summary["pii"]

if pii:
    print("WARNING: PII detected in the following columns:")
    for col, info in pii.items():
        print(f"  {col}: {info}")
else:
    print("No PII detected.")
```

### Masking PII before sharing

Once NowEDA identifies PII columns, you can mask them:

```python
import hashlib

pii_cols = list(df.noweda.summary()["pii"].keys())

df_safe = df.copy()
for col in pii_cols:
    # Replace with SHA-256 hash (consistent but irreversible)
    df_safe[col] = df_safe[col].apply(
        lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:12] if pd.notna(x) else x
    )

# Or simply redact
for col in pii_cols:
    df_safe[col] = "[REDACTED]"
```

---

## Extending the Plugin with Custom PII Patterns

The plugin is designed to be subclassed. Here's how to add phone number detection:

```python
import re
from noweda.plugins.pii import PIIDetectorPlugin

class ExtendedPIIPlugin(PIIDetectorPlugin):
    name = "pii"

    PHONE_REGEX = r"\b(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
    SSN_REGEX   = r"\b\d{3}-\d{2}-\d{4}\b"

    def run(self, df):
        findings = super().run(df)   # get email results

        for col in df.columns:
            if df[col].dtype == "object":
                # Phone detection
                phones = df[col].astype(str).str.contains(
                    self.PHONE_REGEX, regex=True
                ).sum()
                if phones > 0:
                    findings.setdefault(col, {})["phones_detected"] = int(phones)

                # SSN detection
                ssns = df[col].astype(str).str.contains(
                    self.SSN_REGEX, regex=True
                ).sum()
                if ssns > 0:
                    findings.setdefault(col, {})["ssns_detected"] = int(ssns)

        return findings
```

Use it:

```python
from noweda.core.engine import AutoEDAEngine
from noweda.plugins import default_plugins

# Replace the default PII plugin with your extended version
plugins = [p for p in default_plugins() if p.name != "pii"]
plugins.append(ExtendedPIIPlugin())

engine = AutoEDAEngine(plugins)
report = engine.run_df(df)
print(report["results"]["pii"])
```

---

## Using This Plugin Standalone

```python
import pandas as pd
from noweda.plugins.pii import PIIDetectorPlugin

df = pd.DataFrame({
    "email":   ["alice@example.com", "bob@company.org", "not-an-email"],
    "comment": ["Hello world", "Contact me at charlie@test.com", "No email here"],
    "number":  [1, 2, 3],
})

plugin = PIIDetectorPlugin()
result = plugin.run(df)

print(result)
# {'email': {'emails_detected': 2}, 'comment': {'emails_detected': 1}}
```

---

## Important Notes

- The plugin only scans `object`-dtype columns. Numeric or datetime columns are skipped.
- Detection is pattern-based, not context-aware. False positives are possible (e.g., a product code that happens to match the email regex).
- For production data governance, combine NowEDA's detection with a dedicated PII scanning tool.
