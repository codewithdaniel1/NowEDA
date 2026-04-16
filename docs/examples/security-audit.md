# Example: Security & PII Audit

This example shows how to use NowEDA as a data security scanner — finding PII, encoded data, and risk signals before a dataset is shared, exported, or ingested into a pipeline.

---

## Scenario

Your team needs to share a customer transaction export with an external analytics vendor. Before sending it, you need to verify it doesn't contain PII or sensitive encoded data.

---

## Step 1 — Scan the file

```python
import noweda

df = noweda.read("customer_export.csv")
scores = df.noweda.score()

print(f"Risk score: {scores['risk']}")
```

```
Risk score: 40
```

A risk score of 40 = **moderate risk**. Something was detected.

---

## Step 2 — Find exactly what was detected

```python
summary = df.noweda.summary()

# PII findings
pii = summary["pii"]
if pii:
    print("PII DETECTED:")
    for col, findings in pii.items():
        print(f"  Column '{col}': {findings}")
else:
    print("No PII detected.")

# Encoding findings
encoding = summary["encoding"]
if encoding:
    print("\nENCODING DETECTED:")
    for col, enc_type in encoding.items():
        print(f"  Column '{col}': {enc_type}")
```

```
PII DETECTED:
  Column 'contact_email': {'emails_detected': 842}
  Column 'notes': {'emails_detected': 37}

ENCODING DETECTED:
  Column 'payload': possible_base64
```

Three columns need attention before sharing.

---

## Step 3 — Read all risk insights

```python
insights = df.noweda.insights()
risk_insights = [i for i in insights if any(
    kw in i.lower() for kw in ["pii", "risk", "sensitive", "encoded", "mask"]
)]

for insight in risk_insights:
    print(insight)
```

```
PII detected in column 'contact_email': 842 email address(es) found. Mask or remove before sharing.
PII detected in column 'notes': 37 email address(es) found. Mask or remove before sharing.
Column 'payload' may contain encoded data (possible_base64). Inspect for hidden payloads or obfuscation.
High risk level (40). Sensitive data likely present — handle with care.
```

---

## Step 4 — Inspect the encoded column

Before deciding what to do with the `payload` column, decode a sample and inspect it:

```python
import base64

sample = df["payload"].dropna().head(5)
for val in sample:
    try:
        decoded = base64.b64decode(str(val)).decode("utf-8")
        print(f"Original: {val}")
        print(f"Decoded:  {decoded}")
        print()
    except Exception:
        print(f"Could not decode: {val}")
```

```
Original: aGVsbG8=
Decoded:  hello

Original: dXNlcjoxMjM0NTY=
Decoded:  user:123456

Original: cGFzc3dvcmQ6c2VjcmV0
Decoded:  password:secret
```

The `payload` column contains Base64-encoded credentials — this should absolutely not leave your system.

---

## Step 5 — Sanitise the dataset

```python
import hashlib
import pandas as pd

df_safe = df.copy()

# Option 1: Redact PII columns entirely
pii_cols = list(summary["pii"].keys())
for col in pii_cols:
    df_safe[col] = "[REDACTED]"

# Option 2: Hash PII (preserves referential integrity for joining)
# df_safe["contact_email"] = df_safe["contact_email"].apply(
#     lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16] if pd.notna(x) else x
# )

# Drop the encoded/sensitive column entirely
df_safe = df_safe.drop(columns=list(summary["encoding"].keys()))

# Verify the cleaned dataset
df_safe.noweda.score()
```

```python
{'data_quality': 91, 'risk': 0, 'model_readiness': 85}
```

Risk is now 0. Safe to share.

---

## Step 6 — Generate a compliance report

```python
from noweda.report.html import generate_html_report

# Report on original (internal record of what was found)
original_report = df.noweda.report()
generate_html_report(original_report, "audit_findings.html")

# Report on sanitised version (send with the data)
safe_report = df_safe.noweda.report()
generate_html_report(safe_report, "audit_clean.html")
```

---

## Automating This as a Pre-Share Check

```python
def pii_gate(file_path, max_risk=0):
    """Raise an error if the file contains any PII or risk signals."""
    df = noweda.read(file_path)
    scores = df.noweda.score()

    if scores["risk"] > max_risk:
        pii = df.noweda.summary()["pii"]
        encoding = df.noweda.summary()["encoding"]
        raise ValueError(
            f"File '{file_path}' failed PII gate (risk={scores['risk']}).\n"
            f"PII columns: {list(pii.keys())}\n"
            f"Encoded columns: {list(encoding.keys())}"
        )

    return True

# Use it in a pipeline or CI script
pii_gate("export_for_vendor.csv", max_risk=0)
```
