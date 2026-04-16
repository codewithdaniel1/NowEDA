# Encoding Detection Plugin

**Class:** `EncodingDetectionPlugin`  
**Result key:** `encoding`  
**Import:** `from noweda.plugins.encoding import EncodingDetectionPlugin`

---

## What It Does

The Encoding Detection plugin scans string columns for values that appear to be **encoded or obfuscated data** — specifically Base64-encoded strings. This is a security-aware feature unique to NowEDA, useful for detecting hidden payloads, exfiltrated data, or columns that need to be decoded before analysis.

---

## Output Format

```python
{
    "encoded_field": "possible_base64"
}
```

Only columns where encoding is detected appear. Empty dict = no encoding signals.

---

## How Detection Works

The plugin inspects a sample of up to **20 non-null values** from each `object`-dtype column. For each value, it tests whether it is valid Base64:

```python
def is_base64(s):
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False
```

A column is flagged if **more than 5 of the 20 sampled values** are valid Base64 strings. This threshold prevents false positives from short strings that accidentally match.

---

## Scoring Impact

| Encoded columns found | Risk score increase |
|---|---|
| Per column | +10 |

---

## Insight Generated

```
Column 'encoded_field' may contain encoded data (possible_base64).
Inspect for hidden payloads or obfuscation.
```

---

## Why This Matters

**In data analysis:** Encoded columns are unreadable until decoded. NowEDA surfaces them so you know to decode before profiling.

**In security contexts:** Base64 is commonly used to:
- Encode binary payloads in text fields
- Obfuscate malicious strings
- Exfiltrate data that bypasses simple text pattern matching
- Store credentials or tokens in data exports

---

## How to Use the Results

```python
summary = df.noweda.summary()
encoding = summary["encoding"]

if encoding:
    print("Encoded columns detected:")
    for col, enc_type in encoding.items():
        print(f"  {col}: {enc_type}")

    # Attempt to decode Base64 columns
    import base64
    for col in encoding:
        def try_decode(val):
            try:
                return base64.b64decode(str(val)).decode("utf-8")
            except Exception:
                return val  # leave as-is if decoding fails

        df[col + "_decoded"] = df[col].apply(try_decode)
```

---

## Using This Plugin Standalone

```python
import pandas as pd
import base64
from noweda.plugins.encoding import EncodingDetectionPlugin

# Create a DataFrame where one column contains Base64-encoded strings
plaintext_values = ["hello", "world", "foo", "bar", "baz",
                    "alpha", "beta", "gamma", "delta", "epsilon"]
encoded_values = [base64.b64encode(v.encode()).decode() for v in plaintext_values]

df = pd.DataFrame({
    "id":      range(10),
    "label":   plaintext_values,
    "payload": encoded_values,    # this should be flagged
})

plugin = EncodingDetectionPlugin()
result = plugin.run(df)

print(result)
# {'payload': 'possible_base64'}
```

---

## Extending for Other Encodings

The plugin can be subclassed to detect additional encoding schemes:

```python
import binascii
from noweda.plugins.encoding import EncodingDetectionPlugin

class ExtendedEncodingPlugin(EncodingDetectionPlugin):
    name = "encoding"

    def is_hex(self, s):
        try:
            if len(s) % 2 != 0:
                return False
            binascii.unhexlify(s)
            return True
        except Exception:
            return False

    def run(self, df):
        results = super().run(df)   # get Base64 results

        for col in df.columns:
            if df[col].dtype == "object":
                sample = df[col].dropna().astype(str).head(20)
                hex_count = sum(self.is_hex(x) for x in sample)
                if hex_count > 5:
                    results[col] = results.get(col, "") + " possible_hex"

        return results
```

---

## Limitations

- Sampling 20 values may miss encoded columns where only a few rows are encoded.
- Short strings (≤4 characters) can accidentally match Base64 due to how the algorithm works. The threshold of 5+ matches per sample mitigates this.
- The plugin detects encoding signals, not malicious intent. A column of Base64-encoded images is flagged the same as a column of encoded credentials.
