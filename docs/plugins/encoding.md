# Encoding Detection Plugin

**Class:** `EncodingDetectionPlugin`  
**Result key:** `encoding`  
**Import:** `from noweda.plugins.encoding import EncodingDetectionPlugin`

## What It Does

Flags possible Base64 content in object, string and categorical columns. Encoding
is a format signal, not evidence of malicious intent.

## Output Format

The existing result shape is unchanged:

```python
{"payload": "possible_base64"}
```

## How Detection Works

Starting in 0.1.4, the plugin examines the first 20 nonmissing values per text
column. A value must have at least eight characters, pass strict Base64 decoding,
and round-trip to the same encoding. It must also contain padding or Base64
symbols (`=`, `+`, `/`), or decode to printable UTF-8 text (line breaks and tabs
are allowed). This avoids flagging short ordinary names such as `John`.

A column is flagged only when **at least six values match** and **at least 80% of
the sample matches**. The `is_base64()` method tests this heuristic, not just
whether a string can be decoded.

## Confidence and Sample Evidence

```python
report = df.eda.report()
print(report["encoding_details"])
# {'payload': {'sample_size': 20, 'matches': 16, 'confidence': 0.8}}

print(df.eda.encoding_df(include_confidence=True))
# Column, Encoding_Type, Sample_Size, Matches, Confidence
```

`confidence` is exactly `matches / sample_size`. It is an empirical sample
fraction, **not a calibrated probability** that the column is encoded, sensitive,
or malicious. No raw sample values are included in these details.

The default `encoding_df()` still returns only `Column` and `Encoding_Type`.
HTML reports show the match count and sample size.

## Scoring Impact

Each flagged column adds 10 risk points. Review the signal in context: encoded
images and benign payloads can trigger it too.

## Using This Plugin Standalone

```python
import base64
import pandas as pd
from noweda.plugins.encoding import EncodingDetectionPlugin

encoded = base64.b64encode(b"hello world").decode("ascii")
df = pd.DataFrame({"payload": [encoded] * 10, "name": ["John"] * 10})
plugin = EncodingDetectionPlugin()
print(plugin.run(df))  # {'payload': 'possible_base64'}
print(plugin.details)  # evidence from the most recent run
```

## Limitations

- Sampling is limited to the first 20 nonmissing values; later or sparse encoded values may be missed.
- Short encodings, URL-safe variants, whitespace-containing encodings and binary payloads without padding or symbols may be missed.
- Ordinary strings can still satisfy the heuristic. Examine the data before decoding or treating it as sensitive.
