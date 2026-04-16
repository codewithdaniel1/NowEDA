# Example: Writing a Custom Plugin

This example builds a complete custom plugin from scratch — a **Phone Number Detector** — and integrates it into NowEDA's engine with scoring and insights.

---

## Goal

Detect phone numbers in string columns and add them to the risk score.

---

## Step 1 — Write the plugin

```python
# my_plugins/phone.py

import re
from noweda.plugins.base import BasePlugin


class PhoneNumberPlugin(BasePlugin):
    """Detect phone number patterns in string columns."""

    name = "phone"

    # Matches formats like: +1-800-555-1234, (800) 555 1234, 8005551234
    PHONE_REGEX = re.compile(
        r"\b(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
    )

    def run(self, df):
        findings = {}

        for col in df.columns:
            if df[col].dtype != "object":
                continue

            matches = (
                df[col]
                .astype(str)
                .str.contains(self.PHONE_REGEX, regex=True)
                .sum()
            )

            if matches > 0:
                findings[col] = {"phones_detected": int(matches)}

        return findings
```

---

## Step 2 — Use the plugin standalone

```python
import pandas as pd
from my_plugins.phone import PhoneNumberPlugin

df = pd.DataFrame({
    "name":    ["Alice", "Bob", "Carol"],
    "contact": ["Call me at 800-555-1234", "No phone", "+1 (415) 555-9876"],
    "value":   [100, 200, 300],
})

plugin = PhoneNumberPlugin()
result = plugin.run(df)

print(result)
# {'contact': {'phones_detected': 2}}
```

---

## Step 3 — Add it to NowEDA's engine

```python
import noweda as eda
from noweda.core.engine import AutoEDAEngine
from noweda.plugins import default_plugins
from my_plugins.phone import PhoneNumberPlugin

# Add phone plugin to the default stack
plugins = default_plugins() + [PhoneNumberPlugin()]
engine = AutoEDAEngine(plugins)

df = eda.read("contacts.csv")
report = engine.run_df(df)

print(report["results"]["phone"])
# {'contact': {'phones_detected': 2}}
```

---

## Step 4 — Make it affect the risk score

Subclass `Scorer` to include phone numbers in the risk calculation:

```python
from noweda.scoring.scorer import Scorer


class ExtendedScorer(Scorer):
    def compute(self, results):
        scores = super().compute(results)   # run base scoring

        phone = results.get("phone", {})
        scores["risk"] += len(phone) * 10  # +10 per phone column

        return scores
```

---

## Step 5 — Make it generate insights

Subclass `InsightGenerator` to add phone-specific insights:

```python
from noweda.insights.generator import InsightGenerator


class ExtendedInsightGenerator(InsightGenerator):
    def generate(self, results, scores):
        insights = super().generate(results, scores)

        phone = results.get("phone", {})
        for col, info in phone.items():
            insights.append(
                f"Phone numbers detected in column '{col}': "
                f"{info['phones_detected']} match(es). "
                f"Remove or mask before sharing."
            )

        return insights
```

---

## Step 6 — Wire everything into a custom engine

```python
from noweda.core.engine import AutoEDAEngine


class SecureEngine(AutoEDAEngine):
    """Engine with phone detection, extended scoring, and custom insights."""

    def run_df(self, df):
        results = {}
        for plugin in self.plugins:
            results[plugin.name] = plugin.run(df)

        scores = ExtendedScorer().compute(results)
        insights = ExtendedInsightGenerator().generate(results, scores)

        return {"results": results, "scores": scores, "insights": insights}


# Use it
plugins = default_plugins() + [PhoneNumberPlugin()]
engine = SecureEngine(plugins)

df = eda.read("contacts.csv")
report = engine.run_df(df)

for insight in report["insights"]:
    print(insight)
```

---

## Step 7 — Integrate with `df.noweda.*`

To make your custom engine the default used by `df.noweda.*`, patch the accessor before any analysis runs:

```python
import noweda.accessor as _accessor
from noweda.accessor import NowEDAAccessor


class SecureAccessor(NowEDAAccessor):
    def _ensure_analyzed(self):
        if self._report is None:
            plugins = default_plugins() + [PhoneNumberPlugin()]
            engine = SecureEngine(plugins)
            self._report = engine.run_df(self._df)


# Re-register the accessor
import pandas as pd
pd.api.extensions.register_dataframe_accessor("noweda")(SecureAccessor)

# Now df.noweda.* uses your custom engine
df = eda.read("contacts.csv")
df.noweda.insights()   # includes phone number insights
```
