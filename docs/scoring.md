# Scoring System

NowEDA produces three scores for every dataset. Scores give you a fast, quantified summary of data quality, risk exposure, and machine learning readiness.

---

## The Three Scores

```python
scores = df.noweda.score()
# {'data_quality': 77, 'risk': 25, 'model_readiness': 53}
```

| Score | Range | Higher means |
|---|---|---|
| `data_quality` | 0 – 100 | Cleaner, more complete data |
| `risk` | 0 – ∞ | More sensitive or obfuscated data detected |
| `model_readiness` | 0 – 100 | More ready for machine learning without preprocessing |

---

## `data_quality` — How It's Calculated

Starts at **100** and is penalised by data quality issues.

| Issue | Condition | Penalty |
|---|---|---|
| Missing values (critical) | >50% missing per column | −10 per column |
| Missing values (high) | 30–50% missing per column | −5 per column |
| Missing values (low) | >0–30% missing per column | −2 per column |
| Duplicate rows (many) | >10% of rows are duplicates | −10 |
| Duplicate rows (some) | >0% of rows are duplicates | −3 |
| Constant columns | Each zero-variance column | −3 per column |
| Outliers (many) | >5% of observed numeric values flagged | −10 |
| Outliers (some) | >1% and ≤5% of observed numeric values flagged | −5 |

Final score is clamped to `[0, 100]`.

**Example:**

```
Starting score:                   100
Column 'encoded_field' >50% null: -10
Column 'email' 30-50% null:        -5
Minor nulls in 3 columns:          -6
3 outliers / 1,000 numeric values:   0  (0.3%, no penalty)
─────────────────────────────────────
data_quality:                       79
```

---

## `risk` — How It's Calculated

Starts at **0** and is increased by security risk signals.

| Issue | Condition | Risk added |
|---|---|---|
| PII column | Each column with PII detected | +15 |
| Encoded column | Each column with encoding detected | +10 |

There is no upper bound on `risk`. A dataset with 5 PII columns and 3 encoded columns gets `risk = 5×15 + 3×10 = 105`.

**Risk interpretation:**

| Risk value | Level | Recommendation |
|---|---|---|
| 0 | No signals detected | Pattern checks do not establish safety |
| 1 – 20 | Low | Review findings before sharing |
| 21 – 50 | Moderate | Redact or mask identified columns |
| > 50 | High | Do not share without thorough review |

---

## `model_readiness` — How It's Calculated

Starts at **100** and is penalised by anything that would require preprocessing before training a model.

| Issue | Condition | Penalty |
|---|---|---|
| Missing values (critical) | >50% missing per column | −15 per column |
| Missing values (high) | 30–50% missing per column | −8 per column |
| Missing values (low) | >0–30% missing per column | −3 per column |
| Constant columns | Each zero-variance column | −5 per column |
| Heavy skew | Each column with `|skewness| > 2` | −5 per column |
| Outliers (many) | >5% of observed numeric values flagged | −10 |
| Outliers (some) | >1% and ≤5% of observed numeric values flagged | −5 |
| Text/unknown columns | Each column with role `text` or `unknown` | −3 per column |

Final score is clamped to `[0, 100]`.

---

## Reading Score Insights

The insight generator adds human-readable score summaries:

| data_quality | Insight |
|---|---|
| ≥ 90 | "Data quality score is excellent (≥90). Dataset looks clean." |
| 70 – 89 | "Data quality score is acceptable (X). Minor issues present." |
| < 70 | "Data quality score is low (X). Significant cleaning required." |

| risk | Insight |
|---|---|
| 0 | "No risk signals detected." |
| 1 – 20 | "Low risk level (X). Some sensitive indicators found." |
| 21 – 50 | "Moderate risk level (X). Review PII and encoded columns before sharing." |
| > 50 | "High risk level (X). Sensitive data likely present — handle with care." |

---

## Accessing Scores in Code

```python
scores = df.noweda.score()

# Gate on quality
if scores["data_quality"] < 70:
    print("Data needs cleaning before use.")

# Gate on risk
if scores["risk"] > 0:
    print("Sensitive data detected — review before sharing.")

# Gate on model readiness
if scores["model_readiness"] >= 80:
    print("Dataset is ML-ready.")
else:
    print(f"Model readiness: {scores['model_readiness']} — preprocessing needed.")
```

---

## Scorer Source

The scoring logic lives in [noweda/scoring/scorer.py](https://github.com/codewithdaniel1/NowEDA/blob/main/noweda/scoring/scorer.py). To customise scoring for your domain (different thresholds, additional penalties), subclass `Scorer` and call it from a custom engine implementation.


## Outlier Rates and Score Contributions (0.1.4)

The outlier rate is the sum of IQR outlier counts divided by the sum of nonmissing
counts in the same numeric columns. Text cells and missing cells do not enter the
denominator. A row can contribute multiple observed cells and multiple outliers.
The penalty uses this pooled rate, so the same rate incurs the same penalty at
different dataset sizes. Other rules, including duplicate and skewness penalties,
can still produce different overall scores.

If no observed values or no denominator is available (for example in custom
plugin results), the rate is unavailable and no outlier penalty is applied.
Empty DataFrame reports explicitly say their scores are not informative.

```python
report = df.eda.report()
for entry in report["score_breakdown"]:
    print(entry["rule"], entry["contributions"])
    if "evidence" in entry:
        print(entry["evidence"])
```

Start at quality 100, readiness 100 and risk 0, then add each rule's contributions.
The final `clamp` entry accounts for adjustments to keep quality and readiness in
range, so contributions reconcile exactly to the returned scores. Outlier
evidence includes the count, observed numeric value count, rate and thresholds.
HTML reports include the same contributions.
