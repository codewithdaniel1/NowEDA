# Task-Aware ML Guidance

`df.noweda.mlall()` is NowEDA's single ML guidance method. It prints an
explainable assessment and never fits a model or reports estimated accuracy.
Set `plan=True` when code needs the same structured result.

```python
plan = df.noweda.mlall(target="fraud_flag", plan=True)
print(plan["recommendations"])
```

Stars and `/5` values are estimated dataset-fit scores within an analytical task.
They are not cross-validation results or predictions of which model will win.

## Start with the dataset

Call `mlall()` without arguments when you have a dataset but do not yet know the
right objective:

```python
df.noweda.mlall()
```

NowEDA reports usable features, likely identifiers, possible target candidates,
supervised readiness, unsupervised readiness, and ranked directions for clustering,
anomaly detection, and dimensionality reduction. Potential targets are name- and
value-based hints only. NowEDA does not select one for you.

Likely identifiers and possible target columns are excluded from the automatic
unsupervised feature set. If fewer than two usable features remain, NowEDA explains
why it cannot give meaningful unsupervised guidance.

## Supported problem types

| `problem_type` | Target required? | What NowEDA provides |
|---|---:|---|
| `classification` | Yes | Binary or multiclass candidates, label checks, stratified validation and classification metrics |
| `regression` | Yes | Continuous-outcome candidates, label checks, regression validation and metrics |
| `clustering` | No | Clustering candidates, scaling and cluster-stability guidance |
| `anomaly_detection` | No | Unsupervised anomaly candidates and operational evaluation guidance |
| `dimensionality_reduction` | No | Linear and exploratory reduction candidates with stability cautions |

Aliases include `binary`, `multiclass`, `anomaly`, `outlier_detection`,
`dimensionality`, and `reduction`. Hyphens and spaces are normalized to
underscores.

Forecasting is not included yet. It needs a time column, forecast horizon,
optional series identifier, and time-aware validation, so treating it as ordinary
regression would give misleading guidance.

## Supervised objectives

Name the target when you know what you want to predict:

```python
# Let NowEDA infer classification or regression from the target.
df.noweda.mlall(target="fraud_flag")

# State the task explicitly when it is already known.
plan = df.noweda.mlall(
    target="fraud_flag",
    problem_type="classification",
    features=["amount", "channel", "account_age_days"],
    plan=True,
)
```

When `problem_type` is omitted, NowEDA uses these transparent rules:

| Target signal | Inferred task |
|---|---|
| Boolean, string, or categorical values | Classification |
| Exactly two distinct numeric values | Binary classification |
| Low-cardinality integer values | Classification |
| Continuous or high-cardinality numeric values | Regression |

It reports labeled rows, label coverage, and one of these readiness states:

| Readiness | Meaning |
|---|---|
| Ready | Enough observed labels for ordinary supervised guidance |
| Partial labels | Some rows are unlabeled; supervised training uses labeled rows only |
| Limited labels | Too few labeled rows or too few observations in a class for dependable validation |
| No labels | The selected target has no observed values, so supervised training cannot begin |

Partial labels can make semi-supervised learning worth investigating, but NowEDA
does not claim it is appropriate without a user objective and a separate validation
strategy. When no labels are available, `mlall()` presents unsupervised directions
instead of issuing supervised algorithm rankings.

NowEDA rejects missing target names, constant targets, nonnumeric regression
targets, and infinite regression values. It also flags class imbalance, likely
identifier targets, near-perfect target correlations, and temporal features that may
need time-aware validation.

## Unsupervised objectives

Choose an explicit unsupervised task when you already know the analysis you want:

```python
df.noweda.mlall(
    problem_type="clustering",
    features=["annual_spend", "visit_count", "account_age_days"],
)

anomaly_result = df.noweda.mlall(
    problem_type="anomaly_detection",
    features=["transaction_amount", "transactions_per_hour"],
    plan=True,
)
```

Review likely identifiers and fields unavailable when the analysis will run.
Distance-based, margin-based, and component-based methods usually require scaling,
categorical encoding, and a missing-data strategy. Anomalies are unusual observations;
they are not automatically fraud, security incidents, or other business outcomes.

## Structured results

With `plan=True`, `mlall()` returns these fields after printing the guidance:

| Field | Meaning |
|---|---|
| `problem_type` / `problem_subtype` | Resolved task and supervised subtype, when selected |
| `target` / `target_summary` | Selected target and label diagnostics |
| `features` | Input features used for focused guidance, or usable automatic-assessment features |
| `recommendations` | Ranked candidates with a 1–5 `score`, `why`, and `caution` |
| `assessment` | No-argument readiness, possible targets, excluded IDs, and analytical directions |
| `preprocessing` / `evaluation` | Preparation, validation design, and suitable metrics |
| `warnings` | Conditions requiring review before modeling |

Use a leakage-safe train/validation design, fit preprocessing only on training data,
and compare metrics that match the actual decision objective.
