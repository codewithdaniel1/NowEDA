# Task-Aware ML Guidance

NowEDA recommends candidate methods only after you identify the ML objective. A
target column describes what to predict, and that intent cannot be determined
reliably from column statistics alone.

`mlall()` prints guidance for people. `ml_plan()` returns the same information as
a dictionary for notebooks, applications, and automated checks. Neither method
fits a model or reports estimated performance. Stars and `/5` values summarize a
estimated dataset-fit score within the chosen task.

## Supported problem types

| `problem_type` | Target required? | What NowEDA provides |
|---|---:|---|
| `classification` | Yes | Binary or multiclass candidates, class checks, stratified validation and classification metrics |
| `regression` | Yes | Continuous-outcome candidates, target checks, regression validation and metrics |
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

Name the target whenever you want classification or regression guidance:

```python
# Let NowEDA infer classification or regression from the target.
plan = df.noweda.ml_plan(target="fraud_flag")

print(plan["problem_type"])
print(plan["problem_subtype"])
print(plan["inference_reason"])

# Or state the task explicitly.
df.noweda.mlall(
    target="fraud_flag",
    problem_type="classification",
    features=["amount", "channel", "account_age_days"],
)
```

When `problem_type` is omitted, NowEDA uses these transparent rules:

| Target signal | Inferred task |
|---|---|
| Boolean, string, or categorical values | Classification |
| Exactly two distinct numeric values | Binary classification |
| Low-cardinality integer values | Classification |
| Continuous or high-cardinality numeric values | Regression |

Inference is a convenience, not a statement of user intent. The returned plan
sets `inferred=True`, includes `inference_reason`, and the printed output tells you
to override `problem_type` when the result does not match the objective.

NowEDA rejects a missing target name, an all-missing or constant target, a
nonnumeric regression target, and infinite regression values. It reports missing
labels, small labeled samples, class imbalance, and classification targets with
unusually many classes for review. Numeric features with near-perfect target
correlation are called out for possible leakage review.

The target never appears in `features`. If you pass `features=`, every column must
exist, labels must be unique within the list, and the list cannot include the
target.

## Unsupervised objectives

Unsupervised guidance requires an explicit task and does not accept `target=`:

```python
df.noweda.mlall(
    problem_type="clustering",
    features=["annual_spend", "visit_count", "account_age_days"],
)

anomaly_plan = df.noweda.ml_plan(
    problem_type="anomaly_detection",
    features=["transaction_amount", "transactions_per_hour"],
)
```

Review likely identifiers and exclude any fields unavailable when the analysis
will run. Distance-based, margin-based, and component-based methods generally need
numeric scaling and an explicit strategy for categorical features and missing
values.

## Structured plan

`ml_plan()` returns these stable top-level fields:

| Field | Meaning |
|---|---|
| `problem_type` | Resolved canonical objective, or `None` when none was selected |
| `problem_subtype` | `binary`, `multiclass`, or `continuous` for supervised tasks |
| `target` / `target_summary` | Selected target and label diagnostics |
| `features` | Input column labels used for profiling |
| `inferred` / `inference_reason` | Whether the task was inferred and why |
| `recommendations` | Ranked candidates with a 1–5 heuristic `score`, `why`, and `caution` text |
| `preprocessing` | Task-aware preparation steps |
| `evaluation` | Validation design and appropriate metrics |
| `warnings` | Conditions that need review before modeling |
| `supported_problem_types` | Canonical task names accepted by this release |

Calling `mlall()` without an objective lists the supported tasks and an example.
It does not produce a mixed algorithm ranking:

```python
df.noweda.mlall()
```

The ratings depend on dataset characteristics, but they are not accuracy,
cross-validation results, or predictions of which model will win. Use a leakage-safe
train/validation design, fit preprocessing only on training data, and compare the
reported metrics for your actual decision objective.
