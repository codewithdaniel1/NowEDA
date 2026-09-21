# Visual ML Diagnostics

`vizall()` is NowEDA's single visualization entry point. It ranks useful
charts within a fixed panel budget, so wide datasets do not produce an
unbounded wall of plots.

```python
# Explore general structure.
visuals = df.eda.vizall(max_plots=15)

# Add diagnostics for a prediction objective chosen by you.
visuals = df.eda.vizall(
    target="fraud_flag",
    max_plots=15,
)
```

Install the visualization dependencies first:

```bash
pip install "noweda[viz]"
```

## Parameters

```python
df.eda.vizall(sample=None, target=None, max_plots=15)
```

| Parameter | Behavior |
|---|---|
| `sample` | In small mode, `None` uses every loaded row. A positive integer uses a deterministic bounded sample. In large mode, the default is `10_000`. |
| `target` | Optional prediction-target column. NowEDA infers classification or regression from that column, but never chooses a target silently. |
| `max_plots` | Maximum number of individual chart panels, including panels inside a subplot grid. Must be a positive integer. |

`max_plots` is a budget rather than a promise that every panel will be used.
NowEDA may generate fewer panels when the dataset does not contain suitable
numeric, categorical, missing, outlier, or temporal signals.

## General diagnostics

Without a target, `vizall()` selects applicable views of the dataset's
structure. These can include:

- ranked numeric and categorical distributions;
- missing-value rates;
- numeric correlations and the strongest numeric relationships;
- outlier prevalence;
- feature-scale differences;
- categorical association using Cramér's V; and
- temporal relationships when datetime and numeric fields are available; and
- an exploratory PCA projection with a K-Means preview and silhouette scores.

The method excludes detected PII and likely identifier columns from feature
diagnostics. This reduces noisy charts and helps avoid displaying sensitive
values, but it does not replace a privacy review.

## Classification diagnostics

When the target is categorical, boolean, or a low-cardinality integer,
`vizall()` adds classification-focused panels such as:

- target class counts;
- numeric distributions split by class; and
- class shares across categorical feature values;
- a one-feature logistic probability curve for binary targets; and
- side-by-side linear and nonlinear RBF-SVM boundaries using two selected
  numeric features.

The returned evidence can flag class imbalance, weak univariate separation,
possible leakage, approximately linear structure, and curved or threshold-like
structure. These are starting points for model comparison. For example, visible
linear structure may justify testing logistic regression or a linear SVM as a
baseline, while threshold-like structure may justify comparing tree-based or
nonlinear-kernel models.

The SVM panels are two-feature projections. They can reveal a useful boundary,
but they do not describe what the same estimator would learn from every
available feature. The fit and validation score retain nonmissing finite rows;
the displayed axes zoom to the central 98% so isolated outliers do not hide the
boundary.

## Regression diagnostics

When the target is continuous numeric data, `vizall()` adds:

- the target distribution;
- feature-versus-target scatter plots;
- a linear trend and binned target means; and
- mean target values across categorical feature groups; and
- a fitted linear-regression baseline with validation residuals.

Roughly straight, monotonic relationships can support a linear-regression
baseline. Curves, thresholds, and plateaus suggest testing transformations or
nonlinear estimators. Final selection still depends on leakage-safe validation,
residual behavior, and prediction metrics.

## Fitted visual baselines

The regression, logistic, SVM, and K-Means overlays fit small diagnostic
estimators on deterministic bounded samples. Supervised diagnostics use a
train/validation split and display validation R² or balanced accuracy. K-Means
compares silhouette scores over a small range of `k` values.

These scores describe the displayed feature projection and sample. They are
not expected production performance, a full benchmark, hyperparameter search,
or permission to use a feature that will not exist at prediction time. The
estimators are discarded after rendering.

## Returned result

`vizall()` renders the charts and returns a concise dictionary-like
`VizResult`. Keep it when you want to inspect the evidence or reuse the figures.

```python
visuals = df.eda.vizall(target="fraud_flag", max_plots=15)

print(visuals["problem_type"])
print(visuals["target_summary"])
print(visuals["associations"][:5])
print(visuals["model_signals"])
print(visuals["diagnostic_models"])
print(visuals["scope"])

first_figure = visuals["figures"][0]
```

Useful fields include:

| Field | Meaning |
|---|---|
| `problem_type` | Inferred `classification` or `regression` for a selected target; otherwise `None` |
| `target_summary` | Observed labels, missingness, inferred type reason, and class or numeric summary |
| `associations` | Ranked univariate feature-target evidence, or general feature rankings without a target |
| `selected_features` | Fields used in rendered panels |
| `excluded_features` | Likely identifiers, detected PII, and unsupported fields omitted from feature diagnostics |
| `plot_titles` / `figures` | Generated panel groups and Matplotlib figures |
| `model_signals` | Cautious, data-specific interpretations for model families to validate |
| `diagnostic_models` | Disposable fitted baseline, selected features, fitted rows, validation metric, score, and projection scope |
| `observations` | Other findings such as imbalance or feature-scale differences |
| `scope` | Source rows, sampled rows when applicable, and whether results are sample-based |
| `plots_generated` / `max_plots` | Used panel count and requested panel budget |

## Sampling and large mode

Small mode uses all loaded rows unless `sample=` is supplied:

```python
df = eda.read("data.csv")
visuals = df.eda.vizall(target="churned", sample=25_000)
```

Large mode keeps the source on disk and uses 10,000 rows by default. Change the
sample explicitly when the analysis needs a different speed-versus-coverage
tradeoff:

```python
data = eda.read("large.csv", mode="large", chunksize=100_000)
visuals = data.eda.vizall(
    target="churned",
    sample=50_000,
    max_plots=15,
)

print(visuals["scope"])
```

Large-mode output announces the sample size, and `scope["sample_based"]`
identifies whether the result represents a sample. Ordered source files can
produce an unrepresentative leading sample, so review the scope before acting
on small differences.

## How this relates to `mlall()`

`mlall(target=...)` reuses the same association and model-shape evidence in its
structured plan, keeping written recommendations aligned with
`vizall(target=...)`:

```python
plan = df.eda.mlall(target="fraud_flag", plan=True)
print(plan["visual_diagnostics"]["model_signals"])
```

`mlall()` remains model-free. `vizall()` may fit the disposable visual baselines
described above, but does not return a production estimator or claim expected
accuracy. Use their output to choose sensible candidates and preprocessing,
then compare those candidates with validation that matches the real prediction
task.
