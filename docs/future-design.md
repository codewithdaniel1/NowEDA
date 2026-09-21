# Future Design

This page records planned directions. It is not a promise that these features
are available in the current release.

## Deferred large-mode work

Large mode currently keeps the source on disk and uses a bounded, deterministic
sample for exploratory work. The next reliability-focused release should:

- replace the current leading-row sample with a reproducible, representative
  sample for ordered source files;
- benchmark CSV and Parquet inputs at practical sizes and improve recovery from
  malformed input;
- keep every estimated result clearly labelled with its sample size; and
- speed up CI with dependency caching and update GitHub Actions to remove
  runtime deprecation notices.

These improvements are intentionally on hold while the next design direction
for visual ML guidance is defined.

## ML-guided `vizall()`

`vizall()` should remain the single visualization entry point. A future
optional target argument can extend the existing workflow:

```python
df.eda.vizall(target="churned")
```

The user selects the target. NowEDA may infer the target type, but must never
silently choose the prediction objective.

### Diagnostics for continuous targets

For a regression target, `vizall()` can show:

- target distribution and extreme values;
- feature-versus-target scatter plots with a simple linear trend;
- binned feature-versus-target plots to make curves, thresholds, and plateaus
  visible; and
- categorical group distributions and target means.

These charts help assess whether a linear baseline is plausible or whether
nonlinear models and feature transformations are worth testing.

### Diagnostics for categorical targets

For binary or multiclass classification, `vizall()` can show:

- class balance;
- numerical-feature distributions by class;
- category-versus-target-rate charts;
- binned event-rate plots for numerical features; and
- a compact pairwise view of the strongest candidate features when an
interaction or threshold pattern is apparent.

These diagnostics expose imbalance, class overlap, nonlinear boundaries, and
potential interaction effects before model selection.

### Honest model-selection guidance

Charts can suggest useful starting points; they cannot prove that linear
regression, logistic regression, a linear SVM, or a nonlinear SVM is best.
The visual output should therefore use language such as:

- **Linear baseline looks plausible** — roughly monotonic, straight-line
  relationships and limited interaction evidence.
- **Nonlinear effects look plausible** — curves, thresholds, or distinct
  regions are visible.
- **Class separation appears limited** — distributions overlap substantially,
  so validate several model families rather than assuming an SVM will help.
- **Evaluate with cross-validation** — the final decision comes from an
  appropriate validation split and metrics, not the chart alone.

The initial version should stay visualization-only and avoid silently training
or selecting a model. Later, optional and explicit benchmark comparisons can
be considered once the visual guidance is validated.
