# Future Design

This page records planned directions. It is not a promise that these features
are available in the current release.

## Graph-driven model exploration with `vizall()`

The proposed direction is to help users inspect plots and judge which model
families deserve further testing. The priority is graph generation: make the
data's structure, each model's behavior, and its errors visible. Extend the
existing `vizall()` method without adding new public methods:

```python
df.eda.vizall(target="fraud_flag", max_plots=15)
```

Without a target, retain general EDA and applicable unsupervised views. With a
target, prioritize relevant regression or classification comparisons. This plan
builds on the existing fitted diagnostics; it does not describe all of these
capabilities as already implemented.

### Questions the plots should answer

1. What shape or separation is present in the observed data?
2. How do different candidate models represent that structure?
3. Where does each model fail on validation observations?

Raw data can suggest linearity, curvature, or thresholds. Distinguishing a
decision tree, random forest, and boosting usually also requires small fitted
diagnostic models and comparison of their predictions and errors. The graphs
should support a user's judgment, without claiming a guaranteed best model.

### Proposed model-specific graphs

| Candidate | Graphs | Decision supported |
|---|---|---|
| Linear regression | Scatter with a fitted straight line and flexible reference trend; actual versus predicted; validation residuals | Whether a straight relationship captures the signal, and whether errors retain curvature or changing spread |
| Logistic regression | Observed binary event rates with a fitted probability curve; two-feature probability surface; class overlap | Whether a simple probability relationship describes the observations |
| Linear SVM | Class-colored scatter with a straight boundary, margins, and marked validation mistakes | Whether a straight separator provides useful separation |
| Nonlinear SVM | RBF boundary beside the linear boundary on identical features, with marked validation mistakes | Whether curvature captures structure missed by the straight boundary |
| Decision tree | Rectangular classification regions or stepwise regression predictions; compact tree diagram | Whether understandable thresholds describe the pattern, or the tree fragments the data excessively |
| Random forest | Prediction surface or regression curve beside a single tree; validation error comparison | Whether averaging trees reduces brittle behavior and improves predictions |
| Boosting | Prediction surface or regression curve; validation error comparison; training and validation loss across boosting stages | Whether successive corrections improve fit, and when additional complexity stops helping |

Binary probability curves apply only to binary targets. Multiclass examples
need class-specific views and readable legends instead of a forced sigmoid.
For rare-event classification, prioritize precision–recall and confusion
diagnostics alongside boundaries. For regression, prioritize actual-versus-
predicted and residual views.

Logistic regression and linear SVM can produce similar straight boundaries;
probability behavior, margins, and validation errors help explain their
differences. Forest and boosting surfaces can also look similar, making error
comparisons essential.

### Presentation and interpretation

Begin with an observations-only panel before displaying fitted shapes. Use
side-by-side model panels with shared axes, class colors, and visible ranges.
Place validation errors and a compact performance comparison underneath when
the panel budget allows.

Keep two kinds of views clearly distinguished:

| View | Meaning |
|---|---|
| Relationship illustration | A model fitted using one or two selected features so its behavior can be drawn directly |
| Broader prediction diagnostics | A model using the selected usable feature set, shown through validation predictions, errors, and performance |

A two-feature boundary must not imply performance on the entire feature set.
A one-feature logistic curve must not be ranked against a model using many
features as though the comparison were equivalent. Brief captions should
explain the observation, the candidate it suggests testing, and the limits of
the evidence.

### Automatic graph-selection logic

1. Establish the task, usable feature types, target counts, and data-quality
   constraints. Explain ambiguous or unsupported cases rather than forcing an
   inappropriate fitted plot.
2. Identify candidate relationships: strong approximately linear patterns,
   curved or threshold patterns, and useful two-feature interactions. Do not
   always favor nonlinear features or require every pattern to appear.
3. Select informative, complementary views within the plot budget. Avoid
   repeating the same finding through several nearly identical charts. If no
   useful feature pair is found, explain that limitation.
4. Fit applicable bounded diagnostic models and generate matched comparison
   panels. Choose chart candidates before expensive fitting.
5. Show validation mistakes or residuals so the user can assess where each
   model fails. Report insufficient evidence when the data cannot support a
   meaningful comparison.

For `max_plots`, count each analytical subplot toward the limit. A small budget
should show observations, one informative comparison, and its validation
evidence. A larger budget can add model families or another useful relationship.
Keep paired comparisons together and explain which eligible views were omitted
because of the budget. Do not fill unused space with uninformative plots.

### Fairness, sampling, and runtime

- Compare models on the same features, training rows, validation rows, and
  evaluation metric within each comparison group.
- Split before selecting model features or fitting preprocessing. Validation
  observations must not influence those decisions.
- Include a trivial prediction baseline when displaying comparative performance.
  Label scores as diagnostic results rather than expected production accuracy.
- Keep fitting bounded and deterministic. Disclose source rows, analysis rows,
  and diagnostic fitting rows separately, along with transformations and any
  plot zoom or omitted observations.
- Preserve small-mode and large-mode sampling behavior, while disclosing the
  limitations of a leading-row sample until representative large-mode sampling
  is implemented.
- Treat repeated entities and time dependence as validation concerns; do not
  imply that an ordinary random split establishes forecasting performance.
- Keep `mlall()` model-free and its estimated-fit stars separate from measured
  diagnostic scores. Shared descriptive evidence should keep its explanations
  consistent with the plots, without hidden behavior changes after `vizall()`.

### Proposed delivery order and acceptance criteria

1. Improve the existing regression, logistic, and SVM views, feature-pair
   selection, and validation safeguards.
2. Add decision-tree, random-forest, and boosting comparison panels.
3. Add consistent validation-error views beneath the comparisons and refine
   automatic selection under different panel budgets.

Update the playground and documentation with interpretable examples and their
limitations. Validate beyond the showcase CSV using linear, curved,
interaction-only, random-label, rare-class, categorical-only, missing-value,
and repeated-entity datasets. Check that plots remain readable, comparisons
use matched inputs, panel budgets are respected, and weak evidence is reported
honestly. No dataset should be required to display every model or graph type.

## Distribution and product directions

- Publish NowEDA through conda-forge.
- Explore a web dashboard UI for interactive exploration and reports.

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

These improvements remain deferred and are outside the current visualization
scope.

## 2027 maintenance roadmap

- Raise the supported Python baseline to 3.11 after Python 3.10 reaches end
  of life, and test each maintained Python release in CI.
- Maintain pandas 3 compatibility, especially string, categorical, datetime,
  and missing-value inference used by NowEDA's schema rules.
- Review DuckDB, PyArrow, and PySpark dependency support; keep local large-mode
  installation straightforward and leave distributed execution opt-in.
- Automate dependency updates and build checks, including wheel and source
  distribution validation before releases.
