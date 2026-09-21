# Future Design

This page records planned directions. It is not a promise that these features
are available in the current release.

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
