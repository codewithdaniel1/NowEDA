# FAQ

---

## General

**Q: Does NowEDA modify my DataFrame?**

No. NowEDA only reads your DataFrame. Plugins never mutate the input. The accessor stores analysis results on its own instance, not on the DataFrame object.

---

**Q: Does NowEDA work with DataFrames I didn't load with `eda.read()`?** (e.g., `pd.read_sql`, `pd.read_csv`)

Yes. As long as `import noweda as eda` has been executed, `df.noweda.*` is available on any pandas DataFrame, regardless of how it was created.

```python
import noweda as eda
import pandas as pd

df = pd.read_sql("SELECT * FROM orders", conn)
df.noweda.insights()   # works
```

---

**Q: How do I use NowEDA with a DataFrame from a database?**

```python
import noweda as eda
import pandas as pd
import sqlite3

conn = sqlite3.connect("mydb.sqlite")
df = pd.read_sql("SELECT * FROM transactions LIMIT 10000", conn)
df.noweda.score()
```

---

**Q: Why is `df.noweda` not available?**

You need to `import noweda as eda` in the same Python session. The import is what registers the accessor. If you just did `from noweda.plugins.pii import PIIDetectorPlugin` without importing the top-level package, the accessor won't be registered.

```python
import noweda as eda   # this line is required
```

---

**Q: I updated a column and re-ran `df.noweda.insights()` — it returned the same old results. Why?**

Since 0.1.3, changes to DataFrame values and schema automatically invalidate the
cached report. You can also explicitly force a new analysis:

```python
df["col"] = df["col"].fillna(0)
df.noweda.score()       # reflects the modified values
report = df.noweda.refresh()
```

---

## ML Guidance

**Q: How does NowEDA know which column is the target?**

It does not choose one silently. The target represents what you want to predict,
so supervised guidance requires you to name it:

```python
df.noweda.mlall(target="fraud_flag", problem_type="classification")
```

If you omit `problem_type` but provide `target`, NowEDA infers classification or
regression from the target dtype and number of distinct values. The result explains
the inference so you can override it.

---

**Q: What happens if I call `df.noweda.mlall()` with no arguments?**

It assesses the dataset first. The output shows possible target candidates for
review, supervised and unsupervised readiness, likely identifiers, and ranked
unsupervised directions. It does not silently choose a target or mix supervised
algorithms into an unsupervised ranking.

---

**Q: Are the recommended algorithms expected to be the best performers?**

No. They are task-specific candidates based on data characteristics. NowEDA does
not fit or evaluate models in `mlall()`. Stars and `/5` values are
estimated dataset-fit ratings, not expected accuracy. The output includes suitable
validation approaches, metrics, preprocessing, and cautions so you can compare the
candidates without leakage.

---

**Q: Does NowEDA support forecasting?**

Not yet. Forecasting needs a time column, forecast horizon, optional series ID, and
time-aware validation. NowEDA raises a clear error instead of treating forecasting
as ordinary regression.

---

## Formats

**Q: My file has a `.txt` extension but it's tab-separated. How do I load it?**

```python
df = eda.read("data.txt", sep="\t")
```

`.txt` is treated as CSV by default. Pass `sep=` to override.

---

**Q: How does NowEDA handle very large files?**

Large Parquet/ORC files (at least 128 MB) without reader options may use Spark. CSV/JSON and reads with options use pandas. The final DataFrame must fit in RAM; Spark is not guaranteed to be faster. For files that do not fit in memory, use `read_chunked(..., concat=False)` on CSV or line-delimited JSON.

---

**Q: I have a multi-sheet Excel file. How do I load a specific sheet?**

```python
df = eda.read("report.xlsx", sheet_name="Q3 Sales")
```

All `**kwargs` are passed to `pd.read_excel()`.

---

**Q: Can NowEDA read files from a URL?**

Not directly — `eda.read()` / `noweda.read()` requires a local file path. Download first:

```python
import urllib.request
urllib.request.urlretrieve("https://example.com/data.csv", "data.csv")
df = eda.read("data.csv")
```

---

**Q: I get `ImportError: Reading .parquet files requires 'pyarrow'`. How do I fix it?**

```bash
pip install "noweda[parquet]"
```

---

## Plugins & Analysis

**Q: How do I run only specific plugins?**

```python
from noweda.core.engine import AutoEDAEngine
from noweda.plugins.missing import MissingDataPlugin
from noweda.plugins.pii import PIIDetectorPlugin

engine = AutoEDAEngine([MissingDataPlugin(), PIIDetectorPlugin()])
report = engine.run_df(df)
```

---

**Q: My DataFrame has no numeric columns and `correlation` returns `{}`. Is that a bug?**

No. The Correlation plugin only runs on numeric columns. With no numeric data, there's nothing to correlate.

---

**Q: The Schema plugin marked my `age` column as `id_candidate`. Why?**

`id_candidate` for integer columns requires 100% uniqueness. If your `age` column has all distinct values (no repeated ages), it triggers this rule. In practice, adding more rows usually causes some ages to repeat and removes the flag. You can also just ignore this insight for domain columns you know are not IDs.

---

**Q: The PII plugin didn't flag my phone numbers. Why?**

The built-in plugin checks ten-digit US-style phone numbers in text columns, with an optional `+1`. Other countries, short local numbers and numeric-typed columns may be missed. Card-shaped spans are excluded to avoid false phone matches. See [PII detection](plugins/pii.md) for supported formats.

---

## Scoring

**Q: My `model_readiness` score is very low even though my data looks clean. Why?**

Common causes:
- Heavily skewed columns (`|skewness| > 2`) — each one reduces readiness by 5
- Many `text`-role columns (free text needs NLP preprocessing) — each reduces by 3
- Missing values — even small amounts reduce readiness

Check `df.noweda.summary()["stats"]` for skewness values and `df.noweda.summary()["schema"]` for column roles.

---

**Q: How do I interpret a risk score of 25?**

| Risk | Level |
|---|---|
| 0 | No risk signals |
| 1–20 | Low |
| 21–50 | Moderate |
| >50 | High |

A risk of 25 means one PII column (e.g., emails, +15) and one encoded column (+10) were detected. Check `df.noweda.summary()["pii"]` and `["encoding"]` to see exactly what was found.

---

## Reports

**Q: Can I share the HTML report with non-technical stakeholders?**

Yes. The HTML report is fully self-contained — no JavaScript, no external CSS, no server needed. Email the `.html` file directly.

---

**Q: Can I customise the HTML report colours or layout?**

Yes. The HTML generator is in `noweda/report/html.py`. All CSS is inline. Modify the CSS variables at the top of the template or add/remove sections as needed.

---

## Contributing

**Q: How do I contribute a new plugin?**

1. Create your plugin file in `noweda/plugins/`
2. Inherit from `BasePlugin`, set a unique `name`, implement `run(df)`
3. Add it to `tests/`
4. Open a pull request at [github.com/codewithdaniel1/NowEDA](https://github.com/codewithdaniel1/NowEDA)
