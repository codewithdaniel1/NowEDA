from noweda.dtypes import is_textual
import hashlib
import pandas as pd
from functools import wraps
from noweda.core.engine import AutoEDAEngine
from noweda.plugins import default_plugins
from noweda.ml_utils import (
    calculate_vif, cardinality_warning, rare_category_detection,
    get_scaling_recommendation, get_transformation_suggestion, assess_column_quality
)
from noweda.temporal_utils import detect_temporal_columns, stationarity_test, detect_seasonality
from noweda.ml_tasks import build_ml_guidance, format_ml_plan
from noweda.ui import loading


def _analysis_sample(df, sample, method_name):
    """Return a bounded deterministic sample for an expensive analysis method."""
    if sample is None:
        return df
    if isinstance(sample, bool) or not isinstance(sample, int) or sample <= 0:
        raise ValueError("sample must be a positive integer")
    if len(df) <= sample:
        return df
    print(
        f"NowEDA · {method_name}: sample-based results use {sample:,} of "
        f"{len(df):,} rows. Estimates may differ from the full dataset."
    )
    return df.sample(n=sample, random_state=42)

@pd.api.extensions.register_dataframe_accessor("noweda")
@pd.api.extensions.register_dataframe_accessor("eda")
class NowEDAAccessor:

    def __init__(self, pandas_obj):
        self._df = pandas_obj
        # pandas 3 creates a new accessor on each attribute access. Keep state
        # on the DataFrame, outside attrs (which pandas propagates to copies).
        self._cache = pandas_obj.__dict__.setdefault(
            "_noweda_report_cache", {"report": None, "fingerprint": None}
        )

    @property
    def _report(self):
        return self._cache["report"]

    @_report.setter
    def _report(self, value):
        self._cache["report"] = value

    def _ensure_analyzed(self):
        # Check values, row order, index, column names and dtype metadata.
        digest = hashlib.sha256(pd.util.hash_pandas_object(self._df, index=True).values.tobytes())
        digest.update(repr(tuple(self._df.columns)).encode())
        digest.update(repr([repr(dtype) for dtype in self._df.dtypes]).encode())
        fingerprint = digest.digest()
        if self._report is None or fingerprint != self._cache["fingerprint"]:
            engine = AutoEDAEngine(default_plugins())
            self._report = engine.run_df(self._df)
            self._cache["fingerprint"] = fingerprint

    def refresh(self):
        """Force analysis to run again and return the complete report."""
        self._report = None
        self._ensure_analyzed()
        return self._report

    def summary(self):
        """Return raw results from all built-in plugins."""
        self._ensure_analyzed()
        return self._report["results"]

    def insights(self):
        self._ensure_analyzed()
        return self._report["insights"]

    def score(self):
        self._ensure_analyzed()
        return self._report["scores"]

    def report(self):
        """Return the complete analysis report dict (for programmatic use, e.g., HTML export)."""
        self._ensure_analyzed()
        return self._report

    # ────────────────────────────────────────────────────────────────────────
    # Convenience one-liner methods — Quick access to individual tables
    # ────────────────────────────────────────────────────────────────────────

    def scores_df(self):
        """Return data quality, model readiness, and risk scores as a DataFrame.

        Returns:
            DataFrame with shape (3, 1) containing:
                - data_quality (0-100): Score penalised for missing values, duplicates, constants, outliers
                - model_readiness (0-100): Score penalised for skew, untyped columns, high missingness
                - risk (0+): Score incremented per PII column (+15) and encoded column (+10)

        Example:
            scores = df.eda.scores_df()
            print(scores)
            #               Value
            # data_quality    87.5
            # model_readiness  82.1
            # risk            15.0
        """
        self._ensure_analyzed()
        scores = self._report["scores"]
        return pd.DataFrame(scores, index=[0]).T.rename(columns={0: "Value"})

    def insights_df(self, full_line=True):
        """Return human-readable insights about data quality, patterns, and issues as a DataFrame.

        Parameters:
            full_line (bool, default=True): If True, display complete insight text. If False, truncate
                insights to ~50 characters for compact display in notebooks.

        Returns:
            DataFrame with single column 'Insight' containing human-readable findings.

        Example:
            # Full insights (default)
            insights = df.eda.insights_df()

            # Compact insights (truncated for brief review)
            insights_compact = df.eda.insights_df(full_line=False)
        """
        self._ensure_analyzed()
        insights = self._report["insights"]

        if full_line:
            df_result = pd.DataFrame({"Insight": insights})
        else:
            truncated = [text[:50] + "..." if len(text) > 50 else text for text in insights]
            df_result = pd.DataFrame({"Insight": truncated})

        return df_result

    def schema_df(self):
        """Return Column, dtype, role, confidence, unique and uniqueness_ratio."""
        self._ensure_analyzed()
        schema = self._report["results"].get("schema", {})
        df_result = pd.DataFrame(schema).T
        df_result.index.name = "Column"
        return df_result.reset_index()

    def stats_df(self):
        """Return per-column statistics with lowercase field names.

        Numeric fields include mean, median, std, min, max, q25, q75,
        skewness and kurtosis. Categorical fields include top_value and top_freq.
        Undefined numeric statistics are NaN.
        """
        self._ensure_analyzed()
        stats = self._report["results"].get("stats", {})
        df_result = pd.DataFrame(stats).T
        df_result.index.name = "Column"
        return df_result.reset_index()

    def missing_df(self, format="percentage"):
        """Return missing data rates per column as a DataFrame.

        Parameters:
            format (str, default='percentage'): Display format for missing data.
                - 'percentage': Show as % (e.g., "15.3%")
                - 'number': Show as absolute count (e.g., 306 missing values)

        Returns:
            DataFrame with columns: [Column, Missing_Data]

        Example:
            # Default: show missing data as percentage
            missing = df.eda.missing_df()
            print(missing)
            #        Column Missing_Data
            # 0         age          2.5%
            # 1       salary         15.3%

            # Alternative: show absolute missing counts
            missing = df.eda.missing_df(format='number')
            print(missing)
            #        Column Missing_Data
            # 0         age            50
            # 1       salary          306
        """
        self._ensure_analyzed()
        missing = self._report["results"].get("missing", {})
        df_out = pd.DataFrame(list(missing.items()), columns=["Column", "Missing_Data"])

        if format == "number":
            # Convert to counts (missing value count per column)
            df_out["Missing_Data"] = (df_out["Missing_Data"] * len(self._df)).astype(int)
        else:  # percentage
            df_out["Missing_Data"] = df_out["Missing_Data"].apply(lambda x: f"{x*100:.1f}%")

        return df_out

    def duplicates_df(self):
        """Return duplicate rows and constant column information as a DataFrame.

        Returns:
            DataFrame with 3-row structure showing:
                - Total Rows: Total number of rows in dataset
                - Duplicate Rows: Count of exactly duplicate rows
                - Constant Columns: List of columns with only one unique value

        Example:
            duplicates = df.eda.duplicates_df()
            print(duplicates)
            #                   Metric  Value
            # 0             Total Rows  2000
            # 1        Duplicate Rows    15
            # 2      Constant Columns     3
        """
        self._ensure_analyzed()
        dups = self._report["results"].get("duplicates", {})
        return pd.DataFrame({
            "Metric": ["Total Rows", "Duplicate Rows", "Constant Columns"],
            "Value": [
                len(self._df),
                dups.get("duplicate_rows", 0),
                ", ".join(map(str, dups.get("constant_columns", []))) or "None"
            ]
        })

    def correlation_df(self):
        """Return Pearson correlation matrix between all numeric columns as a DataFrame.

        Returns:
            DataFrame: Symmetric correlation matrix where each cell [i,j] represents
                the Pearson correlation coefficient between columns i and j.
                Values range from -1 to 1 (perfect negative to perfect positive correlation).

        Example:
            corr = df.eda.correlation_df()
            print(corr)
            #            age    salary    score
            # age       1.00     0.65     0.42
            # salary    0.65     1.00     0.78
            # score     0.42     0.78     1.00

        Notes:
            - Only numeric columns are included
            - Perfect correlation (±1.0) indicates linear dependence
            - High correlation (|r| > 0.85) may indicate multicollinearity issues
        """
        self._ensure_analyzed()
        corr = self._report["results"].get("correlation", {})
        return pd.DataFrame(corr)

    def outliers_df(self, format="number"):
        """Return outlier counts per numeric column as a DataFrame.

        Uses IQR (Interquartile Range) method to detect outliers:
            - Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are flagged as outliers

        Parameters:
            format (str, default='number'): Display format for outlier counts.
                - 'number': Show absolute outlier counts (e.g., 12 outliers)
                - 'percentage': Show as % of total rows (e.g., "0.6%")

        Returns:
            DataFrame with columns: [Column, Outliers]

        Example:
            # Default: show outlier counts
            outliers = df.eda.outliers_df()
            print(outliers)
            #      Column  Outliers
            # 0       age         12
            # 1    salary         25

            # Alternative: show outliers as percentage
            outliers = df.eda.outliers_df(format='percentage')
            print(outliers)
            #      Column  Outliers
            # 0       age      0.6%
            # 1    salary      1.2%
        """
        self._ensure_analyzed()
        outliers = self._report["results"].get("outliers", {})
        df_out = pd.DataFrame(list(outliers.items()), columns=["Column", "Outliers"])

        if format == "percentage":
            # Convert to percentage of rows
            df_out["Outliers"] = (df_out["Outliers"] / len(self._df) * 100).apply(lambda x: f"{x:.1f}%")

        return df_out

    def pii_df(self):
        """Return detected Personally Identifiable Information (PII) as a DataFrame.

        NowEDA scans for common PII patterns including:
            - Email addresses
            - Phone numbers
            - Social Security Numbers (SSNs)
            - Credit card numbers

        Returns:
            DataFrame with columns: [Column, PII_Type, Count]
                - Column: Column name containing PII
                - PII_Type: Type of PII detected ('email', 'phone', 'ssn', 'credit_card')
                - Count: Number of values matching this PII pattern

            Returns empty DataFrame if no PII detected.

        Example:
            pii = df.eda.pii_df()
            print(pii)
            #        Column     PII_Type  Count
            # 0  customer_email       email    42
            # 1  customer_phone       phone    38
            # 2  ssn_column           ssn     50
        """
        self._ensure_analyzed()
        pii = self._report["results"].get("pii", {})
        if not pii:
            return pd.DataFrame({"Column": [], "PII_Type": [], "Count": []})

        # Flatten the nested structure: for each column, expand findings dict into rows
        rows = []
        for col, findings in pii.items():
            for pii_type, count in findings.items():
                rows.append({"Column": col, "PII_Type": pii_type, "Count": count})

        return pd.DataFrame(rows)

    def encoding_df(self, include_confidence=False):
        """Return Column and Encoding_Type for possible Base64 signals.

        Detection samples up to the first 20 nonmissing values per text column.
        With include_confidence=True, add sample size, matches and confidence
        (the sample match fraction, not a calibrated probability).
        """
        self._ensure_analyzed()
        encoding = self._report["results"].get("encoding", {})
        table = pd.DataFrame(list(encoding.items()), columns=["Column", "Encoding_Type"])
        if include_confidence:
            details = self._report.get("encoding_details", {})
            for label, key in [("Sample_Size", "sample_size"), ("Matches", "matches"),
                               ("Confidence", "confidence")]:
                table[label] = [details.get(col, {}).get(key) for col in encoding]
        return table

    def statsall(self, sample=None):
        """Print a rich full-analysis report to the terminal/notebook.

        Combines: dtypes, describe-style stats per column, scores,
        insights, and a structured summary — all in one call. By default it
        analyzes every input row. Set ``sample`` to a positive integer to use
        a deterministic sample and announce that scope.
        """
        analysis_df = _analysis_sample(self._df, sample, "statsall()")
        if analysis_df is not self._df:
            return analysis_df.eda.statsall(sample=sample)
        self._ensure_analyzed()
        df = self._df
        report = self._report
        results = report["results"]
        scores = report["scores"]
        insights = report["insights"]
        stats = results.get("stats", {})
        schema = results.get("schema", {})

        _BOLD  = "\033[1m"
        _CYAN  = "\033[36m"
        _GREEN = "\033[32m"
        _YELLOW = "\033[33m"
        _RED   = "\033[31m"
        _RESET = "\033[0m"

        def h1(text):
            bar = "=" * 70
            print(f"\n{_BOLD}{_CYAN}{bar}{_RESET}")
            print(f"{_BOLD}{_CYAN}  {text}{_RESET}")
            print(f"{_BOLD}{_CYAN}{bar}{_RESET}")

        def h2(text):
            print(f"\n{_BOLD}{text}{_RESET}")
            print("-" * 60)

        def score_color(val, low=40, mid=70):
            if val >= mid:
                return f"{_GREEN}{val}{_RESET}"
            elif val >= low:
                return f"{_YELLOW}{val}{_RESET}"
            return f"{_RED}{val}{_RESET}"

        # ── Header ──────────────────────────────────────────────────────────
        h1("NowEDA · Full Statistical Report")
        print(f"  Rows    : {_BOLD}{len(df):,}{_RESET}")
        print(f"  Columns : {_BOLD}{len(df.columns)}{_RESET}")

        if df.empty:
            print("  Empty DataFrame: no rows or no columns to analyze.")
            return

        # ── Scores ───────────────────────────────────────────────────────────
        h2("Scores")
        dq  = scores.get("data_quality", "N/A")
        mr  = scores.get("model_readiness", "N/A")
        risk = scores.get("risk", "N/A")
        print(f"  Data Quality    : {score_color(dq)  if isinstance(dq, (int,float))  else dq} out of 100")
        print(f"  Model Readiness : {score_color(mr)  if isinstance(mr, (int,float))  else mr} out of 100")
        print(f"  Risk            : {_RED if isinstance(risk,(int,float)) and risk>0 else _GREEN}{risk}{_RESET}  (0 = no risk)")

        # ── Dtypes ───────────────────────────────────────────────────────────
        h2("Column Types")
        col_w = max((len(str(c)) for c in df.columns), default=6) + 2
        role_w = 20
        print(f"  {'Column':<{col_w}} {'Dtype':<14} {'Role':<{role_w}} {'Conf':>5} {'Unique':>8} {'Missing':>8}")
        print(f"  {'-'*col_w} {'-'*14} {'-'*role_w} {'-----':>5} {'--------':>8} {'--------':>8}")
        for col in df.columns:
            dtype = str(df[col].dtype)
            role  = schema.get(col, {}).get("role", "unknown")
            conf  = schema.get(col, {}).get("confidence", 1.0)
            uniq  = df[col].nunique()
            miss  = int(df[col].isna().sum())
            # Apply colour without breaking right-alignment
            miss_display = f"{_YELLOW}{miss:>8}{_RESET}" if miss > 0 else f"{miss:>8}"
            # Color confidence: high (>0.9) green, medium yellow, low red
            conf_color = _GREEN if conf >= 0.9 else (_YELLOW if conf >= 0.75 else _RED)
            conf_display = f"{conf_color}{conf:>5.2f}{_RESET}"
            print(f"  {str(col):<{col_w}} {dtype:<14} {role:<{role_w}} {conf_display} {uniq:>8} {miss_display}")

        # ── Per-column stats ─────────────────────────────────────────────────
        h2("Descriptive Statistics")

        # Numeric columns
        num_cols = [c for c in df.columns if df[c].dtype.kind in ("i", "u", "f")]
        if num_cols:
            print(f"\n  {'Column':<{col_w}} {'Count':>8} {'Mean':>12} {'Std':>12} {'Min':>10} {'25%':>10} {'50%':>10} {'75%':>10} {'Max':>10} {'Skew':>8}")
            print(f"  {'-'*col_w} {'--------':>8} {'------------':>12} {'------------':>12} {'----------':>10} {'----------':>10} {'----------':>10} {'----------':>10} {'----------':>10} {'--------':>8}")
            for col in num_cols:
                s = stats.get(col, {})
                count  = s.get("count", 0)
                mean   = s.get("mean", float("nan"))
                std    = s.get("std",  float("nan"))
                mn     = s.get("min",  float("nan"))
                q25    = s.get("q25",  float("nan"))
                med    = s.get("median", float("nan"))
                q75    = s.get("q75",  float("nan"))
                mx     = s.get("max",  float("nan"))
                skew   = s.get("skewness", float("nan"))
                skew_str = f"{skew:>8.2f}"
                if abs(skew) > 1:
                    skew_str = f"{_YELLOW}{skew_str}{_RESET}"
                print(f"  {str(col):<{col_w}} {count:>8,} {mean:>12.4g} {std:>12.4g} {mn:>10.4g} {q25:>10.4g} {med:>10.4g} {q75:>10.4g} {mx:>10.4g} {skew_str}")

        # Categorical / text columns
        cat_cols = [c for c in df.columns if is_textual(df[c])]
        if cat_cols:
            try:
                import numpy as np
            except ImportError:
                np = None

            print(f"\n  {'Column':<{col_w}} {'Count':>8} {'Unique':>8} {'Diversity':>10} {'Top Value':<28} {'Freq %':>8}")
            print(f"  {'-'*col_w} {'--------':>8} {'--------':>8} {'----------':>10} {'-'*28} {'--------':>8}")

            for col in cat_cols:
                s = stats.get(col, {})
                count = s.get("count", 0)
                uniq  = s.get("unique", 0)
                top   = str(s.get("top_value", "N/A"))[:26]
                freq  = s.get("top_freq", 0)

                # Calculate diversity: entropy-based (0=one dominant, 1=uniform)
                if count > 0 and uniq > 0 and np is not None:
                    value_counts = df[col].value_counts(normalize=True)
                    # Shannon entropy: higher = more uniform, lower = more imbalanced
                    # Normalize to 0-1 range where 1=perfectly balanced, 0=completely dominated
                    entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
                    max_entropy = np.log2(min(uniq, count))
                    diversity = (entropy / max_entropy) if max_entropy > 0 else 0
                    diversity_color = f"{_GREEN}{diversity:.2f}{_RESET}" if diversity > 0.7 else (f"{_YELLOW}{diversity:.2f}{_RESET}" if diversity > 0.4 else f"{_RED}{diversity:.2f}{_RESET}")
                else:
                    diversity_color = "N/A"

                freq_pct = (freq / count * 100) if count > 0 else 0
                print(f"  {str(col):<{col_w}} {count:>8,} {uniq:>8} {diversity_color:>10} {top:<28} {freq_pct:>7.1f}%")

        # ── Insights ──────────────────────────────────────────────────────────
        h2("Insights")
        if insights:
            for ins in insights:
                print(f"  • {ins}")
        else:
            print("  No issues detected.")

        # ── Column Quality Summary ────────────────────────────────────────────
        h2("Column Quality Summary")
        for col in df.columns:
            role = schema.get(col, {}).get("role", "unknown")
            status, issues = assess_column_quality(df[col], role)

            # Color code status
            if status.startswith("✓"):
                status_colored = f"{_GREEN}{status}{_RESET}"
            elif status.startswith("⚠"):
                status_colored = f"{_YELLOW}{status}{_RESET}"
            else:
                status_colored = f"{_RED}{status}{_RESET}"

            # Print column status
            if issues:
                issues_str = "; ".join(issues)
                print(f"  {str(col):<{col_w}} {status_colored:>15}  ({issues_str})")
            else:
                print(f"  {str(col):<{col_w}} {status_colored}")

        # ── Temporal Analysis ─────────────────────────────────────────────────
        temporal = detect_temporal_columns(df)
        h2("Temporal Data Analysis")
        if temporal:
            for col, (dtype, frequency, confidence) in temporal.items():
                print(f"\n  {_BOLD}{col}{_RESET}")
                print(f"    Type        : {dtype} (confidence: {confidence:.0%})")
                print(f"    Frequency   : {frequency}")

                # Test stationarity for numeric time series
                if df[col].dtype.kind in ("i", "u", "f"):
                    is_stat, p_val = stationarity_test(df[col])
                    if is_stat is not None:
                        status = f"{_GREEN}Stationary{_RESET}" if is_stat else f"{_YELLOW}Non-stationary{_RESET}"
                        print(f"    Stationarity: {status} (p={p_val:.3f})")

                    # Seasonality for longer series
                    has_season, strength = detect_seasonality(df[col])
                    if has_season or strength > 0:
                        print(f"    Seasonality : {_YELLOW}Detected{_RESET} (strength={strength:.2f})" if has_season else f"    Seasonality : None (strength={strength:.2f})")
        else:
            print("  No datetime columns detected.")

        # ── Plugin Summary ────────────────────────────────────────────────────
        h2("Plugin Summary")
        missing = results.get("missing", {})
        missing_columns = [col for col, rate in missing.items() if rate > 0]
        if missing_columns:
            highest_missing = max(missing_columns, key=lambda col: missing[col])
            print(f"  Missing data    : {len(missing_columns)} column(s); highest is {highest_missing} ({missing[highest_missing]:.1%})")
        else:
            print("  Missing data    : none")

        duplicates = results.get("duplicates", {})
        duplicate_rows = duplicates.get("duplicate_rows", 0)
        duplicate_pct = duplicates.get("duplicate_rows_pct", 0.0)
        constants = duplicates.get("constant_columns", [])
        duplicate_summary = f"{duplicate_rows:,} row(s) ({duplicate_pct:.1%})"
        if constants:
            duplicate_summary += f"; constant: {', '.join(map(str, constants[:3]))}"
            if len(constants) > 3:
                duplicate_summary += f" and {len(constants) - 3} more"
        print(f"  Duplicates      : {duplicate_summary}")

        correlation = results.get("correlation", {})
        strong_pairs = []
        seen_pairs = set()
        if isinstance(correlation, dict):
            for left, values in correlation.items():
                if not isinstance(values, dict):
                    continue
                for right, value in values.items():
                    pair = tuple(sorted((str(left), str(right))))
                    if left == right or pair in seen_pairs or not isinstance(value, (int, float)):
                        continue
                    seen_pairs.add(pair)
                    if abs(value) >= 0.7:
                        strong_pairs.append((left, right, value))
        if strong_pairs:
            strong_pairs.sort(key=lambda item: -abs(item[2]))
            examples = ", ".join(
                f"{left} ↔ {right} ({value:.2f})" for left, right, value in strong_pairs[:2]
            )
            suffix = f" and {len(strong_pairs) - 2} more" if len(strong_pairs) > 2 else ""
            print(f"  Correlations    : {len(strong_pairs)} strong pair(s): {examples}{suffix}")
        else:
            print("  Correlations    : no strong pairs (|r| ≥ 0.70)")

        outliers = results.get("outliers", {})
        outlier_columns = {col: count for col, count in outliers.items() if count > 0}
        if outlier_columns:
            print(f"  Outliers        : {sum(outlier_columns.values()):,} finding(s) across {len(outlier_columns)} column(s)")
        else:
            print("  Outliers        : none")

        pii = results.get("pii", {})
        if pii:
            pii_columns = ", ".join(map(str, list(pii)[:4]))
            suffix = f" and {len(pii) - 4} more" if len(pii) > 4 else ""
            print(f"  PII             : {len(pii)} column(s): {pii_columns}{suffix}")
        else:
            print("  PII             : none detected")

        encoding = results.get("encoding", {})
        if encoding:
            encoding_columns = ", ".join(map(str, list(encoding)[:4]))
            suffix = f" and {len(encoding) - 4} more" if len(encoding) > 4 else ""
            print(f"  Encoded values  : {len(encoding)} column(s): {encoding_columns}{suffix}")
        else:
            print("  Encoded values  : none detected")

        # ── ML Preprocessing Guide ────────────────────────────────────────────
        h2("ML Preprocessing Recommendations")

        # Multicollinearity
        num_cols = [c for c in df.columns if df[c].dtype.kind in ("i", "u", "f")]
        preprocessing_cols = []
        vif_cols = []
        excluded_preprocessing_cols = []
        for col in num_cols:
            observed = df[col].dropna()
            role = schema.get(col, {}).get("role", "unknown")
            if observed.nunique() > 2:
                vif_cols.append(col)
            if role == "id_candidate" or observed.nunique() <= 2:
                excluded_preprocessing_cols.append(col)
            else:
                preprocessing_cols.append(col)

        if excluded_preprocessing_cols:
            print(
                "  Excluding binary indicators and likely identifiers from generic "
                "preprocessing: {}".format(", ".join(map(str, excluded_preprocessing_cols)))
            )

        if len(vif_cols) >= 2:
            vif_data = calculate_vif(df, vif_cols)
            if vif_data:
                high_vif = {col: vif for col, vif in vif_data.items() if isinstance(vif, (int, float)) and vif > 5}
                if high_vif:
                    print(f"\n  {_YELLOW}⚠ Multicollinearity Detected (VIF > 5):{_RESET}")
                    for col, vif in sorted(high_vif.items(), key=lambda x: -x[1])[:5]:
                        print(f"    {str(col):20s}: VIF = {vif:>6.1f}  → Consider dropping or combining")
                elif all(pd.notna(vif) for vif in vif_data.values()):
                    print(f"\n  {_GREEN}✓ Low Multicollinearity (all VIF <= 5){_RESET}")
                if any(pd.isna(vif) for vif in vif_data.values()):
                    print("  VIF unavailable for constant columns or insufficient complete observations.")
            else:
                print(f"\n  {_YELLOW}Multicollinearity Assessment: unavailable for the selected numeric columns{_RESET}")
        else:
            print(f"\n  {_GREEN}Multicollinearity Assessment: need at least two numeric columns{_RESET}")

        # Scaling recommendations
        scaling_needed = []
        for col in preprocessing_cols:
            rec = get_scaling_recommendation(df[col])
            if rec and "scale" in rec:
                scaling_needed.append(col)

        if scaling_needed:
            print(f"\n  {_YELLOW}Scaling Recommended:{_RESET}")
            for col in scaling_needed:
                print(f"    {str(col):20s}: Use StandardScaler or MinMaxScaler")
        else:
            print(f"\n  {_GREEN}Scaling Recommendation: no numeric columns need scaling guidance{_RESET}")

        # Transformations
        transform_candidates = []
        for col in preprocessing_cols:
            suggestion = get_transformation_suggestion(df[col])
            if suggestion:
                transform_candidates.append((col, suggestion))

        if transform_candidates:
            print(f"\n  {_YELLOW}Transformation Suggestions:{_RESET}")
            for col, suggestion in transform_candidates[:5]:
                print(f"    {str(col):20s}: {suggestion}")
            if len(transform_candidates) > 5:
                print(f"    … and {len(transform_candidates) - 5} more")
        else:
            print(f"\n  {_GREEN}Transformation Suggestions: none based on the observed distributions{_RESET}")

        # Role-aware feature review. High uniqueness is normal for continuous
        # numeric values and means something different for IDs, PII, dates,
        # free text, and true categorical values.
        pii = results.get("pii", {})
        feature_review = []
        for col in df.columns:
            role = schema.get(col, {}).get("role", "unknown")
            warning = cardinality_warning(
                df[col], role=role, pii_types=pii.get(col)
            )
            if warning:
                feature_review.append((col, warning))

        if feature_review:
            print(f"\n  {_YELLOW}Feature Review (Identifiers, PII, Text, and Cardinality):{_RESET}")
            for col, issue in feature_review:
                print(f"    {str(col):20s}: {issue}")
        else:
            print(f"\n  {_GREEN}Feature Review: no identifier, PII, text, or cardinality concerns detected{_RESET}")

        # Rare categories
        rare_issues = {}
        for col in df.columns:
            role = schema.get(col, {}).get("role", "unknown")
            rare = rare_category_detection(df[col], threshold=0.01, role=role)
            if rare:
                rare_issues[col] = rare

        if rare_issues:
            print(f"\n  {_YELLOW}Rare Categories Detected (<1%):{_RESET}")
            for col, rare_cats in list(rare_issues.items())[:5]:
                rare_str = ", ".join(f"{k}({v:.1%})" for k, v in list(rare_cats.items())[:2])
                print(f"    {str(col):20s}: {rare_str}")
                if len(rare_cats) > 2:
                    print(f"                     {' and ' + str(len(rare_cats) - 2) + ' more rare values'}")
        else:
            print(f"\n  {_GREEN}Rare Categories: none below 1% prevalence{_RESET}")

        # Missing data handling
        cols_with_missing = [c for c in df.columns if df[c].isna().sum() > 0]
        if cols_with_missing:
            print(f"\n  {_YELLOW}Missing Data Strategy:{_RESET}")
            for col in cols_with_missing[:5]:
                missing_pct = df[col].isna().sum() / len(df) * 100
                if missing_pct > 50:
                    rec = f"consider dropping ({missing_pct:.0f}% missing)"
                elif missing_pct > 20:
                    rec = f"impute or drop ({missing_pct:.0f}% missing)"
                else:
                    rec = f"impute ({missing_pct:.0f}% missing)"
                print(f"    {str(col):20s}: {rec}")
        else:
            print(f"\n  {_GREEN}Missing Data Strategy: no missing values detected{_RESET}")

        print(f"\n{_CYAN}{'='*70}{_RESET}\n")

    def vizall(self, sample=None, target=None, max_plots=15):
        """Render ranked, ML-aware visual diagnostics.

        ``vizall()`` selects a bounded set of general diagnostics. Supplying a
        target adds supervised feature-versus-target charts while preserving
        the existing workflow. The returned dictionary-like result contains
        the figures, ranked evidence, selected features, and scope metadata.

        Parameters
        ----------
        sample : int or None, default None
            Analyze every loaded row by default, or use a deterministic sample.
        target : column label or None, default None
            User-selected prediction target. NowEDA infers classification or
            regression from its observed values but never selects a target.
        max_plots : int, default 15
            Maximum number of individual chart panels to render.
        """
        from noweda.visualization import render_vizall

        analysis_df = _analysis_sample(self._df, sample, "vizall()")
        if analysis_df is not self._df:
            result = analysis_df.eda.vizall(
                sample=sample, target=target, max_plots=max_plots
            )
            result["scope"] = {
                "rows": int(len(self._df)),
                "sample_rows": int(len(analysis_df)),
                "sample_based": True,
            }
            return result

        self._ensure_analyzed()
        return render_vizall(
            self._df, self._report, target=target, max_plots=max_plots
        )

    def mlall(self, target=None, problem_type=None, features=None, plan=False, sample=None):
        """Print task-aware ML guidance and optionally return its structured result.

        With no objective, assess likely supervised and unsupervised directions.
        Classification and regression require ``target``; clustering, anomaly
        detection, and dimensionality reduction use ``problem_type``. Set
        ``plan=True`` to return the same structured guidance after it is printed.
        No models are fitted or evaluated.
        """
        analysis_df = _analysis_sample(self._df, sample, "mlall()")
        if analysis_df is not self._df:
            return analysis_df.eda.mlall(
                target=target, problem_type=problem_type, features=features,
                plan=plan, sample=sample,
            )
        self._ensure_analyzed()
        result = build_ml_guidance(
            self._df, self._report, target=target,
            problem_type=problem_type, features=features,
        )
        if target is not None:
            from noweda.visualization import build_visual_diagnostics

            target_values = self._df.iloc[:, self._df.columns.get_loc(target)]
            if target_values.notna().any():
                visual = build_visual_diagnostics(
                    self._df, self._report, target=target
                )
                result["visual_diagnostics"] = {
                    key: value for key, value in visual.items()
                    if key not in {"numeric_scales"}
                }
        format_ml_plan(result)
        visual = result.get("visual_diagnostics")
        if visual and visual.get("model_signals"):
            print("\n  Visual diagnostics to validate with vizall(target={!r}):".format(target))
            for signal in visual["model_signals"]:
                print("    - {}".format(signal))
        if plan:
            return result

    def profile_column(self, column_name):
        """Deep dive into a single column's characteristics and recommendations.

        Shows:
          - Distribution type (normal, skewed, multimodal, etc.)
          - Suggested transformations with statistical justification
          - Outlier explanation
          - Role confidence and alternative interpretations
        """
        self._ensure_analyzed()
        df = self._df
        report = self._report
        results = report["results"]
        stats = results.get("stats", {})
        schema = results.get("schema", {})

        if column_name not in df.columns:
            print(f"Column '{column_name}' not found in DataFrame.")
            return

        col = df[column_name]
        col_stats = stats.get(column_name, {})
        col_schema = schema.get(column_name, {})

        _BOLD  = "\033[1m"
        _CYAN  = "\033[36m"
        _GREEN = "\033[32m"
        _YELLOW = "\033[33m"
        _RED   = "\033[31m"
        _RESET = "\033[0m"

        print(f"\n{_BOLD}{_CYAN}{'='*70}{_RESET}")
        print(f"{_BOLD}{_CYAN}  Column Profile: {column_name}{_RESET}")
        print(f"{_BOLD}{_CYAN}{'='*70}{_RESET}\n")

        # Basic Info
        print(f"{_BOLD}Type Information:{_RESET}")
        print(f"  Data Type       : {col_schema.get('dtype', 'unknown')}")
        print(f"  Inferred Role   : {col_schema.get('role', 'unknown')} (confidence: {col_schema.get('confidence', 'N/A')})")
        print(f"  Unique Values   : {col.nunique()}")
        print(f"  Missing         : {col.isna().sum()} ({col.isna().sum() / len(col) * 100:.1f}%)")

        # Numeric-specific analysis
        if col.dtype.kind in ('i', 'u', 'f'):
            print(f"\n{_BOLD}Numeric Statistics:{_RESET}")
            print(f"  Mean            : {col_stats.get('mean', 'N/A'):.4g}")
            print(f"  Median          : {col_stats.get('median', 'N/A'):.4g}")
            print(f"  Std Dev         : {col_stats.get('std', 'N/A'):.4g}")
            print(f"  Min             : {col_stats.get('min', 'N/A'):.4g}")
            print(f"  Max             : {col_stats.get('max', 'N/A'):.4g}")
            print(f"  Q1 (25%)        : {col_stats.get('q25', 'N/A'):.4g}")
            print(f"  Q3 (75%)        : {col_stats.get('q75', 'N/A'):.4g}")

            # Distribution insights
            skewness = col_stats.get('skewness', 0)
            print(f"\n{_BOLD}Distribution:{_RESET}")
            if abs(skewness) < 0.5:
                print(f"  Shape           : Symmetric/Normal-like")
                print(f"  Skewness        : {skewness:.3f} (minimal skew)")
            elif abs(skewness) < 1:
                print(f"  Shape           : Moderately skewed")
                print(f"  Skewness        : {skewness:.3f}")
                if skewness > 0:
                    print(f"  Direction       : Right-skewed (long tail right)")
                    print(f"  Recommendation  : Try sqrt or log transformation")
                else:
                    print(f"  Direction       : Left-skewed (long tail left)")
                    print(f"  Recommendation  : Try reflecting then log/sqrt")
            else:
                print(f"  Shape           : Highly skewed")
                print(f"  Skewness        : {skewness:.3f}")
                if skewness > 0:
                    print(f"  Direction       : Right-skewed (strong positive skew)")
                else:
                    print(f"  Direction       : Left-skewed (strong negative skew)")
                print(f"  Recommendation  : {_YELLOW}log1p transform strongly recommended{_RESET}")

        # Categorical-specific analysis
        elif is_textual(col):
            print(f"\n{_BOLD}Categorical Statistics:{_RESET}")
            print(f"  Top Value       : {col_stats.get('top_value', 'N/A')}")
            print(f"  Frequency       : {col_stats.get('top_freq', 'N/A')} ({col_stats.get('top_freq', 0) / len(col) * 100:.1f}%)")

            value_counts = col.value_counts()
            if len(value_counts) <= 10:
                print(f"\n{_BOLD}All Values:{_RESET}")
                for val, count in value_counts.items():
                    pct = count / len(col) * 100
                    print(f"  {str(val):<20} : {count:>5} ({pct:>5.1f}%)")
            else:
                print(f"\n{_BOLD}Top 10 Values:{_RESET}")
                for val, count in value_counts.head(10).items():
                    pct = count / len(col) * 100
                    print(f"  {str(val):<20} : {count:>5} ({pct:>5.1f}%)")
                print(f"  ... and {len(value_counts) - 10} more unique values")

            # Cardinality warning
            if col.nunique() > 50:
                print(f"\n{_YELLOW}⚠ High Cardinality Warning:{_RESET}")
                print(f"  This column has {col.nunique()} unique values.")
                print(f"  One-hot encoding will create many features.")
                print(f"  Consider: Target Encoding, Frequency Encoding, or dropping low-frequency values")

        print(f"\n{_CYAN}{'='*70}{_RESET}\n")

    def compare(self, other_df):
        """Compare this dataset with another dataset.

        Shows:
          - Schema differences (new/removed/changed columns)
          - Statistic changes (mean, std, range shifts)
          - Data quality regression
          - New risks or PII patterns
        """
        self._ensure_analyzed()

        if not isinstance(other_df, type(self._df)):
            print("Error: other_df must be a pandas DataFrame")
            return

        df1 = self._df
        df2 = other_df

        # Get reports for both
        from noweda.core.engine import AutoEDAEngine
        from noweda.plugins import default_plugins

        engine = AutoEDAEngine(default_plugins())
        report1 = self._report
        report2 = engine.run_df(df2)

        _BOLD  = "\033[1m"
        _CYAN  = "\033[36m"
        _GREEN = "\033[32m"
        _YELLOW = "\033[33m"
        _RED   = "\033[31m"
        _RESET = "\033[0m"

        def h1(text):
            bar = "=" * 70
            print(f"\n{_BOLD}{_CYAN}{bar}{_RESET}")
            print(f"{_BOLD}{_CYAN}  {text}{_RESET}")
            print(f"{_BOLD}{_CYAN}{bar}{_RESET}")

        h1("Dataset Comparison")

        # Basic dimensions
        print(f"\n{_BOLD}Dataset Dimensions:{_RESET}")
        print(f"  Dataset 1 : {len(df1):>6,} rows × {len(df1.columns):>3} columns")
        print(f"  Dataset 2 : {len(df2):>6,} rows × {len(df2.columns):>3} columns")
        if len(df2) != len(df1):
            change = len(df2) - len(df1)
            change_pct = change / len(df1) * 100 if len(df1) > 0 else 0
            color = _GREEN if change >= 0 else _RED
            print(f"  Change    : {color}{change:+,} ({change_pct:+.1f}%){_RESET}")

        # Score comparison
        print(f"\n{_BOLD}Score Changes:{_RESET}")
        scores1 = report1.get("scores", {})
        scores2 = report2.get("scores", {})

        for score_name in ["data_quality", "model_readiness", "risk"]:
            s1 = scores1.get(score_name, 0)
            s2 = scores2.get(score_name, 0)
            change = s2 - s1
            color = _GREEN if (score_name != "risk" and change > 0) or (score_name == "risk" and change < 0) else _RED
            print(f"  {score_name:<15} : {s1:>3} → {s2:>3}  {color}{change:+}  {_RESET}")

        # Schema changes
        schema1 = report1.get("results", {}).get("schema", {})
        schema2 = report2.get("results", {}).get("schema", {})

        cols1 = set(schema1.keys())
        cols2 = set(schema2.keys())

        if cols1 != cols2:
            print(f"\n{_BOLD}Schema Changes:{_RESET}")
            removed = cols1 - cols2
            added = cols2 - cols1
            if removed:
                print(f"  {_RED}Removed:{_RESET} {', '.join(sorted(map(str, removed)))}")
            if added:
                print(f"  {_GREEN}Added:{_RESET} {', '.join(sorted(map(str, added)))}")

        # Role changes for common columns
        role_changes = {}
        for col in cols1 & cols2:
            role1 = schema1.get(col, {}).get("role")
            role2 = schema2.get(col, {}).get("role")
            if role1 != role2:
                role_changes[col] = (role1, role2)

        if role_changes:
            print(f"\n{_BOLD}Column Role Changes:{_RESET}")
            for col, (r1, r2) in role_changes.items():
                print(f"  {str(col):<20} : {r1} → {r2}")

        # PII comparison
        pii1 = report1.get("results", {}).get("pii", {})
        pii2 = report2.get("results", {}).get("pii", {})
        if pii1 != pii2:
            print(f"\n{_BOLD}PII Detection Changes:{_RESET}")
            new_pii = set(pii2.keys()) - set(pii1.keys())
            if new_pii:
                print(f"  {_RED}New PII detected in:{_RESET} {', '.join(sorted(map(str, new_pii)))}")
            removed_pii = set(pii1.keys()) - set(pii2.keys())
            if removed_pii:
                print(f"  {_GREEN}PII removed from:{_RESET} {', '.join(sorted(map(str, removed_pii)))}")

        print(f"\n{_CYAN}{'='*70}{_RESET}\n")


def _loading_message(method_name, args, kwargs):
    if method_name == "profile_column":
        column_name = args[0] if args else kwargs.get("column_name", "column")
        return f"NowEDA · Profiling column '{column_name}'"

    messages = {
        "insights": "NowEDA · Building insights",
        "score": "NowEDA · Calculating scores",
        "report": "NowEDA · Building report",
        "scores_df": "NowEDA · Building scores table",
        "insights_df": "NowEDA · Building insights table",
        "schema_df": "NowEDA · Building schema table",
        "stats_df": "NowEDA · Building statistics table",
        "missing_df": "NowEDA · Building missing-data table",
        "duplicates_df": "NowEDA · Building duplicate summary",
        "correlation_df": "NowEDA · Building correlation matrix",
        "outliers_df": "NowEDA · Building outlier table",
        "pii_df": "NowEDA · Detecting PII",
        "encoding_df": "NowEDA · Detecting encoding signals",
        "statsall": "NowEDA · Building full statistical report",
        "vizall": "NowEDA · Rendering visualizations",
        "mlall": "NowEDA · Building ML recommendations",
        "compare": "NowEDA · Comparing datasets",
    }
    return messages.get(method_name, f"NowEDA · Running {method_name}")


def _wrap_with_loading(fn, method_name):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        with loading(_loading_message(method_name, args, kwargs)):
            return fn(self, *args, **kwargs)

    return wrapped


for _method_name in (
    "insights",
    "score",
    "report",
    "summary",
    "refresh",
    "scores_df",
    "insights_df",
    "schema_df",
    "stats_df",
    "missing_df",
    "duplicates_df",
    "correlation_df",
    "outliers_df",
    "pii_df",
    "encoding_df",
    "statsall",
    "vizall",
    "mlall",
    "profile_column",
    "compare",
):
    setattr(
        NowEDAAccessor,
        _method_name,
        _wrap_with_loading(getattr(NowEDAAccessor, _method_name), _method_name),
    )
