import pandas as pd
from noweda.core.engine import AutoEDAEngine
from noweda.plugins import default_plugins
from noweda.ml_utils import (
    calculate_vif, cardinality_warning, rare_category_detection,
    get_scaling_recommendation, get_transformation_suggestion, assess_column_quality
)
from noweda.temporal_utils import detect_temporal_columns, stationarity_test, detect_seasonality
from noweda.ml_recommendations import (
    _profile, supervised_recommendations, unsupervised_recommendations,
    preprocessing_pipeline, format_recommendations
)

@pd.api.extensions.register_dataframe_accessor("noweda")
@pd.api.extensions.register_dataframe_accessor("eda")
class NowEDAAccessor:

    def __init__(self, pandas_obj):
        self._df = pandas_obj
        self._report = None

    def _ensure_analyzed(self):
        if self._report is None:
            engine = AutoEDAEngine(default_plugins())
            self._report = engine.run_df(self._df)

    def summary(self):
        self._ensure_analyzed()
        return self._report["results"]

    def insights(self):
        self._ensure_analyzed()
        return self._report["insights"]

    def score(self):
        self._ensure_analyzed()
        return self._report["scores"]

    def report(self):
        self._ensure_analyzed()
        return self._report

    def statsall(self):
        """Print a rich full-analysis report to the terminal/notebook.

        Combines: dtypes, describe-style stats per column, scores,
        insights, and a structured summary — all in one call.
        """
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

        # ── Scores ───────────────────────────────────────────────────────────
        h2("Scores")
        dq  = scores.get("data_quality", "N/A")
        mr  = scores.get("model_readiness", "N/A")
        risk = scores.get("risk", "N/A")
        print(f"  Data Quality    : {score_color(dq)  if isinstance(dq, (int,float))  else dq} / 100")
        print(f"  Model Readiness : {score_color(mr)  if isinstance(mr, (int,float))  else mr} / 100")
        print(f"  Risk            : {_RED if isinstance(risk,(int,float)) and risk>0 else _GREEN}{risk}{_RESET}  (0 = no risk)")

        # ── Dtypes ───────────────────────────────────────────────────────────
        h2("Column Types")
        col_w = max(len(c) for c in df.columns) + 2
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
            print(f"  {col:<{col_w}} {dtype:<14} {role:<{role_w}} {conf_display} {uniq:>8} {miss_display}")

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
                print(f"  {col:<{col_w}} {count:>8,} {mean:>12.4g} {std:>12.4g} {mn:>10.4g} {q25:>10.4g} {med:>10.4g} {q75:>10.4g} {mx:>10.4g} {skew_str}")

        # Categorical / text columns
        cat_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "category"]
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

                # Calculate diversity: entropy-based (0=uniform, 1=one dominant)
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
                print(f"  {col:<{col_w}} {count:>8,} {uniq:>8} {diversity_color:>10} {top:<28} {freq_pct:>7.1f}%")

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
                print(f"  {col:<{col_w}} {status_colored:>15}  ({issues_str})")
            else:
                print(f"  {col:<{col_w}} {status_colored}")

        # ── Temporal Analysis ─────────────────────────────────────────────────
        temporal = detect_temporal_columns(df)
        if temporal:
            h2("Temporal Data Analysis")
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

        # ── Plugin Summary ────────────────────────────────────────────────────
        h2("Plugin Summary")
        for plugin_name, plugin_result in results.items():
            if plugin_name in ("stats", "schema"):
                continue  # already shown above
            print(f"\n  [{plugin_name}]")
            if isinstance(plugin_result, dict):
                for k, v in plugin_result.items():
                    if isinstance(v, dict):
                        inner = ", ".join(f"{ik}={iv}" for ik, iv in list(v.items())[:4])
                        print(f"    {k}: {{{inner}}}")
                    elif isinstance(v, list) and len(v) > 5:
                        print(f"    {k}: [{', '.join(str(x) for x in v[:5])}, … +{len(v)-5} more]")
                    else:
                        print(f"    {k}: {v}")

        # ── ML Preprocessing Guide ────────────────────────────────────────────
        h2("ML Preprocessing Recommendations")

        # Multicollinearity
        num_cols = [c for c in df.columns if df[c].dtype.kind in ("i", "u", "f")]
        if len(num_cols) >= 2:
            vif_data = calculate_vif(df, num_cols)
            if vif_data:
                high_vif = {col: vif for col, vif in vif_data.items() if isinstance(vif, (int, float)) and vif > 5}
                if high_vif:
                    print(f"\n  {_YELLOW}⚠ Multicollinearity Detected (VIF > 5):{_RESET}")
                    for col, vif in sorted(high_vif.items(), key=lambda x: -x[1])[:5]:
                        print(f"    {col:20s}: VIF = {vif:>6.1f}  → Consider dropping or combining")
                else:
                    print(f"\n  {_GREEN}✓ Low Multicollinearity (all VIF < 5){_RESET}")

        # Scaling recommendations
        scaling_needed = []
        for col in num_cols:
            rec = get_scaling_recommendation(df[col])
            if rec and "scale" in rec:
                scaling_needed.append(col)

        if scaling_needed:
            print(f"\n  {_YELLOW}Scaling Recommended:{_RESET}")
            for col in scaling_needed[:5]:
                print(f"    {col:20s}: Use StandardScaler or MinMaxScaler")
            if len(scaling_needed) > 5:
                print(f"    … and {len(scaling_needed) - 5} more")

        # Transformations
        transform_candidates = []
        for col in num_cols:
            suggestion = get_transformation_suggestion(df[col])
            if suggestion:
                transform_candidates.append((col, suggestion))

        if transform_candidates:
            print(f"\n  {_YELLOW}Transformation Suggestions:{_RESET}")
            for col, suggestion in transform_candidates[:5]:
                print(f"    {col:20s}: {suggestion}")
            if len(transform_candidates) > 5:
                print(f"    … and {len(transform_candidates) - 5} more")

        # Cardinality warnings
        cat_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "category"]
        cardinality_issues = []
        for col in cat_cols:
            warning = cardinality_warning(df[col])
            if warning:
                cardinality_issues.append((col, warning))

        if cardinality_issues:
            print(f"\n  {_RED}Cardinality Issues:{_RESET}")
            for col, issue in cardinality_issues:
                print(f"    {col:20s}: {issue}")

        # Rare categories
        rare_issues = {}
        for col in cat_cols:
            rare = rare_category_detection(df[col], threshold=0.01)
            if rare:
                rare_issues[col] = rare

        if rare_issues:
            print(f"\n  {_YELLOW}Rare Categories Detected (<1%):{_RESET}")
            for col, rare_cats in list(rare_issues.items())[:5]:
                rare_str = ", ".join(f"{k}({v:.1%})" for k, v in list(rare_cats.items())[:2])
                print(f"    {col:20s}: {rare_str}")
                if len(rare_cats) > 2:
                    print(f"                     {' and ' + str(len(rare_cats) - 2) + ' more rare values'}")

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
                print(f"    {col:20s}: {rec}")

        print(f"\n{_CYAN}{'='*70}{_RESET}\n")

    def vizall(self):
        """Auto-render the best visualizations for each column type.

        Produces:
          - Histogram + KDE for every numeric column
          - Bar chart of top categories for every categorical column
          - Correlation heatmap when ≥2 numeric columns exist
          - Missing-value bar chart when any missingness detected
          - Time-series line plot for datetime columns (if a numeric col exists)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
        except ImportError:
            raise ImportError(
                "matplotlib is required for vizall(). "
                "Install it with: pip install matplotlib"
            )

        self._ensure_analyzed()
        df = self._df
        results = self._report["results"]
        schema = results.get("schema", {})
        missing_info = results.get("missing", {})

        numeric_cols = [
            c for c in df.columns
            if df[c].dtype.kind in ("i", "u", "f")
        ]
        cat_cols = [
            c for c in df.columns
            if (df[c].dtype == object or str(df[c].dtype) == "category")
            and schema.get(c, {}).get("role") in ("categorical", "categorical_numeric", "text", None)
            and df[c].nunique() <= 30
        ]
        datetime_cols = [
            c for c in df.columns
            if df[c].dtype.kind == "M" or schema.get(c, {}).get("role") == "datetime"
        ]
        cols_with_missing = [
            c for c in df.columns
            if df[c].isna().sum() > 0
        ]

        style = "dark_background" if "dark_background" in plt.style.available else "default"
        plt.style.use(style)

        figs_shown = 0

        # ── Numeric: histogram + KDE ─────────────────────────────────────────
        if numeric_cols:
            n = len(numeric_cols)
            ncols_grid = min(3, n)
            nrows_grid = -(-n // ncols_grid)  # ceiling division
            fig, axes = plt.subplots(
                nrows_grid, ncols_grid,
                figsize=(6 * ncols_grid, 4 * nrows_grid),
                squeeze=False
            )
            fig.suptitle("Numeric Distributions", fontsize=14, fontweight="bold")
            for i, col in enumerate(numeric_cols):
                ax = axes[i // ncols_grid][i % ncols_grid]
                data = df[col].dropna()
                ax.hist(data, bins=30, color="#4C72B0", alpha=0.7, edgecolor="white", linewidth=0.4)
                try:
                    from scipy.stats import gaussian_kde
                    import numpy as np
                    if len(data) >= 2:
                        kde = gaussian_kde(data)
                        x = np.linspace(data.min(), data.max(), 200)
                        ax2 = ax.twinx()
                        ax2.plot(x, kde(x), color="#DD8452", linewidth=2)
                        ax2.set_yticks([])
                except ImportError:
                    pass
                ax.set_title(col, fontsize=11)
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
            # Hide empty subplots
            for j in range(n, nrows_grid * ncols_grid):
                axes[j // ncols_grid][j % ncols_grid].set_visible(False)
            plt.tight_layout()
            plt.show()
            figs_shown += 1

        # ── Categorical: bar charts with distribution ─────────────────────────
        if cat_cols:
            n = len(cat_cols)
            ncols_grid = min(3, n)
            nrows_grid = -(-n // ncols_grid)
            fig, axes = plt.subplots(
                nrows_grid, ncols_grid,
                figsize=(6 * ncols_grid, 4 * nrows_grid),
                squeeze=False
            )
            fig.suptitle("Categorical Distributions (top 15 values)", fontsize=14, fontweight="bold")
            palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
                       "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
            for i, col in enumerate(cat_cols):
                ax = axes[i // ncols_grid][i % ncols_grid]
                counts = df[col].value_counts().head(15)
                total = counts.sum()
                colors = [palette[j % len(palette)] for j in range(len(counts))]
                bars = ax.bar(range(len(counts)), counts.values, color=colors, edgecolor="white", linewidth=0.4)

                # Add percentage labels on bars
                for bar in bars:
                    height = bar.get_height()
                    pct = (height / total * 100) if total > 0 else 0
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{pct:.0f}%', ha='center', va='bottom', fontsize=7, color='white', fontweight='bold')

                ax.set_xticks(range(len(counts)))
                ax.set_xticklabels(
                    [str(x)[:10] for x in counts.index],
                    rotation=45, ha="right", fontsize=8
                )
                ax.set_title(col, fontsize=11)
                ax.set_ylabel("Count")
                ax.set_ylim(0, max(counts.values) * 1.15)  # Space for labels
            for j in range(n, nrows_grid * ncols_grid):
                axes[j // ncols_grid][j % ncols_grid].set_visible(False)
            plt.tight_layout()
            plt.show()
            figs_shown += 1

        # ── Correlation heatmap ───────────────────────────────────────────────
        if len(numeric_cols) >= 2:
            try:
                import numpy as np
                corr = df[numeric_cols].corr()
                n = len(numeric_cols)
                fig_size = max(6, min(n * 1.2, 16))
                fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
                cmap = matplotlib.colormaps.get_cmap("coolwarm") if hasattr(matplotlib, "colormaps") else plt.cm.coolwarm
                im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
                plt.colorbar(im, ax=ax, shrink=0.8)
                ax.set_xticks(range(n))
                ax.set_yticks(range(n))
                ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=9)
                ax.set_yticklabels(numeric_cols, fontsize=9)
                # Annotate cells
                for row in range(n):
                    for col_i in range(n):
                        val = corr.values[row, col_i]
                        ax.text(col_i, row, f"{val:.2f}", ha="center", va="center",
                                fontsize=8, color="white" if abs(val) > 0.5 else "black")
                ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
                plt.tight_layout()
                plt.show()
                figs_shown += 1
            except Exception:
                pass

        # ── Missing value chart ───────────────────────────────────────────────
        if cols_with_missing:
            miss_rates = {
                c: round(df[c].isna().sum() / len(df) * 100, 2)
                for c in cols_with_missing
            }
            miss_rates = dict(sorted(miss_rates.items(), key=lambda x: -x[1]))
            fig, ax = plt.subplots(figsize=(max(6, len(miss_rates) * 0.8), 5))
            colors = ["#C44E52" if v >= 50 else "#DD8452" if v >= 20 else "#4C72B0"
                      for v in miss_rates.values()]
            bars = ax.bar(range(len(miss_rates)), list(miss_rates.values()),
                          color=colors, edgecolor="white", linewidth=0.4)
            ax.set_xticks(range(len(miss_rates)))
            ax.set_xticklabels(list(miss_rates.keys()), rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Missing %")
            ax.set_title("Missing Values by Column", fontsize=13, fontweight="bold")
            ax.set_ylim(0, 100)
            for bar, val in zip(bars, miss_rates.values()):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
            plt.tight_layout()
            plt.show()
            figs_shown += 1

        # ── Missing data correlation heatmap ──────────────────────────────────
        if len(cols_with_missing) >= 2:
            try:
                import numpy as np
                # Create binary matrix: 1 if missing, 0 if not
                missing_matrix = df[cols_with_missing].isna().astype(int)
                # Compute correlation between missing patterns
                missing_corr = missing_matrix.corr()

                if len(cols_with_missing) > 1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    cmap = matplotlib.colormaps.get_cmap("RdYlGn_r") if hasattr(matplotlib, "colormaps") else plt.cm.RdYlGn_r
                    im = ax.imshow(missing_corr.values, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
                    plt.colorbar(im, ax=ax, label="Correlation in Missing Patterns")

                    ax.set_xticks(range(len(cols_with_missing)))
                    ax.set_yticks(range(len(cols_with_missing)))
                    ax.set_xticklabels(cols_with_missing, rotation=45, ha='right', fontsize=9)
                    ax.set_yticklabels(cols_with_missing, fontsize=9)

                    # Annotate
                    for i in range(len(cols_with_missing)):
                        for j in range(len(cols_with_missing)):
                            text = ax.text(j, i, f'{missing_corr.values[i, j]:.2f}',
                                          ha="center", va="center",
                                          color="black" if abs(missing_corr.values[i, j]) < 0.5 else "white",
                                          fontsize=8)

                    ax.set_title("Missing Data Patterns (Correlation)", fontsize=13, fontweight="bold")
                    plt.tight_layout()
                    plt.show()
                    figs_shown += 1
            except Exception:
                pass

        # ── Datetime line plots ───────────────────────────────────────────────
        if datetime_cols and numeric_cols:
            dt_col = datetime_cols[0]
            num_col = numeric_cols[0]
            try:
                sorted_df = df[[dt_col, num_col]].dropna().sort_values(dt_col)
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(sorted_df[dt_col], sorted_df[num_col],
                        color="#4C72B0", linewidth=1.2)
                ax.set_title(f"{num_col} over {dt_col}", fontsize=13, fontweight="bold")
                ax.set_xlabel(dt_col)
                ax.set_ylabel(num_col)
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                plt.show()
                figs_shown += 1
            except Exception:
                pass

        # ── Box plots: numeric vs categorical ──────────────────────────────────
        if numeric_cols and cat_cols:
            try:
                # Select top categorical column (by cardinality, not too many levels)
                cat_by_card = sorted(cat_cols, key=lambda c: df[c].nunique())
                cat_col = cat_by_card[0] if df[cat_by_card[0]].nunique() <= 20 else None

                if cat_col:
                    n = len(numeric_cols)
                    ncols_grid = min(3, n)
                    nrows_grid = -(-n // ncols_grid)
                    fig, axes = plt.subplots(
                        nrows_grid, ncols_grid,
                        figsize=(6 * ncols_grid, 4 * nrows_grid),
                        squeeze=False
                    )
                    fig.suptitle(f"Numeric Distributions by {cat_col}", fontsize=14, fontweight="bold")
                    palette_box = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
                    for i, col in enumerate(numeric_cols):
                        ax = axes[i // ncols_grid][i % ncols_grid]
                        df_plot = df[[col, cat_col]].dropna()
                        if len(df_plot) > 0:
                            df_plot.boxplot(column=col, by=cat_col, ax=ax)
                            ax.set_title(col, fontsize=11)
                            ax.set_xlabel(cat_col)
                            plt.sca(ax)
                            plt.xticks(rotation=45, ha="right", fontsize=8)
                    for j in range(n, nrows_grid * ncols_grid):
                        axes[j // ncols_grid][j % ncols_grid].set_visible(False)
                    plt.tight_layout()
                    plt.show()
                    figs_shown += 1
            except Exception:
                pass

        # ── Feature variance/importance ranking ────────────────────────────────
        if numeric_cols:
            try:
                import numpy as np
                variances = []
                for col in numeric_cols:
                    var = df[col].var()
                    if not np.isnan(var):
                        variances.append((col, var))

                if variances:
                    variances.sort(key=lambda x: -x[1])
                    fig, ax = plt.subplots(figsize=(10, 5))
                    cols, vars = zip(*variances[:15])
                    # Normalize variances to 0-100 scale for readability
                    max_var = max(vars)
                    normalized = [v / max_var * 100 for v in vars]
                    colors_var = ["#4C72B0" if v > 50 else "#DD8452" if v > 20 else "#8C8C8C" for v in normalized]
                    bars = ax.barh(cols, normalized, color=colors_var, edgecolor="white", linewidth=0.5)
                    ax.set_xlabel("Variance (normalized to 0-100)")
                    ax.set_title("Feature Variance Ranking (Higher = More Information)", fontsize=13, fontweight="bold")
                    ax.invert_yaxis()
                    for i, (bar, v) in enumerate(zip(bars, vars)):
                        ax.text(100 + 2, i, f"{v:.2e}", va='center', fontsize=8)
                    plt.tight_layout()
                    plt.show()
                    figs_shown += 1
            except Exception:
                pass

        # ── Pair plot: top correlated pairs ────────────────────────────────────
        if numeric_cols and len(numeric_cols) >= 2:
            try:
                corr_matrix = df[numeric_cols].corr().abs()
                # Find top correlations (excluding diagonal)
                top_pairs = []
                for i in range(len(numeric_cols)):
                    for j in range(i + 1, len(numeric_cols)):
                        corr_val = corr_matrix.iloc[i, j]
                        if corr_val > 0.3:  # threshold
                            top_pairs.append((numeric_cols[i], numeric_cols[j], corr_val))

                top_pairs.sort(key=lambda x: -x[2])
                n_pairs = min(6, len(top_pairs))  # Show up to 6 pairs in 2x3 grid

                if n_pairs > 0:
                    ncols_grid = min(3, n_pairs)
                    nrows_grid = -(-n_pairs // ncols_grid)
                    fig, axes = plt.subplots(
                        nrows_grid, ncols_grid,
                        figsize=(5 * ncols_grid, 4 * nrows_grid),
                        squeeze=False
                    )
                    fig.suptitle("Top Correlated Feature Pairs", fontsize=14, fontweight="bold")

                    for idx, (col1, col2, corr_val) in enumerate(top_pairs[:n_pairs]):
                        ax = axes[idx // ncols_grid][idx % ncols_grid]
                        df_plot = df[[col1, col2]].dropna()
                        ax.scatter(df_plot[col1], df_plot[col2], alpha=0.6, color="#4C72B0", edgecolor="white", linewidth=0.4)

                        # Add regression line
                        z = np.polyfit(df_plot[col1], df_plot[col2], 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(df_plot[col1].min(), df_plot[col1].max(), 100)
                        ax.plot(x_line, p(x_line), color="#DD8452", linewidth=2, label=f"r={corr_val:.2f}")

                        ax.set_xlabel(col1, fontsize=9)
                        ax.set_ylabel(col2, fontsize=9)
                        ax.set_title(f"{col1} vs {col2}", fontsize=10)
                        ax.legend(fontsize=8)
                        ax.grid(alpha=0.3)

                    for j in range(n_pairs, nrows_grid * ncols_grid):
                        axes[j // ncols_grid][j % ncols_grid].set_visible(False)
                    plt.tight_layout()
                    plt.show()
                    figs_shown += 1
            except Exception:
                pass

        # ── Categorical relationship heatmap (Cramér's V) ───────────────────────
        if len(cat_cols) >= 2:
            try:
                from noweda.ml_utils import cramers_v
                import numpy as np

                # Compute Cramér's V for all cat column pairs
                cramers_matrix = np.zeros((len(cat_cols), len(cat_cols)))
                for i, col1 in enumerate(cat_cols):
                    for j, col2 in enumerate(cat_cols):
                        if i == j:
                            cramers_matrix[i, j] = 1.0
                        elif i < j:
                            try:
                                v = cramers_v(df[col1].dropna(), df[col2].dropna())
                                cramers_matrix[i, j] = cramers_matrix[j, i] = v
                            except Exception:
                                pass

                fig, ax = plt.subplots(figsize=(8, 6))
                cmap = matplotlib.colormaps.get_cmap("YlOrRd") if hasattr(matplotlib, "colormaps") else plt.cm.YlOrRd
                im = ax.imshow(cramers_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
                plt.colorbar(im, ax=ax, label="Cramér's V")

                ax.set_xticks(range(len(cat_cols)))
                ax.set_yticks(range(len(cat_cols)))
                ax.set_xticklabels(cat_cols, rotation=45, ha='right', fontsize=9)
                ax.set_yticklabels(cat_cols, fontsize=9)

                # Annotate
                for i in range(len(cat_cols)):
                    for j in range(len(cat_cols)):
                        text = ax.text(j, i, f'{cramers_matrix[i, j]:.2f}',
                                      ha="center", va="center", color="black" if cramers_matrix[i, j] < 0.5 else "white",
                                      fontsize=8)

                ax.set_title("Categorical Association (Cramér's V)", fontsize=13, fontweight="bold")
                plt.tight_layout()
                plt.show()
                figs_shown += 1
            except Exception:
                pass

        if figs_shown == 0:
            print("No visualizations could be generated for this dataset.")

    def mlall(self):
        """Print ML algorithm recommendations and preprocessing pipeline.

        Provides:
          - Supervised Learning recommendations with star ratings and explanations
          - Unsupervised Learning recommendations
          - Step-by-step preprocessing pipeline tailored to the dataset
        """
        self._ensure_analyzed()
        df = self._df
        report = self._report
        results = report["results"]
        scores = report["scores"]
        stats = results.get("stats", {})
        schema = results.get("schema", {})

        # Define colors
        _BOLD  = "\033[1m"
        _CYAN  = "\033[36m"
        _GREEN = "\033[32m"
        _YELLOW = "\033[33m"
        _RED   = "\033[31m"
        _RESET = "\033[0m"

        # Build profile used by ML recommenders
        profile = _profile(df, stats, schema, scores, results)

        # Get recommendations
        supervised = supervised_recommendations(profile)
        unsupervised = unsupervised_recommendations(profile)
        pipeline = preprocessing_pipeline(profile, df)

        # Print header
        bar = "=" * 70
        print(f"\n{_BOLD}{_CYAN}{bar}{_RESET}")
        print(f"{_BOLD}{_CYAN}  NowEDA · ML Algorithm Recommendations & Preprocessing{_RESET}")
        print(f"{_BOLD}{_CYAN}{bar}{_RESET}")

        # Delegate formatting to format_recommendations
        format_recommendations(supervised, unsupervised, pipeline, profile,
                              _BOLD, _CYAN, _GREEN, _YELLOW, _RED, _RESET)

        print(f"{_CYAN}{'='*70}{_RESET}\n")

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
        elif col.dtype == object or str(col.dtype) == "category":
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
                print(f"  {_RED}Removed:{_RESET} {', '.join(sorted(removed))}")
            if added:
                print(f"  {_GREEN}Added:{_RESET} {', '.join(sorted(added))}")

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
                print(f"  {col:<20} : {r1} → {r2}")

        # PII comparison
        pii1 = report1.get("results", {}).get("pii", {})
        pii2 = report2.get("results", {}).get("pii", {})
        if pii1 != pii2:
            print(f"\n{_BOLD}PII Detection Changes:{_RESET}")
            new_pii = set(pii2.keys()) - set(pii1.keys())
            if new_pii:
                print(f"  {_RED}New PII detected in:{_RESET} {', '.join(sorted(new_pii))}")
            removed_pii = set(pii1.keys()) - set(pii2.keys())
            if removed_pii:
                print(f"  {_GREEN}PII removed from:{_RESET} {', '.join(sorted(removed_pii))}")

        print(f"\n{_CYAN}{'='*70}{_RESET}\n")
