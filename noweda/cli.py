import argparse
import json
from noweda import read
from noweda.report.html import generate_html_report

_BOLD   = "\033[1m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_RESET  = "\033[0m"


def _score_color(val, low=40, mid=70):
    if isinstance(val, (int, float)):
        if val >= mid:
            return f"{_GREEN}{val}{_RESET}"
        elif val >= low:
            return f"{_YELLOW}{val}{_RESET}"
        return f"{_RED}{val}{_RESET}"
    return str(val)


def main():
    parser = argparse.ArgumentParser(
        description="NowEDA — Automated Exploratory Data Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  noweda data.csv\n"
            "  noweda data.csv --html report.html\n"
            "  noweda data.csv --html report.html --json report.json\n"
        ),
    )
    parser.add_argument("file", help="Path to the dataset file")
    parser.add_argument("--html", help="Export HTML report to this path", default=None)
    parser.add_argument("--json", help="Export JSON report to this path", default=None)

    args = parser.parse_args()

    df = read(args.file)
    report = df.noweda.report()
    results = report["results"]
    scores  = report["scores"]
    insights = report["insights"]
    stats  = results.get("stats", {})
    schema = results.get("schema", {})

    bar = "=" * 72
    thin = "-" * 60

    # ── Header ──────────────────────────────────────────────────────────────
    print(f"\n{_BOLD}{_CYAN}{bar}{_RESET}")
    print(f"{_BOLD}{_CYAN}  NowEDA · {args.file}{_RESET}")
    print(f"{_BOLD}{_CYAN}{bar}{_RESET}")
    print(f"  Rows    : {_BOLD}{len(df):,}{_RESET}")
    print(f"  Columns : {_BOLD}{len(df.columns)}{_RESET}")

    # ── Scores ───────────────────────────────────────────────────────────────
    print(f"\n{_BOLD}Scores{_RESET}\n{thin}")
    dq   = scores.get("data_quality", "N/A")
    mr   = scores.get("model_readiness", "N/A")
    risk = scores.get("risk", "N/A")
    print(f"  Data Quality    : {_score_color(dq)} / 100")
    print(f"  Model Readiness : {_score_color(mr)} / 100")
    risk_col = f"{_RED}{risk}{_RESET}" if isinstance(risk, (int, float)) and risk > 0 else f"{_GREEN}{risk}{_RESET}"
    print(f"  Risk            : {risk_col}  (0 = no risk)")

    # ── Column Overview ───────────────────────────────────────────────────────
    print(f"\n{_BOLD}Columns{_RESET}\n{thin}")
    col_w = max(len(c) for c in df.columns) + 2
    role_w = 22
    print(f"  {'Column':<{col_w}} {'Dtype':<14} {'Role':<{role_w}} {'Unique':>8} {'Missing':>8}")
    print(f"  {'-'*col_w} {'-'*14} {'-'*role_w} {'--------':>8} {'--------':>8}")
    for col in df.columns:
        dtype  = str(df[col].dtype)
        role   = schema.get(col, {}).get("role", "unknown")
        uniq   = df[col].nunique()
        miss   = int(df[col].isna().sum())
        miss_s = f"{_YELLOW}{miss}{_RESET}" if miss > 0 else f"{miss}"
        print(f"  {col:<{col_w}} {dtype:<14} {role:<{role_w}} {uniq:>8} {miss_s:>8}")

    # ── Numeric Stats ────────────────────────────────────────────────────────
    num_cols = [c for c in df.columns if df[c].dtype.kind in ("i", "u", "f")]
    if num_cols:
        print(f"\n{_BOLD}Numeric Statistics{_RESET}\n{thin}")
        print(f"  {'Column':<{col_w}} {'Count':>8} {'Mean':>12} {'Std':>12} {'Min':>10} {'25%':>10} {'50%':>10} {'75%':>10} {'Max':>10} {'Skew':>8}")
        print(f"  {'-'*col_w} {'--------':>8} {'------------':>12} {'------------':>12} {'----------':>10} {'----------':>10} {'----------':>10} {'----------':>10} {'----------':>10} {'--------':>8}")
        for col in num_cols:
            s    = stats.get(col, {})
            count = s.get("count", 0)
            mean  = s.get("mean", float("nan"))
            std   = s.get("std",  float("nan"))
            mn    = s.get("min",  float("nan"))
            q25   = s.get("q25",  float("nan"))
            med   = s.get("median", float("nan"))
            q75   = s.get("q75",  float("nan"))
            mx    = s.get("max",  float("nan"))
            skew  = s.get("skewness", float("nan"))
            skew_s = f"{skew:>8.2f}"
            if abs(skew) > 1:
                skew_s = f"{_YELLOW}{skew_s}{_RESET}"
            print(f"  {col:<{col_w}} {count:>8,} {mean:>12.4g} {std:>12.4g} {mn:>10.4g} {q25:>10.4g} {med:>10.4g} {q75:>10.4g} {mx:>10.4g} {skew_s}")

    # ── Categorical Stats ─────────────────────────────────────────────────────
    cat_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "category"]
    if cat_cols:
        print(f"\n{_BOLD}Categorical Statistics{_RESET}\n{thin}")
        print(f"  {'Column':<{col_w}} {'Count':>8} {'Unique':>8} {'Top Value':<30} {'Freq':>8}")
        print(f"  {'-'*col_w} {'--------':>8} {'--------':>8} {'-'*30} {'--------':>8}")
        for col in cat_cols:
            s     = stats.get(col, {})
            count = s.get("count", 0)
            uniq  = s.get("unique", 0)
            top   = str(s.get("top_value", "N/A"))[:28]
            freq  = s.get("top_freq", 0)
            print(f"  {col:<{col_w}} {count:>8,} {uniq:>8} {top:<30} {freq:>8,}")

    # ── Insights ─────────────────────────────────────────────────────────────
    print(f"\n{_BOLD}Insights{_RESET}\n{thin}")
    if insights:
        for ins in insights:
            print(f"  • {ins}")
    else:
        print("  No issues detected.")

    print(f"\n{_CYAN}{bar}{_RESET}\n")

    # ── Exports ───────────────────────────────────────────────────────────────
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"JSON report saved → {args.json}")

    if args.html:
        generate_html_report(report, args.html)
        print(f"HTML report saved → {args.html}")
