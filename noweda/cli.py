import argparse
import json
from noweda import read
from noweda.report.html import generate_html_report

def main():
    parser = argparse.ArgumentParser(description="NowEDA CLI")
    parser.add_argument("file", help="Path to dataset")
    parser.add_argument("--html", help="Export HTML report", default=None)
    parser.add_argument("--json", help="Export JSON report", default=None)

    args = parser.parse_args()

    df = read(args.file)
    report = df.noweda.report()

    print("\n=== NOWEDA INSIGHTS ===")
    for insight in report["insights"]:
        print(f"- {insight}")

    print("\n=== SCORES ===")
    print(report["scores"])

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)

    if args.html:
        generate_html_report(report, args.html)
