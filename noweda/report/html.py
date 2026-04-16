import json
from datetime import datetime


def generate_html_report(report, output_path):
    scores = report.get("scores", {})
    insights = report.get("insights", [])
    results = report.get("results", {})

    quality = scores.get("data_quality", 0)
    risk = scores.get("risk", 0)
    readiness = scores.get("model_readiness", 0)

    quality_color = _score_color(quality, invert=False)
    risk_color = _score_color(risk, invert=True, max_val=100)
    readiness_color = _score_color(readiness, invert=False)

    insights_html = "".join(
        f'<li class="insight-item">{_escape(i)}</li>' for i in insights
    )

    schema_html = _schema_table(results.get("schema", {}))
    missing_html = _missing_table(results.get("missing", {}))
    duplicates_html = _duplicates_section(results.get("duplicates", {}))
    outliers_html = _outliers_table(results.get("outliers", {}))
    pii_html = _pii_table(results.get("pii", {}))
    encoding_html = _encoding_table(results.get("encoding", {}))

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>NowEDA Report</title>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #22263a;
    --border: #2e3250;
    --text: #e2e8f0;
    --muted: #8892b0;
    --accent: #7c6af7;
    --green: #4ade80;
    --yellow: #facc15;
    --red: #f87171;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px; }}
  .header {{ margin-bottom: 36px; border-bottom: 1px solid var(--border); padding-bottom: 20px; }}
  .header h1 {{ font-size: 28px; font-weight: 700; color: var(--accent); letter-spacing: -0.5px; }}
  .header .meta {{ color: var(--muted); font-size: 12px; margin-top: 6px; }}
  .scores-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 36px; }}
  .score-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; text-align: center; }}
  .score-card .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}
  .score-card .value {{ font-size: 40px; font-weight: 800; line-height: 1; }}
  .score-card .sub {{ font-size: 11px; color: var(--muted); margin-top: 4px; }}
  .section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
  .section h2 {{ font-size: 16px; font-weight: 600; margin-bottom: 16px; color: var(--text); border-bottom: 1px solid var(--border); padding-bottom: 10px; }}
  .insight-list {{ list-style: none; display: flex; flex-direction: column; gap: 8px; }}
  .insight-item {{ background: var(--surface2); border-left: 3px solid var(--accent); border-radius: 4px; padding: 10px 14px; font-size: 13px; line-height: 1.5; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; padding: 8px 12px; border-bottom: 1px solid var(--border); }}
  td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); color: var(--text); }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: var(--surface2); }}
  .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-transform: uppercase; }}
  .tag-id {{ background: #312e81; color: #a5b4fc; }}
  .tag-cat {{ background: #164e63; color: #67e8f9; }}
  .tag-num {{ background: #14532d; color: #86efac; }}
  .tag-text {{ background: #44403c; color: #d6d3d1; }}
  .tag-dt {{ background: #4a1d96; color: #c4b5fd; }}
  .tag-unk {{ background: #2d2d2d; color: #9ca3af; }}
  .pct-bar {{ background: var(--surface2); border-radius: 4px; height: 6px; width: 100px; display: inline-block; vertical-align: middle; margin-left: 8px; }}
  .pct-fill {{ height: 100%; border-radius: 4px; }}
  .empty {{ color: var(--muted); font-style: italic; font-size: 13px; }}
  .footer {{ text-align: center; color: var(--muted); font-size: 11px; margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--border); }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>NowEDA Report</h1>
    <div class="meta">Generated: {generated_at}</div>
  </div>

  <div class="scores-grid">
    <div class="score-card">
      <div class="label">Data Quality</div>
      <div class="value" style="color:{quality_color}">{quality}</div>
      <div class="sub">out of 100</div>
    </div>
    <div class="score-card">
      <div class="label">Risk Level</div>
      <div class="value" style="color:{risk_color}">{risk}</div>
      <div class="sub">higher = more risk</div>
    </div>
    <div class="score-card">
      <div class="label">Model Readiness</div>
      <div class="value" style="color:{readiness_color}">{readiness}</div>
      <div class="sub">out of 100</div>
    </div>
  </div>

  <div class="section">
    <h2>Insights</h2>
    {"<ul class='insight-list'>" + insights_html + "</ul>" if insights else "<p class='empty'>No insights generated.</p>"}
  </div>

  {schema_html}
  {missing_html}
  {duplicates_html}
  {outliers_html}
  {pii_html}
  {encoding_html}

  <div class="footer">
    Built with <strong>NowEDA</strong> — Automated EDA Framework
  </div>

</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _escape(s):
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _score_color(val, invert=False, max_val=100):
    """Return a CSS color (green/yellow/red) based on value."""
    ratio = val / max_val if max_val else 0
    if invert:
        ratio = 1 - min(ratio, 1)
    if ratio >= 0.75:
        return "var(--green)"
    elif ratio >= 0.4:
        return "var(--yellow)"
    return "var(--red)"


def _role_tag(role):
    mapping = {
        "id_candidate": ("ID", "tag-id"),
        "categorical": ("Categorical", "tag-cat"),
        "categorical_numeric": ("Cat. Numeric", "tag-cat"),
        "numeric": ("Numeric", "tag-num"),
        "text": ("Text", "tag-text"),
        "datetime": ("Datetime", "tag-dt"),
        "unknown": ("Unknown", "tag-unk"),
    }
    label, css = mapping.get(role, (role, "tag-unk"))
    return f'<span class="tag {css}">{label}</span>'


def _schema_table(schema):
    if not schema:
        return ""
    rows = ""
    for col, info in schema.items():
        rows += (
            f"<tr>"
            f"<td>{_escape(col)}</td>"
            f"<td>{_escape(info.get('dtype', ''))}</td>"
            f"<td>{_role_tag(info.get('role', 'unknown'))}</td>"
            f"<td>{info.get('unique', '')}</td>"
            f"<td>{info.get('uniqueness_ratio', ''):.2%}</td>"
            f"</tr>"
        )
    return f"""<div class="section">
  <h2>Column Schema</h2>
  <table>
    <thead><tr><th>Column</th><th>Dtype</th><th>Role</th><th>Unique Values</th><th>Uniqueness</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def _missing_table(missing):
    if not missing:
        return ""
    rows = ""
    for col, pct in sorted(missing.items(), key=lambda x: -x[1]):
        pct_int = int(pct * 100)
        bar_color = "#f87171" if pct > 0.3 else ("#facc15" if pct > 0 else "#4ade80")
        bar = (
            f'<span class="pct-bar"><span class="pct-fill" '
            f'style="width:{pct_int}%;background:{bar_color}"></span></span>'
        )
        rows += f"<tr><td>{_escape(col)}</td><td>{pct:.1%} {bar}</td></tr>"
    return f"""<div class="section">
  <h2>Missing Values</h2>
  <table>
    <thead><tr><th>Column</th><th>Missing %</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def _duplicates_section(duplicates):
    if not duplicates:
        return ""
    dup_rows = duplicates.get("duplicate_rows", 0)
    dup_pct = duplicates.get("duplicate_rows_pct", 0.0)
    const_cols = duplicates.get("constant_columns", [])
    const_str = ", ".join(const_cols) if const_cols else "None"
    return f"""<div class="section">
  <h2>Duplicates &amp; Constants</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>Duplicate rows</td><td>{dup_rows} ({dup_pct:.1%})</td></tr>
      <tr><td>Constant columns</td><td>{_escape(const_str)}</td></tr>
    </tbody>
  </table>
</div>"""


def _outliers_table(outliers):
    if not outliers:
        return ""
    rows = "".join(
        f"<tr><td>{_escape(col)}</td><td>{count}</td></tr>"
        for col, count in sorted(outliers.items(), key=lambda x: -x[1])
        if count > 0
    )
    if not rows:
        return ""
    return f"""<div class="section">
  <h2>Outliers (IQR Method)</h2>
  <table>
    <thead><tr><th>Column</th><th>Outlier Count</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def _pii_table(pii):
    if not pii:
        return ""
    rows = ""
    for col, info in pii.items():
        rows += f"<tr><td>{_escape(col)}</td><td>{_escape(str(info))}</td></tr>"
    return f"""<div class="section" style="border-color:#f87171">
  <h2>PII Detection</h2>
  <table>
    <thead><tr><th>Column</th><th>Finding</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def _encoding_table(encoding):
    if not encoding:
        return ""
    rows = "".join(
        f"<tr><td>{_escape(col)}</td><td>{_escape(enc_type)}</td></tr>"
        for col, enc_type in encoding.items()
    )
    return f"""<div class="section" style="border-color:#facc15">
  <h2>Encoding Detection</h2>
  <table>
    <thead><tr><th>Column</th><th>Detected Encoding</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""
