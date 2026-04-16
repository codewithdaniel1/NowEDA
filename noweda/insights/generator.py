class InsightGenerator:
    """Translate raw plugin results into human-readable, actionable insights."""

    def generate(self, results, scores):
        insights = []

        self._schema_insights(results.get("schema", {}), insights)
        self._missing_insights(results.get("missing", {}), insights)
        self._duplicates_insights(results.get("duplicates", {}), insights)
        self._stats_insights(results.get("stats", {}), insights)
        self._outlier_insights(results.get("outliers", {}), insights)
        self._correlation_insights(results.get("correlation", {}), insights)
        self._pii_insights(results.get("pii", {}), insights)
        self._encoding_insights(results.get("encoding", {}), insights)
        self._score_insights(scores, insights)

        return insights

    # ------------------------------------------------------------------
    # Per-plugin insight rules
    # ------------------------------------------------------------------

    def _schema_insights(self, schema, insights):
        id_cols = [c for c, v in schema.items() if v.get("role") == "id_candidate"]
        cat_cols = [c for c, v in schema.items() if "categorical" in v.get("role", "")]
        datetime_cols = [c for c, v in schema.items() if v.get("role") == "datetime"]

        if id_cols:
            insights.append(
                f"Likely identifier column(s) detected: {', '.join(id_cols)}. "
                "Consider excluding from modelling."
            )
        if cat_cols:
            insights.append(
                f"Column(s) with low cardinality (likely categorical): {', '.join(cat_cols)}."
            )
        if datetime_cols:
            insights.append(
                f"Datetime column(s) detected: {', '.join(datetime_cols)}. "
                "Temporal features may be valuable."
            )

    def _missing_insights(self, missing, insights):
        critical = [(c, v) for c, v in missing.items() if v > 0.5]
        high = [(c, v) for c, v in missing.items() if 0.3 < v <= 0.5]
        low = [(c, v) for c, v in missing.items() if 0 < v <= 0.3]

        for col, pct in critical:
            insights.append(
                f"Column '{col}' is {pct:.0%} missing — consider dropping it."
            )
        for col, pct in high:
            insights.append(
                f"Column '{col}' has high missing rate ({pct:.0%}) — imputation recommended."
            )
        if low:
            cols = ", ".join(f"'{c}'" for c, _ in low)
            insights.append(f"Minor missing values in: {cols}.")

    def _duplicates_insights(self, duplicates, insights):
        dup_rows = duplicates.get("duplicate_rows", 0)
        dup_pct = duplicates.get("duplicate_rows_pct", 0.0)
        const_cols = duplicates.get("constant_columns", [])

        if dup_rows > 0:
            insights.append(
                f"{dup_rows} duplicate row(s) detected ({dup_pct:.1%} of data). "
                "De-duplication is recommended."
            )
        if const_cols:
            insights.append(
                f"Constant (zero-variance) column(s) detected: {', '.join(const_cols)}. "
                "These carry no information and can be dropped."
            )

    def _stats_insights(self, stats, insights):
        for col, s in stats.items():
            skew = s.get("skewness")
            if skew is not None and abs(skew) > 2:
                direction = "right" if skew > 0 else "left"
                insights.append(
                    f"Column '{col}' is heavily {direction}-skewed (skewness={skew:.2f}). "
                    "Log or power transform may help."
                )

    def _outlier_insights(self, outliers, insights):
        for col, count in outliers.items():
            if count > 0:
                insights.append(
                    f"Column '{col}' has {count} outlier(s) (IQR method). "
                    "Review before modelling."
                )

    def _correlation_insights(self, correlation, insights):
        seen = set()
        for col1, corrs in correlation.items():
            for col2, val in corrs.items():
                pair = tuple(sorted([col1, col2]))
                if col1 == col2 or pair in seen:
                    continue
                seen.add(pair)
                if abs(val) > 0.9:
                    insights.append(
                        f"Very strong correlation ({val:.2f}) between '{col1}' and '{col2}'. "
                        "One may be redundant."
                    )
                elif abs(val) > 0.7:
                    insights.append(
                        f"Strong correlation ({val:.2f}) between '{col1}' and '{col2}'."
                    )

    def _pii_insights(self, pii, insights):
        for col, info in pii.items():
            count = info.get("emails_detected", 0)
            insights.append(
                f"PII detected in column '{col}': {count} email address(es) found. "
                "Mask or remove before sharing."
            )

    def _encoding_insights(self, encoding, insights):
        for col, enc_type in encoding.items():
            insights.append(
                f"Column '{col}' may contain encoded data ({enc_type}). "
                "Inspect for hidden payloads or obfuscation."
            )

    def _score_insights(self, scores, insights):
        quality = scores.get("data_quality", 100)
        risk = scores.get("risk", 0)

        if quality >= 90:
            insights.append("Data quality score is excellent (>=90). Dataset looks clean.")
        elif quality >= 70:
            insights.append(f"Data quality score is acceptable ({quality}). Minor issues present.")
        else:
            insights.append(
                f"Data quality score is low ({quality}). Significant cleaning required."
            )

        if risk == 0:
            insights.append("No risk signals detected.")
        elif risk <= 20:
            insights.append(f"Low risk level ({risk}). Some sensitive indicators found.")
        elif risk <= 50:
            insights.append(
                f"Moderate risk level ({risk}). Review PII and encoded columns before sharing."
            )
        else:
            insights.append(
                f"High risk level ({risk}). Sensitive data likely present — handle with care."
            )
