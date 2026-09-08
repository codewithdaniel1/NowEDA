class Scorer:
    """
    Compute data quality and risk scores from plugin results.

    Scores:
        data_quality : 0-100  (higher = cleaner)
        risk         : 0+     (higher = more sensitive/risky)
        model_readiness : 0-100  (higher = more ready for ML)
    """

    def compute(self, results, explain=False):
        scores = {
            "data_quality": 100,
            "risk": 0,
            "model_readiness": 100,
        }

        breakdown = []
        rules = [
            ("missing", self._penalise_missing, "Missing-value penalties per column"),
            ("duplicates", self._penalise_duplicates, "Duplicate rows and constant columns"),
            ("outliers", self._penalise_outliers, "IQR outliers as a fraction of observed numeric values"),
            ("stats", self._penalise_skew, "Readiness penalty for columns with absolute skewness > 2"),
            ("pii", self._add_pii_risk, "15 risk points per column with PII signals"),
            ("encoding", self._add_encoding_risk, "10 risk points per column with encoding signals"),
            ("schema", self._penalise_schema, "3 readiness points deducted per text or unknown column"),
        ]
        for key, rule, reason in rules:
            before = scores.copy()
            evidence = None
            if key == "outliers":
                evidence = rule(results.get(key, {}), scores, results.get("stats", {}))
            else:
                rule(results.get(key, {}), scores)
            entry = {"rule": key, "reason": reason,
                     "contributions": {k: scores[k] - before[k] for k in scores}}
            if evidence is not None:
                entry["evidence"] = evidence
            breakdown.append(entry)

        # Clamp quality and readiness to [0, 100]
        before = scores.copy()
        scores["data_quality"] = max(0, min(100, scores["data_quality"]))
        scores["model_readiness"] = max(0, min(100, scores["model_readiness"]))
        breakdown.append({"rule": "clamp", "reason": "Keep quality and readiness within 0–100",
                          "contributions": {k: scores[k] - before[k] for k in scores}})

        return (scores, breakdown) if explain else scores

    # ------------------------------------------------------------------

    def _penalise_missing(self, missing, scores):
        for pct in missing.values():
            if pct > 0.5:
                scores["data_quality"] -= 10
                scores["model_readiness"] -= 15
            elif pct > 0.3:
                scores["data_quality"] -= 5
                scores["model_readiness"] -= 8
            elif pct > 0:
                scores["data_quality"] -= 2
                scores["model_readiness"] -= 3

    def _penalise_duplicates(self, duplicates, scores):
        dup_pct = duplicates.get("duplicate_rows_pct", 0.0)
        const_cols = duplicates.get("constant_columns", [])

        if dup_pct > 0.1:
            scores["data_quality"] -= 10
        elif dup_pct > 0:
            scores["data_quality"] -= 3

        scores["data_quality"] -= len(const_cols) * 3
        scores["model_readiness"] -= len(const_cols) * 5

    def _penalise_outliers(self, outliers, scores, stats):
        # Count observed cells in the same columns that have outlier results.
        # Missing denominators from custom plugins must not imply zero outliers.
        counts = [stats.get(col, {}).get("count") for col in outliers]
        observed = sum(counts) if all(c is not None for c in counts) else None
        total = sum(outliers.values())
        rate = total / observed if observed else None
        if rate is not None and rate > 0.05:
            scores["data_quality"] -= 10
            scores["model_readiness"] -= 10
        elif rate is not None and rate > 0.01:
            scores["data_quality"] -= 5
            scores["model_readiness"] -= 5
        return {"outlier_count": total, "observed_numeric_values": observed,
                "rate": rate, "thresholds": {"minor_above": 0.01, "major_above": 0.05}}

    def _penalise_skew(self, stats, scores):
        heavy_skew_cols = [
            col for col, s in stats.items()
            if s.get("skewness") is not None and abs(s["skewness"]) > 2
        ]
        scores["model_readiness"] -= len(heavy_skew_cols) * 5

    def _add_pii_risk(self, pii, scores):
        scores["risk"] += len(pii) * 15

    def _add_encoding_risk(self, encoding, scores):
        scores["risk"] += len(encoding) * 10

    def _penalise_schema(self, schema, scores):
        # All-text or all-unknown columns reduce model readiness
        untyped = sum(
            1 for v in schema.values()
            if v.get("role") in ("text", "unknown")
        )
        scores["model_readiness"] -= untyped * 3
