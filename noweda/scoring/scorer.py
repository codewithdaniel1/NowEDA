class Scorer:
    """
    Compute data quality and risk scores from plugin results.

    Scores:
        data_quality : 0-100  (higher = cleaner)
        risk         : 0+     (higher = more sensitive/risky)
        model_readiness : 0-100  (higher = more ready for ML)
    """

    def compute(self, results):
        scores = {
            "data_quality": 100,
            "risk": 0,
            "model_readiness": 100,
        }

        self._penalise_missing(results.get("missing", {}), scores)
        self._penalise_duplicates(results.get("duplicates", {}), scores)
        self._penalise_outliers(results.get("outliers", {}), scores)
        self._penalise_skew(results.get("stats", {}), scores)
        self._add_pii_risk(results.get("pii", {}), scores)
        self._add_encoding_risk(results.get("encoding", {}), scores)
        self._penalise_schema(results.get("schema", {}), scores)

        # Clamp quality and readiness to [0, 100]
        scores["data_quality"] = max(0, min(100, scores["data_quality"]))
        scores["model_readiness"] = max(0, min(100, scores["model_readiness"]))

        return scores

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

    def _penalise_outliers(self, outliers, scores):
        total = sum(outliers.values())
        if total > 50:
            scores["data_quality"] -= 10
            scores["model_readiness"] -= 10
        elif total > 10:
            scores["data_quality"] -= 5
            scores["model_readiness"] -= 5

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
