from noweda.scoring.scorer import Scorer
from noweda.insights.generator import InsightGenerator

class AutoEDAEngine:

    def __init__(self, plugins):
        self.plugins = plugins

    def run_df(self, df):
        if not df.columns.is_unique:
            raise ValueError("NowEDA requires unique column labels; rename duplicate columns before analysis.")
        results = {}
        encoding_details = {}

        for plugin in self.plugins:
            results[plugin.name] = plugin.run(df)
            if plugin.name == "encoding":
                encoding_details = getattr(plugin, "details", {})

        # A most common value, a minimum, or even the mean of one record can
        # expose a contact detail. Keep counts for flagged columns in reports.
        stats = results.get("stats", {})
        counts = {"dtype", "count", "missing", "unique", "top_freq"}
        for column in results.get("pii", {}):
            if column in stats:
                stats[column] = {key: value for key, value in stats[column].items() if key in counts}

        scores, breakdown = Scorer().compute(results, explain=True)
        insights = (["Empty DataFrame: no rows or no columns to analyze. Scores are not informative."]
                    if df.empty else InsightGenerator().generate(results, scores))

        return {
            "results": results,
            "scores": scores,
            "insights": insights,
            "score_breakdown": breakdown,
            "encoding_details": encoding_details,
        }
