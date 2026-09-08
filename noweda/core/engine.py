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
