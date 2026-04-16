from noweda.scoring.scorer import Scorer
from noweda.insights.generator import InsightGenerator

class AutoEDAEngine:

    def __init__(self, plugins):
        self.plugins = plugins

    def run_df(self, df):
        results = {}

        for plugin in self.plugins:
            results[plugin.name] = plugin.run(df)

        scores = Scorer().compute(results)
        insights = InsightGenerator().generate(results, scores)

        return {
            "results": results,
            "scores": scores,
            "insights": insights
        }
