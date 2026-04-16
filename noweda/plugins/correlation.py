from .base import BasePlugin

class CorrelationPlugin(BasePlugin):
    name = "correlation"

    def run(self, df):
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty:
            return {}
        return numeric_df.corr().to_dict()
