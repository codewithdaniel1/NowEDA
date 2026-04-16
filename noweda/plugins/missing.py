from .base import BasePlugin

class MissingDataPlugin(BasePlugin):
    name = "missing"

    def run(self, df):
        return {
            col: float(df[col].isna().mean())
            for col in df.columns
        }
