from .base import BasePlugin

class OutlierPlugin(BasePlugin):
    name = "outliers"

    def run(self, df):
        result = {}

        for col in df.select_dtypes(include="number").columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            outliers = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)]
            result[col] = int(len(outliers))

        return result
