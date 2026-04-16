import base64
from .base import BasePlugin

class EncodingDetectionPlugin(BasePlugin):
    name = "encoding"

    def is_base64(self, s):
        try:
            return base64.b64encode(base64.b64decode(s)) == s.encode()
        except:
            return False

    def run(self, df):
        results = {}

        for col in df.columns:
            if df[col].dtype == "object":
                sample = df[col].dropna().astype(str).head(20)
                count = sum(self.is_base64(x) for x in sample)

                if count > 5:
                    results[col] = "possible_base64"

        return results
