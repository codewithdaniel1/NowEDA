import re
from .base import BasePlugin

class PIIDetectorPlugin(BasePlugin):
    name = "pii"

    EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

    def run(self, df):
        findings = {}

        for col in df.columns:
            if df[col].dtype == "object":
                matches = df[col].astype(str).str.contains(self.EMAIL_REGEX, regex=True).sum()
                if matches > 0:
                    findings[col] = {"emails_detected": int(matches)}

        return findings
