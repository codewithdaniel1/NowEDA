"""Conservative Base64 signals with inspectable sample evidence."""
import base64
import binascii

from noweda.dtypes import is_textual
from .base import BasePlugin


class EncodingDetectionPlugin(BasePlugin):
    name = "encoding"

    def is_base64(self, s):
        # Short ordinary words often round-trip through Base64 by accident.
        if len(s) < 8:
            return False
        try:
            decoded = base64.b64decode(s, validate=True)
            if base64.b64encode(decoded).decode("ascii") != s:
                return False
            # Padding/symbols support binary payloads. Otherwise require decoded
            # printable UTF-8 text as evidence beyond the Base64 alphabet.
            if any(char in s for char in "=+/"):
                return True
            text = decoded.decode("utf-8")
            return bool(text) and all(c.isprintable() or c in "\n\r\t" for c in text)
        except (ValueError, UnicodeError, binascii.Error):
            return False

    def run(self, df):
        results = {}
        self.details = {}
        for col in df.columns:
            if is_textual(df[col]):
                sample = df[col].dropna().astype(str).head(20)
                count = sum(self.is_base64(x) for x in sample)
                rate = count / len(sample) if len(sample) else 0.0
                if count >= 6 and rate >= 0.8:
                    results[col] = "possible_base64"
                    self.details[col] = {
                        "sample_size": len(sample),
                        "matches": count,
                        "confidence": rate,
                    }
        return results
