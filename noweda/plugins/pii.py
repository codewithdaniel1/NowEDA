"""Pattern-based PII signals, counted once per matching cell and type."""
import re
import math

from noweda.dtypes import is_textual
from .base import BasePlugin


class PIIDetectorPlugin(BasePlugin):
    name = "pii"

    PATTERNS = [
        ("email", r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
        ("credit_card", r"(?<![0-9])(?:[0-9][ -]?){12,18}[0-9](?![ -]?[0-9])"),
        ("ssn", r"\b[0-9]{3}[-\s][0-9]{2}[-\s][0-9]{4}\b"),
        ("phone", r"(?<![0-9])(?:\+1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})(?![0-9])"),
    ]
    _CARD_NUMBER = re.compile(
        r"(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})"
    )

    def run(self, df):
        findings = {}
        patterns = dict(self.PATTERNS)
        for col in df.columns:
            if df[col].dtype.kind in ("i", "u", "f"):
                finding = self._numeric_signal(col, df[col])
                if finding:
                    findings[col] = finding
                continue
            if not is_textual(df[col]):
                continue
            counts = dict.fromkeys(patterns, 0)
            # Repeated index labels still represent distinct cells.
            for value in df[col].dropna().astype(str):
                candidates = list(re.finditer(patterns["credit_card"], value))
                cards = [m for m in candidates if self._valid_card(m.group())]
                counts["credit_card"] += bool(cards)
                for label, pattern in patterns.items():
                    if label not in ("credit_card", "phone"):
                        counts[label] += bool(re.search(pattern, value))
                # Mask card-shaped spans, including invalid checksums, so a suffix
                # cannot become a phone. Separate phones in the same cell survive.
                phone_text = list(value)
                for match in candidates:
                    phone_text[match.start():match.end()] = " " * (match.end() - match.start())
                counts["phone"] += bool(re.search(patterns["phone"], "".join(phone_text)))
            if any(counts.values()):
                findings[col] = {kind: count for kind, count in counts.items() if count}
        return findings

    def _numeric_signal(self, column, series):
        # CSV readers often infer phone and card columns as numbers.
        name = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(column)).lower()
        words = set(re.findall(r"[a-z]+", name))
        if words & {"phone", "telephone", "mobile", "cellphone"}:
            kind = "phone"
        elif "card" in words:
            kind = "credit_card"
        elif "ssn" in words:
            kind = "ssn"
        else:
            return {}
        count = 0
        for value in series.dropna():
            if series.dtype.kind == "f" and (
                not math.isfinite(value) or not value.is_integer() or abs(value) >= 2**53
            ):
                continue
            digits = str(int(value))
            if kind == "credit_card":
                count += self._valid_card(digits)
            elif kind == "phone":
                count += bool(re.fullmatch(r"1?[0-9]{10}", digits))
            else:
                count += bool(re.fullmatch(r"[0-9]{9}", digits))
        return {kind: count} if count else {}

    def _valid_card(self, value):
        digits = re.sub(r"[ -]", "", value)
        return bool(self._CARD_NUMBER.fullmatch(digits)) and self._luhn_check(digits)

    def _luhn_check(self, number_str):
        digits = [int(d) for d in reversed(number_str)]
        total = sum(
            d if i % 2 == 0 else (d * 2 if d * 2 < 10 else d * 2 - 9)
            for i, d in enumerate(digits)
        )
        return total % 10 == 0
