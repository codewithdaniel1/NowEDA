import re
from .base import BasePlugin


class PIIDetectorPlugin(BasePlugin):
    name = "pii"

    # Regex patterns for different PII types (all non-capturing groups)
    # Note: Order matters! Check credit_card before phone to avoid false positives
    # Credit card patterns are checked first, then excluded from phone matches
    PATTERNS = [
        ("email", r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
        ("credit_card", r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13}|6(?:011|5\d{2})\d{12})\b"),
        ("ssn", r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"),
        # Phone: US/intl formats. Avoid matching against credit cards by requiring:
        # - Starts with ( or digit or +
        # - Has 3-digit area code
        # - Has 3-digit exchange + 4-digit number (phone structure)
        # - NOT a raw 16-digit sequence (credit card)
        ("phone", r"(?:\+1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})(?![0-9])"),
    ]

    def run(self, df):
        findings = {}

        for col in df.columns:
            if df[col].dtype != "object":
                continue

            # Drop NaN values and convert to string
            series = df[col].dropna().astype(str)
            col_findings = {}

            # Track which rows have credit cards to skip false positive phone matches
            cc_rows = set()

            for label, pattern in self.PATTERNS:
                matches = series.str.contains(pattern, regex=True, na=False).sum()
                if matches > 0:
                    # Apply Luhn check for credit cards to reduce false positives
                    if label == "credit_card":
                        matches = self._count_valid_credit_cards(series, pattern)
                        if matches > 0:
                            col_findings[label] = int(matches)
                            # Track which rows matched as credit cards
                            cc_mask = series.str.contains(pattern, regex=True, na=False)
                            cc_rows = set(cc_mask[cc_mask].index)

                    # Skip phone detection in credit card rows
                    elif label == "phone" and cc_rows:
                        # Count phone matches excluding credit card rows
                        phone_mask = series.str.contains(pattern, regex=True, na=False)
                        phone_rows = set(phone_mask[phone_mask].index)
                        unique_phone_rows = phone_rows - cc_rows
                        if len(unique_phone_rows) > 0:
                            col_findings[label] = len(unique_phone_rows)

                    elif label != "credit_card" and matches > 0:
                        col_findings[label] = int(matches)

            if col_findings:
                findings[col] = col_findings

        return findings

    def _count_valid_credit_cards(self, series, pattern):
        """
        Re-count only cells where the extracted number passes the Luhn check.
        This reduces false positives from sequences that look like credit cards.
        """
        count = 0
        for value in series:
            for match in re.finditer(pattern, value):
                digits = re.sub(r"\D", "", match.group())
                if len(digits) >= 13 and self._luhn_check(digits):
                    count += 1
        return count

    def _luhn_check(self, number_str):
        """
        Validate a credit card number using the Luhn algorithm.
        """
        digits = [int(d) for d in number_str]
        digits.reverse()
        total = sum(
            d if i % 2 == 0 else (d * 2 if d * 2 < 10 else d * 2 - 9)
            for i, d in enumerate(digits)
        )
        return total % 10 == 0
