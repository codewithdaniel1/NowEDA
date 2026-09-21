"""Generate the deterministic synthetic dataset used by ``playground.ipynb``.

The data deliberately contains signals for NowEDA demonstrations: identifiers,
PII-like values, Base64 text, missing values, duplicate rows, rare categories,
outliers, correlated numeric fields, dates, categorical targets, and a numeric
target with an intentionally suspicious proxy feature. No real personal data is
used.
"""

import base64
import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path


ROWS = 100_000
DUPLICATE_ROWS = 300
PREVIEW_DUPLICATE_ROWS = 25
OUTPUT = Path(__file__).resolve().parents[1] / "test_data_large_with_pii.csv"
RANDOM_SEED = 20_260_912

FIRST_NAMES = ("Alex", "Blair", "Casey", "Devon", "Emery", "Jordan", "Morgan", "Riley")
LAST_NAMES = ("Bennett", "Chen", "Garcia", "Hughes", "Khan", "Patel", "Rivera", "Taylor")
REGIONS = ("North", "South", "East", "West", "Central")
COUNTRIES = ("US", "CA", "GB", "AU", "DE", "IN")
CHANNELS = ("web", "mobile", "partner", "retail")
DEVICES = ("desktop", "ios", "android", "tablet")
TIERS = ("starter", "standard", "premium", "enterprise")
STATUSES = ("active", "pending", "suspended", "closed")
SOURCES = ("organic", "email", "paid_search", "referral", "partner_beta")

COLUMNS = (
    "record_id", "customer_id", "email", "phone", "ssn",
    "payment_card", "signup_timestamp", "last_login_timestamp", "event_timestamp",
    "region", "country", "acquisition_channel", "device_type", "plan_tier",
    "account_status", "churned", "fraud_flag", "account_balance", "monthly_income",
    "credit_limit", "transaction_count",
    "avg_transaction_amount", "lifetime_value", "account_age_days",
    "days_since_last_login", "support_tickets", "satisfaction_score",
    "promo_source", "notes", "data_source", "encoded_reference", "fraud_risk_score",
)


def _card_number(index):
    """Return a clearly synthetic but Luhn-valid, hyphenated card number."""
    digits = "4{:014d}".format(index % 10**14)
    check = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 0:
            value *= 2
            if value > 9:
                value -= 9
        check += value
    number = digits + str((10 - check % 10) % 10)
    return "{}-{}-{}-{}".format(number[:4], number[4:8], number[8:12], number[12:])


def _timestamp(value):
    return value.isoformat(timespec="seconds")


def _sigmoid(value):
    """Return a numerically stable logistic probability."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _row(index, rng):
    customer_number = 1_000_000 + index
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    domain = rng.choice(("example.test", "synthetic.invalid", "demo.example"))
    email = "{}.{}{}@{}".format(first.lower(), last.lower(), customer_number, domain)
    phone = "+1-{}-{:03d}-{:04d}".format(
        200 + index % 700, (index * 37) % 1000, (index * 7919) % 10000
    )
    ssn = "{:03d}-{:02d}-{:04d}".format(900 + index % 100, index % 100, index % 10000)
    signup = datetime(2019, 1, 1) + timedelta(days=index % 2_000, hours=index % 24)
    event = datetime(2026, 1, 1) + timedelta(minutes=index * 11)
    # A latent behavioral group creates realistic unsupervised structure without
    # exposing a synthetic "correct cluster" label to the playground user.
    behavior = rng.choices(("casual", "steady", "power"), (34, 46, 20))[0]
    behavior_profile = {
        "casual": {
            "income_log": 10.45, "transactions": 42, "transaction_sd": 14,
            "amount_log": 3.30, "tickets": 0.50, "satisfaction": 3.05,
            "login_weights": (15, 13, 13, 18, 17, 14, 10),
            "tiers": (65, 25, 8, 2),
        },
        "steady": {
            "income_log": 10.85, "transactions": 125, "transaction_sd": 24,
            "amount_log": 3.85, "tickets": 0.42, "satisfaction": 3.85,
            "login_weights": (26, 21, 17, 15, 10, 7, 4),
            "tiers": (18, 52, 24, 6),
        },
        "power": {
            "income_log": 11.25, "transactions": 235, "transaction_sd": 32,
            "amount_log": 4.45, "tickets": 0.34, "satisfaction": 4.35,
            "login_weights": (39, 25, 17, 10, 5, 3, 1),
            "tiers": (4, 20, 48, 28),
        },
    }[behavior]
    days_since_login = rng.choices(
        (0, 1, 2, 7, 14, 30, 90), behavior_profile["login_weights"]
    )[0]
    last_login = event - timedelta(days=days_since_login, hours=rng.randrange(24))

    monthly_income = round(
        rng.lognormvariate(behavior_profile["income_log"], 0.30), 2
    )
    credit_limit = round(monthly_income * 2.35 + rng.gauss(0, 50), 2)
    utilization_shape = {
        "casual": (2.7, 4.8), "steady": (2.2, 5.5), "power": (1.8, 6.3)
    }[behavior]
    utilization = min(1.65, max(0, rng.betavariate(*utilization_shape)))
    balance = round(credit_limit * utilization + rng.gauss(0, 110), 2)
    transaction_count = max(
        0,
        int(rng.gauss(
            behavior_profile["transactions"], behavior_profile["transaction_sd"]
        )),
    )
    average_transaction = round(
        max(1, rng.lognormvariate(behavior_profile["amount_log"], 0.34)), 2
    )
    account_age = (event.date() - signup.date()).days
    tickets = min(15, int(rng.expovariate(behavior_profile["tickets"])))
    satisfaction = round(max(
        1,
        min(5, rng.gauss(behavior_profile["satisfaction"] - 0.08 * tickets, 0.48)),
    ), 1)

    # Churn follows an interpretable logistic relationship. Transaction volume
    # and satisfaction create a useful linear-classifier baseline; inactivity
    # and support demand add overlap so the result is not artificially perfect.
    churn_logit = (
        -1.40
        - 0.018 * (transaction_count - 85)
        - 0.95 * (satisfaction - 3.2)
        + 0.024 * days_since_login
        + 0.10 * tickets
    )
    churned = int(rng.random() < _sigmoid(churn_logit))
    status = "closed" if churned and rng.random() < 0.55 else rng.choices(
        STATUSES, (78, 10, 7, 5)
    )[0]

    # Fraud contains a curved two-feature boundary. The post-outcome risk score
    # below remains intentionally leaky so NowEDA can demonstrate that warning.
    transaction_signal = (transaction_count - 125.0) / 72.0
    amount_signal = (math.log(max(average_transaction, 1.0)) - 3.85) / 0.58
    risk_radius = transaction_signal ** 2 + amount_signal ** 2
    fraud_probability = 0.006 + 0.32 * _sigmoid(2.4 * (risk_radius - 5.0))
    fraud = int(rng.random() < min(fraud_probability, 0.70))

    # A mostly linear continuous target supports regression diagnostics while
    # retaining enough noise for honest residual and validation examples.
    lifetime_value = round(max(
        0,
        48 * transaction_count
        + 7.5 * account_age
        + 11 * average_transaction
        + 0.035 * monthly_income
        + rng.gauss(0, 2_500),
    ), 2)
    promo_source = "partner_beta" if index % 401 == 0 else rng.choice(SOURCES[:-1])
    fraud_risk = round(min(
        100,
        max(0, fraud * 82 + 2.4 * risk_radius + rng.gauss(4, 3)),
    ), 2)

    # A small, intentional outlier rate makes IQR detection visible.
    if index % 173 == 0:
        balance = round(balance * 22, 2)
        average_transaction = round(average_transaction * 35, 2)

    note = (
        "Synthetic account; contact {}".format(email)
        if index % 11 == 0
        else "Synthetic support note: {} ticket(s), {} channel.".format(tickets, rng.choice(CHANNELS))
    )
    encoded_reference = base64.b64encode(
        "synthetic-reference-{:06d}".format(customer_number).encode("ascii")
    ).decode("ascii")

    return (
        "REC-{:06d}".format(customer_number),
        "CUST-{:06d}".format(customer_number),
        "" if index % 29 == 0 else email,
        "" if index % 17 == 0 else phone,
        "" if index % 23 == 0 else ssn,
        "" if index % 31 == 0 else _card_number(index),
        _timestamp(signup),
        "" if index % 19 == 0 else _timestamp(last_login),
        _timestamp(event),
        rng.choice(REGIONS),
        rng.choice(COUNTRIES),
        rng.choice(CHANNELS),
        rng.choice(DEVICES),
        rng.choices(TIERS, behavior_profile["tiers"])[0],
        status,
        "yes" if churned else "no",
        fraud,
        balance,
        monthly_income,
        credit_limit,
        transaction_count,
        average_transaction,
        lifetime_value,
        account_age,
        days_since_login,
        tickets,
        satisfaction,
        promo_source,
        note,
        "noweda_playground_synthetic_v2",
        encoded_reference,
        fraud_risk,
    )


def main():
    rng = random.Random(RANDOM_SEED)
    seeds = []
    unique_rows = ROWS - DUPLICATE_ROWS
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        for index in range(unique_rows):
            row = _row(index, rng)
            writer.writerow(row)
            if len(seeds) < DUPLICATE_ROWS:
                seeds.append(row)
            # Keep a modest duplicate sample inside the notebook's 5,000-row preview.
            if index == 4_974:
                writer.writerows(seeds[:PREVIEW_DUPLICATE_ROWS])
        writer.writerows(seeds[PREVIEW_DUPLICATE_ROWS:])
    print("Wrote {:,} rows × {} columns to {}".format(ROWS, len(COLUMNS), OUTPUT))


if __name__ == "__main__":
    main()
