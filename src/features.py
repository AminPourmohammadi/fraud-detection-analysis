"""Features and the decision rule, shared by both notebooks.

Notebooks 01 and 02 import from this file, so training and monitoring
use exactly the same inputs and the same decision rule.
"""

import numpy as np

# The 28 anonymised bank features, then amount, then hour of day.
FEATURES = []
for i in range(1, 29):
    FEATURES.append("V" + str(i))
FEATURES.append("Amount")
FEATURES.append("hour_of_day")

# What it costs to send one transaction to a human reviewer (EUR).
REVIEW_COST_EUR = 5.0

SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24


def add_hour_of_day(df):
    """Turn 'seconds since the first transaction' into an hour, 0-23."""
    df = df.copy()
    hours_elapsed = df["Time"] // SECONDS_PER_HOUR
    df["hour_of_day"] = (hours_elapsed % HOURS_PER_DAY).astype(int)
    return df


def expected_loss(fraud_prob, amount):
    """Money we expect to lose if we let the transaction through."""
    return fraud_prob * amount


def flag_for_review(fraud_prob, amount, review_cost=REVIEW_COST_EUR):
    """Review when the expected loss is bigger than the cost of a review.

    Works on single numbers and on whole arrays.
    """
    return np.asarray(expected_loss(fraud_prob, amount)) > review_cost
