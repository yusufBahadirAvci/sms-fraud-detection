"""Tests for the data pipeline.

We use a tiny fake DataFrame, so the tests:
- do not need the internet,
- run very fast,
- always give the same result.
"""

import pandas as pd

from src.sms_fraud.data import split_data, validate_data


def make_fake_df() -> pd.DataFrame:
    """A small DataFrame with some "bad" rows on purpose."""
    return pd.DataFrame(
        {
            "label": ["ham", "spam", "ham", "ham", "unknown", "SPAM "],
            "message": [
                "See you tomorrow",
                "You won a prize, click here",
                "See you tomorrow",  # duplicate of row 0
                "   ",               # empty message (only spaces)
                "Some text",         # invalid label
                "Free gift for you",  # label needs cleaning -> "spam"
            ],
        }
    )


def test_validate_removes_bad_rows():
    clean = validate_data(make_fake_df())
    # 6 rows - 1 empty - 1 invalid label - 1 duplicate = 3 rows.
    assert len(clean) == 3


def test_validate_keeps_only_valid_labels():
    clean = validate_data(make_fake_df())
    # "SPAM " must be cleaned to "spam"; "unknown" must be gone.
    assert set(clean["label"]) == {"ham", "spam"}


def test_validate_does_not_change_input():
    # The function must work on a copy, not change our original data.
    df = make_fake_df()
    validate_data(df)
    assert len(df) == 6


def test_split_keeps_spam_ratio():
    # 40 ham + 10 spam = 20% spam.
    df = pd.DataFrame(
        {
            "label": ["ham"] * 40 + ["spam"] * 10,
            "message": [f"message {i}" for i in range(50)],
        }
    )
    train, test = split_data(df, test_size=0.2, random_state=42)

    # 20% of 50 = 10 rows in test, 40 in train, no rows lost.
    assert len(test) == 10
    assert len(train) == 40
    # Stratified split: both parts keep exactly 20% spam.
    assert (train["label"] == "spam").mean() == 0.2
    assert (test["label"] == "spam").mean() == 0.2


def test_split_has_no_overlap():
    # A row must never be in both train and test.
    df = pd.DataFrame(
        {
            "label": ["ham"] * 8 + ["spam"] * 2,
            "message": [f"message {i}" for i in range(10)],
        }
    )
    train, test = split_data(df, test_size=0.2, random_state=42)
    assert set(train["message"]).isdisjoint(set(test["message"]))
