"""Tests for the near-duplicate leakage check and the cluster split (Step 7a)."""

import pandas as pd

from src.sms_fraud.data import split_data
from src.sms_fraud.leakage import check_leakage, cluster_ids, normalize_text


def test_normalize_text_lowercase_url_digits_punctuation_spaces():
    text = "WIN £1000!  Call 0800-123 now:\n http://x.com/a"
    # "0800-123" -> "0000 000": every digit is "0", the "-" becomes a space.
    assert normalize_text(text) == "win 0000 call 0000 000 now <url>"


def test_normalize_text_makes_template_copies_equal():
    # Same spam, other number and other link -> the SAME normalized text.
    a = "Kargonuz bekliyor! Tikla: www.example.com/a1 Tel 0555 111 22 33"
    b = "kargonuz bekliyor tikla example.com/zz9 tel 0532 987 65 43"
    assert normalize_text(a) == normalize_text(b)


def test_normalize_text_turkish_capital_i():
    # "İ".lower() would add an extra dot mark; we want a plain "i".
    assert normalize_text("İNDİRİM") == "indirim"


def test_cluster_ids_groups_near_duplicates_only():
    messages = [
        "Free entry in a weekly comp to win an ipod, txt POD to 80182",
        "Free entry in a weekly comp to win an ipod! Txt POD to 80999",  # near-copy of 0
        "See you at the cinema tonight, bring popcorn",
    ]
    ids = cluster_ids(messages, threshold=0.9)
    assert ids[0] == ids[1]
    assert ids[0] != ids[2]


def make_clustered_data() -> pd.DataFrame:
    """60 rows in 20 clusters of 3 near-copies (only the number changes)."""
    words = ["alpha", "bravo", "delta", "gamma", "kilo", "lima", "oscar", "papa", "romeo", "tango"]
    rows = []
    for c in range(20):
        label = "spam" if c % 4 == 0 else "ham"
        # Two different words per cluster, so clusters are NOT near-copies of each other.
        base = f"{words[c % 10]} {words[(c * 3 + 1) % 10]} message number {'x' * c}"
        rows += [{"label": label, "message": f"{base} {k}{k}{k}"} for k in range(3)]
    return pd.DataFrame(rows)


def test_cluster_split_never_puts_a_cluster_on_both_sides():
    df = make_clustered_data()
    groups = cluster_ids(df["message"], threshold=0.9)
    # The 3 copies of each cluster must get the same id.
    assert len(set(groups)) <= 20

    train, test = split_data(df, test_size=0.2, random_state=42, groups=groups)
    cluster_of = dict(zip(df["message"], groups, strict=True))
    train_clusters = {cluster_of[m] for m in train["message"]}
    test_clusters = {cluster_of[m] for m in test["message"]}
    assert train_clusters.isdisjoint(test_clusters)
    # Nothing lost, nothing added.
    assert len(train) + len(test) == len(df)
    # And the leakage check agrees: no test message has a near-copy in train.
    result = check_leakage(train["message"], test["message"], threshold=0.9)
    assert not result["near"].any()


def test_random_split_does_leak_on_the_same_data():
    # The OLD split (no groups) puts copies on both sides: this is the bug
    # that Step 7a fixes. If this ever fails, the test data is too easy.
    df = make_clustered_data()
    train, test = split_data(df, test_size=0.2, random_state=42)
    result = check_leakage(train["message"], test["message"], threshold=0.9)
    assert result["near"].any()
