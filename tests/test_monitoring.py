"""Tests for the monitoring functions. Small fake data, no model file needed."""

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.sms_fraud.monitoring import compare, psi, write_alert
from src.sms_fraud.text_stats import find_vectorizer, oov_ratio

THRESHOLDS = {"oov_ratio": 0.5, "ks_pvalue": 0.05, "psi": 0.2, "f1_drop": 0.10}


# ----- PSI -----


def test_psi_is_zero_for_identical_data():
    rng = np.random.default_rng(42)
    probs = rng.uniform(0, 1, 1000)
    assert psi(probs, probs) < 1e-9


def test_psi_small_random_window_is_not_an_alert():
    # 100 values drawn from the SAME distribution must not look like drift,
    # even if some bins are empty by chance. (This was a real false alarm.)
    rng = np.random.default_rng(0)
    reference = np.concatenate(
        [rng.uniform(0, 0.1, 880), rng.uniform(0.1, 0.9, 40), rng.uniform(0.9, 1, 80)]
    )
    window = rng.choice(reference, 100, replace=False)
    assert psi(reference, window) < 0.2


def test_psi_is_large_for_very_different_data():
    low = np.full(500, 0.05)  # model says "surely normal"
    high = np.full(500, 0.95)  # model says "surely spam"
    # Way above the 0.2 "big change" rule of thumb.
    assert psi(low, high) > 1.0


# ----- OOV ratio -----


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer().fit(["free prize call now", "see you at dinner"])


def test_oov_ratio_is_zero_for_known_words():
    assert oov_ratio(make_vectorizer(), "Free PRIZE, call now!") == 0.0


def test_oov_ratio_is_one_for_unknown_words():
    assert oov_ratio(make_vectorizer(), "Bu mesaja tiklayiniz") == 1.0


def test_oov_ratio_counts_the_share():
    # "free" is known, "bedava" is not -> 1 of 2 unknown.
    assert oov_ratio(make_vectorizer(), "free bedava") == 0.5


def test_oov_ratio_for_char_model_counts_letter_pieces():
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3)).fit(["tıklayın"])
    # Same word -> every piece known. A new suffix still shares most pieces.
    assert oov_ratio(char, "tıklayın") == 0.0
    assert 0.0 < oov_ratio(char, "tıklayınız") < 0.5


def test_find_vectorizer_picks_word_part_of_feature_union():
    union = FeatureUnion(
        [("word", TfidfVectorizer()), ("char", TfidfVectorizer(analyzer="char_wb"))]
    )
    pipeline = Pipeline([("tfidf", union), ("clf", LogisticRegression())])
    assert find_vectorizer(pipeline).analyzer == "word"
    # A plain pipeline: the vectorizer itself.
    single = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    assert find_vectorizer(single) is single.named_steps["tfidf"]


# ----- Alerts -----


def make_window(oov: float, length: int, prob: float, n: int = 200) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "length": np.full(n, length) + np.arange(n) % 5,  # small spread
            "oov_ratio": np.full(n, oov),
            "probability": np.full(n, prob),
            "prediction": np.full(n, prob >= 0.5),
        }
    )


def test_no_alert_when_windows_are_the_same():
    ref = make_window(oov=0.1, length=70, prob=0.1)
    _, alerts = compare(ref, ref.copy(), THRESHOLDS, psi_bins=10)
    assert alerts == []


def test_alert_when_oov_threshold_is_crossed():
    ref = make_window(oov=0.1, length=70, prob=0.1)
    cur = make_window(oov=0.9, length=70, prob=0.1)  # only OOV is different
    rows, alerts = compare(ref, cur, THRESHOLDS, psi_bins=10)
    assert len(alerts) == 1 and "OOV" in alerts[0]
    assert next(r for r in rows if r["metric"].startswith("OOV"))["status"] == "ALERT"


def test_alert_when_f1_drops():
    ref = make_window(oov=0.1, length=70, prob=0.1)
    good = {"accuracy": 0.98, "precision": 0.98, "recall": 0.90, "f1": 0.94}
    bad = {"accuracy": 0.50, "precision": 0.90, "recall": 0.10, "f1": 0.18}
    _, alerts = compare(ref, ref.copy(), THRESHOLDS, psi_bins=10, ref_metrics=good, cur_metrics=bad)
    assert any("F1 dropped" in a for a in alerts)


def test_alert_file_is_written_and_removed(tmp_path):
    path = write_alert(["OOV ratio too high"], tmp_path, "test window")
    assert json.loads(path.read_text(encoding="utf-8"))["alerts"] == ["OOV ratio too high"]
    # No alerts on the next run -> the old file must be deleted.
    assert write_alert([], tmp_path, "test window") is None
    assert not path.exists()
