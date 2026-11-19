"""Tests for the monitoring functions. Small fake data, no model file needed."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.sms_fraud.monitoring import compare, compare_by_language, psi, write_alert, write_report
from src.sms_fraud.text_stats import detect_language, find_vectorizer, oov_ratio

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


# ----- Language detector (Step 7a) -----


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        # Turkish-only letters -> "tr" at once.
        ("Kargonuz teslim edilemedi, adresinizi güncelleyin", "tr"),
        # Turkish typed WITHOUT Turkish letters: found by stopwords.
        ("Yarin sabah bulusalim mi, simdi cok yorgunum", "tr"),
        # Apostrophe suffix "'a" must not count as the English word "a".
        ("KAMPANYA BITTI. SMS ALMAMAK ICIN RET YAZIP 7889'A GONDERIN", "tr"),
        ("Are you coming to the party tonight?", "en"),
        # "ü" means "you" in the English UCI data: must stay English.
        ("How ü doing? I will call u later", "en"),
        # No known word at all -> the default, "en".
        (":)", "en"),
    ],
)
def test_detect_language(message, expected):
    assert detect_language(message) == expected


# ----- Per-language monitoring (Step 7a) -----


def test_each_language_gets_its_own_table(tmp_path):
    # English: short messages; Turkish: long ones. Each current window looks
    # like ITS OWN reference, so there must be no alert at all. (With ONE
    # mixed reference, the length difference would look like drift.)
    en = make_window(oov=0.1, length=60, prob=0.1)
    tr = make_window(oov=0.1, length=130, prob=0.6)
    metrics = {"accuracy": 0.99, "precision": 0.99, "recall": 0.95, "f1": 0.97}
    refs = {"en": (en, metrics), "tr": (tr, metrics)}
    cur = pd.concat([en.assign(lang="en"), tr.assign(lang="tr")], ignore_index=True)

    sections = compare_by_language(refs, cur, THRESHOLDS, 10, {"en": metrics, "tr": metrics})
    assert set(sections) == {"en", "tr"}
    assert all(sec["alerts"] == [] for sec in sections.values())
    assert sections["en"]["n_cur"] == len(en) and sections["tr"]["n_cur"] == len(tr)

    report = write_report(sections, tmp_path, "test window").read_text(encoding="utf-8")
    assert "## Language: en" in report and "## Language: tr" in report
    # Two tables -> two header lines.
    assert report.count("| Metric | Reference | Current |") == 2


def test_alert_names_its_language():
    en = make_window(oov=0.1, length=60, prob=0.1)
    refs = {"en": (en, None), "tr": (make_window(oov=0.1, length=130, prob=0.6), None)}
    # Only the Turkish current window is strange (most words unknown).
    cur = pd.concat(
        [en.assign(lang="en"), make_window(oov=0.9, length=130, prob=0.6).assign(lang="tr")]
    )
    sections = compare_by_language(refs, cur, THRESHOLDS, 10, {})
    assert sections["en"]["alerts"] == []
    assert len(sections["tr"]["alerts"]) == 1 and sections["tr"]["alerts"][0].startswith("[tr]")
