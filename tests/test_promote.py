"""Tests for champion / challenger promotion and rollback.

The rule tests use FAKE metrics (plain dicts): no model, no MLflow.
The registry tests use a TEMPORARY MLflow database in a pytest tmp folder,
so the real mlruns/ folder is never touched.
"""

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
from mlflow import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.sms_fraud.promote import (
    CHAMPION,
    PREVIOUS,
    alias_version,
    decide,
    pick_challenger,
    promote,
    rollback,
    unseen_only,
)

MAX_DROP = 0.02
CHAMPION_METRICS = {"en_f1": 0.94, "tr_f1": 0.36}


# ----- The rules (fake metrics) -----


def test_promote_when_turkish_better_and_english_ok():
    ok, reasons = decide(CHAMPION_METRICS, {"en_f1": 0.93, "tr_f1": 0.95}, MAX_DROP)
    assert ok is True
    assert len(reasons) == 2


def test_reject_when_turkish_not_better():
    # Equal is not enough: the challenger must be strictly better.
    ok, _ = decide(CHAMPION_METRICS, {"en_f1": 0.99, "tr_f1": 0.36}, MAX_DROP)
    assert ok is False


def test_reject_when_english_drops_too_much():
    # Great on Turkish, but English falls by 0.05 (> 0.02): English users would suffer.
    ok, reasons = decide(CHAMPION_METRICS, {"en_f1": 0.89, "tr_f1": 0.99}, MAX_DROP)
    assert ok is False
    assert "FAIL" in reasons[1]


def test_english_drop_of_exactly_the_limit_is_allowed():
    # "Not more than 0.02 below" -> a drop of exactly 0.02 is still OK.
    ok, _ = decide(CHAMPION_METRICS, {"en_f1": 0.92, "tr_f1": 0.90}, MAX_DROP)
    assert ok is True


def test_challenger_is_the_run_with_best_turkish_f1():
    runs = {
        "A": {"metrics": {"en_f1": 0.99, "tr_f1": 0.80}},
        "B": {"metrics": {"en_f1": 0.90, "tr_f1": 0.97}},
        "C": {"metrics": {"en_f1": 0.95, "tr_f1": 0.96}},
    }
    assert pick_challenger(runs) == "B"


def test_unseen_only_drops_test_rows_the_model_has_seen():
    # Step 7b: the old champion saw these train messages. A test row that is
    # a near-copy (only the number changes) must not count in its score.
    seen = {"en": pd.Series(["WIN a prize now, call 0800 111", "see you at the cinema tonight"])}
    X = pd.Series(["WIN a prize now, call 0800 999", "my train is late again, sorry"])
    y = pd.Series([1, 0])

    result = unseen_only({"en": (X, y)}, seen, threshold=0.9)

    X_left, y_left = result["en"]
    assert list(X_left) == ["my train is late again, sorry"]
    assert list(y_left) == [0]


# ----- The registry: promote + rollback (temporary MLflow) -----


def tiny_model(spam_word: str) -> Pipeline:
    """A tiny model that calls messages with `spam_word` spam."""
    messages = [f"{spam_word} now", f"get {spam_word}", "see you soon", "call me later"]
    return Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())]).fit(
        messages, [1, 1, 0, 0]
    )


@pytest.fixture
def registry(tmp_path):
    """A fresh MLflow registry with 2 versions of "test-model"."""
    old_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    # Own artifact folder, so nothing is written into the project's mlruns/.
    exp_id = mlflow.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())
    for word in ("prize", "bedava"):  # version 1, version 2
        with mlflow.start_run(experiment_id=exp_id):
            mlflow.sklearn.log_model(
                tiny_model(word), name="model", registered_model_name="test-model"
            )
    yield MlflowClient()
    # Put the global MLflow setting back for the other tests.
    mlflow.set_tracking_uri(old_uri)


def test_promote_moves_aliases_and_copies_file(registry, tmp_path):
    registry.set_registered_model_alias("test-model", CHAMPION, "1")
    candidate, model_path = tmp_path / "candidate.joblib", tmp_path / "model.joblib"
    joblib.dump(tiny_model("bedava"), candidate)

    promote(registry, "test-model", candidate, model_path, new_version="2", old_version="1")

    assert alias_version(registry, "test-model", CHAMPION) == "2"
    assert alias_version(registry, "test-model", PREVIOUS) == "1"
    assert model_path.read_bytes() == candidate.read_bytes()


def test_rollback_restores_previous_model(registry, tmp_path):
    # State after a promotion: champion = v2 ("bedava" model), previous = v1 ("prize" model).
    registry.set_registered_model_alias("test-model", CHAMPION, "2")
    registry.set_registered_model_alias("test-model", PREVIOUS, "1")
    model_path = tmp_path / "model.joblib"

    restored, demoted = rollback(registry, "test-model", model_path)

    assert (restored, demoted) == ("1", "2")
    assert alias_version(registry, "test-model", CHAMPION) == "1"
    assert alias_version(registry, "test-model", PREVIOUS) == "2"
    # The file on disk is now the old model: it gives exactly the same
    # probabilities as version 1 in the registry.
    messages = ["prize now", "bedava now", "see you soon"]
    restored_model = joblib.load(model_path)
    version_1 = mlflow.sklearn.load_model("models:/test-model/1")
    assert (restored_model.predict_proba(messages) == version_1.predict_proba(messages)).all()


def test_rollback_without_previous_fails_clearly(registry, tmp_path):
    registry.set_registered_model_alias("test-model", CHAMPION, "1")
    with pytest.raises(RuntimeError, match="nothing to roll back"):
        rollback(registry, "test-model", tmp_path / "model.joblib")
