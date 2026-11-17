"""Tests for model training.

We train on 10 fake messages. The goal is NOT a good model; the goal is to
check that the code runs and returns values of the right type and range.
No MLflow, no files, no internet.
"""

import pytest

from src.sms_fraud.train import MODEL_NAMES, build_classifier, build_pipeline, evaluate

# 5 spam (1) and 5 normal (0) messages.
MESSAGES = [
    "You won a free prize, click here",
    "Winner! Claim your free gift now",
    "Urgent: verify your account today",
    "Free cash prize, call now",
    "Click here to win a free phone",
    "See you at dinner tonight",
    "The meeting starts at 10",
    "Can you call me later?",
    "I am on the bus, home soon",
    "Thanks for the book, see you",
]
LABELS = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]


def train_tiny_model(model_name: str):
    model = build_pipeline(model_name, max_features=100, ngram_range=(1, 1), C=1.0, alpha=1.0)
    model.fit(MESSAGES, LABELS)
    return model


# parametrize runs the SAME test once for each model name.
# So every test below runs twice: once for "logreg", once for "nb".
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_predictions_are_zero_or_one(model_name):
    predictions = train_tiny_model(model_name).predict(MESSAGES)
    assert len(predictions) == len(MESSAGES)
    assert set(predictions) <= {0, 1}


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_metrics_are_between_zero_and_one(model_name):
    predictions = train_tiny_model(model_name).predict(MESSAGES)
    metrics = evaluate(LABELS, predictions)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_factory_uses_the_given_settings():
    # The factory must pass C and alpha to the right model.
    assert build_classifier("logreg", C=10.0).C == 10.0
    assert build_classifier("nb", alpha=0.1).alpha == 0.1


def test_factory_rejects_unknown_model():
    # A typo in the model name must fail loudly, not train something random.
    with pytest.raises(ValueError):
        build_classifier("random_forest")
