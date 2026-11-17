"""Tests for model training.

We train on 10 fake messages. The goal is NOT a good model; the goal is to
check that the code runs and returns values of the right type and range.
No MLflow, no files, no internet.
"""

from src.sms_fraud.train import build_pipeline, evaluate

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


def train_tiny_model():
    model = build_pipeline(max_features=100, ngram_range=(1, 1), C=1.0)
    model.fit(MESSAGES, LABELS)
    return model


def test_predictions_are_zero_or_one():
    predictions = train_tiny_model().predict(MESSAGES)
    assert len(predictions) == len(MESSAGES)
    assert set(predictions) <= {0, 1}


def test_metrics_are_between_zero_and_one():
    predictions = train_tiny_model().predict(MESSAGES)
    metrics = evaluate(LABELS, predictions)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0
