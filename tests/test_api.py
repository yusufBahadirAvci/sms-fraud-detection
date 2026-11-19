"""Tests for the API, using FastAPI's TestClient.

TestClient calls the API in memory: no real server, no network.
We use a tiny FAKE model, so these tests do not need models/model.joblib
(that file is made by "dvc repro" and may not exist, e.g. on a new clone).
"""

import json

import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.sms_fraud import api

FRAUD_TEXT = "You won a free prize, click here to claim cash now"
NORMAL_TEXT = "See you at dinner tonight, I will call you later"


def make_fake_service() -> api.ModelService:
    """Train a very small model in a few milliseconds."""
    messages = [
        "You won a free prize, click here",
        "Claim your free cash prize now",
        "Winner! Free gift, call now to claim",
        "See you at dinner tonight",
        "Can you call me later?",
        "I am on the bus, home soon",
    ]
    labels = [1, 1, 1, 0, 0, 0]
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(C=10.0))])
    pipeline.fit(messages, labels)
    return api.ModelService(pipeline, version="test-version", threshold=0.5)


# A fixture is a helper that pytest runs before each test that asks for it.
# "monkeypatch" changes things only during one test, then puts them back.
@pytest.fixture
def client(monkeypatch, tmp_path):
    # 1. When the app starts, load our fake model instead of the real file.
    monkeypatch.setattr(api, "load_model_service", make_fake_service)
    # 2. Write predictions to a temporary file, not to the real log.
    monkeypatch.setattr(api, "PREDICTIONS_LOG", tmp_path / "predictions.jsonl")
    # "with" runs the lifespan code (startup), like a real server start.
    with TestClient(api.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_valid_message(client):
    response = client.post("/predict", json={"message": FRAUD_TEXT})
    assert response.status_code == 200
    body = response.json()
    assert body["is_fraud"] is True
    assert 0.5 <= body["probability"] <= 1.0
    assert body["model_version"] == "test-version"


def test_predict_normal_message(client):
    body = client.post("/predict", json={"message": NORMAL_TEXT}).json()
    assert body["is_fraud"] is False
    assert 0.0 <= body["probability"] < 0.5


@pytest.mark.parametrize("bad_message", ["", "   "])
def test_empty_message_returns_422(client, bad_message):
    # 422 = "Unprocessable Content": the request has the wrong data.
    response = client.post("/predict", json={"message": bad_message})
    assert response.status_code == 422


def test_batch_endpoint(client):
    response = client.post("/predict/batch", json={"messages": [FRAUD_TEXT, NORMAL_TEXT]})
    assert response.status_code == 200
    results = response.json()["results"]
    # One result per message, in the same order.
    assert [r["is_fraud"] for r in results] == [True, False]


def test_predictions_are_logged(client):
    client.post("/predict/batch", json={"messages": [FRAUD_TEXT, NORMAL_TEXT]})
    lines = api.PREDICTIONS_LOG.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert set(record) == {
        "timestamp",
        "message_length",
        "oov_ratio",
        "probability",
        "prediction",
        "model_version",
    }
    # We log the LENGTH, never the text itself (privacy).
    assert record["message_length"] == len(FRAUD_TEXT)
    assert "message" not in record
    assert 0.0 <= record["oov_ratio"] <= 1.0


def test_missing_model_returns_503(monkeypatch, tmp_path):
    # If the model file is missing, the server must still start and tell us.
    def no_model():
        raise FileNotFoundError("no model")

    monkeypatch.setattr(api, "load_model_service", no_model)
    with TestClient(api.app) as test_client:
        assert test_client.get("/health").json()["model_loaded"] is False
        assert test_client.post("/predict", json={"message": "hi"}).status_code == 503
