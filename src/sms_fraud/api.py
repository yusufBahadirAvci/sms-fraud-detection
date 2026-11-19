"""REST API that serves the trained SMS fraud model.

Run it locally with:
    uvicorn src.sms_fraud.api:app --port 8000
Then open http://localhost:8000/docs to try it in the browser.
"""

import hashlib
import json
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
from fastapi import Depends, FastAPI, HTTPException, Request

from src.sms_fraud.config import get_config, project_path
from src.sms_fraud.logger import get_logger
from src.sms_fraud.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from src.sms_fraud.text_stats import find_vectorizer, oov_ratio

logger = get_logger("sms_fraud.api")

# Path of the prediction log. It is a module variable, so tests can point it
# to a temporary file and never mix test data into the real log.
PREDICTIONS_LOG = project_path(get_config()["api"]["predictions_log"])

# The server can answer several requests at the same time (in threads).
# A lock makes sure only ONE thread writes to the log file at a time,
# so two lines never get mixed together.
_log_lock = threading.Lock()


class ModelService:
    """The loaded model plus the small extras the API needs."""

    def __init__(self, pipeline, version: str, threshold: float):
        self.pipeline = pipeline
        self.version = version
        self.threshold = threshold
        # The TF-IDF vectorizer of the pipeline. We need its vocabulary to
        # compute the OOV ratio (share of unknown words) for monitoring.
        # find_vectorizer also works for word+char models (Step 5b).
        self.vectorizer = find_vectorizer(pipeline)

    def oov_ratios(self, messages: list[str]) -> list[float]:
        """Share of unknown words for each message (0.0 = all known, 1.0 = all unknown)."""
        return [oov_ratio(self.vectorizer, m) for m in messages]

    def predict(self, messages: list[str]) -> list[PredictResponse]:
        """Predict many messages at once (faster than one by one)."""
        # predict_proba gives 2 numbers per message: [P(ham), P(spam)].
        # We take column 1 = probability of spam.
        spam_probs = self.pipeline.predict_proba(messages)[:, 1]
        return [
            PredictResponse(
                is_fraud=bool(p >= self.threshold),
                # round: 4 digits are enough and the JSON stays easy to read.
                probability=round(float(p), 4),
                model_version=self.version,
            )
            for p in spam_probs
        ]


def file_md5(path) -> str:
    """Return the MD5 hash of a file.

    We use it as the model version. Why MD5? DVC uses the same hash for
    models/model.joblib in dvc.lock. So you can find exactly which DVC
    pipeline run built the model that answered a request.
    """
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        # Read in blocks of 1 MB, so big files do not fill the memory.
        for block in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(block)
    return md5.hexdigest()


def load_model_service() -> ModelService:
    """Load the model file from disk and wrap it in a ModelService."""
    cfg = get_config()
    model_path = project_path(cfg["model"]["model_path"])
    pipeline = joblib.load(model_path)
    version = file_md5(model_path)
    logger.info("Loaded model %s (version %s)", model_path, version)
    return ModelService(pipeline, version=version, threshold=cfg["api"]["threshold"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Code that runs ONCE when the server starts (before "yield") and once
    when it stops (after "yield").

    Why load the model here and not in each request? Loading takes time.
    Loading once makes every request fast.
    """
    try:
        app.state.model_service = load_model_service()
    except FileNotFoundError:
        # Do not crash: the server still starts, /health says
        # model_loaded=false, and /predict answers 503. This makes the
        # problem easy to see from outside.
        logger.error("Model file not found. Run 'dvc repro' first.")
        app.state.model_service = None
    yield
    logger.info("API is shutting down")


app = FastAPI(
    title="SMS Fraud Detection API",
    description="Classifies SMS messages as fraud (spam) or normal (ham).",
    version="1.0.0",
    lifespan=lifespan,
)


# PATTERN: Dependency Injection — endpoints do not find or load the model themselves; FastAPI "injects" it with Depends(get_model_service), so tests can replace it with a fake model.
def get_model_service(request: Request) -> ModelService:
    """Give the loaded model to an endpoint, or answer 503 if there is none."""
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        # 503 = "Service Unavailable": the server runs, but cannot do the job.
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return service


def log_predictions(messages: list[str], results: list[PredictResponse], oov: list[float]) -> None:
    """Add one JSON line per prediction to logs/predictions.jsonl.

    We save NUMBERS about the message (length, share of unknown words), never
    the message text. Why? SMS can contain private data (names, codes).
    For monitoring, numbers are enough.
    JSONL = one JSON object per line: easy to append and easy to read later.
    """
    PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    lines = [
        json.dumps(
            {
                "timestamp": timestamp,
                "message_length": len(message),
                # round: 4 digits are enough and keep the log small.
                "oov_ratio": round(ratio, 4),
                "probability": result.probability,
                "prediction": result.is_fraud,
                "model_version": result.model_version,
            }
        )
        for message, result, ratio in zip(messages, results, oov)
    ]
    with _log_lock:
        with open(PREDICTIONS_LOG, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


# Note: these endpoints use "def", not "async def". The model does heavy CPU
# work; FastAPI runs "def" endpoints in a thread pool, so one slow request
# does not block the whole server.
@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Is the server alive, and is the model ready?"""
    loaded = getattr(request.app.state, "model_service", None) is not None
    return HealthResponse(status="ok", model_loaded=loaded)


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, service: ModelService = Depends(get_model_service)) -> PredictResponse:
    """Classify one SMS message."""
    result = service.predict([body.message])[0]
    log_predictions([body.message], [result], service.oov_ratios([body.message]))
    return result


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(
    body: BatchPredictRequest, service: ModelService = Depends(get_model_service)
) -> BatchPredictResponse:
    """Classify many SMS messages in one request."""
    results = service.predict(body.messages)
    log_predictions(body.messages, results, service.oov_ratios(body.messages))
    return BatchPredictResponse(results=results)
