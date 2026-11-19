# SMS Fraud Detection

[![CI](https://github.com/yusufBahadirAvci/sms-fraud-detection/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/yusufBahadirAvci/sms-fraud-detection/actions/workflows/ci.yml)

## Goal
Detect fraud (spam) SMS messages. The project has:
- a rule-based baseline (`src/sms_fraud/rule_based.py`),
- a data pipeline and EDA (`data.py`, `eda.py`),
- an ML model (TF-IDF + Logistic Regression) tracked with MLflow and built with DVC,
- a FastAPI service (`api.py`) and a Streamlit UI (`ui/app.py`), both runnable with Docker.

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt       # runtime only
pip install -r requirements-dev.txt   # + pytest, pytest-cov, ruff, httpx (for development)
```

## Data & models (DVC remote on DagsHub)
Data and models are not in Git. DVC keeps them in an S3-compatible storage on DagsHub:
`https://dagshub.com/yusufBahadirAvci/sms-fraud-detection` (remote `origin` in `.dvc/config`).
The access token is personal: store it with `--local`, so it goes to `.dvc/config.local`
(ignored by Git) and is never committed.
```powershell
pip install -r requirements-dev.txt     # includes dvc[s3]
# DagsHub uses the token as BOTH the key id and the secret:
dvc remote modify origin --local access_key_id <your DagsHub token>
dvc remote modify origin --local secret_access_key <your DagsHub token>
dvc pull                                # download data + model into the working folder
dvc status -c                           # local cache vs remote: should be in sync
dvc push                                # after "dvc repro" or "dvc add": upload new versions
```

## Build data and model (DVC pipeline)
```powershell
dvc repro                         # prepare, prepare_turkish, leakage, combine, eda, train (A/B/C), challenge, monitor
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlruns/mlflow.db   # http://127.0.0.1:5000
```
The model is trained on English + Turkish (`combined_train.csv`) and scored on each
language's test set separately (`en_*` / `tr_*` metrics). Training only writes
candidates (`models/candidates/`); it never replaces the served model.

## Trustworthy results (Step 7a)
- **Leakage check** (`leakage.py` -> `reports/leakage.md`): the old random split had
  near-copies of test messages in train (English 5.1%, Turkish 11.4% of test rows;
  similarity >= 0.9 on normalized text).
- **Clean split**: near-duplicates are grouped into clusters, and a whole cluster goes to
  train OR test (`data.py`, `leakage_safe_split`). The clean split has 0 leaked rows.
- **Retrained on the clean split** (`reports/metrics.json`; old numbers in
  `reports/metrics_old_split.json`): Turkish F1 stays very high (B_char 0.998), so the
  high score was NOT mainly caused by leakage. The Kaggle Turkish data is simply easy
  (mostly advertising spam vs. personal messages).
- **Caveat**: the current champion was trained on the OLD split, which contains ~80% of the
  new test rows. Its scores on the clean test sets are therefore too optimistic, and the
  promotion rule compares it unfairly (`reports/promotion.md`).
- **Challenge set** (`data/challenge/tr_phishing.csv`, evaluation only, never trained on):
  40 hand-written Turkish messages (modern phishing vs. normal). Results in `reports/challenge.md`.

## Promote a new model (champion / challenger)
```powershell
python -m src.sms_fraud.promote            # best new run (by tr_f1) vs current model
dvc add models/model.joblib                # only if it was promoted
dvc repro monitor                          # re-check drift with the new model
python -m src.sms_fraud.promote rollback   # go back to the MLflow alias "previous"
```
Rules (`config.yaml`, `promotion:`): Turkish F1 must be higher **and** English F1 may drop
by at most 0.02. The decision and the comparison table are in `reports/promotion.md`.
MLflow aliases: `champion` = served model, `previous` = the one before (for rollback).

## Run locally (2 terminals, venv active)
```powershell
# Terminal 1: the API
uvicorn src.sms_fraud.api:app --port 8000
# Terminal 2: the UI
streamlit run ui/app.py
```
- API docs (try the endpoints in the browser): http://localhost:8000/docs
- UI: http://localhost:8501

## Run with Docker
`models/model.joblib` must exist first (run `dvc repro`), because it is copied into the API image.
```powershell
docker compose up --build -d      # build and start api + ui
docker compose ps                 # api should be "healthy"
docker compose logs -f api        # watch the API logs
docker compose down               # stop and remove the containers
```
Predictions are appended to `logs/predictions.jsonl` (kept on your computer, also after `down`).

## Example request
```powershell
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{\"message\": \"You won a FREE prize! Call now\"}'
```

## Monitoring (drift)
```powershell
dvc repro monitor                                           # per language: half of each test set vs the other half -> reports/monitoring/
python scripts/simulate_traffic.py                          # API must run: 100 English + 100 Turkish requests
python -m src.sms_fraud.monitoring --mode log               # live log, split by the logged "lang"
```
Since Step 7a every language has its **own** reference window, and `drift_report.md` has one
table per language. The API logs the detected language (`lang`, from `text_stats.detect_language`:
Turkish-only letters + small stopword lists; about 99% right on English, 93% on Turkish).
A crossed threshold logs `DRIFT ALERT: [lang] ...` and writes `alert.json` next to `drift_report.md`.
Thresholds are in `config/config.yaml` (`monitoring:`).

The raw Turkish file is tracked with DVC (`data/external/TurkishSMSCollection.csv.dvc`), not Git.

## Test and lint
```powershell
ruff check .                  # lint (rules in pyproject.toml)
ruff format .                 # format the code
pytest -v --cov=src           # tests + coverage (no real data or model needed)
```

## Deployment (images on GitHub Container Registry)
After CI is green on `main`, the **CD** workflow (`.github/workflows/cd.yml`, job `deliver`)
pulls the real model from DagsHub, builds both images, tests the API (it must serve the real
model) and pushes them to GHCR with two tags: the short commit sha (never changes) and `latest`.
```powershell
# Only needed while the packages are private (GitHub -> Packages -> package settings):
# log in with a GitHub token that has the "read:packages" scope.
docker login ghcr.io -u <github user>

docker pull ghcr.io/yusufbahadiravci/sms-fraud-api:latest
docker pull ghcr.io/yusufbahadiravci/sms-fraud-ui:latest

docker network create sms
docker run -d --name api --network sms -p 8000:8000 ghcr.io/yusufbahadiravci/sms-fraud-api:latest
docker run -d --name ui --network sms -p 8501:8501 -e API_URL=http://api:8000 ghcr.io/yusufbahadiravci/sms-fraud-ui:latest
```
Use a sha tag (e.g. `:fae0fea`) instead of `latest` to run one exact version, or to roll back.

## Scheduled monitoring
The CD job `monitor` runs every Monday 06:00 UTC (and by the "Run workflow" button). It pulls
the model and the test sets, runs `python -m src.sms_fraud.monitoring`, uploads
`reports/monitoring` as an artifact and, if `alert.json` exists, opens a
**"Drift alert &lt;date&gt;"** issue (or comments on the open one, so there are no duplicates).

## Continuous Integration
`.github/workflows/ci.yml` runs on every push and pull request to `main`:
- **test**: `ruff check`, `ruff format --check`, `pytest` with coverage (must be at least 70%); `coverage.xml` is uploaded as an artifact.
- **docker**: builds the UI image, builds the API image with a dummy model
  (`scripts/make_dummy_model.py`, CI only), starts it, waits for the HEALTHCHECK and calls `/health` and `/predict`.
