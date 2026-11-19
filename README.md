# SMS Fraud Detection

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
pip install -r requirements.txt
```

## Build data and model (DVC pipeline)
```powershell
dvc repro                         # prepare, prepare_turkish, combine, eda, train (runs A/B/C), monitor
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlruns/mlflow.db   # http://127.0.0.1:5000
```
The model is trained on English + Turkish (`combined_train.csv`) and scored on each
language's test set separately (`en_*` / `tr_*` metrics). Training only writes
candidates (`models/candidates/`); it never replaces the served model.

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
dvc repro monitor                                           # combined test (en+tr) vs Turkish test -> reports/monitoring/
python scripts/simulate_traffic.py                          # API must run: 100 English + 100 Turkish requests
python -m src.sms_fraud.monitoring --mode log --take 100 --out-dir reports/monitoring_live/english
python -m src.sms_fraud.monitoring --mode log --skip 100 --take 100 --out-dir reports/monitoring_live/turkish
```
A crossed threshold logs `DRIFT ALERT: ...` and writes `alert.json` next to `drift_report.md`.
Thresholds are in `config/config.yaml` (`monitoring:`).

The raw Turkish file is tracked with DVC (`data/external/TurkishSMSCollection.csv.dvc`), not Git.

## Test
```powershell
pytest -v
```
