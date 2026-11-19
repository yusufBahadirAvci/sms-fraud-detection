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
dvc repro                         # prepare -> eda -> train
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlruns/mlflow.db   # http://127.0.0.1:5000
```

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
dvc repro monitor                                           # English test vs Turkish data -> reports/monitoring/
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
