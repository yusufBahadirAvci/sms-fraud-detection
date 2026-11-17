# SMS Fraud Detection

## Goal
Detect fraud (scam) SMS messages. Step 1 is a simple **rule-based** checker
that looks for suspicious keywords (Turkish and English). It is the baseline
we will compare a future ML model against.

## Setup
```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## How to run
```powershell
python -m src.sms_fraud.rule_based
```

## How to test
```powershell
pytest -v
```
