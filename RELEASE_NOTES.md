# v1.0.0 — Fair champion, documentation

First complete release of the SMS fraud detection MLOps project (English + Turkish).

## What is in v1.0.0

- **Model:** TF-IDF on letter pieces (char 3–5 grams) + Logistic Regression (`B_char`,
  MLflow version 19, alias `champion`). Trained on English + Turkish SMS.
- **Fair training data:** the model is trained on the **clean split**. Near-duplicate messages are
  grouped, and a group is never in both train and test (old split leaked 5.1% EN / 11.4% TR of test rows).
- **Fair promotion:** the old champion (v13, old split) was scored only on test rows it never saw.
  `B_char` won under the existing rule. `previous` = v13, so rollback is one command.
- **Pipeline:** DVC stages `prepare`, `prepare_turkish`, `leakage`, `combine`, `eda`, `train`,
  `challenge`, `monitor`. Data and model are stored on the DagsHub DVC remote.
- **Serving:** FastAPI (`/health`, `/predict`, `/predict/batch`) and a Streamlit UI, both in Docker
  with healthchecks. Images on GHCR: `ghcr.io/yusufbahadiravci/sms-fraud-api` and `sms-fraud-ui`.
- **CI/CD:** GitHub Actions. CI: ruff, pytest (71 tests, 93% coverage), Docker smoke test.
  CD: builds images with the real model and pushes them to GHCR.
- **Monitoring:** per-language drift checks (OOV, KS test, PSI, F1 drop). Weekly job opens a GitHub
  issue on drift.
- **Docs:** new `README.md` (architecture diagram, results, limitations) and `docs/CONCEPTS.md`
  (study guide for every concept and design pattern).

## Final metrics (champion B_char, v19)

| Test set | Precision | Recall | F1 |
|---|---|---|---|
| English, clean split (1032 messages) | 0.983 | 0.883 | 0.930 |
| Turkish, clean split (948 messages) | 1.000 | 0.996 | 0.998 |
| Turkish phishing challenge set (40 messages) | 1.000 | 0.850 | 0.919 |
| Rule-based baseline, English | 1.000 | 0.094 | 0.171 |
| Rule-based baseline, Turkish | 0.941 | 0.063 | 0.119 |
| Rule-based baseline, challenge set | 1.000 | 0.350 | 0.519 |

Old champion v13 on unseen-only rows: English F1 0.884, Turkish F1 0.994.
On the same rows B_char scores exactly the same, so the main gain of v1.0.0 is a
**fair and leakage-free** model, not a higher score.

## Known limitations

- The Turkish data is mostly advertising spam vs. personal messages, so Turkish F1 is too optimistic.
- Phishing recall is low: 3 of 20 hand-written phishing messages are missed (account login alert,
  SIM card verification, "mom, I have a new number, send money").
- English recall (0.88) is lower than precision (0.98).
- Monitoring depends on a simple rule-based language detector (about 97% correct).
- Small datasets (about 5,100 EN and 4,700 TR messages; challenge set 40).
- MLflow uses a local SQLite file; the registry is not shared.

## Run it

```bash
docker network create sms
docker run -d --name api --network sms -p 8000:8000 ghcr.io/yusufbahadiravci/sms-fraud-api:latest
docker run -d --name ui --network sms -p 8501:8501 -e API_URL=http://api:8000 ghcr.io/yusufbahadiravci/sms-fraud-ui:latest
```
