# Model Promotion Report

- Created: 2026-11-19 11:51 UTC
- Champion: version 11 (models/model.joblib before this run)
- Challenger (best tr_f1): B_char (version 13)
- Decision: **PROMOTED**

## Rules

1. challenger tr_f1 > champion tr_f1
2. challenger en_f1 >= champion en_f1 - 0.02

## Checks

- Turkish F1: 0.9970 vs 0.3223 -> better (OK)
- English F1: 0.9350 vs 0.9388 (drop +0.0038, limit 0.02) -> OK

## Comparison (en = English test set, tr = Turkish test set)

| Metric | champion (v11) | A_word (v12) | B_char (v13) ← challenger | C_word_char (v14) |
|---|---|---|---|---|
| en_accuracy | 0.9855 | 0.9806 | 0.9845 | 0.9835 |
| en_precision | 0.9829 | 0.9655 | 0.9746 | 0.9664 |
| en_recall | 0.8984 | 0.8750 | 0.8984 | 0.8984 |
| en_f1 | 0.9388 | 0.9180 | 0.9350 | 0.9312 |
| tr_accuracy | 0.5696 | 0.9895 | 0.9968 | 0.9958 |
| tr_precision | 1.0000 | 0.9940 | 1.0000 | 0.9980 |
| tr_recall | 0.1921 | 0.9861 | 0.9941 | 0.9941 |
| tr_f1 | 0.3223 | 0.9901 | 0.9970 | 0.9960 |

Rollback: `python -m src.sms_fraud.promote rollback` (then `dvc add models/model.joblib`).

## Rollback (2026-11-19 11:51 UTC)

Champion is version 11 again; version 13 is now 'previous'.

## Rollback (2026-11-19 11:51 UTC)

Champion is version 13 again; version 11 is now 'previous'.
