# Model Promotion Report

- Created: 2026-11-19 15:33 UTC
- Champion: version 13 (models/model.joblib before this run)
- Challenger (best tr_f1): B_char (version 19)
- Decision: **REJECTED**

## Rules

1. challenger tr_f1 > champion tr_f1
2. challenger en_f1 >= champion en_f1 - 0.02

## Checks

- Turkish F1: 0.9980 vs 0.9990 -> not better (FAIL)
- English F1: 0.9300 vs 0.9680 (drop +0.0380, limit 0.02) -> too big a drop (FAIL)

## Comparison (en = English test set, tr = Turkish test set)

| Metric | champion (v13) | A_word (v18) | B_char (v19) ← challenger | C_word_char (v20) |
|---|---|---|---|---|
| en_accuracy | 0.9922 | 0.9797 | 0.9835 | 0.9845 |
| en_precision | 0.9918 | 0.9652 | 0.9826 | 0.9828 |
| en_recall | 0.9453 | 0.8672 | 0.8828 | 0.8906 |
| en_f1 | 0.9680 | 0.9136 | 0.9300 | 0.9344 |
| tr_accuracy | 0.9989 | 0.9916 | 0.9979 | 0.9958 |
| tr_precision | 1.0000 | 0.9980 | 1.0000 | 0.9980 |
| tr_recall | 0.9980 | 0.9861 | 0.9960 | 0.9941 |
| tr_f1 | 0.9990 | 0.9920 | 0.9980 | 0.9960 |

Rollback: `python -m src.sms_fraud.promote rollback` (then `dvc add models/model.joblib`).
