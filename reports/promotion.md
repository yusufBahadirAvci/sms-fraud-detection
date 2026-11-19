# Model Promotion Report

- Created: 2026-11-19 15:54 UTC
- Champion: version 13 (models/model.joblib before this run)
- Challenger (best tr_f1): B_char (version 19)
- Decision: **PROMOTED**

## Rules

1. challenger tr_f1 > champion tr_f1
2. challenger en_f1 >= champion en_f1 - 0.02

## Checks

- Turkish F1: 0.9980 vs 0.9939 -> better (OK)
- English F1: 0.9300 vs 0.8837 (drop -0.0463, limit 0.02) -> OK

## Comparison (en = English test set, tr = Turkish test set)

| Metric | champion (v13), unseen-only | A_word (v18) | B_char (v19) ← challenger | C_word_char (v20) | B_char, same unseen rows |
|---|---|---|---|---|---|
| en_accuracy | 0.9747 | 0.9797 | 0.9835 | 0.9845 | 0.9747 |
| en_precision | 0.9500 | 0.9652 | 0.9826 | 0.9828 | 0.9500 |
| en_recall | 0.8261 | 0.8672 | 0.8828 | 0.8906 | 0.8261 |
| en_f1 | 0.8837 | 0.9136 | 0.9300 | 0.9344 | 0.8837 |
| tr_accuracy | 0.9936 | 0.9916 | 0.9979 | 0.9958 | 0.9936 |
| tr_precision | 1.0000 | 0.9980 | 1.0000 | 0.9980 | 1.0000 |
| tr_recall | 0.9880 | 0.9861 | 0.9960 | 0.9941 | 0.9880 |
| tr_f1 | 0.9939 | 0.9920 | 0.9980 | 0.9960 | 0.9939 |

- Champion was trained on the OLD random split, so it is scored only on clean test rows with no near-copy in its training data: en 198 of 1032 rows, tr 157 of 948 rows.
- Same rows for both: champion en_f1 0.8837 / tr_f1 0.9939, B_char en_f1 0.8837 / tr_f1 0.9939.

Rollback: `python -m src.sms_fraud.promote rollback` (then `dvc add models/model.joblib`).
