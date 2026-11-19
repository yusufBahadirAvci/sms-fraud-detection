"""Score the champion on a small, hand-written Turkish phishing set.

Run it with:
    python -m src.sms_fraud.challenge

Why? The Kaggle Turkish test set is mostly old ADVERTISING spam (shops,
hotels, credit offers). Today's real danger is PHISHING: fake cargo
messages, fake e-Devlet refunds, "your account is blocked, click here".
data/challenge/tr_phishing.csv has 40 such messages (20 phishing, 20 normal),
written by hand, with fake links (example.com) and fake numbers.

RULE: this set is for EVALUATION ONLY. It is never used for training or
for tuning settings. If we tuned on it, its score would stop telling us
the truth about new phishing messages.

Output:
    reports/challenge.json  precision, recall, f1 (DVC metrics, kept in Git)
    reports/challenge.md    the scores + EVERY misclassified message
    MLflow                  one run "challenge_tr_phishing" with challenge_* metrics
"""

import json

import joblib
import mlflow
import pandas as pd
from mlflow import MlflowClient

from src.sms_fraud.config import get_config, project_path
from src.sms_fraud.logger import get_logger
from src.sms_fraud.promote import CHAMPION, alias_version
from src.sms_fraud.train import LABEL_MAP, evaluate, setup_mlflow

logger = get_logger("sms_fraud.challenge")


def score_challenge(model, df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Return (challenge_* metrics, table of misclassified rows)."""
    y_true = df["label"].map(LABEL_MAP)
    y_pred = model.predict(df["message"])
    # Prefix "challenge_", so these never mix with the en_/tr_ test metrics.
    metrics = {f"challenge_{k}": float(v) for k, v in evaluate(y_true, y_pred).items()}

    wrong = df[y_true.to_numpy() != y_pred].copy()
    # The spam probability helps to see if a mistake was close or far off.
    wrong["spam_prob"] = model.predict_proba(wrong["message"])[:, 1] if len(wrong) else []
    return metrics, wrong


def write_report(path, metrics: dict, wrong: pd.DataFrame, n_rows: int, model_note: str) -> None:
    """Markdown report: scores and the full list of mistakes."""
    lines = [
        "# Turkish Phishing Challenge Set",
        "",
        f"- Model: {model_note}",
        f"- Messages: {n_rows} (hand-written; evaluation only, never trained on)",
        "",
        "| Metric | Value |",
        "|---|---|",
        *[f"| {k} | {v:.4f} |" for k, v in metrics.items()],
        "",
        f"## Misclassified messages ({len(wrong)})",
        "",
        "- true **spam** -> predicted ham = phishing we MISSED",
        "- true **ham** -> predicted spam = normal SMS we would block",
        "",
        "| True label | Spam prob | Message |",
        "|---|---|---|",
        *[f"| {r.label} | {r.spam_prob:.3f} | {r.message} |" for r in wrong.itertuples()],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = get_config()
    ch_cfg = cfg["challenge"]
    df = pd.read_csv(project_path(ch_cfg["file"]), encoding="utf-8", keep_default_na=False)
    model = joblib.load(project_path(cfg["model"]["model_path"]))

    metrics, wrong = score_challenge(model, df)
    for name, value in metrics.items():
        logger.info("%s = %.4f", name, value)

    # Which registered version is the champion? Only for the report and tags.
    setup_mlflow(cfg["mlflow"])
    version = alias_version(MlflowClient(), cfg["mlflow"]["registered_model_name"], CHAMPION)
    model_note = f"champion (registered version {version}, models/model.joblib)"

    with mlflow.start_run(run_name="challenge_tr_phishing"):
        mlflow.set_tags({"step": "7a", "eval_only": "true", "champion_version": str(version)})
        mlflow.log_metrics(metrics)

    with open(project_path(ch_cfg["metrics_path"]), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    write_report(project_path(ch_cfg["report_path"]), metrics, wrong, len(df), model_note)
    logger.info("%d of %d messages misclassified", len(wrong), len(df))


if __name__ == "__main__":
    main()
