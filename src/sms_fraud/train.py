"""Train the ML model, compare it with the rule-based baseline, track with MLflow.

Run it with:
    python -m src.sms_fraud.train            (C from config.yaml)
    python -m src.sms_fraud.train --C 0.1    (try another C value)
"""

import argparse
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from src.sms_fraud.config import get_config, project_path
from src.sms_fraud.logger import get_logger
from src.sms_fraud.rule_based import is_fraud

logger = get_logger("sms_fraud.train")

# Models work with numbers, not words. spam is our "positive" class (1),
# because spam is the thing we want to find.
LABEL_MAP = {"ham": 0, "spam": 1}


def load_split(name: str) -> tuple[pd.Series, pd.Series]:
    """Load train.csv or test.csv. Return (messages, labels as 0/1)."""
    path = project_path(get_config()["data"]["processed_dir"]) / f"{name}.csv"
    # keep_default_na=False: by default pandas reads text like "NA" or "null"
    # as a missing value. An SMS can really say "NA", so we keep it as text.
    df = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
    return df["message"], df["label"].map(LABEL_MAP)


def build_pipeline(max_features: int, ngram_range: tuple[int, int], C: float) -> Pipeline:
    """Create the model: text -> TF-IDF numbers -> Logistic Regression."""
    # PATTERN: Pipeline — the text cleaning step (TF-IDF) and the model are joined into one object, so training and prediction always use the same steps in the same order.
    return Pipeline(
        [
            # Step 1: turn each message into numbers. TF-IDF gives a high
            # score to words that are common in THIS message but rare overall.
            ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
            # Step 2: a simple, fast classifier that works well on text.
            # max_iter=1000 gives it enough steps to finish learning.
            ("clf", LogisticRegression(C=C, max_iter=1000)),
        ]
    )


def evaluate(y_true, y_pred) -> dict:
    """Return accuracy, precision, recall and f1 (spam = positive class)."""
    # zero_division=0: if the model predicts no spam at all, precision would be
    # "0 divided by 0". We return 0 instead of a warning.
    return {
        # Share of ALL messages labeled correctly.
        "accuracy": accuracy_score(y_true, y_pred),
        # Of the messages we CALLED spam, how many really were spam?
        "precision": precision_score(y_true, y_pred, zero_division=0),
        # Of the REAL spam messages, how many did we find?
        "recall": recall_score(y_true, y_pred, zero_division=0),
        # One number that balances precision and recall.
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_baseline(messages: pd.Series, y_true: pd.Series) -> dict:
    """Score the rule-based checker from Step 1 on the same messages."""
    # is_fraud returns True/False; int() turns it into 1/0 like our labels.
    y_pred = [int(is_fraud(m)) for m in messages]
    return evaluate(y_true, y_pred)


def setup_mlflow(mlflow_cfg: dict) -> None:
    """Tell MLflow to save everything in the local ./mlruns folder."""
    tracking_dir = project_path(mlflow_cfg["tracking_dir"])
    tracking_dir.mkdir(parents=True, exist_ok=True)

    # Runs, metrics and the Model Registry go into a SQLite database file.
    # SQLite is a full database inside ONE file - no server needed.
    db_path = (tracking_dir / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    # Create the experiment the first time. We choose where model files go
    # (mlruns/artifacts), so nothing is written outside the mlruns folder.
    name = mlflow_cfg["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(tracking_dir / "artifacts").as_uri())
    mlflow.set_experiment(name)


def parse_args() -> argparse.Namespace:
    """Read options from the command line, e.g. --C 0.1."""
    parser = argparse.ArgumentParser(description="Train the SMS fraud model.")
    # default=None means: "if not given, use the value from config.yaml".
    parser.add_argument("--C", type=float, default=None, help="Regularization strength.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    model_cfg = cfg["model"]
    mlflow_cfg = cfg["mlflow"]

    # Collect the hyperparameters. The CLI value wins over the config value.
    params = {
        "max_features": model_cfg["max_features"],
        # YAML gives a list [1, 2]; scikit-learn needs a tuple (1, 2).
        "ngram_range": tuple(model_cfg["ngram_range"]),
        "C": args.C if args.C is not None else model_cfg["C"],
    }

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    logger.info("Train rows: %d | Test rows: %d", len(X_train), len(X_test))

    setup_mlflow(mlflow_cfg)

    # Everything inside this "with" block belongs to ONE MLflow run.
    with mlflow.start_run(run_name=f"logreg_C={params['C']}"):
        model = build_pipeline(**params)
        model.fit(X_train, y_train)

        ml_metrics = evaluate(y_test, model.predict(X_test))
        baseline_metrics = evaluate_baseline(X_test, y_test)

        # Show both results side by side, so we see if ML beats the rules.
        logger.info("%-10s %10s %10s", "metric", "baseline", "ml_model")
        for name in ml_metrics:
            logger.info("%-10s %10.4f %10.4f", name, baseline_metrics[name], ml_metrics[name])

        # Log to MLflow. We add a prefix so the two models do not mix up.
        mlflow.log_params(params)
        mlflow.log_metrics({f"ml_{k}": v for k, v in ml_metrics.items()})
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline_metrics.items()})

        # Save the model in MLflow AND register it in the Model Registry.
        # Each run creates a new "version" of "sms-fraud-classifier".
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=mlflow_cfg["registered_model_name"],
        )

    # Also save files on disk, so DVC can track them.
    model_path = project_path(model_cfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Saved model to %s", model_path)

    metrics_path = project_path(model_cfg["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"C": params["C"], "ml_model": ml_metrics, "baseline": baseline_metrics}, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
