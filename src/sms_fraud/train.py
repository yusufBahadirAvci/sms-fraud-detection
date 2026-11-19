"""Train candidate models on English + Turkish data and track them with MLflow.

Run it with:
    python -m src.sms_fraud.train               (all runs from config.yaml: A, B, C)
    python -m src.sms_fraud.train --only B_char (one run only)

Each run:
1. trains on the COMBINED train set (English + Turkish),
2. is scored SEPARATELY on the English test set and the Turkish test set
   (metrics with prefix en_ and tr_),
3. is logged to MLflow and registered as a new model version,
4. is saved as models/candidates/<run name>.joblib.

Training does NOT change models/model.joblib (the model the API serves).
That is the job of promote.py, which compares the best new run with the
current model and replaces it only if the new one is really better.
"""

import argparse
import json

import joblib
import matplotlib

# "Agg" draws charts into files only, without opening a window.
# Why? Training often runs where there is no screen (DVC, servers, tests).
# This line must come BEFORE "import matplotlib.pyplot".
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.pipeline import FeatureUnion, Pipeline  # noqa: E402

from src.sms_fraud.config import get_config, project_path  # noqa: E402
from src.sms_fraud.logger import get_logger  # noqa: E402
from src.sms_fraud.rule_based import is_fraud  # noqa: E402

logger = get_logger("sms_fraud.train")

# Models work with numbers, not words. spam is our "positive" class (1),
# because spam is the thing we want to find.
LABEL_MAP = {"ham": 0, "spam": 1}

# The model names and feature kinds we accept.
MODEL_NAMES = ["logreg", "nb"]
FEATURE_KINDS = ["word", "char", "word+char"]


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------


def load_labelled(path) -> tuple[pd.Series, pd.Series]:
    """Load a CSV with "label" and "message". Return (messages, labels as 0/1)."""
    # keep_default_na=False: by default pandas reads text like "NA" or "null"
    # as a missing value. An SMS can really say "NA", so we keep it as text.
    df = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
    return df["message"], df["label"].map(LABEL_MAP)


def load_split(name: str) -> tuple[pd.Series, pd.Series]:
    """Load the English train.csv or test.csv."""
    return load_labelled(project_path(get_config()["data"]["processed_dir"]) / f"{name}.csv")


def load_test_sets() -> dict[str, tuple[pd.Series, pd.Series]]:
    """The two test sets, kept SEPARATE: {"en": (X, y), "tr": (X, y)}."""
    return {
        "en": load_split("test"),
        "tr": load_labelled(project_path(get_config()["data"]["turkish_test_file"])),
    }


# ---------------------------------------------------------------------------
# 2. Build the model
# ---------------------------------------------------------------------------


# PATTERN: Factory — one function decides WHICH classifier to create from a name ("logreg" or "nb"),
#   so main() and the tests never need to know the details of each model.
def build_classifier(model_name: str, C: float = 1.0, alpha: float = 1.0):
    """Create a new, untrained classifier from its name."""
    if model_name == "logreg":
        # A simple, fast linear model that works well on text.
        # max_iter=1000 gives it enough steps to finish learning.
        return LogisticRegression(C=C, max_iter=1000)
    if model_name == "nb":
        # Multinomial Naive Bayes: a classic, very fast model for word counts.
        # It learns how often each word appears in spam and in ham.
        return MultinomialNB(alpha=alpha)
    # Fail early and clearly if someone asks for a model we do not have.
    raise ValueError(f"Unknown model: {model_name!r}. Use one of {MODEL_NAMES}.")


# PATTERN: Factory — one function creates the text-to-numbers step from a name ("word", "char",
#   "word+char"), so a new experiment is one line in config.yaml, not new code.
def build_features(
    features: str,
    max_features: int,
    ngram_range: tuple[int, int],
    char_max_features: int = 20000,
    char_ngram_range: tuple[int, int] = (3, 5),
):
    """Create the TF-IDF step: words, letter pieces, or both."""

    def word_tfidf():
        # Words and word pairs, as in the Step 4 model.
        return TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    def char_tfidf():
        # analyzer="char_wb": letter pieces taken only INSIDE words
        # (wb = word boundary). "tıklayın" -> " tı", "tık", "ıkl", ...
        # So "tıkla", "tıklayın" and "tıklayınız" share most of their pieces,
        # and a new suffix does not make the whole word unknown.
        # sublinear_tf: use 1 + log(count) instead of the raw count, so one
        # piece repeated many times in a long message does not dominate.
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ngram_range,
            max_features=char_max_features,
            sublinear_tf=True,
        )

    if features == "word":
        return word_tfidf()
    if features == "char":
        return char_tfidf()
    if features == "word+char":
        # FeatureUnion runs both vectorizers on the same text and puts their
        # numbers side by side in one long row: [word numbers | char numbers].
        return FeatureUnion([("word", word_tfidf()), ("char", char_tfidf())])
    raise ValueError(f"Unknown features: {features!r}. Use one of {FEATURE_KINDS}.")


def build_pipeline(
    model_name: str,
    max_features: int,
    ngram_range: tuple[int, int],
    C: float = 1.0,
    alpha: float = 1.0,
    features: str = "word",
    char_max_features: int = 20000,
    char_ngram_range: tuple[int, int] = (3, 5),
) -> Pipeline:
    """Create the full model: text -> TF-IDF numbers -> classifier."""
    # PATTERN: Pipeline — the text step (TF-IDF) and the classifier are joined into one object, so
    #   training and prediction always use the same steps in the same order.
    return Pipeline(
        [
            # Step 1: turn each message into numbers. The step keeps the name
            # "tfidf" for every feature kind, so the API and monitoring code
            # always find it in the same place.
            (
                "tfidf",
                build_features(
                    features, max_features, ngram_range, char_max_features, char_ngram_range
                ),
            ),
            # Step 2: the classifier, chosen by the factory above.
            ("clf", build_classifier(model_name, C=C, alpha=alpha)),
        ]
    )


# ---------------------------------------------------------------------------
# 3. Score the model
# ---------------------------------------------------------------------------


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


def evaluate_by_language(model, test_sets: dict) -> dict:
    """Score a model on each test set. Keys get the language as prefix:
    {"en_accuracy": ..., "en_f1": ..., "tr_accuracy": ..., "tr_f1": ...}.
    """
    scores = {}
    for lang, (X, y) in test_sets.items():
        for name, value in evaluate(y, model.predict(X)).items():
            scores[f"{lang}_{name}"] = float(value)
    return scores


def evaluate_baseline(messages: pd.Series, y_true: pd.Series) -> dict:
    """Score the rule-based checker from Step 1 on the same messages."""
    # is_fraud returns True/False; int() turns it into 1/0 like our labels.
    y_pred = [int(is_fraud(m)) for m in messages]
    return evaluate(y_true, y_pred)


def save_confusion_matrix(y_true, y_pred, title: str, path) -> None:
    """Draw a confusion matrix and save it as a PNG file.

    A confusion matrix is a 2x2 table: real class (rows) vs predicted class
    (columns). It shows WHERE the model makes mistakes:
    - spam predicted as ham  = spam we missed (bad for the user)
    - ham predicted as spam  = normal SMS we blocked by mistake
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["ham", "spam"], cmap="Blues", colorbar=False
    )
    display.ax_.set_title(title)
    display.figure_.tight_layout()
    display.figure_.savefig(path, dpi=120)
    # Close the figure to free memory. Without this, many runs would keep
    # many open figures in memory.
    plt.close(display.figure_)


# ---------------------------------------------------------------------------
# 4. MLflow and the main program
# ---------------------------------------------------------------------------


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


def run_experiment(experiment: dict, X_train, y_train, test_sets: dict, cfg: dict) -> dict:
    """Train, score, log and save ONE run. Return its results for metrics.json."""
    model_cfg = cfg["model"]
    run_name = experiment["name"]
    params = {
        "model": experiment.get("model", model_cfg["name"]),
        "features": experiment["features"],
        "C": model_cfg["C"],
        "alpha": model_cfg["alpha"],
        "max_features": model_cfg["max_features"],
        # YAML gives a list [1, 2]; scikit-learn needs a tuple (1, 2).
        "ngram_range": tuple(model_cfg["ngram_range"]),
        "char_max_features": model_cfg["char_max_features"],
        "char_ngram_range": tuple(model_cfg["char_ngram_range"]),
    }

    # Everything inside this "with" block belongs to ONE MLflow run.
    with mlflow.start_run(run_name=run_name) as run:
        model = build_pipeline(
            model_name=params["model"],
            max_features=params["max_features"],
            ngram_range=params["ngram_range"],
            C=params["C"],
            alpha=params["alpha"],
            features=params["features"],
            char_max_features=params["char_max_features"],
            char_ngram_range=params["char_ngram_range"],
        )
        model.fit(X_train, y_train)
        scores = evaluate_by_language(model, test_sets)

        mlflow.log_params(params)
        # Tags are free text labels. They help to find these runs later.
        # "clean_split": since Step 7a, near-duplicates never cross train/test.
        mlflow.set_tags({"step": "7a", "train_data": "combined_en_tr", "split": "clean_split"})
        mlflow.log_metrics(scores)

        # One confusion matrix per language, saved on disk AND in MLflow.
        for lang, (X, y) in test_sets.items():
            cm_path = project_path(model_cfg["confusion_dir"]) / f"{run_name}_{lang}.png"
            save_confusion_matrix(y, model.predict(X), f"{run_name} on {lang} test", cm_path)
            mlflow.log_artifact(str(cm_path))

        # Save the model in MLflow AND register it in the Model Registry.
        # Each run creates a new "version" of "sms-fraud-classifier".
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=cfg["mlflow"]["registered_model_name"],
        )

    # Save the candidate on disk too. promote.py copies the winner from here.
    candidate_path = project_path(model_cfg["candidates_dir"]) / f"{run_name}.joblib"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, candidate_path)

    logger.info(
        "%-12s en_f1=%.4f tr_f1=%.4f (registered version %s)",
        run_name,
        scores["en_f1"],
        scores["tr_f1"],
        info.registered_model_version,
    )
    return {
        "run_id": run.info.run_id,
        "model_version": str(info.registered_model_version),
        # Relative path: the file stays valid on another computer.
        "candidate_path": f"{model_cfg['candidates_dir']}/{run_name}.joblib",
        "params": {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()},
        "metrics": scores,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SMS fraud candidate models.")
    parser.add_argument(
        "--only", default=None, help="Run only the experiment with this name (e.g. B_char)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    experiments = cfg["model"]["experiments"]
    if args.only:
        experiments = [e for e in experiments if e["name"] == args.only]
        if not experiments:
            raise ValueError(f"No experiment named {args.only!r} in config.yaml")

    X_train, y_train = load_labelled(project_path(cfg["data"]["combined_train_file"]))
    test_sets = load_test_sets()
    logger.info(
        "Train rows: %d (combined) | en test: %d | tr test: %d",
        len(X_train),
        len(test_sets["en"][0]),
        len(test_sets["tr"][0]),
    )

    setup_mlflow(cfg["mlflow"])
    runs = {e["name"]: run_experiment(e, X_train, y_train, test_sets, cfg) for e in experiments}

    # The rule-based checker from Step 1, for comparison (it has no training).
    baseline = {}
    for lang, (X, y) in test_sets.items():
        baseline.update({f"{lang}_{k}": float(v) for k, v in evaluate_baseline(X, y).items()})

    # metrics.json is kept in Git (DVC "metrics"), and promote.py reads it.
    metrics_path = project_path(cfg["model"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"runs": runs, "baseline_rules": baseline}, f, indent=2)
    logger.info("Saved metrics of %d runs to %s", len(runs), metrics_path)


if __name__ == "__main__":
    main()
