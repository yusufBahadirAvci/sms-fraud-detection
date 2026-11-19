"""Create a TINY fake model at models/model.joblib. FOR CI ONLY.

Why? The real model lives only in the local DVC cache (no DVC remote yet),
so GitHub Actions cannot download it. But Dockerfile.api copies
models/model.joblib into the image. With this dummy model, CI can still
check that the API image builds, starts, and answers /health and /predict.

The dummy model is NOT useful for real predictions (10 training messages).

Run it with:
    python scripts/make_dummy_model.py           (stops if the file exists)
    python scripts/make_dummy_model.py --force   (overwrites it)

It only needs scikit-learn and joblib (requirements-api.txt), so CI does
not have to install the big training packages (MLflow, DVC, ...).
"""

import argparse
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# <project root>/scripts/make_dummy_model.py -> go up 1 folder.
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"

# 5 spam (1) and 5 normal (0) messages.
MESSAGES = [
    "WINNER! Click here to claim your prize",
    "You won a free gift, call now",
    "Urgent: verify your bank account today",
    "%50 indirim fırsatı, hemen tıklayın",
    "Free cash prize, reply YES now",
    "See you at dinner tonight",
    "The meeting starts at 10",
    "Akşam yemeğe geliyor musun?",
    "Can you call me later?",
    "Thanks for the book, see you soon",
]
LABELS = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]


# PATTERN: Test Double (Dummy) — a tiny stand-in with the same shape as the real model, so CI can
#   build and start the API image without the real model file.
def build_dummy_model() -> Pipeline:
    """Train the same SHAPE of model as the real one: a Pipeline whose text
    step is called "tfidf" (the API looks for this name), then a classifier."""
    model = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    return model.fit(MESSAGES, LABELS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny dummy model (CI only).")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing model file.")
    args = parser.parse_args()

    # Safety: on a developer's computer this file is the REAL model.
    # Never replace it by accident.
    if MODEL_PATH.exists() and not args.force:
        raise SystemExit(f"{MODEL_PATH} already exists. Use --force to overwrite it.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(build_dummy_model(), MODEL_PATH)
    print(f"Saved dummy model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
