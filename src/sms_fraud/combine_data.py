"""Build the combined train set: English train + Turkish train.

Run it with:
    python -m src.sms_fraud.combine_data

Input:  data/processed/train.csv, data/processed/turkish_train.csv
Output: data/processed/combined_train.csv  (columns: label, message, lang)

We combine ONLY the train sets. The test sets stay separate (test.csv and
turkish_test.csv), so we can score the model on each language alone.
Why? One mixed score would hide a bad result in one language.
"""

import pandas as pd

from src.sms_fraud.config import get_config, project_path
from src.sms_fraud.data import spam_percent
from src.sms_fraud.logger import get_logger

logger = get_logger("sms_fraud.combine_data")


def read_csv(path) -> pd.DataFrame:
    # keep_default_na=False: keep text like "NA" as text (see train.py).
    return pd.read_csv(path, encoding="utf-8", keep_default_na=False)


def combine(english: pd.DataFrame, turkish: pd.DataFrame) -> pd.DataFrame:
    """Stack the two tables. Every row keeps its "lang" value (en / tr)."""
    return pd.concat([english, turkish], ignore_index=True)


def main() -> None:
    cfg = get_config()["data"]
    english = read_csv(project_path(cfg["processed_dir"]) / "train.csv")
    turkish = read_csv(project_path(cfg["turkish_train_file"]))
    combined = combine(english, turkish)

    out_path = project_path(cfg["combined_train_file"])
    combined.to_csv(out_path, index=False, encoding="utf-8")

    # Log rows and spam share per language, so we see the mix we train on.
    for lang, part in combined.groupby("lang"):
        logger.info("%s: %d rows | spam %.2f%%", lang, len(part), spam_percent(part))
    logger.info("Saved %s (%d rows)", out_path, len(combined))


if __name__ == "__main__":
    main()
