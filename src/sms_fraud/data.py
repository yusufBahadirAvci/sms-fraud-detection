"""Data pipeline: download -> load -> validate -> split -> save.

Run it with:
    python -m src.sms_fraud.data

Each step is a small, separate function. Why?
- Small functions are easy to read and easy to test one by one.
- Later steps (training, DVC) can reuse single functions.
"""

import csv
import html
import urllib.request
import zipfile

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.sms_fraud.config import get_config, project_path
from src.sms_fraud.leakage import cluster_ids
from src.sms_fraud.logger import get_logger

# We write the name by hand instead of using __name__.
# Why? When we run "python -m src.sms_fraud.data", Python sets __name__
# to "__main__", and the logs would show "__main__" instead of the module name.
logger = get_logger("sms_fraud.data")

# The only labels we accept. "ham" means a normal message.
VALID_LABELS = {"ham", "spam"}


def download_data() -> None:
    """Download the zip file and unzip it into data/raw/.

    If the data file is already there, we skip the download.
    Why? Downloading again every run is slow and not needed.
    """
    cfg = get_config()["data"]
    raw_dir = project_path(cfg["raw_dir"])
    raw_file = raw_dir / cfg["raw_file"]

    if raw_file.exists():
        logger.info("Raw data already exists, skipping download: %s", raw_file)
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "sms_spam_collection.zip"

    logger.info("Downloading data from %s", cfg["url"])
    urllib.request.urlretrieve(cfg["url"], zip_path)

    # Unzip all files into data/raw/.
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    logger.info("Unzipped data into %s", raw_dir)


def load_data() -> pd.DataFrame:
    """Read the raw file into a DataFrame with columns ["label", "message"]."""
    cfg = get_config()["data"]
    raw_file = project_path(cfg["raw_dir"]) / cfg["raw_file"]

    # The file has no header row. Each line is: label <TAB> message.
    # quoting=QUOTE_NONE: messages can contain " characters. We tell pandas
    # to treat them as normal text, not as the start of a quoted field.
    df = pd.read_csv(
        raw_file,
        sep="\t",
        header=None,
        names=["label", "message"],
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )
    logger.info("Loaded %d rows from %s", len(df), raw_file)
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data and return a new DataFrame.

    Rules:
    1. No empty messages.
    2. Labels must be "ham" or "spam".
    3. No duplicate rows.
    We log how many rows each rule removed, so we know what happened.
    """
    # Work on a copy, so we never change the caller's DataFrame by accident.
    df = df.copy()

    # Clean extra spaces and make labels lowercase ("Spam " -> "spam").
    # fillna("") turns missing values into empty text, so rule 1 catches them.
    df["message"] = df["message"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].fillna("").astype(str).str.strip().str.lower()

    # Decode HTML codes: "&lt;" -> "<", "&gt;" -> ">", "&amp;" -> "&".
    # Why? The raw file has text like "&lt;#&gt;". Without this step the model
    # learns fake "words" like "lt" and "gt" (our EDA showed them as the top
    # ham words). Real users never type "&lt;", so this would not help.
    decoded = df["message"].map(html.unescape)
    changed = int((decoded != df["message"]).sum())
    df["message"] = decoded
    logger.info("Decoded HTML codes in %d messages", changed)

    # Rule 1: remove empty messages.
    before = len(df)
    df = df[df["message"] != ""]
    logger.info("Removed %d rows with empty messages", before - len(df))

    # Rule 2: remove rows with unknown labels.
    before = len(df)
    df = df[df["label"].isin(VALID_LABELS)]
    logger.info("Removed %d rows with invalid labels", before - len(df))

    # Rule 3: remove duplicates. Same label + same message = duplicate.
    # Why? Duplicates can end up in BOTH train and test. Then the test
    # score looks better than it really is (this is called "data leakage").
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Removed %d duplicate rows", before - len(df))

    # Give the rows new numbers 0, 1, 2, ... after removing some rows.
    return df.reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    test_size: float | None = None,
    random_state: int | None = None,
    groups=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into train and test sets.

    stratify=label keeps the spam ratio the same in train and test.
    Why? Spam is a small part of the data (~13%). A pure random split
    could put too few spam messages into the test set.

    groups (Step 7a): one cluster number per row (from leakage.cluster_ids).
    - None:  plain random split (used up to Step 6; leakage.py still uses
             it to rebuild the OLD split for comparison).
    - given: GROUP split. All rows of one cluster go to the same side, so a
             near-copy of a test message is never in train (no leakage).

    If test_size or random_state is not given, we use the config values.
    Tests can pass their own values.
    """
    cfg = get_config()["split"]
    if test_size is None:
        test_size = cfg["test_size"]
    if random_state is None:
        random_state = cfg["random_state"]

    if groups is None:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["label"],
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    # StratifiedGroupKFold cuts the data into n_splits parts ("folds"):
    # - a group never goes to two folds (no leakage), and
    # - each fold gets about the same spam share (stratified).
    # We use the FIRST fold as the test set. test_size 0.2 -> 5 folds, so the
    # test set is about 1/5 = 20% (not exactly: groups have different sizes).
    n_splits = round(1 / test_size)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, df["label"], groups=groups))
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


# PATTERN: Repeatable Splitting — the split is decided by cluster (not by single row) with a fixed
#   seed, so near-copies always land on the same side and every run gives the same split.
def leakage_safe_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster near-duplicates, then split by cluster. Used for both languages."""
    threshold = get_config()["leakage"]["similarity_threshold"]
    groups = cluster_ids(df["message"], threshold)
    logger.info(
        "Found %d clusters in %d rows (biggest cluster: %d rows)",
        len(set(groups)),
        len(df),
        pd.Series(groups).value_counts().max(),
    )
    return split_data(df, groups=groups)


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Write train.csv and test.csv into data/processed/."""
    out_dir = project_path(get_config()["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # index=False: do not write the row numbers as an extra column.
    train_df.to_csv(out_dir / "train.csv", index=False, encoding="utf-8")
    test_df.to_csv(out_dir / "test.csv", index=False, encoding="utf-8")
    logger.info("Saved train.csv and test.csv into %s", out_dir)


def spam_percent(df: pd.DataFrame) -> float:
    """Return the share of spam rows, as a percent (0-100)."""
    return (df["label"] == "spam").mean() * 100


# PATTERN: Pipeline — the data goes through fixed steps in order (download, load, validate, split,
#   save); each step's output is the next step's input.
def main() -> None:
    """Run all steps in order and log a short summary."""
    download_data()
    raw_df = load_data()
    clean_df = validate_data(raw_df)
    # Mark the language of every row. Since Step 5b we mix English and
    # Turkish data, and need to know where each row came from.
    clean_df["lang"] = "en"
    # Save ALL clean rows too (like turkish.csv). leakage.py uses them to
    # rebuild the old random split and compare it with the new one.
    english_file = project_path(get_config()["data"]["english_file"])
    english_file.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(english_file, index=False, encoding="utf-8")
    # Step 7a: split by near-duplicate cluster (no leakage), not row by row.
    train_df, test_df = leakage_safe_split(clean_df)
    save_data(train_df, test_df)

    logger.info("----- Summary -----")
    logger.info("Raw rows:     %d", len(raw_df))
    logger.info("Clean rows:   %d (removed %d)", len(clean_df), len(raw_df) - len(clean_df))
    logger.info("Train rows:   %d | spam %.2f%%", len(train_df), spam_percent(train_df))
    logger.info("Test rows:    %d | spam %.2f%%", len(test_df), spam_percent(test_df))


if __name__ == "__main__":
    main()
