"""Prepare the Turkish SMS dataset (for monitoring, NOT for training yet).

Run it with:
    python -m src.sms_fraud.turkish_data

Input:  data/external/TurkishSMSCollection.csv  (tracked with DVC)
Output: data/processed/turkish.csv  (columns: label, message)

Raw file format: separator ";", columns Message;Group;GroupText.
Group 1 = Spam, Group 2 = Normal.
"""

import pandas as pd

from src.sms_fraud.config import get_config, project_path
from src.sms_fraud.data import spam_percent, validate_data
from src.sms_fraud.logger import get_logger

logger = get_logger("sms_fraud.turkish_data")

# The raw file uses numbers for the class. We turn them into the same
# labels as the English data, so validate_data() and the model can use them.
GROUP_TO_LABEL = {"1": "spam", "2": "ham"}

# Encodings we try, in this order.
# - utf-8: the modern standard.
# - cp1254: the old Windows encoding for Turkish. Many older Turkish files use it.
ENCODINGS = ["utf-8", "cp1254"]


def read_with_fallback(path) -> tuple[pd.DataFrame, str]:
    """Read the CSV with the first encoding that works. Return (df, encoding).

    Why? With the wrong encoding, Turkish letters like "ı" or "ş" become
    broken characters, and the model would see different "words".
    """
    for encoding in ENCODINGS:
        try:
            df = pd.read_csv(
                path,
                sep=";",
                encoding=encoding,
                # Read everything as text: "1" stays the text "1", not a number.
                dtype=str,
                # Keep text like "NA" as text, not as a missing value.
                keep_default_na=False,
                # Messages that contain ";" or line breaks are wrapped in
                # quotes ("..."). pandas handles these quotes by default.
            )
            return df, encoding
        except UnicodeDecodeError:
            logger.warning("Could not read %s as %s, trying the next encoding", path, encoding)
    raise ValueError(f"Could not read {path} with any of {ENCODINGS}")


def load_turkish_data() -> pd.DataFrame:
    """Load the raw Turkish file into columns ["label", "message"]."""
    path = project_path(get_config()["data"]["turkish_raw_file"])
    raw, encoding = read_with_fallback(path)
    logger.info("Loaded %d rows from %s (encoding: %s)", len(raw), path, encoding)

    # .map() gives an empty value for unknown groups. validate_data() then
    # removes those rows as "invalid labels", and logs how many.
    labels = raw["Group"].str.strip().map(GROUP_TO_LABEL).fillna("")
    return pd.DataFrame({"label": labels, "message": raw["Message"]})


def main() -> None:
    df = load_turkish_data()
    # Reuse the SAME cleaning rules as for the English data (empty messages,
    # labels, HTML codes, duplicates). Same rules = fair comparison.
    clean = validate_data(df)

    out_path = project_path(get_config()["data"]["turkish_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_path, index=False, encoding="utf-8")

    logger.info("----- Summary -----")
    logger.info("Raw rows:   %d", len(df))
    logger.info("Clean rows: %d (removed %d)", len(clean), len(df) - len(clean))
    logger.info("Spam:       %.2f%%", spam_percent(clean))
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
