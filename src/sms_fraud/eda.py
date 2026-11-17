"""EDA (Exploratory Data Analysis): look at the data before trusting a model.

Run it with:
    python -m src.sms_fraud.eda

It creates:
    reports/figures/class_balance.png
    reports/figures/message_length.png
    reports/figures/top_words.png
    reports/eda_summary.md
"""

import matplotlib

# "Agg" draws charts into files only, without opening a window.
# This line must come BEFORE "import matplotlib.pyplot".
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402

from src.sms_fraud.config import get_config, project_path  # noqa: E402
from src.sms_fraud.logger import get_logger  # noqa: E402
from src.sms_fraud.rule_based import SUSPICIOUS_KEYWORDS  # noqa: E402

logger = get_logger("sms_fraud.eda")

FIGURES_DIR = project_path("reports/figures")
SUMMARY_PATH = project_path("reports/eda_summary.md")

# One color per class, used in every chart, so the charts are easy to compare.
COLORS = {"ham": "tab:blue", "spam": "tab:red"}


def load_clean_data() -> pd.DataFrame:
    """Load train.csv + test.csv together = all clean data."""
    processed_dir = project_path(get_config()["data"]["processed_dir"])
    # keep_default_na=False: keep text like "NA" as text, not as missing.
    parts = [
        pd.read_csv(processed_dir / f"{name}.csv", encoding="utf-8", keep_default_na=False)
        for name in ("train", "test")
    ]
    df = pd.concat(parts, ignore_index=True)
    # Message length in characters. We add it as a new column.
    df["length"] = df["message"].str.len()
    return df


def top_words(messages: pd.Series, n: int) -> pd.Series:
    """Return the n most common words, without English stopwords.

    Stopwords are very common words like "the", "and", "to". They appear in
    every kind of message, so they tell us nothing about spam.
    """
    # CountVectorizer splits text into words and counts them.
    # stop_words="english" removes a built-in list of English stopwords.
    vectorizer = CountVectorizer(stop_words="english")
    counts = vectorizer.fit_transform(messages)
    # Sum each word's count over all messages.
    totals = pd.Series(counts.sum(axis=0).A1, index=vectorizer.get_feature_names_out())
    return totals.sort_values(ascending=False).head(n)


def plot_class_balance(df: pd.DataFrame) -> None:
    """Bar chart: how many ham and how many spam messages."""
    counts = df["label"].value_counts().reindex(["ham", "spam"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values, color=[COLORS[c] for c in counts.index])
    # Write the number on top of each bar, so we do not have to guess.
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value}\n({value / len(df):.1%})", ha="center", va="bottom")
    ax.set_title("Class balance")
    ax.set_ylabel("Number of messages")
    ax.set_ylim(0, counts.max() * 1.2)
    save_figure(fig, "class_balance.png")


def plot_message_length(df: pd.DataFrame) -> None:
    """Two overlaid histograms of message length: ham vs spam."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for label in ("ham", "spam"):
        lengths = df.loc[df["label"] == label, "length"]
        # density=True: show SHARE of messages, not raw counts. Why? There
        # are ~7x more ham messages; with raw counts spam would look tiny.
        # alpha=0.5 makes bars half-transparent so both colors stay visible.
        ax.hist(lengths, bins=60, range=(0, 400), density=True, alpha=0.5, label=label, color=COLORS[label])
    ax.set_title("Message length (characters)")
    ax.set_xlabel("Length (messages longer than 400 are not shown)")
    ax.set_ylabel("Share of messages")
    ax.legend()
    save_figure(fig, "message_length.png")


def plot_top_words(spam_words: pd.Series, ham_words: pd.Series) -> None:
    """Two horizontal bar charts side by side: top words in spam and in ham."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, words, label in ((axes[0], spam_words, "spam"), (axes[1], ham_words, "ham")):
        # [::-1] reverses the order, so the most common word is at the TOP.
        ax.barh(words.index[::-1], words.values[::-1], color=COLORS[label])
        ax.set_title(f"Top {len(words)} words in {label}")
        ax.set_xlabel("Count")
    save_figure(fig, "top_words.png")


def save_figure(fig, filename: str) -> None:
    """Save a chart as PNG and close it (closing frees memory)."""
    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved chart %s", path)


def write_summary(df: pd.DataFrame, spam_words: pd.Series) -> None:
    """Write a short Markdown report with numbers and observations."""
    counts = df["label"].value_counts()
    avg_len = df.groupby("label")["length"].mean()
    spam_share = counts["spam"] / len(df)

    # Which top spam words are also keywords in our Step 1 rules?
    # We only compare single-word keywords; phrases like "click here" are
    # not single words, so they cannot match one word from the chart.
    in_rules = [w for w in spam_words.index if w in SUSPICIOUS_KEYWORDS]

    # The observations use the REAL numbers from the data, so the text
    # stays correct if the data changes.
    observations = [
        f"The data is imbalanced: only {spam_share:.1%} of messages are spam. "
        f"A model that always says \"ham\" would already get {1 - spam_share:.1%} accuracy, "
        f"so we must look at precision, recall and F1, not only accuracy.",
        f"Spam messages are much longer: {avg_len['spam']:.0f} characters on average "
        f"vs {avg_len['ham']:.0f} for ham. Length alone is already a useful signal.",
        f"The most common spam words are {', '.join(repr(w) for w in spam_words.index[:5])}. "
        f"Of the top {len(spam_words)} spam words, the keyword list in rule_based.py contains only "
        f"{len(in_rules)}: {', '.join(in_rules) if in_rules else 'none'}. "
        f"This explains why the rule-based baseline finds so little spam (low recall).",
    ]

    lines = [
        "# EDA Summary",
        "",
        "## Row counts",
        "",
        "| Class | Rows | Share |",
        "|---|---|---|",
        *[f"| {c} | {counts[c]} | {counts[c] / len(df):.1%} |" for c in ("ham", "spam")],
        f"| **total** | **{len(df)}** | 100% |",
        "",
        "## Average message length (characters)",
        "",
        "| Class | Average length |",
        "|---|---|",
        *[f"| {c} | {avg_len[c]:.1f} |" for c in ("ham", "spam")],
        "",
        "## Top 10 spam words (stopwords removed)",
        "",
        "| # | Word | Count |",
        "|---|---|---|",
        *[f"| {i} | {word} | {count} |" for i, (word, count) in enumerate(spam_words.head(10).items(), start=1)],
        "",
        "## Observations",
        "",
        *[f"{i}. {text}" for i, text in enumerate(observations, start=1)],
        "",
        "## Charts",
        "",
        "- `figures/class_balance.png`",
        "- `figures/message_length.png`",
        "- `figures/top_words.png`",
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved summary %s", SUMMARY_PATH)


def main() -> None:
    # DVC deletes old outputs before it runs a stage, so we create the folder.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_clean_data()
    logger.info("Loaded %d clean rows", len(df))

    spam_words = top_words(df.loc[df["label"] == "spam", "message"], n=20)
    ham_words = top_words(df.loc[df["label"] == "ham", "message"], n=20)

    plot_class_balance(df)
    plot_message_length(df)
    plot_top_words(spam_words, ham_words)
    write_summary(df, spam_words)


if __name__ == "__main__":
    main()
