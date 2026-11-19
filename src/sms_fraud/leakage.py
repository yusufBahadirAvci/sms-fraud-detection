"""Near-duplicate leakage: are (almost) the same messages in train AND test?

Run it with:
    python -m src.sms_fraud.leakage

Why does this matter? Spam is often sent as a TEMPLATE: the same text with
a different number, name or link. If one copy is in the train set and
another copy is in the test set, the model has "seen the answer" and the
test score looks better than it really is. This is called data leakage.
validate_data() only removes EXACT duplicates, so these near-copies stay.

What this file does:
1. normalize_text(): make small differences disappear (case, links,
   digits, punctuation, spaces).
2. check_leakage(): for every test message, find the most similar train
   message (char TF-IDF + cosine similarity). Count exact and near matches.
3. cluster_ids(): put near-duplicates into groups ("clusters"). data.py
   uses them for a leakage-safe split: a whole cluster goes to train OR test.
4. main(): write reports/leakage.md for the OLD random split (before Step
   7a) and for the NEW clean split, per language.
"""

import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

from src.sms_fraud.config import get_config, project_path
from src.sms_fraud.logger import get_logger

logger = get_logger("sms_fraud.leakage")

# A link: "http://...", "www...." or a bare domain like "n11.com/iphone".
URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|\b[\w.-]+\.(com|net|org|tr|uk|ly|info|biz|me)\b\S*)",
    re.IGNORECASE,
)
# A placeholder word for links. Punctuation is removed AFTER we insert it,
# so it must be plain letters. At the very end it becomes "<url>".
URL_TOKEN = "urltoken"

# How many test rows we compare at once. Comparing all rows in one step
# would need a very big table in memory; 500 rows at a time is small.
CHUNK_SIZE = 500


# ---------------------------------------------------------------------------
# 1. Normalize
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Make a message "canonical", so near-copies look exactly the same.

    "WIN £1000! Call 0800-123 now: http://x.com/a" -> "win 0000 call 0000 now <url>"
    """
    # Turkish capital "İ" would become "i" + an extra dot mark with .lower().
    # We turn it into a plain "i" first.
    text = text.replace("İ", "i").lower()
    # Links -> one placeholder word. Two spam copies with different links
    # are still the same spam.
    text = URL_RE.sub(f" {URL_TOKEN} ", text)
    # Every digit -> "0". Phone numbers, prices and codes change in every copy.
    text = re.sub(r"\d", "0", text)
    # Punctuation -> space. [^\w\s] = "not a letter/digit and not a space".
    text = re.sub(r"[^\w\s]", " ", text)
    # Many spaces / line breaks -> one space, and none at the start or end.
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(rf"\b{URL_TOKEN}\b", "<url>", text)


# ---------------------------------------------------------------------------
# 2. Similarity
# ---------------------------------------------------------------------------


def fit_vectorizer(messages: list[str]) -> TfidfVectorizer:
    """Char TF-IDF on NORMALIZED text (same letter pieces as the B_char model).

    We fit it on ALL messages of one language. Both the leakage check and
    the clustering use this same setup, so they always agree about which
    pairs are "near-duplicates".
    """
    ngram_range = tuple(get_config()["model"]["char_ngram_range"])
    return TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range).fit(messages)


def best_matches(X_query, X_pool, skip_self: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """For every row of X_query: the most similar row of X_pool.

    TF-IDF rows have length 1, so "row @ row" is the cosine similarity
    (1.0 = same letter pieces, 0.0 = nothing in common).
    Returns (index of best match, its similarity).
    """
    best_idx, best_sim = [], []
    for start in range(0, X_query.shape[0], CHUNK_SIZE):
        sims = (X_query[start : start + CHUNK_SIZE] @ X_pool.T).toarray()
        if skip_self:
            # Comparing a set with itself: a row is always 100% similar to itself.
            rows = np.arange(sims.shape[0])
            sims[rows, rows + start] = -1.0
        best_idx.append(sims.argmax(axis=1))
        best_sim.append(sims.max(axis=1))
    return np.concatenate(best_idx), np.concatenate(best_sim)


def check_leakage(train_messages, test_messages, threshold: float) -> pd.DataFrame:
    """One row per TEST message: its closest train message and how close it is.

    Columns: test, train (closest train message), similarity,
             exact (same normalized text), near (similarity >= threshold OR exact).
    """
    train_list, test_list = list(train_messages), list(test_messages)
    train_norm = [normalize_text(m) for m in train_list]
    test_norm = [normalize_text(m) for m in test_list]

    vectorizer = fit_vectorizer(train_norm + test_norm)
    idx, sim = best_matches(vectorizer.transform(test_norm), vectorizer.transform(train_norm))

    # A set gives a fast "is this text anywhere in train?" lookup.
    train_texts = set(train_norm)
    exact = np.array([t in train_texts for t in test_norm])
    return pd.DataFrame(
        {
            "test": test_list,
            "train": [train_list[i] for i in idx],
            "similarity": sim,
            "exact": exact,
            # "exact" is also "near": two messages that are empty after
            # normalizing ("!!!" and "...") have no letter pieces, so their
            # similarity is 0, but they are still the same text.
            "near": exact | (sim >= threshold),
        }
    )


# ---------------------------------------------------------------------------
# 3. Clusters (used by data.py for the leakage-safe split)
# ---------------------------------------------------------------------------


def cluster_ids(messages, threshold: float) -> np.ndarray:
    """Give every message a cluster number. Near-duplicates get the SAME number.

    Think of a graph: each message is a dot; we draw a line between two dots
    if their similarity is >= threshold (or their normalized text is equal).
    A cluster = all dots you can reach by following lines. So if A~B and
    B~C, then A, B and C are one cluster, even if A and C are less similar.
    """
    norm = [normalize_text(m) for m in messages]
    n = len(norm)
    X = fit_vectorizer(norm).transform(norm)

    # Lines from similarity, found chunk by chunk (memory stays small).
    rows, cols = [], []
    for start in range(0, n, CHUNK_SIZE):
        sims = (X[start : start + CHUNK_SIZE] @ X.T).toarray()
        r, c = np.nonzero(sims >= threshold)
        rows.append(r + start)
        cols.append(c)

    # Lines for exact normalized copies: link each message to the FIRST
    # message with the same text. pd.factorize gives one code per text.
    codes, _ = pd.factorize(pd.Series(norm))
    first_of_code = pd.Series(np.arange(n)).groupby(codes).transform("min").to_numpy()
    rows.append(np.arange(n))
    cols.append(first_of_code)

    r, c = np.concatenate(rows), np.concatenate(cols)
    graph = csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    # directed=False: a line from A to B also goes from B to A.
    _, labels = connected_components(graph, directed=False)
    return labels


# ---------------------------------------------------------------------------
# 4. The report
# ---------------------------------------------------------------------------


def short(text: str, width: int = 110) -> str:
    """One line, no "|" (it would break the Markdown table), cut if too long."""
    text = re.sub(r"\s+", " ", text).replace("|", "/")
    return text if len(text) <= width else text[: width - 3] + "..."


def summarize(result: pd.DataFrame, labels: pd.Series) -> dict:
    """Counts and percents for one language and one split."""
    n = len(result)
    labels = labels.reset_index(drop=True)
    return {
        "n_test": n,
        "exact": int(result["exact"].sum()),
        "exact_pct": 100 * result["exact"].mean(),
        "near": int(result["near"].sum()),
        "near_pct": 100 * result["near"].mean(),
        # Leak share inside each class: is spam leaking more than ham?
        "near_pct_spam": 100 * result["near"][labels == "spam"].mean(),
        "near_pct_ham": 100 * result["near"][labels == "ham"].mean(),
    }


def example_rows(result: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    """Pick n leaked pairs at random (fixed seed = same examples every run)."""
    leaked = result[result["near"]]
    return leaked.sample(n=min(n, len(leaked)), random_state=random_state)


def write_report(path, sections: dict, threshold: float, n_examples: int) -> None:
    """sections: {lang: {"old": (summary, examples), "clean": (summary, examples)}}"""
    lines = [
        "# Near-Duplicate Leakage Report",
        "",
        "For every TEST message we look for the most similar TRAIN message.",
        "",
        "- **exact**: same text after normalizing (lowercase, links -> `<url>`, "
        "digits -> `0`, no punctuation, single spaces).",
        f"- **near**: char TF-IDF cosine similarity >= {threshold} (includes exact).",
        "- **old split**: the random stratified split used up to Step 6 "
        "(rebuilt here with the same seed).",
        "- **clean split**: Step 7a split by near-duplicate cluster "
        "(a cluster is never in both train and test).",
        "",
        "## Summary",
        "",
        "| Language | Split | Test rows | Exact dupes | Exact % | Near dupes | Near % "
        "| Near % (spam) | Near % (ham) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for lang, parts in sections.items():
        for split_name, (s, _) in parts.items():
            lines.append(
                f"| {lang} | {split_name} | {s['n_test']} | {s['exact']} | {s['exact_pct']:.1f} "
                f"| {s['near']} | {s['near_pct']:.1f} | {s['near_pct_spam']:.1f} "
                f"| {s['near_pct_ham']:.1f} |"
            )
    for lang, parts in sections.items():
        _, examples = parts["old"]
        lines += [
            "",
            f"## Examples: {lang}, old split ({n_examples} random leaked pairs)",
            "",
            "| Similarity | Exact | Test message | Closest train message |",
            "|---|---|---|---|",
            *[
                f"| {r.similarity:.3f} | {'yes' if r.exact else 'no'} "
                f"| {short(r.test)} | {short(r.train)} |"
                for r in examples.itertuples()
            ],
        ]
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def read_csv(path) -> pd.DataFrame:
    # keep_default_na=False: keep text like "NA" as text (see train.py).
    return pd.read_csv(path, encoding="utf-8", keep_default_na=False)


def main() -> None:
    # Imported here, not at the top: data.py imports THIS file (for
    # cluster_ids). Importing data.py at the top too would be a circle.
    from src.sms_fraud.data import split_data

    cfg = get_config()
    data_cfg, leak_cfg = cfg["data"], cfg["leakage"]
    threshold = leak_cfg["similarity_threshold"]
    n_examples = leak_cfg["n_examples"]
    random_state = cfg["split"]["random_state"]
    processed = project_path(data_cfg["processed_dir"])

    # (all clean rows, clean train, clean test) for each language.
    files = {
        "en": (data_cfg["english_file"], processed / "train.csv", processed / "test.csv"),
        "tr": (
            data_cfg["turkish_file"],
            data_cfg["turkish_train_file"],
            data_cfg["turkish_test_file"],
        ),
    }
    sections = {}
    for lang, (all_file, train_file, test_file) in files.items():
        # OLD split: split_data() without groups is the random split used up
        # to Step 6. Same rows, same order, same seed -> the same split.
        old_train, old_test = split_data(read_csv(project_path(all_file)))
        clean_train, clean_test = (
            read_csv(project_path(train_file)),
            read_csv(project_path(test_file)),
        )

        sections[lang] = {}
        for split_name, (train, test) in {
            "old": (old_train, old_test),
            "clean": (clean_train, clean_test),
        }.items():
            result = check_leakage(train["message"], test["message"], threshold)
            summary = summarize(result, test["label"])
            sections[lang][split_name] = (summary, example_rows(result, n_examples, random_state))
            logger.info(
                "%s %-5s: %d test rows | exact %d (%.1f%%) | near %d (%.1f%%)",
                lang,
                split_name,
                summary["n_test"],
                summary["exact"],
                summary["exact_pct"],
                summary["near"],
                summary["near_pct"],
            )

    report = project_path(leak_cfg["report_path"])
    write_report(report, sections, threshold, n_examples)
    logger.info("Saved %s", report)


if __name__ == "__main__":
    main()
