"""Small text statistics shared by the API and the monitoring code.

This file has NO heavy imports (no pandas, scipy, matplotlib), so the small
API Docker image can use it too.
"""

import re

# ---------------------------------------------------------------------------
# Language detection (Step 7a): "en" or "tr"
# ---------------------------------------------------------------------------

# Letters that exist ONLY in Turkish (not in English or German). One of
# them is enough. "ö" and "ü" are NOT here: the English UCI data uses "ü"
# as a short form of "you" ("how ü doing"), so they only add one vote below.
TURKISH_ONLY_LETTERS = set("çğışÇĞİŞ")
TURKISH_VOTE_LETTERS = set("öüÖÜ")

# Very common short words. Many Turkish SMS are typed WITHOUT Turkish
# letters ("icin" instead of "için"), so we also need words. We keep both
# spellings. Words that also exist in English (like "a", "o", "sms") are left out.
TURKISH_STOPWORDS = {
    "ve", "bu", "bir", "icin", "ile", "da", "de", "mi", "mu", "ne", "cok", "ama",
    "gibi", "var", "yok", "ben", "sen", "biz", "siz", "evet", "hayir", "tamam",
    "degil", "daha", "kadar", "simdi", "nasil", "olarak", "size", "sizi", "hemen",
    "tl", "lira", "her", "abi", "yarin", "bugun", "iyi", "misin", "musun", "neden",
    "veya", "ucretsiz", "indirim", "kampanya", "sizin", "bizim", "merhaba", "selam",
    "tamamdir", "tesekkurler", "sagol", "canim", "ay", "gun", "hafta", "sonra", "en",
    "yeni", "ozel", "ilk", "bilgi", "iptal", "yazip",
}  # fmt: skip
ENGLISH_STOPWORDS = {
    "the", "to", "you", "and", "is", "a", "i", "it", "in", "for", "your", "my",
    "me", "of", "on", "have", "are", "be", "at", "this", "that", "with", "u",
    "ur", "we", "not", "will", "can", "do", "so", "but", "get", "call", "now",
    "free", "just", "what", "how", "if", "or", "no", "ok", "go", "from", "all",
}  # fmt: skip


# PATTERN: Strategy — language detection is one small pure function. The API and monitoring only
#   call detect_language(), so a better detector (e.g. a real model) can replace it in one place.
def detect_language(message: str) -> str:
    """Guess the language of one message: "tr" or "en".

    Simple rules (no model, no extra package, very fast):
    1. Any Turkish-only letter (ç, ğ, ı, ş, İ) -> "tr".
    2. Otherwise count votes: each Turkish stopword and each "ö"/"ü" is one
       Turkish vote, each English stopword one English vote.
       More Turkish votes -> "tr".
    3. A tie (also: no known word at all, like ":)") -> "en".
    Not perfect, but good enough to send each message to the right
    monitoring window. Its accuracy is in reports/monitoring/drift_report.md.
    """
    if any(ch in TURKISH_ONLY_LETTERS for ch in message):
        return "tr"
    # Turkish adds suffixes after an apostrophe: "7889'a", "Ankara'da".
    # Drop them, so "a" is not counted as the English word "a".
    words = re.findall(r"\w+", re.sub(r"['’]\w+", "", message.lower()))
    turkish = sum(word in TURKISH_STOPWORDS for word in words)
    turkish += sum(ch in TURKISH_VOTE_LETTERS for ch in message)
    english = sum(word in ENGLISH_STOPWORDS for word in words)
    return "tr" if turkish > english else "en"


def find_vectorizer(pipeline):
    """Return the TF-IDF vectorizer we use to measure unknown words.

    Since Step 5b the "tfidf" step of a model can be one of two things:
    - one TfidfVectorizer (word OR char features), or
    - a FeatureUnion (word AND char features side by side).
    For a FeatureUnion we pick the WORD part. Why? "Share of unknown words"
    is easy to understand and stays comparable with older reports.
    """
    step = pipeline.named_steps["tfidf"]
    # A FeatureUnion keeps its parts in "transformer_list": [(name, vectorizer), ...].
    parts = dict(getattr(step, "transformer_list", []))
    if parts:
        return parts.get("word", next(iter(parts.values())))
    return step


def oov_ratio(vectorizer, message: str) -> float:
    """Share of words in the message that the model does NOT know (0.0 - 1.0).

    OOV = "out of vocabulary". The TF-IDF step only knows the words it saw
    during training (its "vocabulary"). Unknown words are simply ignored by
    the model. So if most words are unknown, the model is almost blind,
    and its prediction cannot be trusted.

    Example: a model trained on English sees "Bu mesaja tiklayiniz"
    -> 3 words, 0 known -> oov_ratio = 1.0

    For a char model (analyzer "char" / "char_wb") the vocabulary holds
    letter pieces, not words. Then we count unknown letter PIECES instead.
    """
    if vectorizer.analyzer == "word":
        # Use the SAME text cleaning and word splitting as the vectorizer, so
        # we count words exactly the way the model sees them.
        preprocess = vectorizer.build_preprocessor()  # e.g. makes text lowercase
        tokenize = vectorizer.build_tokenizer()  # splits text into words
        units = tokenize(preprocess(message))
    else:
        # build_analyzer() gives the letter pieces exactly as the model sees them.
        units = vectorizer.build_analyzer()(message)

    # A message with no words at all (e.g. "!!!") gives the model nothing it
    # knows, so we treat it as fully unknown.
    if not units:
        return 1.0

    vocabulary = vectorizer.vocabulary_
    unknown = sum(1 for unit in units if unit not in vocabulary)
    return unknown / len(units)
