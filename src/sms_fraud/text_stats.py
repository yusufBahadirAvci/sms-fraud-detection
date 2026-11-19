"""Small text statistics shared by the API and the monitoring code.

This file has NO heavy imports (no pandas, scipy, matplotlib), so the small
API Docker image can use it too.
"""


def oov_ratio(vectorizer, message: str) -> float:
    """Share of words in the message that the model does NOT know (0.0 - 1.0).

    OOV = "out of vocabulary". The TF-IDF step only knows the words it saw
    during training (its "vocabulary"). Unknown words are simply ignored by
    the model. So if most words are unknown, the model is almost blind,
    and its prediction cannot be trusted.

    Example: a model trained on English sees "Bu mesaja tiklayiniz"
    -> 3 words, 0 known -> oov_ratio = 1.0
    """
    # Use the SAME text cleaning and word splitting as the vectorizer, so we
    # count words exactly the way the model sees them.
    preprocess = vectorizer.build_preprocessor()  # e.g. makes text lowercase
    tokenize = vectorizer.build_tokenizer()  # splits text into words
    words = tokenize(preprocess(message))

    # A message with no words at all (e.g. "!!!") gives the model nothing it
    # knows, so we treat it as fully unknown.
    if not words:
        return 1.0

    vocabulary = vectorizer.vocabulary_
    unknown = sum(1 for word in words if word not in vocabulary)
    return unknown / len(words)
